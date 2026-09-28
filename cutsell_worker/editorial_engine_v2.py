"""Experimental whole-video editorial authority for CutSell.

V2 deliberately has one semantic decision seam.  Perception and candidate
construction happen upstream; the whole-video reasoner then settles the full
candidate universe.  Once that plan is accepted, only physical Boundary work
may run.  Provider failure is fail-closed because silently falling back to the
legacy rule chain would make an A/B run impossible to interpret.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
import re
import unicodedata
from typing import Callable

from .contracts import DraftClip, ProcessingResult
from .selection_boundary_contract import (
    enforce_selection_contract,
    freeze_selection_contract,
)
from .unified_selection_reasoner import (
    UnifiedSelectionReasoner,
    apply_unified_selection_reasoner,
)


SCHEMA_VERSION = "cutsell.editorial_engine_v2.v1"
ENV_FLAG = "CUTSELL_EDITORIAL_ENGINE_V2"

PHASES = (
    "perceive_complete_source",
    "understand_story_and_attempts",
    "reconstruct_attempts",
    "clean_cut_and_best_take",
    "compose_story",
    "resolve_keep_discard",
    "freeze_selection",
    "execute_boundaries",
    "post_render_review",
)


def _true(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def enabled(explicit: bool | None = None) -> bool:
    return _true(os.environ.get(ENV_FLAG, "0")) if explicit is None else bool(explicit)


def _candidate_ids(draft) -> tuple[str, ...]:
    return tuple(sorted({
        str(clip.clip_id)
        for clip in (*tuple(draft.selected), *tuple(draft.alternates), *tuple(draft.discarded))
    }))


def _discard_signature(draft) -> str:
    values = sorted(str(clip.clip_id) for clip in tuple(draft.discarded))
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)


def _ordered_semantic_signature(draft) -> str:
    """Seal V2 story order as well as content; legacy Freeze sorts by source."""
    tokens = []
    for clip in tuple(draft.selected):
        for token in _TOKEN_RE.findall(str(clip.text or "")):
            raw = unicodedata.normalize("NFKD", token.casefold())
            tokens.append("".join(ch for ch in raw if not unicodedata.combining(ch)))
    return hashlib.sha256("\x1f".join(tokens).encode("utf-8")).hexdigest()


def _require_whole_video_evidence(draft) -> dict:
    diagnostics = dict(getattr(draft, "diagnostics", None) or {})
    whole = diagnostics.get("whole_video_context") or {}
    status = (whole.get("status") or {}).get("status")
    av_status = whole.get("audiovisual_input_status")
    if status not in {"ok", "available", "complete", "success", "applied"}:
        raise RuntimeError("Editorial Engine V2 requires available whole-video context")
    if av_status != "received_and_parsed":
        raise RuntimeError("Editorial Engine V2 requires verified audiovisual Watch + Listen input")
    sources = tuple(whole.get("sources") or ())
    if not sources or any(not str(source.get("audiovisual_evidence") or "") for source in sources):
        raise RuntimeError("Editorial Engine V2 requires audiovisual evidence for every source")
    return whole


def _fold_alternates(draft):
    """V2 has a final KEEP/DISCARD decision, never a hidden third bucket."""
    discarded_by_id = {str(clip.clip_id): replace(clip, selected=False) for clip in draft.discarded}
    for clip in draft.alternates:
        discarded_by_id.setdefault(str(clip.clip_id), replace(clip, selected=False))
    discarded = tuple(sorted(
        discarded_by_id.values(),
        key=lambda clip: (clip.source_order, float(clip.start), float(clip.end), clip.clip_id),
    ))
    return replace(draft, alternates=(), discarded=discarded)


def _add_focused_visual_action_candidates(draft, whole: dict):
    """Expose independently observed wordless operations to the one selector.

    The AV provider must have intersected a local visual observation with
    objective source silence. This method does not select footage; the same
    whole-video reasoner sees the action candidate alongside speech, and can
    discard it if the operation is redundant or interrupts the story.
    """
    existing = {clip.clip_id for clip in (*draft.selected, *draft.alternates, *draft.discarded)}
    order = {clip.source_asset_id: clip.source_order for clip in
             (*draft.selected, *draft.alternates, *draft.discarded)}
    candidates, audit = [], []
    for source in whole.get('sources') or ():
        source_id = str(source.get('source_asset_id') or '')
        try:
            evidence = json.loads(str(source.get('audiovisual_evidence') or '{}'))
        except (ValueError, TypeError):
            continue
        for action in evidence.get('focused_silent_visual_actions') or ():
            try:
                start, end = float(action['start']), float(action['end'])
                silent_start, silent_end = (float(action['measured_silence_start']),
                                            float(action['measured_silence_end']))
                seen_start, seen_end = float(action['observed_start']), float(action['observed_end'])
                confidence = float(action['confidence'])
            except (KeyError, TypeError, ValueError):
                continue
            if (source_id not in order or action.get('basis') !=
                    'focused_av_action_intersect_source_measured_silence'
                    or not re.fullmatch(r'[0-9a-f]{64}', str(action.get('source_sha256') or ''))
                    or evidence.get('source_sha256') != action.get('source_sha256')
                    or not .85 <= confidence <= 1.0
                    or not all(map(lambda x: x >= 0 and x < float('inf'),
                                   (start, end, silent_start, silent_end, seen_start, seen_end)))
                    or not 1.5 <= end - start <= 18.0
                    or not (silent_start <= start < end <= silent_end
                            and seen_start <= start < end <= seen_end)):
                continue
            clip_id = 'v2_visual_' + hashlib.sha256(
                f'{source_id}\x1f{start:.3f}\x1f{end:.3f}\x1f{action["source_sha256"]}'.encode()
            ).hexdigest()[:20]
            if clip_id in existing:
                continue
            existing.add(clip_id)
            candidates.append(DraftClip(
                clip_id=clip_id, source_asset_id=source_id, source_order=order[source_id],
                start=start, end=end, text='', caption_text='', words=(),
                selected=False, audio_muted=True,
            ))
            audit.append({'clip_id': clip_id, 'source_asset_id': source_id,
                          'start': start, 'end': end, 'confidence': confidence,
                          'visual_observation': str(action.get('visual_observation') or '')[:240]})
    diagnostics = dict(draft.diagnostics or {})
    diagnostics['v2_focused_visual_action_candidates'] = audit
    return replace(draft, discarded=(*draft.discarded, *candidates), diagnostics=diagnostics)


def _run_speech_boundary_preserving_visual_actions(result, boundary):
    """Do not feed silent action footage to ASR word-envelope/silence trimmers.

    The same Boundary stage still operates on every spoken selection. Insert
    the explicit visual scenes at their pre-stage story positions, unchanged;
    fail closed if a speech fragment cannot be attributed to its parent.
    """
    original = tuple(result.draft.selected)
    visuals = {clip.clip_id for clip in original
               if clip.audio_muted and not clip.words and not clip.text.strip()}
    if not visuals:
        return boundary(result)
    spoken = tuple(clip for clip in original if clip.clip_id not in visuals)
    if not spoken:
        return result
    processed = boundary(replace(result, draft=replace(result.draft, selected=spoken)))
    by_parent: dict[str, list] = {clip.clip_id: [] for clip in spoken}
    for clip in processed.draft.selected:
        parent = (clip.parent_semantic_clip_id
                  if clip.parent_semantic_clip_id in by_parent else clip.clip_id)
        if parent not in by_parent:
            raise RuntimeError('Boundary produced a speech fragment with unknown V2 parent')
        by_parent[parent].append(clip)
    if any(not fragments for fragments in by_parent.values()):
        raise RuntimeError('Boundary removed a frozen spoken selection')
    merged = tuple(part for original_clip in original
                   for part in ([original_clip] if original_clip.clip_id in visuals
                                else by_parent[original_clip.clip_id]))
    return replace(processed, draft=replace(processed.draft, selected=merged))


def _reconcile_spoken_visual_source_overlap(result: ProcessingResult, whole: dict) -> ProcessingResult:
    """Cut a verified silent action at its speech boundary before Freeze.

    An AV action candidate may begin inside a speech candidate's generous
    trailing handle. Only measured source silence and complete word timing
    authorize shortening that handle. Never remove a word to make QC pass.
    """
    selected = list(result.draft.selected)
    verified = {str(row.get('clip_id')): row for row in
                (result.draft.diagnostics or {}).get('v2_focused_visual_action_candidates', ())}
    events = {str(source.get('source_asset_id')): tuple(source.get('events') or ())
              for source in whole.get('sources') or ()}
    audit = []
    for index in range(1, len(selected)):
        spoken, visual = selected[index - 1:index + 1]
        if (visual.clip_id not in verified or not visual.audio_muted or visual.words or visual.text.strip()
                or spoken.audio_muted or not spoken.words or not spoken.text.strip()
                or spoken.source_asset_id != visual.source_asset_id
                or spoken.source_order != visual.source_order
                or not spoken.start < visual.start < spoken.end <= visual.end):
            continue
        measured = any(isinstance(event, dict)
                       and event.get('kind') == 'audio_silence_interval'
                       and float(event.get('start') or -1) <= visual.start + .06
                       and float(event.get('end') or -1) >= spoken.end - .06
                       for event in events.get(spoken.source_asset_id, ()))
        last_word_end = max(word.end for word in spoken.words)
        if not measured or last_word_end > visual.start + 1e-6:
            continue
        selected[index - 1] = replace(spoken, end=visual.start)
        audit.append({'spoken_clip_id': spoken.clip_id, 'visual_clip_id': visual.clip_id,
                      'old_spoken_end': round(spoken.end, 3),
                      'new_spoken_end': round(visual.start, 3),
                      'last_word_end': round(last_word_end, 3),
                      'basis': 'verified_visual_source_silence_after_complete_speech'})
    if not audit:
        return result
    diagnostics = dict(result.draft.diagnostics or {})
    diagnostics['v2_spoken_visual_overlap_reconciliation'] = audit
    return replace(result, draft=replace(result.draft, selected=tuple(selected), diagnostics=diagnostics))


def _coalesce_overlapping_selected_speech(draft):
    """Represent overlapping selections as one continuous source-word delivery.

    Both candidates may contain unique leading/trailing words; do not choose
    one based on length. Only coalesce if their shared source words agree and
    their union covers the complete spoken interval without a disputed word.
    Otherwise leave the hard Boundary/Freeze failure in place for review.
    """
    selected = list(draft.selected)
    audit = []
    index = 0
    while index < len(selected) - 1:
        first, second = selected[index:index + 2]
        left, right = sorted((first, second), key=lambda clip: (clip.start, clip.end))
        if (left.source_asset_id != right.source_asset_id or
                left.source_order != right.source_order or
                not left.start < right.start < left.end or right.end <= left.end or
                not left.words or not right.words):
            index += 1
            continue
        def keys(words):
            return [(round(w.start, 3), round(w.end, 3), w.text.casefold()) for w in words]
        if (" ".join(w.text.strip() for w in left.words).casefold().split() != left.text.casefold().split() or
                " ".join(w.text.strip() for w in right.words).casefold().split() != right.text.casefold().split()):
            index += 1
            continue
        left_overlap = {key for word, key in zip(left.words, keys(left.words))
                        if word.end > right.start and word.start < left.end}
        right_overlap = {key for word, key in zip(right.words, keys(right.words))
                         if word.end > right.start and word.start < left.end}
        if not left_overlap or left_overlap != right_overlap:
            index += 1
            continue
        by_key = {key: word for word, key in zip(left.words, keys(left.words))}
        by_key.update(zip(keys(right.words), right.words))
        joined = tuple(by_key[key] for key in sorted(by_key))
        if (not joined or joined[0].start > left.start + .1 or joined[-1].end < right.end - .1):
            index += 1
            continue
        text = " ".join(word.text.strip() for word in joined).strip()
        selected[index:index + 2] = [replace(
            left, end=right.end, words=joined, text=text, caption_text=text,
            word_indices=tuple(sorted(set((*left.word_indices, *right.word_indices)))),
        )]
        audit.append({"left_clip_id": left.clip_id, "right_clip_id": right.clip_id,
                      "start": round(left.start, 3), "end": round(right.end, 3),
                      "action": "source_word_union"})
    diagnostics = dict(draft.diagnostics or {})
    diagnostics["v2_overlapping_selection_word_union"] = audit
    return replace(draft, selected=tuple(selected), diagnostics=diagnostics)


def _audience_regions(whole: dict) -> dict[str, tuple[tuple[float, float, bool], ...]]:
    """Return high-confidence AV audience spans, failing closed on bad evidence."""
    output: dict[str, tuple[tuple[float, float], ...]] = {}
    for source in tuple(whole.get("sources") or ()):
        source_id = str(source.get("source_asset_id") or "")
        try:
            evidence = json.loads(str(source.get("audiovisual_evidence") or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        regions = []
        for region in evidence.get("regions") or ():
            if not isinstance(region, dict) or region.get("role") != "audience":
                continue
            try:
                start, end = float(region["start"]), float(region["end"])
                confidence = float(region.get("confidence", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            if end > start and confidence >= 0.70:
                description = " ".join(str(region.get(key) or "") for key in (
                    "audio_observation", "visual_observation", "reason",
                )).casefold()
                demonstration = any(token in description for token in (
                    "demonstrat", "show", "display", "mix", "pour", "scoop", "apply", "use the product",
                    "prepar", "instruction", "bottle", "container", "product",
                ))
                operation = any(token in description for token in (
                    "mix", "pour", "scoop", "stir", "blend", "apply", "unbox", "assemble",
                    "mezcla", "mezclando", "vertiendo", "aplicando",
                ))
                regions.append((start, end, demonstration, operation))
        if source_id and regions:
            output[source_id] = tuple(regions)
    return output


def _restore_safe_audience_continuity(result: ProcessingResult, whole: dict) -> ProcessingResult:
    """Keep short in-take AV gaps when Selection approved both surrounding pieces.

    Candidate segmentation is allowed to split a polished delivery, but those
    physical splits must not become destructive edits.  Rejoin only a short,
    chronological gap wholly verified as audience delivery and containing no
    explicitly discarded candidate.  This remains Boundary work: membership,
    text, IDs, and story order do not change.
    """
    selected = list(result.draft.selected)
    audience_by_source = _audience_regions(whole)
    discarded = tuple(result.draft.discarded)
    rows = []
    reasoner_rows = {
        str(row.get("clip_id")): row
        for row in ((result.draft.diagnostics or {}).get("unified_selection_reasoner") or {}).get("decisions", ())
        if isinstance(row, dict)
    }
    source_events = {str(source.get("source_asset_id") or ""): tuple(source.get("events") or ())
                     for source in whole.get("sources") or () if isinstance(source, dict)}
    for index in range(len(selected) - 1):
        left, right = selected[index], selected[index + 1]
        if (left.audio_muted and not left.words and not left.text.strip()
                or right.audio_muted and not right.words and not right.text.strip()):
            continue
        if left.source_asset_id != right.source_asset_id or left.source_order != right.source_order:
            continue
        gap_start, gap_end = float(left.end), float(right.start)
        gap = gap_end - gap_start
        if gap <= 1e-6 or gap > 8.0:
            continue
        # A broad AV audience label cannot overrule measured dead air. Do
        # not stretch selected speech across a source silence that QC would
        # reject after render, even if both neighboring takes are selected.
        containing = [region for region in audience_by_source.get(left.source_asset_id, ())
                      if region[0] <= gap_start + 1e-6 and region[1] >= gap_end - 1e-6]
        if not containing:
            continue
        measured_dead_air = any(
            isinstance(event, dict) and event.get("kind") == "audio_silence_interval"
            and min(gap_end, float(event.get("end") or 0)) -
            max(gap_start, float(event.get("start") or 0)) >= 1.20
            for event in source_events.get(left.source_asset_id, ()))
        visually_observed_operation = any(region[3] for region in containing)
        if measured_dead_air and not visually_observed_operation:
            continue
        # Ordinary speech gaps stay tightly bounded. A longer bridge is safe
        # only when Watch + Listen explicitly observed an audience-facing
        # product demonstration: action-only footage between approved spoken
        # instructions is part of the story, not dead air.
        if gap > 2.25 and not any(region[2] for region in containing):
            continue
        if gap > 2.25:
            left_decision = reasoner_rows.get(str(left.clip_id), {})
            right_decision = reasoner_rows.get(str(right.clip_id), {})
            continuous_demo = left_decision.get("safety_override") == "av_continuous_demonstration_preserved"
            if not continuous_demo and (str(left_decision.get("relation") or "").startswith("retry_") or str(
                right_decision.get("relation") or ""
            ).startswith("retry_")):
                continue
            left_tokens = {token.casefold() for token in _TOKEN_RE.findall(str(left.text or "")) if len(token) >= 4}
            right_tokens = {token.casefold() for token in _TOKEN_RE.findall(str(right.text or "")) if len(token) >= 4}
            if not left_tokens.intersection(right_tokens):
                continue
        if any(item.source_asset_id == left.source_asset_id
               and float(item.start) < gap_end - 1e-6
               and float(item.end) > gap_start + 1e-6
               for item in discarded):
            continue
        selected[index] = replace(left, end=gap_end)
        rows.append({
            "left_clip_id": left.clip_id,
            "source_asset_id": left.source_asset_id,
            "right_clip_id": right.clip_id,
            "gap_start": round(gap_start, 3),
            "gap_end": round(gap_end, 3),
            "restored_sec": round(gap, 3),
            "basis": (
                "selected_neighbors_inside_high_confidence_audience_demonstration"
                if gap > 2.25 else
                "selected_neighbors_inside_high_confidence_audience_region"
            ),
        })
    if not rows:
        return result
    diagnostics = dict(result.draft.diagnostics or {})
    diagnostics["editorial_engine_v2_continuity_restoration"] = rows
    return replace(result, draft=replace(result.draft, selected=tuple(selected), diagnostics=diagnostics))


def run_editorial_engine_v2(
    result: ProcessingResult,
    *,
    selection_reasoner: UnifiedSelectionReasoner | None,
    recover_complete_boundaries: Callable[[ProcessingResult], ProcessingResult],
    execute_boundaries: Callable[[ProcessingResult], ProcessingResult],
) -> ProcessingResult:
    """Resolve once, freeze once, and reject every post-freeze semantic mutation."""
    if selection_reasoner is None:
        raise RuntimeError("Editorial Engine V2 requires a whole-video selection reasoner")

    whole = _require_whole_video_evidence(result.draft)
    result = replace(result, draft=_add_focused_visual_action_candidates(result.draft, whole))
    candidate_ids = _candidate_ids(result.draft)
    if not candidate_ids:
        raise RuntimeError("Editorial Engine V2 received no editorial candidates")

    request_diagnostics = dict(result.draft.diagnostics or {})
    request_diagnostics["editorial_engine_v2_request"] = {
        "schema_version": SCHEMA_VERSION,
        "require_audiovisual_evidence": True,
        "allow_global_story_reordering": True,
    }
    request_draft = replace(result.draft, diagnostics=request_diagnostics)
    resolved = apply_unified_selection_reasoner(request_draft, selection_reasoner)
    reasoner_diag = dict((resolved.diagnostics or {}).get("unified_selection_reasoner") or {})
    if reasoner_diag.get("status") != "applied":
        detail = reasoner_diag.get("error") or reasoner_diag.get("status") or "unknown"
        raise RuntimeError(f"Editorial Engine V2 whole-video plan failed: {detail}")
    if _candidate_ids(resolved) != candidate_ids:
        raise RuntimeError("Editorial Engine V2 plan changed the candidate universe")

    resolved = _fold_alternates(resolved)
    diagnostics = dict(resolved.diagnostics or {})
    diagnostics["editorial_engine_v2"] = {
        "schema_version": SCHEMA_VERSION,
        "status": "selection_resolved",
        "authority": "whole_video_reasoner_single_semantic_seam",
        "whole_video_source_count": len(tuple(whole.get("sources") or ())),
        "candidate_count": len(candidate_ids),
        "selected_count": len(tuple(resolved.selected)),
        "discarded_count": len(tuple(resolved.discarded)),
        "phases": [
            {"phase": phase, "status": "complete" if phase != "post_render_review" else "pending_render"}
            for phase in PHASES
        ],
        "post_freeze_semantic_mutation_allowed": False,
        "repair_policy": "localized_family_reopen_only_after_blocking_post_render_review",
    }
    result = replace(result, draft=replace(resolved, diagnostics=diagnostics))

    # Complete source-proven word edges before the semantic phase barrier.
    result = _run_speech_boundary_preserving_visual_actions(result, recover_complete_boundaries)
    result = replace(result, draft=_fold_alternates(result.draft))
    result = replace(result, draft=_coalesce_overlapping_selected_speech(result.draft))
    result = _reconcile_spoken_visual_source_overlap(result, whole)
    result = _restore_safe_audience_continuity(result, whole)
    discard_signature = _discard_signature(result.draft)
    ordered_semantic_signature = _ordered_semantic_signature(result.draft)

    result = replace(result, draft=freeze_selection_contract(result.draft))
    frozen_selected = tuple(result.draft.selected)
    result = _run_speech_boundary_preserving_visual_actions(result, execute_boundaries)
    # Boundary polish may rebuild a clip from its spoken-word envelope and
    # unintentionally erase an already-approved action-only demonstration
    # bridge. Reassert the same evidence-gated physical continuity after the
    # last Boundary mutation; semantic membership and text remain frozen.
    result = _restore_safe_audience_continuity(result, whole)
    try:
        result = replace(result, draft=enforce_selection_contract(result.draft))
    except RuntimeError as exc:
        # Preserve replayable source words and physical-operation audits. Do
        # not weaken Freeze, restore text over changed media, or auto-approve.
        diagnostic_keys = (
            "final_boundary_authority",
            "post_selection_edge_only_boundary", "post_selection_interior_gap_trim",
            "post_selection_interior_gap_trace", "boundary_engine_pass",
            "human_boundary_polish", "editorial_engine_v2_continuity_restoration",
            "selection_boundary_contract", "v2_recording_tail",
            "unified_selection_reasoner", "v2_take_competitions", "v2_overlapping_selection_word_union",
        )
        exc.boundary_failure_evidence = {
            "schema_version": "cutsell.v2.boundary_failure.v1",
            "before": [asdict(clip) for clip in frozen_selected],
            "after": [asdict(clip) for clip in result.draft.selected],
            "diagnostics": {key: result.draft.diagnostics[key] for key in diagnostic_keys
                            if key in result.draft.diagnostics},
        }
        exc.boundary_failure_evidence["diagnostics"]["attempt_reconstruction"] = {
            "positioned_performance_evidence": (result.draft.diagnostics.get("attempt_reconstruction") or {}).get(
                "positioned_performance_evidence", []),
        }
        raise

    if tuple(result.draft.alternates):
        raise RuntimeError("Editorial Engine V2 post-freeze stage recreated alternates")
    if _discard_signature(result.draft) != discard_signature:
        raise RuntimeError("Editorial Engine V2 post-freeze stage changed DISCARD membership")
    if _ordered_semantic_signature(result.draft) != ordered_semantic_signature:
        raise RuntimeError("Editorial Engine V2 post-freeze stage changed story order")

    diagnostics = dict(result.draft.diagnostics or {})
    diagnostics["editorial_engine_v2"] = {
        **diagnostics["editorial_engine_v2"],
        "status": "frozen_boundary_verified_pending_post_render_review",
        "discard_membership_sha256": discard_signature,
        "ordered_semantic_sha256": ordered_semantic_signature,
        "selection_contract_status": (
            diagnostics.get("selection_boundary_contract") or {}
        ).get("status"),
    }
    return replace(
        result,
        draft=replace(result.draft, diagnostics=diagnostics),
        stage_status={
            **result.stage_status,
            "brain_mode": "editorial_engine_v2_whole_video",
            "selection_phase_authority": "v2_whole_video_reasoner_frozen",
            "unified_selection_reasoner": "required_applied",
            "selection_boundary_contract": "verified",
            "editorial_engine_v2": "pending_post_render_review",
        },
    )
