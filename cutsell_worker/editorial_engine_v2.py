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

from .contracts import ProcessingResult
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
                regions.append((start, end, demonstration))
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
    for index in range(len(selected) - 1):
        left, right = selected[index], selected[index + 1]
        if left.source_asset_id != right.source_asset_id or left.source_order != right.source_order:
            continue
        gap_start, gap_end = float(left.end), float(right.start)
        gap = gap_end - gap_start
        if gap <= 1e-6 or gap > 8.0:
            continue
        containing = [region for region in audience_by_source.get(left.source_asset_id, ())
                      if region[0] <= gap_start + 1e-6 and region[1] >= gap_end - 1e-6]
        if not containing:
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
    result = recover_complete_boundaries(result)
    result = replace(result, draft=_fold_alternates(result.draft))
    result = _restore_safe_audience_continuity(result, whole)
    discard_signature = _discard_signature(result.draft)
    ordered_semantic_signature = _ordered_semantic_signature(result.draft)

    result = replace(result, draft=freeze_selection_contract(result.draft))
    frozen_selected = tuple(result.draft.selected)
    result = execute_boundaries(result)
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
