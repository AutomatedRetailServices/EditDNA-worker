"""Speech-safe interior performance-gap trimming after Best Take.

This pass may refine boundaries inside an already-selected take, but it must never
change the spoken information chosen by Selection. It therefore cuts only between
aligned Word envelopes and preserves every word on the left and right child clips.

Normal gaps require multimodal reset evidence. Narrow fallbacks handle completed
sentences when physical reset evidence is strong even if face/camera detection misses
the reset. The anticipatory fallback may use an earlier reset as evidence, but the
actual edit remains confined to the speech-free word gap.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Iterable

from .contracts import DraftClip

_PHYSICAL_KINDS = frozenset({"hand_motion_reset_candidate", "body_reset_candidate"})
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})
_TERMINAL_MARKS = (".", "!", "?", "…")

# D-046 FIX A: this hook may divide an already-SELECTED, already-winning
# realization into two physical pieces (a legitimate Boundary-flavored
# operation -- it never changes which spoken material won). Before this
# fix, the resulting pieces carried no D-036 fragment provenance
# (`parent_semantic_clip_id`), so any downstream consumer that determines
# "did this take_judge_groups member survive?" by exact `clip_id` equality
# against `draft.selected` (canonical_edit_plan.py's winning/discarded
# derivation, final_story_coherence_validation.py's `_missing_idea_
# coverage`) could no longer find the original id anywhere in `selected`
# and wrongly concluded the whole idea vanished -- see D-045 Case A.
# Stamping the same provenance fields human_boundary_polish_v5.py already
# uses lets those consumers recognize a split realization as still
# covered, via the general `effective_parent_semantic_clip_id` contract
# instead of a Video00-specific patch.
BOUNDARY_REASON_INTERIOR_PERFORMANCE_GAP = "remove_interior_performance_gap"
# D-095.2: a proven audio silence (ffmpeg silencedetect, the SAME measurement
# post_render_media_qc uses to invalidate a render) lying inside a selected
# clip is objective dead air -- recording-process material, never spoken
# content. Word-gap evidence cannot see it when ASR word timestamps are
# stretched over the silence (run 33995806350: a 2.36 s interior silence
# with no word gap >= 0.26 s around it). Cut inside the silence, leaving a
# natural pause edge on both sides; words are partitioned by their midpoint
# and their timings clamped to the piece so no later envelope pass can
# re-expand across the removed dead air.
BOUNDARY_REASON_INTERIOR_AUDIO_SILENCE = "remove_interior_audio_silence"
AUDIO_SILENCE_EVENT_KIND = "audio_silence_interval"
# Same threshold post_render_media_qc applies as LINGERING_ACCIDENTAL_SILENCE:
# anything at or above it inside a kept take is guaranteed to invalidate the
# render later, so it is removed here, at the Boundary-flavored stage that
# already owns interior physical gaps.
LONG_AUDIO_SILENCE_SEC = 1.20
AUDIO_SILENCE_EDGE_PAD_SEC = 0.12


def _clamp_word(word, start: float, end: float):
    new_start = min(max(float(word.start), start), end)
    new_end = max(min(float(word.end), end), new_start)
    if abs(new_start - float(word.start)) < 1e-9 and abs(new_end - float(word.end)) < 1e-9:
        return word
    return replace(word, start=new_start, end=new_end)


def _audio_silence_split_candidate(
    clip: DraftClip,
    words,
    events,
    *,
    long_audio_silence_sec: float,
    minimum_edge_margin_sec: float,
    audio_pad_sec: float,
    minimum_piece_sec: float,
):
    """Return (left_words, right_words, left_end, right_start, silence, rejections)
    for the longest qualifying audio silence inside ``clip``; the first
    element is None when no silence qualifies. ``rejections`` lists every
    interior audio silence >= long_audio_silence_sec that did not qualify,
    with the reason (observability)."""
    silences = sorted(
        (
            (float(event.get("start") or 0.0), float(event.get("end") or 0.0))
            for event in events
            if _kind(event.get("kind")) == AUDIO_SILENCE_EVENT_KIND
        ),
        key=lambda item: -(item[1] - item[0]),
    )
    rejections: list[dict] = []
    clip_start, clip_end = float(clip.start), float(clip.end)
    for start, end in silences:
        duration = end - start
        if duration < long_audio_silence_sec:
            continue
        if end <= clip_start or start >= clip_end:
            continue
        if start < clip_start + minimum_edge_margin_sec or end > clip_end - minimum_edge_margin_sec:
            rejections.append({"reason": "audio_silence_edge_margin", "gap_start": start, "gap_end": end})
            continue
        left_end = start + audio_pad_sec
        right_start = end - audio_pad_sec
        if right_start - left_end < 0.10:
            rejections.append({"reason": "audio_silence_too_short_after_padding", "gap_start": start, "gap_end": end})
            continue
        midpoint = (start + end) / 2.0
        left_words = tuple(w for w in words if (float(w.start) + float(w.end)) / 2.0 <= midpoint)
        right_words = tuple(w for w in words if (float(w.start) + float(w.end)) / 2.0 > midpoint)
        if not left_words or not right_words:
            rejections.append({"reason": "audio_silence_no_words_on_side", "gap_start": start, "gap_end": end})
            continue
        if left_end - clip_start < minimum_piece_sec or clip_end - right_start < minimum_piece_sec:
            rejections.append({"reason": "audio_silence_piece_too_short", "gap_start": start, "gap_end": end})
            continue
        left_words = tuple(_clamp_word(w, clip_start, left_end) for w in left_words)
        right_words = tuple(_clamp_word(w, right_start, clip_end) for w in right_words)
        return left_words, right_words, left_end, right_start, (start, end), rejections
    return None, (), 0.0, 0.0, None, rejections


def _kind(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _events_for_source(diagnostics: dict, source_asset_id: str) -> tuple[dict, ...]:
    whole = diagnostics.get("whole_video_context") or {}
    for source in whole.get("sources") or ():
        if isinstance(source, dict) and source.get("source_asset_id") == source_asset_id:
            return tuple(event for event in (source.get("events") or ()) if isinstance(event, dict))
    return ()


def _child_id(clip: DraftClip, side: str, start: float, end: float) -> str:
    digest = hashlib.sha256(
        f"{clip.clip_id}|post-selection-interior-gap|{side}|{start:.3f}|{end:.3f}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{clip.clip_id}__psig{side}{digest}"


def _text(words) -> str:
    return " ".join(str(word.text or "").strip() for word in words).strip()


def _is_completed_left_delivery(word) -> bool:
    return str(getattr(word, "text", "") or "").strip().endswith(_TERMINAL_MARKS)


def split_selected_interior_performance_gaps(
    selected: Iterable[DraftClip],
    diagnostics: dict,
    *,
    minimum_word_gap_sec: float = 0.18,
    long_gap_without_break_sec: float = 1.00,
    anticipatory_minimum_gap_sec: float = 0.40,
    anticipatory_lookback_sec: float = 2.00,
    anticipatory_near_gap_sec: float = 0.55,
    evidence_radius_sec: float = 0.75,
    minimum_edge_margin_sec: float = 0.35,
    max_splits_per_clip: int = 3,
    include_rejected_diagnostics: bool = False,
    long_audio_silence_sec: float = LONG_AUDIO_SILENCE_SEC,
    audio_pad_sec: float = AUDIO_SILENCE_EDGE_PAD_SEC,
) -> tuple[tuple[DraftClip, ...], tuple[dict, ...]]:
    """Split selected clips only around speech-free performance resets.

    ``include_rejected_diagnostics`` is observability-only. It records why otherwise
    valid interior word gaps were rejected, but never changes the split decision.
    """
    output: list[DraftClip] = []
    audit: list[dict] = []

    for original in selected:
        # D-046 FIX A: the ROOT semantic clip every physical sibling this
        # original ultimately produces reconstructs together -- never a
        # fragment id, even if `original` itself already carries fragment
        # provenance from an earlier physical pass (chained splits), so all
        # descendants stay discoverable under one shared key. Mirrors
        # human_boundary_polish_v5.py's own `root_parent` pattern exactly.
        root_parent = getattr(original, "parent_semantic_clip_id", None) or original.clip_id
        # D-050A: mirrors `root_parent` exactly, for `realization_id`
        # instead of `clip_id` -- see canonical_identity.py's ID
        # OWNERSHIP table. `original.realization_id` itself is preserved
        # unchanged on every fragment below (dataclasses.replace() never
        # touches it); this is only the explicit "a split happened" marker.
        root_realization = getattr(original, "realization_id", None)
        original_pieces: list[DraftClip] = []
        pending = [original]
        split_count = 0
        while pending:
            clip = pending.pop(0)
            words = tuple(sorted(clip.words, key=lambda word: (float(word.start), float(word.end))))
            events = _events_for_source(diagnostics, clip.source_asset_id)

            # D-095.2: objective audio dead air first. It needs no visual
            # reset corroboration -- a proven >= long_audio_silence_sec
            # silence contains no speech by measurement -- and it does not
            # depend on ASR word gaps at all.
            if split_count < max_splits_per_clip and len(words) >= 2:
                left_words, right_words, left_end, right_start, silence, audio_rejections = (
                    _audio_silence_split_candidate(
                        clip, words, events,
                        long_audio_silence_sec=long_audio_silence_sec,
                        minimum_edge_margin_sec=minimum_edge_margin_sec,
                        audio_pad_sec=audio_pad_sec,
                        minimum_piece_sec=minimum_edge_margin_sec,
                    )
                )
                if include_rejected_diagnostics:
                    for rejection in audio_rejections:
                        audit.append({
                            "authority": "post_selection_interior_gap_trace",
                            "decision": "reject",
                            "evidence_mode": "long_audio_silence",
                            "parent_clip_id": original.clip_id,
                            "parent_start": round(float(original.start), 3),
                            "parent_end": round(float(original.end), 3),
                            "gap_start": round(float(rejection["gap_start"]), 3),
                            "gap_end": round(float(rejection["gap_end"]), 3),
                            "gap_sec": round(float(rejection["gap_end"]) - float(rejection["gap_start"]), 3),
                            "reason": rejection["reason"],
                        })
                if left_words is not None:
                    left = replace(
                        clip,
                        clip_id=_child_id(clip, "l", float(clip.start), float(left_end)),
                        end=float(left_end),
                        text=_text(left_words),
                        caption_text=_text(left_words),
                        words=left_words,
                        parent_semantic_clip_id=root_parent,
                        parent_realization_id=root_realization,
                        boundary_reason=BOUNDARY_REASON_INTERIOR_AUDIO_SILENCE,
                    )
                    right = replace(
                        clip,
                        clip_id=_child_id(clip, "r", float(right_start), float(clip.end)),
                        start=float(right_start),
                        text=_text(right_words),
                        caption_text=_text(right_words),
                        words=right_words,
                        parent_semantic_clip_id=root_parent,
                        parent_realization_id=root_realization,
                        boundary_reason=BOUNDARY_REASON_INTERIOR_AUDIO_SILENCE,
                    )
                    left = replace(left, render_fragment_id=left.clip_id)
                    right = replace(right, render_fragment_id=right.clip_id)
                    pending[0:0] = [left, right]
                    split_count += 1
                    audit.append({
                        "authority": "post_selection_interior_gap_trim",
                        "decision": "split",
                        "parent_clip_id": original.clip_id,
                        "parent_text": str(original.text or ""),
                        "evidence_mode": "long_audio_silence",
                        "removed_gap_start": round(float(left_end), 3),
                        "removed_gap_end": round(float(right_start), 3),
                        "removed_gap_sec": round(float(right_start) - float(left_end), 3),
                        "audio_silence_start": round(float(silence[0]), 3),
                        "audio_silence_end": round(float(silence[1]), 3),
                        "physical_event_count": 0,
                        "break_event_count": 0,
                        "left_word": str(left_words[-1].text),
                        "right_word": str(right_words[0].text),
                    })
                    continue

            if len(words) < 4 or split_count >= max_splits_per_clip:
                original_pieces.append(clip)
                continue

            best = None
            for index in range(len(words) - 1):
                left_word = words[index]
                right_word = words[index + 1]
                gap_start = float(left_word.end)
                gap_end = float(right_word.start)
                gap = gap_end - gap_start
                if gap < minimum_word_gap_sec:
                    continue

                rejection = None
                if gap_start <= float(clip.start) + minimum_edge_margin_sec:
                    rejection = "left_edge_margin"
                elif gap_end >= float(clip.end) - minimum_edge_margin_sec:
                    rejection = "right_edge_margin"
                elif index + 1 < 2 or len(words) - (index + 1) < 2:
                    rejection = "insufficient_words_on_side"

                window_start = gap_start - evidence_radius_sec
                window_end = gap_end + evidence_radius_sec
                physical = [
                    event for event in events
                    if float(event.get("end") or 0.0) >= window_start
                    and float(event.get("start") or 0.0) <= window_end
                    and _kind(event.get("kind")) in _PHYSICAL_KINDS
                    and float(event.get("confidence") or 0.0) >= 0.90
                ]
                breaks = [
                    event for event in events
                    if float(event.get("end") or 0.0) >= window_start
                    and float(event.get("start") or 0.0) <= window_end
                    and _kind(event.get("kind")) in _BREAK_KINDS
                    and float(event.get("confidence") or 0.0) >= (
                        0.72 if _kind(event.get("kind")) == "facial_expression_shift_candidate" else 0.80
                    )
                ]
                hand_count = sum(
                    1 for event in physical
                    if _kind(event.get("kind")) == "hand_motion_reset_candidate"
                )
                physical_ok = len(physical) >= 2 and hand_count >= 1
                multimodal_ok = physical_ok and bool(breaks)
                completed_left = _is_completed_left_delivery(left_word)
                long_gap_physical_ok = (
                    physical_ok
                    and gap >= long_gap_without_break_sec
                    and completed_left
                )

                anticipatory_window_start = gap_start - anticipatory_lookback_sec
                anticipatory_physical = [
                    event for event in events
                    if float(event.get("end") or 0.0) >= anticipatory_window_start
                    and float(event.get("start") or 0.0) <= window_end
                    and _kind(event.get("kind")) in _PHYSICAL_KINDS
                    and float(event.get("confidence") or 0.0) >= 0.90
                ]
                anticipatory_hand_count = sum(
                    1 for event in anticipatory_physical
                    if _kind(event.get("kind")) == "hand_motion_reset_candidate"
                )
                near_gap_reset_count = sum(
                    1 for event in anticipatory_physical
                    if float(event.get("end") or 0.0) >= gap_start - anticipatory_near_gap_sec
                    and float(event.get("start") or 0.0) <= gap_end + evidence_radius_sec
                )
                anticipatory_ok = (
                    completed_left
                    and anticipatory_minimum_gap_sec <= gap < long_gap_without_break_sec
                    and len(anticipatory_physical) >= 2
                    and anticipatory_hand_count >= 1
                    and near_gap_reset_count >= 1
                )

                if rejection is None and not multimodal_ok and not long_gap_physical_ok and not anticipatory_ok:
                    if completed_left and anticipatory_minimum_gap_sec <= gap < long_gap_without_break_sec:
                        if len(anticipatory_physical) < 2:
                            rejection = "insufficient_anticipatory_reset_evidence"
                        elif anticipatory_hand_count < 1:
                            rejection = "anticipatory_reset_missing_hand_evidence"
                        elif near_gap_reset_count < 1:
                            rejection = "anticipatory_reset_not_near_gap"
                        else:
                            rejection = "anticipatory_reset_guard_rejected"
                    elif not physical_ok:
                        rejection = "insufficient_physical_reset_evidence"
                    elif gap < long_gap_without_break_sec:
                        rejection = "gap_below_physical_reset_threshold"
                    elif not completed_left:
                        rejection = "left_delivery_not_terminal"
                    else:
                        rejection = "missing_multimodal_break"

                if include_rejected_diagnostics and rejection is not None:
                    audit.append({
                        "authority": "post_selection_interior_gap_trace",
                        "decision": "reject",
                        "reason": rejection,
                        "parent_clip_id": original.clip_id,
                        "parent_start": round(float(original.start), 3),
                        "parent_end": round(float(original.end), 3),
                        "parent_text": str(original.text or ""),
                        "gap_start": round(gap_start, 3),
                        "gap_end": round(gap_end, 3),
                        "gap_sec": round(gap, 3),
                        "left_word": str(left_word.text),
                        "right_word": str(right_word.text),
                        "left_terminal": bool(completed_left),
                        "physical_event_count": len(physical),
                        "hand_event_count": hand_count,
                        "break_event_count": len(breaks),
                        "physical_ok": bool(physical_ok),
                        "multimodal_ok": bool(multimodal_ok),
                        "long_gap_physical_ok": bool(long_gap_physical_ok),
                        "anticipatory_physical_event_count": len(anticipatory_physical),
                        "anticipatory_hand_event_count": anticipatory_hand_count,
                        "anticipatory_near_gap_reset_count": near_gap_reset_count,
                        "anticipatory_ok": bool(anticipatory_ok),
                        "physical_events": [
                            {
                                "kind": _kind(event.get("kind")),
                                "start": round(float(event.get("start") or 0.0), 3),
                                "end": round(float(event.get("end") or 0.0), 3),
                                "confidence": round(float(event.get("confidence") or 0.0), 3),
                            }
                            for event in physical
                        ],
                        "anticipatory_physical_events": [
                            {
                                "kind": _kind(event.get("kind")),
                                "start": round(float(event.get("start") or 0.0), 3),
                                "end": round(float(event.get("end") or 0.0), 3),
                                "confidence": round(float(event.get("confidence") or 0.0), 3),
                            }
                            for event in anticipatory_physical
                        ],
                        "break_events": [
                            {
                                "kind": _kind(event.get("kind")),
                                "start": round(float(event.get("start") or 0.0), 3),
                                "end": round(float(event.get("end") or 0.0), 3),
                                "confidence": round(float(event.get("confidence") or 0.0), 3),
                            }
                            for event in breaks
                        ],
                    })

                if rejection is not None:
                    continue

                if multimodal_ok:
                    evidence_mode = "multimodal_break"
                    selected_physical = physical
                elif long_gap_physical_ok:
                    evidence_mode = "long_gap_physical_reset"
                    selected_physical = physical
                else:
                    evidence_mode = "completed_sentence_anticipatory_reset"
                    selected_physical = anticipatory_physical

                score = (
                    2 if multimodal_ok else 1 if long_gap_physical_ok else 0,
                    len(selected_physical),
                    len(breaks),
                    max(float(event.get("confidence") or 0.0) for event in selected_physical),
                    gap,
                )
                if best is None or score > best[0]:
                    best = (
                        score,
                        index,
                        gap_start,
                        gap_end,
                        selected_physical,
                        breaks,
                        evidence_mode,
                    )

            if best is None:
                original_pieces.append(clip)
                continue

            _, index, gap_start, gap_end, physical, breaks, evidence_mode = best
            left_words = words[: index + 1]
            right_words = words[index + 1 :]
            left = replace(
                clip,
                clip_id=_child_id(clip, "l", float(clip.start), float(left_words[-1].end)),
                end=float(left_words[-1].end),
                text=_text(left_words),
                caption_text=_text(left_words),
                words=left_words,
                # D-046 FIX A: D-036 fragment provenance -- render_fragment_id
                # is filled in once the final clip_id is known below;
                # parent_semantic_clip_id always points at the true ROOT
                # semantic clip so canonical_edit_plan.py/final_story_
                # coherence_validation.py can recognize this piece as still
                # covering that idea even though its own clip_id differs.
                parent_semantic_clip_id=root_parent,
                parent_realization_id=root_realization,
                boundary_reason=BOUNDARY_REASON_INTERIOR_PERFORMANCE_GAP,
            )
            right = replace(
                clip,
                clip_id=_child_id(clip, "r", float(right_words[0].start), float(clip.end)),
                start=float(right_words[0].start),
                text=_text(right_words),
                caption_text=_text(right_words),
                words=right_words,
                parent_semantic_clip_id=root_parent,
                parent_realization_id=root_realization,
                boundary_reason=BOUNDARY_REASON_INTERIOR_PERFORMANCE_GAP,
            )
            left = replace(left, render_fragment_id=left.clip_id)
            right = replace(right, render_fragment_id=right.clip_id)
            pending[0:0] = [left, right]
            split_count += 1
            audit.append({
                "authority": "post_selection_interior_gap_trim",
                "decision": "split",
                "parent_clip_id": original.clip_id,
                "parent_text": str(original.text or ""),
                "evidence_mode": evidence_mode,
                "removed_gap_start": round(gap_start, 3),
                "removed_gap_end": round(gap_end, 3),
                "removed_gap_sec": round(gap_end - gap_start, 3),
                "physical_event_count": len(physical),
                "break_event_count": len(breaks),
                "left_word": str(left_words[-1].text),
                "right_word": str(right_words[0].text),
            })

        if len(original_pieces) > 1:
            # D-046 FIX A: fragment_index/fragment_count are only meaningful
            # once every gap candidate for THIS original has been applied
            # and the final piece count is known -- stamped here as one
            # last pass, mirroring human_boundary_polish_v5.py's own
            # end-of-splitting stamp.
            total = len(original_pieces)
            original_pieces = [
                replace(piece, fragment_index=index, fragment_count=total)
                for index, piece in enumerate(original_pieces)
            ]
        output.extend(original_pieces)

    output.sort(key=lambda clip: (clip.source_order, float(clip.start), float(clip.end), clip.clip_id))
    return tuple(output), tuple(audit)


def install_post_selection_interior_gap_trim() -> None:
    from . import pipeline

    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_post_selection_interior_gap_trim", False):
        return

    def build_with_post_selection_interior_gap_trim(*args, **kwargs):
        result = original(*args, **kwargs)
        draft = result.draft
        diagnostics = dict(draft.diagnostics or {})
        selected, audit = split_selected_interior_performance_gaps(
            draft.selected,
            diagnostics,
            include_rejected_diagnostics=True,
        )
        if not audit:
            return result
        diagnostics["post_selection_interior_gap_trim"] = [
            item for item in audit if item.get("decision") == "split"
        ]
        diagnostics["post_selection_interior_gap_trace"] = [
            item for item in audit if item.get("decision") == "reject"
        ]
        repaired = replace(draft, selected=selected, diagnostics=diagnostics)
        return replace(result, draft=repaired)

    build_with_post_selection_interior_gap_trim._cutsell_post_selection_interior_gap_trim = True
    pipeline.build_flow_b_draft = build_with_post_selection_interior_gap_trim
