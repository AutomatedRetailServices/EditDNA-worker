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
) -> tuple[tuple[DraftClip, ...], tuple[dict, ...]]:
    """Split selected clips only around speech-free performance resets.

    ``include_rejected_diagnostics`` is observability-only. It records why otherwise
    valid interior word gaps were rejected, but never changes the split decision.
    """
    output: list[DraftClip] = []
    audit: list[dict] = []

    for original in selected:
        pending = [original]
        split_count = 0
        while pending:
            clip = pending.pop(0)
            words = tuple(sorted(clip.words, key=lambda word: (float(word.start), float(word.end))))
            if len(words) < 4 or split_count >= max_splits_per_clip:
                output.append(clip)
                continue

            events = _events_for_source(diagnostics, clip.source_asset_id)
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
                output.append(clip)
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
            )
            right = replace(
                clip,
                clip_id=_child_id(clip, "r", float(right_words[0].start), float(clip.end)),
                start=float(right_words[0].start),
                text=_text(right_words),
                caption_text=_text(right_words),
                words=right_words,
            )
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
