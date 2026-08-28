"""Speech-safe interior performance-gap trimming after Best Take.

This pass may refine boundaries inside an already-selected take, but it must never
change the spoken information chosen by Selection. It therefore cuts only between
aligned Word envelopes and preserves every word on the left and right child clips.

Normal gaps still require multimodal reset evidence. A second, deliberately narrow
fallback handles unusually long speech-free gaps after a completed sentence when the
speaker performs multiple strong physical resets even if face/camera detection misses
that reset. This models a common human edit: remove the performance pause, keep both
spoken thoughts.
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
    long_gap_without_break_sec: float = 1.50,
    evidence_radius_sec: float = 0.75,
    minimum_edge_margin_sec: float = 0.35,
    max_splits_per_clip: int = 3,
) -> tuple[tuple[DraftClip, ...], tuple[dict, ...]]:
    """Split selected clips only around speech-free performance resets."""
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
                if gap_start <= float(clip.start) + minimum_edge_margin_sec:
                    continue
                if gap_end >= float(clip.end) - minimum_edge_margin_sec:
                    continue
                if index + 1 < 2 or len(words) - (index + 1) < 2:
                    continue

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
                long_gap_physical_ok = (
                    physical_ok
                    and gap >= long_gap_without_break_sec
                    and _is_completed_left_delivery(left_word)
                )
                if not multimodal_ok and not long_gap_physical_ok:
                    continue

                evidence_mode = (
                    "multimodal_break" if multimodal_ok else "long_gap_physical_reset"
                )
                score = (
                    1 if multimodal_ok else 0,
                    len(physical),
                    len(breaks),
                    max(float(event.get("confidence") or 0.0) for event in physical),
                    gap,
                )
                if best is None or score > best[0]:
                    best = (
                        score,
                        index,
                        gap_start,
                        gap_end,
                        physical,
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
        selected, audit = split_selected_interior_performance_gaps(draft.selected, diagnostics)
        if not audit:
            return result
        diagnostics["post_selection_interior_gap_trim"] = list(audit)
        repaired = replace(draft, selected=selected, diagnostics=diagnostics)
        return replace(result, draft=repaired)

    build_with_post_selection_interior_gap_trim._cutsell_post_selection_interior_gap_trim = True
    pipeline.build_flow_b_draft = build_with_post_selection_interior_gap_trim
