"""Trim internal note/script-consult pauses using conservative multimodal evidence.

A creator may finish a phrase, look away/down to recover the next line from notes, then
resume the same delivery. ASR often keeps both spoken portions inside one candidate, so
ordinary take-level cleanup cannot remove the visible consultation pause. This module
splits only at real word gaps that are independently corroborated by visual disengagement
plus a physical/expression reset, or by an authoritative whole-video word-search/retry
signal. Spoken words on both sides are preserved; only the non-speech consultation gap
is removed.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Iterable, Tuple

from .contracts import CandidateTake, Word
from .whole_video_analysis import WholeVideoContext

_CAMERA_BREAK_KINDS = frozenset({"camera_disengagement_candidate"})
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_FACE_BREAK_KINDS = frozenset({"facial_expression_shift_candidate"})
_AUTHORITATIVE_CONSULT_KINDS = frozenset({
    "searching_for_words",
    "retry_setup",
    "unintentional_dead_air",
})


def _kind(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _consult_evidence(
    take: CandidateTake,
    gap_start: float,
    gap_end: float,
    context: WholeVideoContext | None,
) -> tuple[bool, tuple[str, ...]]:
    events = tuple(
        event for event in _source_events(context, take.source_asset_id)
        if event.end >= gap_start - 0.30 and event.start <= gap_end + 0.30
    )
    reasons: list[str] = []

    authoritative = [
        event for event in events
        if _kind(event.kind) in _AUTHORITATIVE_CONSULT_KINDS and event.confidence >= 0.78
    ]
    if authoritative:
        strongest = max(authoritative, key=lambda item: item.confidence)
        reasons.append(f"event:{_kind(strongest.kind)}:{strongest.confidence:.2f}")
        return True, tuple(reasons)

    camera = [
        event for event in events
        if _kind(event.kind) in _CAMERA_BREAK_KINDS and event.confidence >= 0.82
    ]
    resets = [
        event for event in events
        if _kind(event.kind) in _RESET_KINDS and event.confidence >= 0.86
    ]
    face_breaks = [
        event for event in events
        if _kind(event.kind) in _FACE_BREAK_KINDS and event.confidence >= 0.80
    ]

    if camera and (resets or face_breaks):
        reasons.append(f"camera_disengagement:{max(item.confidence for item in camera):.2f}")
        if resets:
            reasons.append(f"physical_reset:{max(item.confidence for item in resets):.2f}")
        if face_breaks:
            reasons.append(f"expression_break:{max(item.confidence for item in face_breaks):.2f}")
        return True, tuple(reasons)
    return False, ()


def _child_id(take: CandidateTake, start: float, end: float, index: int) -> str:
    digest = hashlib.sha256(
        f"{take.clip_id}|script-consult|{index}|{start:.3f}|{end:.3f}".encode("utf-8")
    ).hexdigest()[:14]
    return f"{take.clip_id}__scg{digest}"


def _build_child(
    take: CandidateTake,
    words: tuple[Word, ...],
    *,
    start: float,
    end: float,
    index: int,
) -> CandidateTake:
    text = " ".join(word.text.strip() for word in words if str(word.text or "").strip()).strip()
    signals = take.signals
    if signals is not None:
        signals = replace(signals, start=start, end=end)
    return replace(
        take,
        clip_id=_child_id(take, start, end, index),
        start=start,
        end=end,
        text=text,
        words=words,
        signals=signals,
    )


def trim_script_consult_pauses(
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    *,
    minimum_gap_sec: float = 0.65,
    maximum_gap_sec: float = 6.0,
    minimum_words_per_side: int = 2,
) -> tuple[Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    """Split candidates around visually corroborated internal note/script consultations."""
    output: list[CandidateTake] = []
    diagnostics: list[dict] = []

    for take in takes:
        words = tuple(sorted(take.words, key=lambda item: (float(item.start), float(item.end))))
        if len(words) < minimum_words_per_side * 2:
            output.append(take)
            continue

        split_after: list[int] = []
        split_meta: dict[int, tuple[float, float, tuple[str, ...]]] = {}
        for index in range(len(words) - 1):
            left = words[index]
            right = words[index + 1]
            gap_start = float(left.end)
            gap_end = float(right.start)
            gap = gap_end - gap_start
            if gap < minimum_gap_sec or gap > maximum_gap_sec:
                continue
            if index + 1 < minimum_words_per_side:
                continue
            if len(words) - (index + 1) < minimum_words_per_side:
                continue
            confirmed, reasons = _consult_evidence(take, gap_start, gap_end, context)
            if not confirmed:
                continue
            split_after.append(index)
            split_meta[index] = (gap_start, gap_end, reasons)

        if not split_after:
            output.append(take)
            continue

        boundaries = [-1, *split_after, len(words) - 1]
        children: list[CandidateTake] = []
        for child_index in range(len(boundaries) - 1):
            first_word = boundaries[child_index] + 1
            last_word = boundaries[child_index + 1]
            child_words = words[first_word : last_word + 1]
            if not child_words:
                continue
            start = take.start if child_index == 0 else float(child_words[0].start)
            end = take.end if child_index == len(boundaries) - 2 else float(child_words[-1].end)
            if end <= start:
                continue
            children.append(_build_child(
                take,
                tuple(child_words),
                start=start,
                end=end,
                index=child_index,
            ))

        if len(children) < 2:
            output.append(take)
            continue

        output.extend(children)
        diagnostics.append({
            "original_clip_id": take.clip_id,
            "action": "split_script_consult_pause",
            "result_clip_ids": [child.clip_id for child in children],
            "removed_gaps": [
                {
                    "start": split_meta[index][0],
                    "end": split_meta[index][1],
                    "duration_sec": round(split_meta[index][1] - split_meta[index][0], 3),
                    "evidence": list(split_meta[index][2]),
                }
                for index in split_after
            ],
        })

    return tuple(output), tuple(diagnostics)


def install_script_consult_pause_trim() -> None:
    """Install after base temporal refinement and preserve all spoken words."""
    from . import temporal_editing

    original = temporal_editing.refine_takes_with_temporal_context
    if getattr(original, "_cutsell_script_consult_pause_trim", False):
        return

    def refine_with_script_consult_trim(takes, context, **kwargs):
        refined, diagnostics = original(takes, context, **kwargs)
        refined, consult_diagnostics = trim_script_consult_pauses(refined, context)
        return refined, tuple(diagnostics) + tuple(consult_diagnostics)

    refine_with_script_consult_trim._cutsell_script_consult_pause_trim = True
    temporal_editing.refine_takes_with_temporal_context = refine_with_script_consult_trim
