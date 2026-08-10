"""Temporal performance editing helpers for CutSell Watch + Listen.

This module turns whole-video observations into safe, timestamp-aware editing
signals. It intentionally trims only high-confidence bad-performance material at
take boundaries. Interior events are preserved as diagnostics for a later mixed
trim/provider pass rather than blindly cutting through speech.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Iterable, Tuple

from .contracts import CandidateTake, MediaSignals, Word
from .whole_video_analysis import TemporalEvent, WholeVideoContext


# Event taxonomy, not phrase matching. These are human-performance states that a
# competent editor would normally remove when they are unambiguously part of the
# recording process rather than intentional content.
BAD_PERFORMANCE_KINDS = frozenset({
    "false_start",
    "wrong_take",
    "verbal_fumble",
    "visual_fumble",
    "body_reset",
    "retry_setup",
    "frustration",
    "breaking_character",
    "recording_joke",
    "accidental_laughter",
    "camera_adjustment",
    "product_handling_mistake",
    "searching_for_words",
    "unintentional_dead_air",
})


def _kind(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def events_for_take(take: CandidateTake, context: WholeVideoContext | None) -> Tuple[TemporalEvent, ...]:
    if context is None:
        return ()
    events = []
    for source in context.sources:
        if source.source_asset_id != take.source_asset_id:
            continue
        for event in source.events:
            if event.end <= take.start or event.start >= take.end:
                continue
            events.append(event)
    return tuple(sorted(events, key=lambda item: (item.start, item.end)))


def harmful_events_for_take(
    take: CandidateTake,
    context: WholeVideoContext | None,
    *,
    minimum_confidence: float = 0.72,
) -> Tuple[TemporalEvent, ...]:
    return tuple(
        event for event in events_for_take(take, context)
        if _kind(event.kind) in BAD_PERFORMANCE_KINDS and event.confidence >= minimum_confidence
    )


def harmful_coverage_ratio(take: CandidateTake, events: Iterable[TemporalEvent]) -> float:
    """Return union coverage of harmful events inside a take."""
    duration = take.duration_sec
    if duration <= 0.0:
        return 0.0
    intervals = []
    for event in events:
        start = max(take.start, event.start)
        end = min(take.end, event.end)
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return 0.0
    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    covered = sum(end - start for start, end in merged)
    return min(1.0, max(0.0, covered / duration))


def _trim_words(words: Tuple[Word, ...], start: float, end: float) -> Tuple[Word, ...]:
    return tuple(word for word in words if word.end > start and word.start < end)


def _signals_for_trim(signals: MediaSignals | None, start: float, end: float) -> MediaSignals | None:
    if signals is None:
        return None
    return replace(signals, start=start, end=end)


def _child_id(take: CandidateTake, start: float, end: float) -> str:
    digest = hashlib.sha256(
        f"{take.clip_id}|temporal|{start:.3f}|{end:.3f}".encode("utf-8")
    ).hexdigest()[:14]
    return f"{take.clip_id}__t{digest}"


def refine_takes_with_temporal_context(
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    *,
    edge_tolerance_sec: float = 0.30,
    minimum_keep_sec: float = 0.30,
) -> tuple[Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    """Trim high-confidence recording-process events from take edges.

    This safely handles the common editor behavior of keeping a good spoken line
    while removing the awkward reaction/body reset immediately before or after it.
    Interior bad events are reported but are not cut here because doing so without
    a mixed-trim reasoning pass could mutilate valid speech.
    """
    refined = []
    diagnostics = []
    for take in takes:
        events = harmful_events_for_take(take, context)
        new_start, new_end = take.start, take.end
        applied = []
        interior = []

        for event in events:
            touches_start = event.start <= take.start + edge_tolerance_sec
            touches_end = event.end >= take.end - edge_tolerance_sec
            if touches_start and event.end < new_end - minimum_keep_sec:
                candidate = max(new_start, min(new_end, event.end))
                if candidate > new_start:
                    new_start = candidate
                    applied.append((event, "trim_start"))
                continue
            if touches_end and event.start > new_start + minimum_keep_sec:
                candidate = min(new_end, max(new_start, event.start))
                if candidate < new_end:
                    new_end = candidate
                    applied.append((event, "trim_end"))
                continue
            interior.append(event)

        if new_end - new_start < minimum_keep_sec:
            new_start, new_end = take.start, take.end
            applied = []

        if applied:
            words = _trim_words(take.words, new_start, new_end)
            text = " ".join(word.text for word in words).strip() if words else take.text
            child = replace(
                take,
                clip_id=_child_id(take, new_start, new_end),
                start=new_start,
                end=new_end,
                text=text,
                words=words,
                signals=_signals_for_trim(take.signals, new_start, new_end),
            )
            refined.append(child)
        else:
            child = take
            refined.append(take)

        diagnostics.append({
            "original_clip_id": take.clip_id,
            "result_clip_id": child.clip_id,
            "original_start": take.start,
            "original_end": take.end,
            "result_start": child.start,
            "result_end": child.end,
            "applied": [
                {
                    "action": action,
                    "kind": event.kind,
                    "start": event.start,
                    "end": event.end,
                    "confidence": event.confidence,
                    "description": event.description,
                }
                for event, action in applied
            ],
            "interior_bad_events": [event.__dict__ for event in interior],
        })
    return tuple(refined), tuple(diagnostics)
