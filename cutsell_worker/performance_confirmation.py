"""Confirm dense local performance candidates using speech/retry context.

MediaPipe/OpenCV deliberately emit *_candidate observations.  This module is the
bridge from measurement to editorial evidence: it only promotes a candidate when
multiple independent signals agree (performance trajectory + timing + a likely
retry of the same communication attempt).  A single head turn or hand movement is
never enough to delete content.
"""
from __future__ import annotations

from dataclasses import replace
from difflib import SequenceMatcher
import re
from typing import Iterable, Tuple

from .contracts import CandidateTake
from .local_performance import LocalPerformanceTimeline
from .whole_video_analysis import TemporalEvent, WholeVideoContext

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})
_BREAK_KINDS = frozenset({"camera_disengagement_candidate", "facial_expression_shift_candidate"})


def _normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN_RE.findall(str(text or "")) if len(token) > 1)


def retry_similarity(first: str, second: str) -> float:
    """Language-agnostic lexical/sequence proxy for the same communication attempt.

    This is intentionally conservative and only corroborates visual evidence; it is
    not the final semantic grouping model.
    """
    a = _normalized_tokens(first)
    b = _normalized_tokens(second)
    if not a or not b:
        return 0.0
    aset, bset = set(a), set(b)
    union = aset | bset
    jaccard = len(aset & bset) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, " ".join(a), " ".join(b)).ratio()
    prefix = 0
    for left, right in zip(a, b):
        if left != right:
            break
        prefix += 1
    prefix_ratio = prefix / max(1, min(len(a), len(b)))
    return max(jaccard, 0.72 * sequence + 0.28 * prefix_ratio)


def _events_near_transition(
    timeline: LocalPerformanceTimeline,
    current: CandidateTake,
    following: CandidateTake,
    *,
    before_sec: float = 0.55,
    after_sec: float = 1.50,
) -> tuple[TemporalEvent, ...]:
    start = max(current.start, current.end - before_sec)
    end = min(following.start, current.end + after_sec)
    if end < start:
        end = current.end + after_sec
    return tuple(
        event for event in timeline.events
        if event.end >= start and event.start <= end
        and event.kind in (_RESET_KINDS | _BREAK_KINDS)
    )


def confirm_local_performance_events(
    takes: Iterable[CandidateTake],
    timelines: Iterable[LocalPerformanceTimeline],
    context: WholeVideoContext,
    *,
    max_retry_gap_sec: float = 4.0,
    minimum_retry_similarity: float = 0.58,
) -> tuple[WholeVideoContext, Tuple[dict, ...]]:
    """Promote strong candidate clusters into actionable temporal evidence.

    Confirmation requires a nearby later take from the same source that strongly
    resembles the same communication attempt plus local reset/break evidence.  Two
    visual evidence families (reset + disengagement/expression) confirm a rejected
    take.  A single family is retained as a non-destructive retry_setup edge event.
    """
    take_tuple = tuple(sorted(takes, key=lambda item: (item.source_order, item.start, item.end)))
    timeline_by_source = {item.source_asset_id: item for item in timelines}
    additions_by_source: dict[str, list[TemporalEvent]] = {}
    diagnostics = []

    by_source: dict[str, list[CandidateTake]] = {}
    for take in take_tuple:
        by_source.setdefault(take.source_asset_id, []).append(take)

    for source_id, source_takes in by_source.items():
        timeline = timeline_by_source.get(source_id)
        if timeline is None or not timeline.events:
            continue
        for index, current in enumerate(source_takes[:-1]):
            # Search only the next few temporally-close takes; this avoids pairing
            # a recurring topic minutes later with a recording retry.
            best = None
            for following in source_takes[index + 1:index + 4]:
                gap = following.start - current.end
                if gap > max_retry_gap_sec:
                    break
                if gap < -0.15:
                    continue
                similarity = retry_similarity(current.text, following.text)
                if similarity < minimum_retry_similarity:
                    continue
                if best is None or similarity > best[0]:
                    best = (similarity, following)
            if best is None:
                continue

            similarity, following = best
            nearby = _events_near_transition(timeline, current, following)
            reset_events = tuple(event for event in nearby if event.kind in _RESET_KINDS)
            break_events = tuple(event for event in nearby if event.kind in _BREAK_KINDS)
            if not reset_events and not break_events:
                continue

            evidence_conf = max(event.confidence for event in nearby)
            families = int(bool(reset_events)) + int(bool(break_events))
            confirmed_kind = "wrong_take" if families >= 2 else "retry_setup"
            confidence = min(
                0.97,
                0.66 + 0.16 * similarity + 0.07 * families + 0.06 * evidence_conf,
            )

            if confirmed_kind == "wrong_take":
                # Strong multimodal rejection + near retry means the earlier
                # attempt itself is the failed take, even if the words are valid.
                confirmed = TemporalEvent(
                    source_asset_id=source_id,
                    start=current.start,
                    end=current.end,
                    kind="wrong_take",
                    confidence=confidence,
                    description=(
                        "dense local reset/break evidence followed by a temporally-close "
                        f"retry of the same communication attempt (similarity={similarity:.2f})"
                    ),
                )
            else:
                strongest = max(nearby, key=lambda event: event.confidence)
                # Edge-only evidence remains useful for precision trimming but is
                # not enough to reject a semantically complete take by itself.
                confirmed = TemporalEvent(
                    source_asset_id=source_id,
                    start=max(current.start, min(strongest.start, current.end - 0.08)),
                    end=max(current.end, strongest.end),
                    kind="retry_setup",
                    confidence=min(0.86, confidence),
                    description=(
                        "local performance change adjacent to a likely retry; preserve "
                        "the spoken take unless stronger evidence rejects it"
                    ),
                )

            additions_by_source.setdefault(source_id, []).append(confirmed)
            diagnostics.append({
                "source_asset_id": source_id,
                "take_id": current.clip_id,
                "retry_take_id": following.clip_id,
                "retry_similarity": round(similarity, 4),
                "gap_sec": round(max(0.0, following.start - current.end), 4),
                "candidate_event_kinds": [event.kind for event in nearby],
                "confirmed_kind": confirmed.kind,
                "confidence": round(confirmed.confidence, 4),
            })

    if not additions_by_source or not context.sources:
        return context, tuple(diagnostics)

    sources = []
    for source in context.sources:
        additions = tuple(additions_by_source.get(source.source_asset_id, ()))
        if not additions:
            sources.append(source)
            continue
        combined = tuple(sorted(source.events + additions, key=lambda event: (event.start, event.end, event.kind)))
        sources.append(replace(source, events=combined))
    return replace(context, sources=tuple(sources)), tuple(diagnostics)
