"""Conservative mini-session boundaries for compilation-style raw footage.

A single uploaded MP4 may be a compilation containing multiple creators/products. Retry
reasoning must never connect takes across those internal edits. This module derives only
high-confidence boundaries from dense local performance events already present in the
whole-video context, then scopes take grouping to the resulting mini-sessions.

It is intentionally fail-open: a boundary is used only when multiple independent visual
discontinuity families coincide and the inferred cut falls between candidate takes.
Nothing is deleted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from .contracts import CandidateTake
from .providers import ProviderStatus
from .take_grouping_provider import (
    TakeGroupingProvider,
    TakeGroupingProviderResult,
    safe_group_takes,
)
from .whole_video_analysis import WholeVideoContext

_CAMERA_KINDS = frozenset({"camera_disengagement_candidate"})
_FACE_KINDS = frozenset({"facial_expression_shift_candidate"})
_RESET_KINDS = frozenset({"body_reset_candidate", "hand_motion_reset_candidate"})


@dataclass(frozen=True)
class SessionBoundary:
    source_asset_id: str
    timestamp: float
    confidence: float
    evidence_kinds: Tuple[str, ...]


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def infer_session_boundaries(
    context: WholeVideoContext | None,
    source_asset_id: str,
    *,
    cluster_radius_sec: float = 0.22,
) -> Tuple[SessionBoundary, ...]:
    """Infer only dense multi-family visual discontinuities.

    A creator gesture/reset alone is never a session boundary. We require camera
    disengagement, facial-geometry shift, and a body/hand reset in the same very tight
    temporal cluster. This is characteristic of hard compilation edits while remaining
    conservative for ordinary delivery motion.
    """
    events = tuple(sorted(_source_events(context, source_asset_id), key=lambda e: (e.start, e.end)))
    candidates = []
    for camera in events:
        if camera.kind not in _CAMERA_KINDS or camera.confidence < 0.78:
            continue
        center = (camera.start + camera.end) / 2.0
        nearby = tuple(
            event for event in events
            if abs(((event.start + event.end) / 2.0) - center) <= cluster_radius_sec
        )
        face = [event for event in nearby if event.kind in _FACE_KINDS and event.confidence >= 0.78]
        reset = [event for event in nearby if event.kind in _RESET_KINDS and event.confidence >= 0.88]
        if not face or not reset:
            continue
        confidence = min(0.99, (camera.confidence + max(e.confidence for e in face) + max(e.confidence for e in reset)) / 3.0)
        kinds = tuple(sorted({camera.kind, *(e.kind for e in face), *(e.kind for e in reset)}))
        candidates.append(SessionBoundary(source_asset_id, center, confidence, kinds))

    # Collapse duplicate detections around the same edit.
    collapsed = []
    for boundary in candidates:
        if collapsed and boundary.timestamp - collapsed[-1].timestamp <= 0.35:
            if boundary.confidence > collapsed[-1].confidence:
                collapsed[-1] = boundary
            continue
        collapsed.append(boundary)
    return tuple(collapsed)


def _usable_boundaries_between_takes(
    takes: Tuple[CandidateTake, ...],
    boundaries: Tuple[SessionBoundary, ...],
    *,
    maximum_gap_sec: float = 3.0,
) -> Tuple[SessionBoundary, ...]:
    if len(takes) < 2:
        return ()
    ordered = tuple(sorted(takes, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    usable = []
    for left, right in zip(ordered, ordered[1:]):
        if left.source_asset_id != right.source_asset_id:
            continue
        gap = right.start - left.end
        if gap < -0.02 or gap > maximum_gap_sec:
            continue
        for boundary in boundaries:
            if left.end - 0.03 <= boundary.timestamp <= right.start + 0.03:
                usable.append(boundary)
                break
    return tuple(usable)


def partition_takes_by_sessions(
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
) -> Tuple[Tuple[CandidateTake, ...], ...]:
    take_tuple = tuple(takes)
    if not take_tuple:
        return ()

    partitions = []
    by_source: dict[str, list[CandidateTake]] = {}
    for take in take_tuple:
        by_source.setdefault(take.source_asset_id, []).append(take)

    for source_id, source_takes in sorted(
        by_source.items(),
        key=lambda item: min((t.source_order, t.start) for t in item[1]),
    ):
        ordered = tuple(sorted(source_takes, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
        inferred = infer_session_boundaries(context, source_id)
        usable = _usable_boundaries_between_takes(ordered, inferred)
        boundary_times = tuple(boundary.timestamp for boundary in usable)

        current = []
        for index, take in enumerate(ordered):
            if current and index > 0:
                previous = ordered[index - 1]
                if any(previous.end - 0.03 <= timestamp <= take.start + 0.03 for timestamp in boundary_times):
                    partitions.append(tuple(current))
                    current = []
            current.append(take)
        if current:
            partitions.append(tuple(current))

    return tuple(partitions)


def safe_group_takes_by_sessions(
    provider: TakeGroupingProvider | None,
    takes: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    *,
    context_text: str = "",
) -> TakeGroupingProviderResult:
    """Run retry grouping independently inside each inferred mini-session."""
    take_tuple = tuple(takes)
    partitions = partition_takes_by_sessions(take_tuple, context)
    if len(partitions) <= 1:
        return safe_group_takes(provider, take_tuple, context_text=context_text)

    results = tuple(
        safe_group_takes(provider, partition, context_text=context_text)
        for partition in partitions
    )
    groups = tuple(group for result in results for group in result.groups)
    statuses = {result.status.status for result in results}
    reasons = [result.reason for result in results if result.reason]
    status_name = "applied" if "applied" in statuses else "baseline"
    status = ProviderStatus(
        provider="session_scoped_take_grouping",
        requested=provider is not None,
        available=True,
        status=status_name,
        reason=f"mini_sessions={len(partitions)}",
    )
    reason = "; ".join([f"session_boundary_scoped:{len(partitions)}", *reasons])
    return TakeGroupingProviderResult(groups=groups, status=status, reason=reason)
