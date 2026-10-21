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

    @property
    def at(self) -> float:
        """Backward-compatible timestamp alias for boundary consumers."""
        return self.timestamp


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
    """Infer only dense multi-family visual discontinuities."""
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
        confidence = min(
            0.99,
            (camera.confidence + max(e.confidence for e in face) + max(e.confidence for e in reset)) / 3.0,
        )
        kinds = tuple(sorted({camera.kind, *(e.kind for e in face), *(e.kind for e in reset)}))
        candidates.append(SessionBoundary(source_asset_id, center, confidence, kinds))

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


def _repair_session_group_coverage(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
) -> tuple[Tuple[Tuple[str, ...], ...], int, int]:
    """Guarantee every kept take survives session-scoped grouping exactly once.

    Grouping is not deletion authority. Provider/baseline partitioning may choose retry
    families, but it must never silently drop a candidate. Unknown IDs and duplicate IDs
    are removed; any omitted real candidate is restored as a singleton in natural source
    order so Best Take / Hybrid can make the explicit selection decision later.
    """
    natural_ids = tuple(take.clip_id for take in takes)
    allowed = set(natural_ids)
    seen: set[str] = set()
    normalized: list[Tuple[str, ...]] = []
    duplicate_or_unknown = 0

    for group in groups:
        kept: list[str] = []
        for raw_id in group:
            clip_id = str(raw_id)
            if clip_id not in allowed or clip_id in seen:
                duplicate_or_unknown += 1
                continue
            kept.append(clip_id)
            seen.add(clip_id)
        if kept:
            normalized.append(tuple(kept))

    missing = [clip_id for clip_id in natural_ids if clip_id not in seen]
    for clip_id in missing:
        normalized.append((clip_id,))
        seen.add(clip_id)

    order = {clip_id: index for index, clip_id in enumerate(natural_ids)}
    normalized.sort(key=lambda group: min(order.get(clip_id, 10**9) for clip_id in group))
    return tuple(normalized), len(missing), duplicate_or_unknown


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
        result = safe_group_takes(provider, take_tuple, context_text=context_text)
        repaired_groups, missing_count, duplicate_count = _repair_session_group_coverage(
            result.groups, take_tuple
        )
        if not missing_count and not duplicate_count:
            return result
        return TakeGroupingProviderResult(
            groups=repaired_groups,
            status=ProviderStatus(
                provider=result.status.provider,
                requested=result.status.requested,
                available=result.status.available,
                status="coverage_repaired",
                reason=f"missing_restored={missing_count};duplicates_removed={duplicate_count}",
            ),
            reason="; ".join(
                part for part in (
                    result.reason,
                    f"global_group_coverage_repaired:missing={missing_count}:duplicates={duplicate_count}",
                ) if part
            ),
        )

    results = tuple(
        safe_group_takes(provider, partition, context_text=context_text)
        for partition in partitions
    )
    raw_groups = tuple(group for result in results for group in result.groups)
    groups, missing_count, duplicate_count = _repair_session_group_coverage(raw_groups, take_tuple)
    statuses = {result.status.status for result in results}
    reasons = [result.reason for result in results if result.reason]
    status_name = "applied" if "applied" in statuses else "baseline"
    if missing_count or duplicate_count:
        status_name = "coverage_repaired"
    status = ProviderStatus(
        provider="session_scoped_take_grouping",
        requested=provider is not None,
        available=True,
        status=status_name,
        reason=(
            f"mini_sessions={len(partitions)};missing_restored={missing_count};"
            f"duplicates_removed={duplicate_count}"
        ),
    )
    reason = "; ".join([
        f"session_boundary_scoped:{len(partitions)}",
        *reasons,
        f"global_group_coverage:missing_restored={missing_count}:duplicates_removed={duplicate_count}",
    ])
    return TakeGroupingProviderResult(groups=groups, status=status, reason=reason)
