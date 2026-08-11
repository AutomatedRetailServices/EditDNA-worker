"""Provider boundary for semantic retry/take grouping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

from .contracts import CandidateTake
from .providers import ProviderStatus
from .take_grouping import group_takes, retry_similarity


@dataclass(frozen=True)
class TakeGroupingProviderResult:
    groups: Tuple[Tuple[str, ...], ...]
    status: ProviderStatus
    reason: str = ""


class TakeGroupingProvider(Protocol):
    def group(
        self,
        takes: Tuple[CandidateTake, ...],
        context_text: str = "",
    ) -> TakeGroupingProviderResult: ...


def _baseline_groups(takes: Tuple[CandidateTake, ...]) -> Tuple[Tuple[str, ...], ...]:
    grouped = group_takes(takes)
    return tuple(tuple(item.clip_id for item in members) for members in grouped.values())


def _provider_members_compatible(left: CandidateTake, right: CandidateTake) -> bool:
    """Require lexical retry evidence before accepting a provider merge.

    Temporal proximity is supporting evidence, never sufficient evidence by itself.
    Nearby sequential story beats can be only fractions of a second apart, so they
    still need the same lexical retry signal used by the conservative local grouper.
    Distant material needs even stronger similarity.
    """
    if left.source_asset_id != right.source_asset_id:
        return False
    score = retry_similarity(left.text, right.text)
    gap = max(0.0, max(left.start, right.start) - min(left.end, right.end))
    if gap <= 8.0:
        return score >= 0.72
    return score >= 0.82


def _constrain_provider_group(
    group: Tuple[str, ...],
    take_map: dict[str, CandidateTake],
) -> Tuple[Tuple[str, ...], ...]:
    """Split provider groups using complete-link retry compatibility.

    A new member must be compatible with every take already in a cluster. This avoids
    transitive chains where A resembles B and B resembles C but A and C are distinct
    story beats that should never become one destructive Best Take group.
    """
    members = [take_map[clip_id] for clip_id in group if clip_id in take_map]
    members.sort(key=lambda take: (take.source_order, take.start, take.end, take.clip_id))
    if len(members) <= 1:
        return (tuple(take.clip_id for take in members),) if members else ()

    clusters: list[list[CandidateTake]] = []
    for take in members:
        placed = False
        for cluster in clusters:
            if all(_provider_members_compatible(take, existing) for existing in cluster):
                cluster.append(take)
                placed = True
                break
        if not placed:
            clusters.append([take])
    return tuple(tuple(take.clip_id for take in cluster) for cluster in clusters)


def _repair_groups(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
) -> tuple[Tuple[Tuple[str, ...], ...], bool]:
    """Constrain provider output without throwing away useful grouping signal.

    Unknown ids and duplicate memberships are ignored. Any omitted real candidate is
    appended as a singleton group in natural source order. Provider multi-take groups
    are additionally split when they lack concrete retry evidence, so uncertain
    semantic similarity cannot silently delete unique material downstream.
    """
    natural_ids = tuple(take.clip_id for take in takes)
    take_map = {take.clip_id: take for take in takes}
    allowed = set(natural_ids)
    seen: set[str] = set()
    repaired = False
    normalized: list[Tuple[str, ...]] = []

    for raw_group in groups:
        kept: list[str] = []
        for raw_id in raw_group:
            clip_id = str(raw_id)
            if clip_id not in allowed or clip_id in seen:
                repaired = True
                continue
            seen.add(clip_id)
            kept.append(clip_id)
        if kept:
            constrained = _constrain_provider_group(tuple(kept), take_map)
            if len(constrained) > 1:
                repaired = True
            normalized.extend(group for group in constrained if group)
        elif raw_group:
            repaired = True

    for clip_id in natural_ids:
        if clip_id not in seen:
            normalized.append((clip_id,))
            seen.add(clip_id)
            repaired = True

    return tuple(normalized), repaired


def _group_gap(
    left_group: Tuple[str, ...],
    right_group: Tuple[str, ...],
    take_map: dict[str, CandidateTake],
) -> float:
    """Return the smallest temporal separation between two same-source groups."""
    gaps = []
    for left_id in left_group:
        left = take_map[left_id]
        for right_id in right_group:
            right = take_map[right_id]
            if left.source_asset_id != right.source_asset_id:
                continue
            gaps.append(max(0.0, max(left.start, right.start) - min(left.end, right.end)))
    return min(gaps) if gaps else float("inf")


def _reconcile_similarity_threshold(group_gap_sec: float) -> float:
    if group_gap_sec <= 8.0:
        return 0.80
    if group_gap_sec <= 30.0:
        return 0.90
    return 0.97


def _groups_should_reconcile(
    left_group: Tuple[str, ...],
    right_group: Tuple[str, ...],
    take_map: dict[str, CandidateTake],
) -> bool:
    """Recover provider splits with group-level timing + complete-link similarity.

    Timing is measured between the groups as recording attempts, not independently
    for each old member. Every cross-pair must still satisfy the lexical threshold,
    which prevents transitive A~B~C collapse while allowing a genuine multi-attempt
    retry group to absorb a later near-verbatim attempt.
    """
    left_members = []
    right_members = []
    for clip_id in left_group:
        take = take_map.get(clip_id)
        if take is None:
            return False
        left_members.append(take)
    for clip_id in right_group:
        take = take_map.get(clip_id)
        if take is None:
            return False
        right_members.append(take)
    if not left_members or not right_members:
        return False
    source_ids = {take.source_asset_id for take in (*left_members, *right_members)}
    if len(source_ids) != 1:
        return False

    threshold = _reconcile_similarity_threshold(_group_gap(left_group, right_group, take_map))
    return all(
        retry_similarity(left.text, right.text) >= threshold
        for left in left_members
        for right in right_members
    )


def _reconcile_missed_retries(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
) -> tuple[Tuple[Tuple[str, ...], ...], bool]:
    if len(groups) <= 1:
        return groups, False
    take_map = {take.clip_id: take for take in takes}
    merged: list[list[str]] = []
    changed = False

    for group in groups:
        target_index = None
        for index, existing in enumerate(merged):
            if _groups_should_reconcile(tuple(existing), group, take_map):
                target_index = index
                break
        if target_index is None:
            merged.append(list(group))
        else:
            merged[target_index].extend(group)
            changed = True

    ordered_groups: list[Tuple[str, ...]] = []
    for group in merged:
        unique = {clip_id for clip_id in group}
        ordered = sorted(
            unique,
            key=lambda clip_id: (
                take_map[clip_id].source_order,
                take_map[clip_id].start,
                take_map[clip_id].end,
                clip_id,
            ),
        )
        ordered_groups.append(tuple(ordered))
    ordered_groups.sort(
        key=lambda group: (
            take_map[group[0]].source_order,
            take_map[group[0]].start,
            take_map[group[0]].end,
            group[0],
        )
    )
    return tuple(ordered_groups), changed


def interstitial_retry_debris_ids(
    groups: Tuple[Tuple[str, ...], ...],
    takes: Tuple[CandidateTake, ...],
    *,
    max_retry_span_sec: float = 15.0,
    max_fragment_sec: float = 2.5,
    max_fragment_words: int = 5,
) -> frozenset[str]:
    """Identify only short incomplete speech trapped inside a confirmed retry span.

    This is deliberately structural rather than phrase-based: a fragment is removable
    only when two members of the same validated retry group bracket it in time. A
    short line outside that retry envelope is preserved, even when it is incomplete.
    """
    take_map = {take.clip_id: take for take in takes}
    ordered = tuple(sorted(takes, key=lambda item: (item.source_order, item.start, item.end, item.clip_id)))
    grouped_ids = {clip_id for group in groups for clip_id in group if len(group) >= 2}
    debris: set[str] = set()

    for group in groups:
        if len(group) < 2:
            continue
        members = [take_map[clip_id] for clip_id in group if clip_id in take_map]
        members.sort(key=lambda item: (item.source_order, item.start, item.end, item.clip_id))
        for left, right in zip(members, members[1:]):
            if left.source_asset_id != right.source_asset_id:
                continue
            if right.start - left.end > max_retry_span_sec:
                continue
            for candidate in ordered:
                if candidate.clip_id in grouped_ids:
                    continue
                if candidate.source_asset_id != left.source_asset_id:
                    continue
                if candidate.start < left.end or candidate.end > right.start:
                    continue
                words = len(candidate.text.split())
                if (
                    not candidate.complete_idea
                    and candidate.duration_sec <= max_fragment_sec
                    and words <= max_fragment_words
                ):
                    debris.add(candidate.clip_id)
    return frozenset(debris)


def safe_group_takes(
    provider: TakeGroupingProvider | None,
    takes: Tuple[CandidateTake, ...],
    context_text: str = "",
) -> TakeGroupingProviderResult:
    """Use semantic grouping while preserving every real candidate exactly once."""
    baseline = _baseline_groups(takes)
    if provider is None or len(takes) <= 1:
        return TakeGroupingProviderResult(
            baseline,
            ProviderStatus("baseline", False, True, "lexical_fallback"),
            "baseline",
        )
    try:
        result = provider.group(takes, context_text=context_text)
        if not result.groups:
            raise ValueError("take grouping returned no groups")
        normalized_input = tuple(tuple(str(item) for item in group) for group in result.groups if group)
        repaired_groups, repaired = _repair_groups(normalized_input, takes)
        if not repaired_groups:
            raise ValueError("take grouping produced no valid candidates")
        reconciled_groups, reconciled = _reconcile_missed_retries(repaired_groups, takes)
        reason = result.reason
        if repaired:
            reason = (reason + "; " if reason else "") + "provider_output_repaired"
        if reconciled:
            reason = (reason + "; " if reason else "") + "local_retry_reconciled"
        return TakeGroupingProviderResult(
            reconciled_groups,
            ProviderStatus("openai", True, True, "applied"),
            reason,
        )
    except Exception as exc:
        return TakeGroupingProviderResult(
            baseline,
            ProviderStatus(
                provider=provider.__class__.__name__,
                requested=True,
                available=False,
                status="provider_error_fallback",
                reason=f"{exc.__class__.__name__}:{str(exc)[:160]}",
            ),
            "baseline_fallback",
        )
