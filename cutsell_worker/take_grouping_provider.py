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
    """Require concrete retry evidence before accepting a provider merge.

    Nearby attempts can legitimately use different wording, while distant material
    needs very strong lexical similarity to be treated as the same retry. This errs
    toward preserving unique content instead of letting a broad semantic grouping
    create destructive overcut.
    """
    if left.source_asset_id == right.source_asset_id:
        gap = max(0.0, max(left.start, right.start) - min(left.end, right.end))
        if gap <= 8.0:
            return True
    return retry_similarity(left.text, right.text) >= 0.82


def _constrain_provider_group(
    group: Tuple[str, ...],
    take_map: dict[str, CandidateTake],
) -> Tuple[Tuple[str, ...], ...]:
    members = [take_map[clip_id] for clip_id in group if clip_id in take_map]
    members.sort(key=lambda take: (take.source_order, take.start, take.end, take.clip_id))
    if len(members) <= 1:
        return (tuple(take.clip_id for take in members),) if members else ()

    clusters: list[list[CandidateTake]] = []
    for take in members:
        placed = False
        for cluster in clusters:
            if any(_provider_members_compatible(take, existing) for existing in cluster):
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
        reason = result.reason
        if repaired:
            reason = (reason + "; " if reason else "") + "provider_output_repaired"
        return TakeGroupingProviderResult(
            repaired_groups,
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
