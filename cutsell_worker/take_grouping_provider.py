"""Provider boundary for semantic retry/take grouping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

from .contracts import CandidateTake
from .providers import ProviderStatus
from .take_grouping import group_takes


@dataclass(frozen=True)
class TakeGroupingProviderResult:
    groups: Tuple[Tuple[str, ...], ...]
    status: ProviderStatus
    reason: str = ""


class TakeGroupingProvider(Protocol):
    def group(self, takes: Tuple[CandidateTake, ...]) -> TakeGroupingProviderResult: ...


def _baseline_groups(takes: Tuple[CandidateTake, ...]) -> Tuple[Tuple[str, ...], ...]:
    grouped = group_takes(takes)
    return tuple(tuple(item.clip_id for item in members) for members in grouped.values())


def safe_group_takes(
    provider: TakeGroupingProvider | None,
    takes: Tuple[CandidateTake, ...],
) -> TakeGroupingProviderResult:
    """Use semantic grouping only when it preserves every real candidate exactly once."""
    baseline = _baseline_groups(takes)
    if provider is None or len(takes) <= 1:
        return TakeGroupingProviderResult(
            baseline,
            ProviderStatus("baseline", False, True, "lexical_fallback"),
            "baseline",
        )
    try:
        result = provider.group(takes)
        expected = {take.clip_id for take in takes}
        flattened = [clip_id for group in result.groups for clip_id in group]
        if not result.groups or any(not group for group in result.groups):
            raise ValueError("take grouping returned empty group")
        if len(flattened) != len(expected):
            raise ValueError("take grouping changed candidate count")
        if set(flattened) != expected:
            raise ValueError("take grouping added/dropped candidate")
        if len(set(flattened)) != len(flattened):
            raise ValueError("take grouping duplicated candidate")
        return TakeGroupingProviderResult(
            tuple(tuple(str(item) for item in group) for group in result.groups),
            result.status,
            result.reason,
        )
    except Exception as exc:
        return TakeGroupingProviderResult(
            baseline,
            ProviderStatus(
                provider=provider.__class__.__name__,
                requested=True,
                available=False,
                status="provider_error_fallback",
                reason=exc.__class__.__name__,
            ),
            "baseline_fallback",
        )
