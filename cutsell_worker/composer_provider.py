"""Provider boundary for sales-aware flexible composition without destructive edits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

from .contracts import CandidateTake, EditStrategy, SemanticLabel
from .providers import ProviderStatus


@dataclass(frozen=True)
class ComposerProviderResult:
    ordered_clip_ids: Tuple[str, ...]
    status: ProviderStatus
    reason: str = ""


class ComposerProvider(Protocol):
    def order(
        self,
        takes: Tuple[CandidateTake, ...],
        labels: Tuple[SemanticLabel, ...],
        strategy: EditStrategy,
    ) -> ComposerProviderResult: ...


def safe_compose_order(
    provider: ComposerProvider | None,
    takes: Tuple[CandidateTake, ...],
    labels: Tuple[SemanticLabel, ...],
    strategy: EditStrategy,
) -> ComposerProviderResult:
    """Allow AI to reorder only; never add, drop, or duplicate creator speech."""
    natural_ids = tuple(take.clip_id for take in takes)
    if provider is None or len(takes) <= 1:
        return ComposerProviderResult(
            natural_ids,
            ProviderStatus("none", False, False, "not_requested"),
            "natural_order",
        )
    try:
        result = provider.order(takes, labels, strategy)
        proposed = tuple(str(item) for item in result.ordered_clip_ids)
        if len(proposed) != len(natural_ids):
            raise ValueError("composer changed clip count")
        if set(proposed) != set(natural_ids):
            raise ValueError("composer added/dropped unknown clip")
        if len(set(proposed)) != len(proposed):
            raise ValueError("composer duplicated clip")
        return ComposerProviderResult(proposed, result.status, result.reason)
    except Exception as exc:
        return ComposerProviderResult(
            natural_ids,
            ProviderStatus(
                provider=provider.__class__.__name__,
                requested=True,
                available=False,
                status="provider_error_fallback",
                reason=exc.__class__.__name__,
            ),
            "natural_order_fallback",
        )
