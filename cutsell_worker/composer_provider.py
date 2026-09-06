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
        context_text: str = "",
    ) -> ComposerProviderResult: ...


def _repair_order(proposed: Tuple[str, ...], natural: Tuple[str, ...]) -> tuple[Tuple[str, ...], bool]:
    """Keep valid provider ordering, ignore invalid ids/duplicates, append omissions naturally."""
    allowed = set(natural)
    seen: set[str] = set()
    repaired = False
    ordered: list[str] = []
    for raw_id in proposed:
        clip_id = str(raw_id)
        if clip_id not in allowed or clip_id in seen:
            repaired = True
            continue
        seen.add(clip_id)
        ordered.append(clip_id)
    for clip_id in natural:
        if clip_id not in seen:
            ordered.append(clip_id)
            seen.add(clip_id)
            repaired = True
    return tuple(ordered), repaired


def safe_compose_order(
    provider: ComposerProvider | None,
    takes: Tuple[CandidateTake, ...],
    labels: Tuple[SemanticLabel, ...],
    strategy: EditStrategy,
    context_text: str = "",
) -> ComposerProviderResult:
    """Allow AI to reorder only; never add, drop, duplicate, or alter creator speech."""
    natural_ids = tuple(take.clip_id for take in takes)
    if provider is None or len(takes) <= 1:
        return ComposerProviderResult(
            natural_ids,
            ProviderStatus("none", False, False, "not_requested"),
            "natural_order",
        )
    try:
        result = provider.order(takes, labels, strategy, context_text=context_text)
        proposed = tuple(str(item) for item in result.ordered_clip_ids)
        if not proposed:
            raise ValueError("composer returned empty order")
        repaired_order, repaired = _repair_order(proposed, natural_ids)
        if set(repaired_order) != set(natural_ids) or len(repaired_order) != len(natural_ids):
            raise ValueError("composer repair failed to preserve candidates")
        reason = result.reason
        if repaired:
            reason = (reason + "; " if reason else "") + "provider_output_repaired"
        return ComposerProviderResult(
            repaired_order,
            ProviderStatus("openai", True, True, "applied"),
            reason,
        )
    except Exception as exc:
        return ComposerProviderResult(
            natural_ids,
            ProviderStatus(
                provider=provider.__class__.__name__,
                requested=True,
                available=False,
                status="provider_error_fallback",
                reason=f"{exc.__class__.__name__}:{str(exc)[:160]}",
            ),
            "natural_order_fallback",
        )
