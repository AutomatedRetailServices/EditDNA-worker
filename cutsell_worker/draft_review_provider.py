"""Constrained global draft-review boundary for CutSell.

The reviewer sees the complete proposed story after local cleanup and Best Take.
It may only keep/reorder existing selected clips. It cannot invent material, add a
clip that was not selected, duplicate clips, or alter source speech.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

from .contracts import CandidateTake, EditStrategy, SemanticLabel
from .providers import ProviderStatus


@dataclass(frozen=True)
class DraftReviewResult:
    ordered_clip_ids: Tuple[str, ...]
    postable: bool
    issues: Tuple[str, ...]
    reason: str
    status: ProviderStatus


class DraftReviewProvider(Protocol):
    def review(
        self,
        takes: Tuple[CandidateTake, ...],
        labels: Tuple[SemanticLabel, ...],
        strategy: EditStrategy,
        context_text: str = "",
    ) -> DraftReviewResult: ...


def safe_review_draft(
    provider: DraftReviewProvider | None,
    takes: Tuple[CandidateTake, ...],
    labels: Tuple[SemanticLabel, ...],
    strategy: EditStrategy,
    *,
    context_text: str = "",
) -> DraftReviewResult:
    natural = tuple(take.clip_id for take in takes)
    if provider is None or len(takes) <= 1:
        return DraftReviewResult(
            natural,
            True,
            (),
            "not_requested",
            ProviderStatus("none", False, False, "not_requested"),
        )
    try:
        result = provider.review(takes, labels, strategy, context_text=context_text)
        allowed = set(natural)
        proposed = tuple(str(item) for item in result.ordered_clip_ids)
        if not proposed:
            raise ValueError("draft review returned empty edit")
        if len(set(proposed)) != len(proposed):
            raise ValueError("draft review duplicated clip")
        if any(item not in allowed for item in proposed):
            raise ValueError("draft review introduced unknown clip")
        # Review may conservatively remove redundant/incoherent selected clips, but
        # cannot remove everything or reintroduce an alternate/discarded take.
        return DraftReviewResult(
            proposed,
            bool(result.postable),
            tuple(str(issue)[:240] for issue in result.issues[:20]),
            str(result.reason or "")[:800],
            result.status,
        )
    except Exception as exc:
        return DraftReviewResult(
            natural,
            False,
            ("review_provider_failed",),
            exc.__class__.__name__,
            ProviderStatus(
                provider=provider.__class__.__name__,
                requested=True,
                available=False,
                status="provider_error_fallback",
                reason=exc.__class__.__name__,
            ),
        )
