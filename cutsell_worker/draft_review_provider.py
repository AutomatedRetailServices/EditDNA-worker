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


def _sanitize_review_order(proposed: Tuple[str, ...], allowed: set[str]) -> tuple[Tuple[str, ...], bool]:
    """Keep valid unique selected ids; drop unknown ids and duplicates conservatively."""
    seen: set[str] = set()
    repaired = False
    kept: list[str] = []
    for raw_id in proposed:
        clip_id = str(raw_id)
        if clip_id not in allowed or clip_id in seen:
            repaired = True
            continue
        seen.add(clip_id)
        kept.append(clip_id)
    return tuple(kept), repaired


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
        sanitized, repaired = _sanitize_review_order(proposed, allowed)
        if not sanitized:
            raise ValueError("draft review returned no valid selected clips")
        issues = tuple(str(issue)[:240] for issue in result.issues[:20])
        reason = str(result.reason or "")[:800]
        if repaired:
            issues = (*issues, "review_output_repaired")
            reason = (reason + "; " if reason else "") + "unknown/duplicate ids removed"
        # Review may conservatively remove redundant/incoherent selected clips, but
        # cannot remove everything or reintroduce an alternate/discarded take.
        return DraftReviewResult(
            sanitized,
            bool(result.postable),
            issues,
            reason,
            ProviderStatus("openai", True, True, "applied"),
        )
    except Exception as exc:
        return DraftReviewResult(
            natural,
            False,
            ("review_provider_failed",),
            f"{exc.__class__.__name__}:{str(exc)[:240]}",
            ProviderStatus(
                provider=provider.__class__.__name__,
                requested=True,
                available=False,
                status="provider_error_fallback",
                reason=f"{exc.__class__.__name__}:{str(exc)[:160]}",
            ),
        )
