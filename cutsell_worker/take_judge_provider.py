"""Provider boundary for Best Take ranking with deterministic fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

from .contracts import CandidateTake, RankedTake
from .providers import ProviderStatus
from .take_judge import rank_takes


@dataclass(frozen=True)
class TakeJudgeProviderResult:
    ranked: Tuple[RankedTake, ...]
    status: ProviderStatus


class TakeJudgeProvider(Protocol):
    def rank(self, takes: Tuple[CandidateTake, ...]) -> TakeJudgeProviderResult: ...


def safe_rank_takes(
    takes: Tuple[CandidateTake, ...],
    provider: TakeJudgeProvider | None = None,
) -> TakeJudgeProviderResult:
    baseline = rank_takes(takes)
    if provider is None or len(takes) < 2:
        return TakeJudgeProviderResult(
            baseline,
            ProviderStatus("baseline", False, True, "baseline"),
        )
    try:
        result = provider.rank(takes)
        expected = {take.clip_id for take in takes}
        actual = [item.clip_id for item in result.ranked]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("take judge must rank every candidate exactly once")
        if any(not 0.0 <= item.score <= 1.0 for item in result.ranked):
            raise ValueError("take judge score outside 0..1")
        return result
    except Exception as exc:
        return TakeJudgeProviderResult(
            baseline,
            ProviderStatus(
                provider=provider.__class__.__name__,
                requested=True,
                available=False,
                status="provider_error_fallback",
                reason=exc.__class__.__name__,
            ),
        )
