"""Provider boundaries for optional CutSell intelligence layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

from .contracts import CandidateTake, SemanticLabel


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    requested: bool
    available: bool
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class SemanticProviderResult:
    labels: Tuple[SemanticLabel, ...]
    status: ProviderStatus


class SemanticProvider(Protocol):
    def classify(self, takes: Tuple[CandidateTake, ...]) -> SemanticProviderResult: ...


class NoopSemanticProvider:
    """Safe default until an external semantic provider is configured."""

    def classify(self, takes: Tuple[CandidateTake, ...]) -> SemanticProviderResult:
        return SemanticProviderResult(
            labels=(),
            status=ProviderStatus(
                provider="none",
                requested=False,
                available=False,
                status="not_requested",
                reason=None,
            ),
        )


def safe_semantic_classify(provider: SemanticProvider, takes: Tuple[CandidateTake, ...]) -> SemanticProviderResult:
    try:
        return provider.classify(takes)
    except Exception as exc:
        return SemanticProviderResult(
            labels=(),
            status=ProviderStatus(
                provider=provider.__class__.__name__,
                requested=True,
                available=False,
                status="provider_error",
                reason=exc.__class__.__name__,
            ),
        )
