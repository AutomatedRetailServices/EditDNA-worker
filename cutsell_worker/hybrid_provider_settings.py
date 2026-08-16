"""Provider policy for the CutSell Hybrid Editorial Brain.

The production provider remains disabled by default. These settings describe the
approved architecture and budget ceilings only; no SDK or network call is made here.

Current candidate strategy (Aug 2026):
- primary: Gemini 3.5 Flash-Lite for high-volume structured editorial classification;
- escalation: Gemini 3.6 Flash for harder semantic conflicts;
- no OpenAI provider is enabled by this policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import os


def _env_bool(values: dict[str, str], key: str, default: bool = False) -> bool:
    raw = values.get(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(values: dict[str, str], key: str, default: float) -> float:
    try:
        return float(values.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class HybridProviderSettings:
    enabled: bool = False
    provider: str = "google"
    primary_model: str = "gemini-3.5-flash-lite"
    escalation_model: str = "gemini-3.6-flash"
    primary_input_per_million_usd: float = 0.30
    primary_output_per_million_usd: float = 2.50
    escalation_input_per_million_usd: float = 1.50
    escalation_output_per_million_usd: float = 7.50
    max_cost_per_session_usd: float = 0.02
    max_test_budget_usd: float = 2.00
    max_daily_budget_usd: float = 5.00
    escalation_conflict_score: float = 0.65
    escalation_local_confidence: float = 0.55

    def estimate_cost_usd(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        escalation: bool = False,
    ) -> float:
        in_tokens = max(0, int(input_tokens))
        out_tokens = max(0, int(output_tokens))
        if escalation:
            in_rate = self.escalation_input_per_million_usd
            out_rate = self.escalation_output_per_million_usd
        else:
            in_rate = self.primary_input_per_million_usd
            out_rate = self.primary_output_per_million_usd
        return (in_tokens / 1_000_000.0) * in_rate + (out_tokens / 1_000_000.0) * out_rate

    def should_escalate(self, *, local_confidence: float, conflict_score: float) -> bool:
        return bool(
            float(conflict_score) >= self.escalation_conflict_score
            or float(local_confidence) <= self.escalation_local_confidence
        )

    def allows_estimated_session_cost(self, cost_usd: float) -> bool:
        return 0.0 <= float(cost_usd) <= self.max_cost_per_session_usd


def load_hybrid_provider_settings(env: dict[str, str] | None = None) -> HybridProviderSettings:
    values = env if env is not None else os.environ
    provider = str(values.get("CUTSELL_HYBRID_PROVIDER", "google")).strip().lower()
    # Provider selection is intentionally constrained while the first production
    # integration is being validated. Unknown values fail closed instead of silently
    # enabling a network transport.
    if provider not in {"google", "none"}:
        provider = "none"
    enabled = _env_bool(values, "CUTSELL_HYBRID_LLM_ENABLED", False) and provider == "google"
    return HybridProviderSettings(
        enabled=enabled,
        provider=provider,
        primary_model=str(values.get("CUTSELL_HYBRID_PRIMARY_MODEL", "gemini-3.5-flash-lite")).strip(),
        escalation_model=str(values.get("CUTSELL_HYBRID_ESCALATION_MODEL", "gemini-3.6-flash")).strip(),
        max_cost_per_session_usd=max(0.0, _env_float(values, "CUTSELL_HYBRID_MAX_SESSION_USD", 0.02)),
        max_test_budget_usd=max(0.0, _env_float(values, "CUTSELL_HYBRID_TEST_BUDGET_USD", 2.00)),
        max_daily_budget_usd=max(0.0, _env_float(values, "CUTSELL_HYBRID_DAILY_BUDGET_USD", 5.00)),
        escalation_conflict_score=min(1.0, max(0.0, _env_float(values, "CUTSELL_HYBRID_ESCALATION_CONFLICT", 0.65))),
        escalation_local_confidence=min(1.0, max(0.0, _env_float(values, "CUTSELL_HYBRID_ESCALATION_LOCAL_CONFIDENCE", 0.55))),
    )
