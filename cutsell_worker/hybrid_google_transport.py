"""Explicitly gated Gemini HTTP transport for CutSell Hybrid Editorial Brain.

Importing this module cannot make a request. The caller must provide an API key,
explicitly-enabled provider settings, and a dollar ledger with remaining budget.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import requests

from .hybrid_google import (
    build_gemini_generate_content_request,
    parse_gemini_generate_content_response,
)
from .hybrid_payload import estimate_tokens_from_chars
from .hybrid_provider_settings import HybridProviderSettings


def _compact_output_token_ceiling(compact_payload: Mapping[str, Any], requested_max: int) -> int:
    """Derive a bounded output ceiling for the compact structured decision schema.

    Run #31 proved the previous 192-token ceiling for six real candidate decisions was
    too aggressive: nearly every six-candidate response failed validation while smaller
    2-4 candidate responses remained available. Keep the compact schema and six-item
    batching, but reserve enough room for the real clip IDs plus JSON structure. Actual
    provider usage is still reconciled after every successful response, so this raises
    reliability without turning the old flat 500-token reservation back on for all calls.
    """
    hard_max = max(1, int(requested_max))
    raw_candidates = compact_payload.get("candidates")
    candidate_count = len(raw_candidates) if isinstance(raw_candidates, (list, tuple)) else 0
    if candidate_count <= 0:
        return hard_max
    if candidate_count <= 2:
        schema_ceiling = 192
    elif candidate_count <= 4:
        schema_ceiling = 256
    elif candidate_count <= 6:
        schema_ceiling = 320
    else:
        schema_ceiling = min(500, 80 + (32 * candidate_count))
    return min(hard_max, schema_ceiling)


def _raw_output_tokens(raw: Mapping[str, Any]) -> int:
    """Read Gemini-reported billed candidate tokens even when structured parsing fails."""
    usage = raw.get("usageMetadata") or {}
    if not isinstance(usage, Mapping):
        return 0
    try:
        return max(0, int(usage.get("candidatesTokenCount") or 0))
    except (TypeError, ValueError):
        return 0


@dataclass
class DollarBudgetLedger:
    max_usd: float
    reserved_usd: float = 0.0

    @property
    def remaining_usd(self) -> float:
        return max(0.0, float(self.max_usd) - float(self.reserved_usd))

    def reserve(self, estimated_usd: float) -> bool:
        cost = max(0.0, float(estimated_usd))
        if cost > self.remaining_usd:
            return False
        self.reserved_usd += cost
        return True

    def release(self, unused_usd: float) -> None:
        """Return unused preflight reservation without ever making the ledger negative."""
        amount = max(0.0, float(unused_usd))
        self.reserved_usd = max(0.0, float(self.reserved_usd) - amount)


@dataclass
class GoogleGeminiTransport:
    api_key: str
    model: str
    settings: HybridProviderSettings
    ledger: DollarBudgetLedger
    escalation: bool = False
    timeout_sec: float = 30.0
    session: Any = requests

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("Gemini API key required")
        if self.model not in {self.settings.primary_model, self.settings.escalation_model}:
            raise ValueError("Gemini model not approved by hybrid provider policy")

    def _reconcile_reported_usage(self, *, input_tokens: int, output_tokens: int, estimated_cost: float) -> None:
        actual_cost = self.settings.estimate_cost_usd(
            input_tokens=input_tokens,
            output_tokens=max(0, int(output_tokens)),
            escalation=self.escalation,
        )
        if actual_cost < estimated_cost:
            self.ledger.release(estimated_cost - actual_cost)

    def __call__(self, compact_payload: Mapping[str, Any], max_output_tokens: int) -> Mapping[str, Any]:
        if not self.settings.enabled:
            raise RuntimeError("hybrid paid transport is disabled")

        input_tokens = estimate_tokens_from_chars(len(json.dumps(dict(compact_payload), ensure_ascii=False)))
        effective_output_tokens = _compact_output_token_ceiling(compact_payload, max_output_tokens)
        estimated_cost = self.settings.estimate_cost_usd(
            input_tokens=input_tokens,
            output_tokens=effective_output_tokens,
            escalation=self.escalation,
        )
        if not self.settings.allows_estimated_session_cost(estimated_cost):
            raise RuntimeError("hybrid session cost cap exceeded")
        if not self.ledger.reserve(estimated_cost):
            raise RuntimeError("hybrid edit/test dollar budget exhausted")

        # Reserve the bounded worst-case request before HTTP so the hard edit cap can
        # never be crossed optimistically. Network/HTTP failures retain the full reserve
        # because billed usage is unknown. Once Gemini returns an HTTP-success JSON body,
        # however, usageMetadata is authoritative enough to reconcile the reservation
        # even if the structured decision payload itself is malformed or truncated.
        body = build_gemini_generate_content_request(
            compact_payload,
            max_output_tokens=effective_output_tokens,
            thinking_level="minimal",
        )
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        response = self.session.post(
            endpoint,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json=body,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, Mapping):
            raise ValueError("Gemini HTTP response must be an object")

        reported_output_tokens = _raw_output_tokens(raw)
        try:
            parsed = parse_gemini_generate_content_response(raw)
        except ValueError:
            self._reconcile_reported_usage(
                input_tokens=input_tokens,
                output_tokens=reported_output_tokens,
                estimated_cost=estimated_cost,
            )
            raise

        actual_output_tokens = max(0, int(parsed.get("output_tokens") or reported_output_tokens))
        self._reconcile_reported_usage(
            input_tokens=input_tokens,
            output_tokens=actual_output_tokens,
            estimated_cost=estimated_cost,
        )
        return parsed
