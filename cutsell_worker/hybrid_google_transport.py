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

    def __call__(self, compact_payload: Mapping[str, Any], max_output_tokens: int) -> Mapping[str, Any]:
        if not self.settings.enabled:
            raise RuntimeError("hybrid paid transport is disabled")

        input_tokens = estimate_tokens_from_chars(len(json.dumps(dict(compact_payload), ensure_ascii=False)))
        estimated_cost = self.settings.estimate_cost_usd(
            input_tokens=input_tokens,
            output_tokens=max_output_tokens,
            escalation=self.escalation,
        )
        if not self.settings.allows_estimated_session_cost(estimated_cost):
            raise RuntimeError("hybrid session cost cap exceeded")
        if not self.ledger.reserve(estimated_cost):
            raise RuntimeError("hybrid edit/test dollar budget exhausted")

        # The bake-off found no quality gain from spending extra thinking on the
        # escalation model, so both paths start at minimal thinking. A future measured
        # quality gain may change this explicitly.
        body = build_gemini_generate_content_request(
            compact_payload,
            max_output_tokens=max_output_tokens,
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
        return parse_gemini_generate_content_response(raw)
