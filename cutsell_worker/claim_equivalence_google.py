"""Gemini-backed claim-equivalence arbiter for CutSell (D-061 Phase 2).

This is a narrow arbiter, not a Selection authority: it answers exactly one
question per call -- "does the winning realization preserve this specific
audience-facing claim, even if paraphrased?" -- and nothing else. It never
sees clip_ids, timestamps, or source/video identity, mirroring every other
bounded arbiter in this codebase (`semantic_idea_equivalence_google.py`'s
`GoogleSemanticEquivalenceArbiter`, this module's own direct model).

Implements `semantic_claims.ClaimEquivalenceArbiter`: `claim_covered(claim_
text, winning_realization_text) -> (covered: bool, confidence: float,
reason: str)`. Called ONLY by `semantic_claims.resolve_ambiguous_coverage`,
which itself is only reached when deterministic claim_coverage already
placed the claim in the genuinely ambiguous band (`AMBIGUOUS_COVERAGE_FLOOR
<= coverage < COVERAGE_THRESHOLD`) -- a confidently-covered or confidently-
mismatched claim (including every case `claim_coverage`'s own deterministic
number/negation/causal-direction guards catch) never reaches this arbiter at
all, and `resolve_ambiguous_coverage` fails open to NOT COVERED on any
exception or non-True verdict. This arbiter cannot override a deterministic
hard mismatch even if it wanted to -- it is never consulted for one.

The prompt itself additionally, defensively instructs the model to answer
NOT covered whenever numbers, negation polarity, diagnosis/entity identity,
or causal direction differ, or whenever genuinely uncertain -- belt and
braces on top of the structural guarantee above, not a substitute for it.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import requests

from .hybrid_google_transport import DollarBudgetLedger
from .hybrid_payload import estimate_tokens_from_chars
from .hybrid_provider_settings import HybridProviderSettings


class ClaimEquivalenceUnreliableResponseError(ValueError):
    """The provider response could not be trusted: truncated/malformed
    JSON or a missing field. Always raised instead of ever treating a
    partial response as an applied result -- the caller
    (`semantic_claims.resolve_ambiguous_coverage`) catches any exception
    and fails open to NOT COVERED, never to COVERED."""


_MAX_REASON_CHARS = 60


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "covered": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string"},
        },
        "required": ["covered", "confidence", "reason"],
        "additionalProperties": False,
    }


def build_claim_equivalence_request(
    claim_text: str, winning_realization_text: str, *, max_output_tokens: int,
) -> dict[str, Any]:
    prompt = (
        "You are checking whether a SECOND text (the winning realization) preserves a "
        "specific factual claim made in a FIRST text (the claim), even if reworded. "
        "Answer covered=true only if the winning realization states the SAME proposition, "
        "allowing for paraphrase, different word order, or different length. "
        "Answer covered=false whenever ANY of the following differ between the two texts: "
        "a stated number, percentage, or quantity; whether the statement is affirmed or "
        "negated; the specific diagnosis, entity, or named thing being discussed; or the "
        "direction of a stated cause-and-effect relationship (X causes Y is NOT the same as "
        "Y causes X). When genuinely uncertain whether the same proposition is preserved, "
        "answer covered=false -- preserving information is always safer than assuming it. "
        "Provide a confidence 0-1 and one short general reason (no more than a dozen words, "
        "no quoting the input verbatim). Do not echo the input text back. "
        "Output only the requested JSON schema.\n\n"
        + json.dumps(
            {"claim": claim_text, "winning_realization": winning_realization_text},
            ensure_ascii=False, separators=(",", ":"),
        )
    )
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": int(max_output_tokens),
            "thinkingConfig": {"thinkingLevel": "low"},
            "responseMimeType": "application/json",
            "responseJsonSchema": _response_schema(),
        },
    }


def parse_claim_equivalence_response(raw: Mapping[str, Any]) -> tuple[bool, float, str, int]:
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ClaimEquivalenceUnreliableResponseError("Gemini claim-equivalence response missing candidates")
    first = candidates[0]
    if not isinstance(first, Mapping):
        raise ClaimEquivalenceUnreliableResponseError("Gemini claim-equivalence candidate malformed")
    finish_reason = str(first.get("finishReason") or "")
    content = first.get("content")
    if not isinstance(content, Mapping):
        raise ClaimEquivalenceUnreliableResponseError(
            f"Gemini claim-equivalence response missing content (finishReason={finish_reason!r})"
        )
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ClaimEquivalenceUnreliableResponseError(
            f"Gemini claim-equivalence response missing parts (finishReason={finish_reason!r})"
        )
    text = "".join(str(part.get("text") or "") for part in parts if isinstance(part, Mapping))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClaimEquivalenceUnreliableResponseError(
            f"Gemini claim-equivalence response was not valid JSON (finishReason={finish_reason!r}): {exc}"
        ) from exc
    if not isinstance(parsed, Mapping) or "covered" not in parsed:
        raise ClaimEquivalenceUnreliableResponseError(
            f"Gemini claim-equivalence response missing 'covered' (finishReason={finish_reason!r})"
        )
    usage = raw.get("usageMetadata") or {}
    try:
        output_tokens = max(0, int(usage.get("candidatesTokenCount") or 0)) if isinstance(usage, Mapping) else 0
    except (TypeError, ValueError):
        output_tokens = 0
    try:
        confidence = float(parsed.get("confidence", -1.0))
    except (TypeError, ValueError):
        confidence = -1.0
    return bool(parsed.get("covered")), confidence, str(parsed.get("reason") or ""), output_tokens


@dataclass
class GoogleClaimEquivalenceArbiter:
    api_key: str
    model: str
    settings: HybridProviderSettings
    ledger: DollarBudgetLedger
    timeout_sec: float = 60.0
    session: Any = requests
    max_input_tokens: int = 4_000
    max_output_tokens: int = 300
    max_retries: int = 1

    def _call_once(
        self, claim_text: str, winning_realization_text: str, *, output_tokens_requested: int,
    ) -> tuple[bool, float, str, int]:
        body = build_claim_equivalence_request(
            claim_text, winning_realization_text, max_output_tokens=output_tokens_requested,
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
            raise ClaimEquivalenceUnreliableResponseError("Gemini claim-equivalence HTTP response must be an object")
        return parse_claim_equivalence_response(raw)

    def claim_covered(self, claim_text: str, winning_realization_text: str) -> tuple[bool, float, str]:
        if not self.api_key:
            raise ValueError("Gemini API key required")
        if not self.settings.enabled or self.settings.provider != "google":
            raise RuntimeError("claim equivalence paid transport is disabled")
        if self.model not in {self.settings.primary_model, self.settings.escalation_model}:
            raise ValueError("Gemini model not approved by provider policy")

        payload_chars = len(json.dumps(
            {"claim": claim_text, "winning_realization": winning_realization_text}, ensure_ascii=False,
        ))
        input_tokens = estimate_tokens_from_chars(payload_chars)
        if input_tokens > self.max_input_tokens:
            raise ValueError("claim equivalence input token budget exceeded")

        output_reserve = self.max_output_tokens
        for attempt in range(self.max_retries + 1):
            estimated_cost = self.settings.estimate_cost_usd(
                input_tokens=input_tokens, output_tokens=output_reserve, escalation=False,
            )
            if not self.settings.allows_estimated_session_cost(estimated_cost):
                raise RuntimeError("claim equivalence session cost cap exceeded")
            if not self.ledger.reserve(estimated_cost):
                raise RuntimeError("claim equivalence dollar budget exhausted")

            try:
                covered, confidence, reason, output_tokens = self._call_once(
                    claim_text, winning_realization_text, output_tokens_requested=output_reserve,
                )
            except (requests.RequestException, ClaimEquivalenceUnreliableResponseError):
                self.ledger.release(estimated_cost)
                if attempt < self.max_retries:
                    continue
                raise
            else:
                actual_cost = self.settings.estimate_cost_usd(
                    input_tokens=input_tokens, output_tokens=output_tokens, escalation=False,
                )
                if actual_cost < estimated_cost:
                    self.ledger.release(estimated_cost - actual_cost)
                return covered, confidence, reason[:_MAX_REASON_CHARS]
        # Unreachable (loop above always returns or raises), kept for mypy/completeness.
        raise RuntimeError("claim equivalence call did not complete")
