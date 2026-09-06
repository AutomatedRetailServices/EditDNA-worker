"""Gemini-backed semantic idea-equivalence arbiter for CutSell.

This is a narrow arbiter, not a Selection authority: it answers exactly one
question per candidate pair -- "do these two deliveries represent recording
attempts of the same intended idea/message?" -- and nothing else. It never
sees clip_ids, timestamps, or source/video identity (see
semantic_idea_equivalence.IdeaEquivalencePair), so it cannot encode a
Video00-specific or clip-specific rule even by construction.

The output-token budget below is derived the same way unified_selection_
google.py's was corrected to be derived (see that module's
_worst_case_decision_json_chars): from the exact marginal cost of one
additional decision object as it actually appears in Gemini's real
pretty-printed response, not a guessed or compact-JSON estimate. That
truncation bug is not being repeated here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import requests

from .hybrid_google_transport import DollarBudgetLedger
from .hybrid_payload import estimate_tokens_from_chars
from .hybrid_provider_settings import HybridProviderSettings
from .semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalenceRequest,
    IdeaEquivalenceResult,
)


class SemanticEquivalenceUnreliableResponseError(ValueError):
    """The provider response could not be trusted as one complete decision
    per pair: truncated/malformed JSON, a missing field, or a decision
    count that does not match the request. Always raised instead of ever
    treating a partial response as an applied result -- safe_check_idea_
    equivalence's fail-open path is the only place this may take effect,
    and it discards the response entirely (same_idea=False everywhere)."""


_REASON_CODES_HINT_MAX_CHARS = 60  # prompt asks for a concise reason; budgeted generously below


def _worst_case_pair_decision_json_chars() -> int:
    """Exact marginal cost, in characters, of one additional pair decision
    as it actually appears embedded in a real pretty-printed (indent=2)
    `{"decisions": [...]}` array -- derived by diffing a 1-item and 2-item
    array, exactly as unified_selection_google.py's corrected estimate is,
    rather than assuming compact serialization."""
    sample = {
        "pair_index": 999,
        "same_idea": False,
        "confidence": 0.95,
        "reason": "x" * _REASON_CODES_HINT_MAX_CHARS,
    }
    one = json.dumps({"decisions": [sample]}, indent=2)
    two = json.dumps({"decisions": [sample, sample]}, indent=2)
    return len(two) - len(one)


# Same margin rationale as unified_selection_google.py's
# _JSON_FORMATTING_SAFETY_MARGIN: a margin on top of the exact measurement,
# not a replacement for it -- protects against tokenizer/formatting drift.
_JSON_FORMATTING_SAFETY_MARGIN = 1.20
_TOKENS_PER_PAIR_DECISION = estimate_tokens_from_chars(
    int(_worst_case_pair_decision_json_chars() * _JSON_FORMATTING_SAFETY_MARGIN)
)
_DECISION_ARRAY_OVERHEAD_TOKENS = estimate_tokens_from_chars(len('{"decisions":[]}') + 8)


def output_token_reserve(pair_count: int, *, ceiling: int) -> int:
    """Worst-case output token budget for `pair_count` decisions, capped at
    `ceiling`. Mirrors unified_selection_google.output_token_reserve."""
    return min(
        ceiling,
        max(320, _TOKENS_PER_PAIR_DECISION * max(0, int(pair_count)) + _DECISION_ARRAY_OVERHEAD_TOKENS),
    )


def _response_schema() -> dict[str, Any]:
    # No array length bound -- see unified_selection_google.py's own
    # documented isolation-probe finding that an exact/loose length bound
    # 400s at scale on this same model. The caller already validates the
    # decision count/index coverage in Python after the response returns.
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pair_index": {"type": "integer", "minimum": 0},
                        "same_idea": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reason": {"type": "string"},
                    },
                    "required": ["pair_index", "same_idea", "confidence", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def build_semantic_equivalence_request(
    request: IdeaEquivalenceRequest, *, max_output_tokens: int,
) -> dict[str, Any]:
    pairs_payload = [
        {"pair_index": index, "left_text": pair.left_text, "right_text": pair.right_text}
        for index, pair in enumerate(request.pairs)
    ]
    prompt = (
        "You are checking whether pairs of spoken deliveries are recording attempts of "
        "the SAME intended idea or message, or two DIFFERENT ideas. You are not selecting "
        "or ranking anything -- answer only same_idea (true/false) and confidence (0-1) "
        "for each pair, with one concise general reason (no more than a dozen words, no "
        "quoting the input verbatim). "
        "Two texts are the SAME idea if a human editor would consider them competing "
        "recording attempts of one intended statement, even with very different wording, "
        "different length, or one being incomplete/stumbled. Two texts are DIFFERENT ideas "
        "if they convey distinct information, topics, or story beats, even if they share "
        "vocabulary. When genuinely uncertain, answer same_idea=false (different) -- "
        "preserving a distinct beat is always safer than merging two unrelated ones. "
        "Return exactly one decision per pair, in the same order, with pair_index equal to "
        "its zero-based position in that order. Do not echo the input text back. Output "
        "only the requested JSON schema.\n\n"
        + json.dumps({"pairs": pairs_payload}, ensure_ascii=False, separators=(",", ":"))
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


def parse_semantic_equivalence_response(raw: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], int, str]:
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise SemanticEquivalenceUnreliableResponseError("Gemini semantic-equivalence response missing candidates")
    first = candidates[0]
    if not isinstance(first, Mapping):
        raise SemanticEquivalenceUnreliableResponseError("Gemini semantic-equivalence candidate malformed")
    finish_reason = str(first.get("finishReason") or "")
    content = first.get("content")
    if not isinstance(content, Mapping):
        raise SemanticEquivalenceUnreliableResponseError(
            f"Gemini semantic-equivalence response missing content (finishReason={finish_reason!r})"
        )
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise SemanticEquivalenceUnreliableResponseError(
            f"Gemini semantic-equivalence response missing parts (finishReason={finish_reason!r})"
        )
    text = "".join(str(part.get("text") or "") for part in parts if isinstance(part, Mapping))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SemanticEquivalenceUnreliableResponseError(
            f"Gemini semantic-equivalence response was not valid JSON (finishReason={finish_reason!r}): {exc}"
        ) from exc
    decisions = parsed.get("decisions") if isinstance(parsed, Mapping) else None
    if not isinstance(decisions, list):
        raise SemanticEquivalenceUnreliableResponseError(
            f"Gemini semantic-equivalence response missing decisions (finishReason={finish_reason!r})"
        )
    usage = raw.get("usageMetadata") or {}
    try:
        output_tokens = max(0, int(usage.get("candidatesTokenCount") or 0)) if isinstance(usage, Mapping) else 0
    except (TypeError, ValueError):
        output_tokens = 0
    return decisions, output_tokens, finish_reason


@dataclass
class GoogleSemanticEquivalenceArbiter:
    api_key: str
    model: str
    settings: HybridProviderSettings
    ledger: DollarBudgetLedger
    timeout_sec: float = 60.0
    session: Any = requests
    max_input_tokens: int = 8_000
    max_output_tokens: int = 1_500
    max_retries: int = 1

    def _call_once(
        self, request: IdeaEquivalenceRequest, *, output_tokens_requested: int,
    ) -> tuple[list[IdeaEquivalenceDecision], int]:
        body = build_semantic_equivalence_request(request, max_output_tokens=output_tokens_requested)
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
            raise SemanticEquivalenceUnreliableResponseError("Gemini semantic-equivalence HTTP response must be an object")
        raw_decisions, output_tokens, finish_reason = parse_semantic_equivalence_response(raw)
        if len(raw_decisions) != len(request.pairs):
            raise SemanticEquivalenceUnreliableResponseError(
                "semantic equivalence ordered decision count mismatch "
                f"(expected {len(request.pairs)}, got {len(raw_decisions)}, finishReason={finish_reason!r})"
            )
        mismatches = [
            (i, item.get("pair_index") if isinstance(item, Mapping) else "<malformed>")
            for i, item in enumerate(raw_decisions)
            if not isinstance(item, Mapping) or item.get("pair_index") != i
        ]
        if mismatches:
            raise SemanticEquivalenceUnreliableResponseError(
                "semantic equivalence pair_index mismatch (expected sequential "
                f"0..{len(request.pairs) - 1}, mismatches={mismatches[:5]}, finishReason={finish_reason!r})"
            )
        decisions = [
            IdeaEquivalenceDecision(
                pair_index=i,
                same_idea=bool(item.get("same_idea")),
                confidence=float(item.get("confidence", -1.0)),
                reason=str(item.get("reason") or ""),
            )
            for i, item in enumerate(raw_decisions)
        ]
        return decisions, output_tokens

    def _max_affordable_output_tokens(self, input_tokens: int) -> int:
        input_cost = self.settings.estimate_cost_usd(input_tokens=input_tokens, output_tokens=0, escalation=False)
        budget_for_output = max(0.0, self.ledger.remaining_usd - input_cost)
        rate = self.settings.primary_output_per_million_usd
        if rate <= 0:
            return self.max_output_tokens
        return int(budget_for_output / (rate / 1_000_000.0))

    def check(self, request: IdeaEquivalenceRequest) -> IdeaEquivalenceResult:
        if not self.api_key:
            raise ValueError("Gemini API key required")
        if not self.settings.enabled or self.settings.provider != "google":
            raise RuntimeError("semantic equivalence paid transport is disabled")
        if self.model not in {self.settings.primary_model, self.settings.escalation_model}:
            raise ValueError("Gemini model not approved by provider policy")
        if not request.pairs:
            return IdeaEquivalenceResult((), "google", self.model, True, True, 0, 0)

        payload_chars = len(json.dumps(
            {"pairs": [{"left_text": p.left_text, "right_text": p.right_text} for p in request.pairs]},
            ensure_ascii=False,
        ))
        input_tokens = estimate_tokens_from_chars(payload_chars)
        if input_tokens > self.max_input_tokens:
            raise ValueError("semantic equivalence input token budget exceeded")

        output_reserve = output_token_reserve(len(request.pairs), ceiling=self.max_output_tokens)

        for attempt in range(self.max_retries + 1):
            estimated_cost = self.settings.estimate_cost_usd(
                input_tokens=input_tokens, output_tokens=output_reserve, escalation=False,
            )
            if not self.settings.allows_estimated_session_cost(estimated_cost):
                raise RuntimeError("semantic equivalence session cost cap exceeded")
            if not self.ledger.reserve(estimated_cost):
                raise RuntimeError("semantic equivalence dollar budget exhausted")

            try:
                decisions, output_tokens = self._call_once(request, output_tokens_requested=output_reserve)
            except (requests.RequestException, SemanticEquivalenceUnreliableResponseError):
                self.ledger.release(estimated_cost)
                if attempt < self.max_retries:
                    bumped = max(output_reserve, int(output_reserve * 1.5))
                    affordable = self._max_affordable_output_tokens(input_tokens)
                    next_reserve = min(self.max_output_tokens, bumped, affordable)
                    if next_reserve < output_reserve:
                        raise
                    output_reserve = next_reserve
                    continue
                raise
            else:
                actual_cost = self.settings.estimate_cost_usd(
                    input_tokens=input_tokens, output_tokens=output_tokens, escalation=False,
                )
                if actual_cost < estimated_cost:
                    self.ledger.release(estimated_cost - actual_cost)
                return IdeaEquivalenceResult(
                    decisions=tuple(decisions),
                    provider="google",
                    model=self.model,
                    requested=True,
                    available=True,
                    estimated_input_tokens=input_tokens,
                    estimated_output_tokens=output_tokens,
                )
