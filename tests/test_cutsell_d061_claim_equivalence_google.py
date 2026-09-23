"""D-061 Phase 2 -- provider-reliability coverage for the claim-equivalence
Gemini transport.

Mirrors test_cutsell_semantic_idea_equivalence_google.py's approach: these
tests never exercise real StoryValidator/ClaimCoverageBestTake logic, only
the request/response contract and the retry/cost-ledger interactions every
new Gemini transport in this codebase must cover.
"""
from __future__ import annotations

import json

import pytest
import requests

from cutsell_worker.claim_equivalence_google import (
    ClaimEquivalenceUnreliableResponseError,
    GoogleClaimEquivalenceArbiter,
    build_claim_equivalence_request,
    parse_claim_equivalence_response,
)
from cutsell_worker.hybrid_google_transport import DollarBudgetLedger
from cutsell_worker.hybrid_provider_settings import HybridProviderSettings


def decision_json(*, covered: bool = True, confidence: float = 0.85, reason: str = "same fact, reworded") -> str:
    return json.dumps({"covered": covered, "confidence": confidence, "reason": reason})


def gemini_response(text: str, *, finish_reason: str = "STOP", output_tokens: int = 40) -> dict:
    return {
        "candidates": [{
            "finishReason": finish_reason,
            "content": {"parts": [{"text": text}]},
        }],
        "usageMetadata": {"candidatesTokenCount": output_tokens},
    }


def truncated_response(*, finish_reason: str = "MAX_TOKENS") -> dict:
    return gemini_response('{"covered":true,"confidence":0.9,"rea', finish_reason=finish_reason)


class FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, bodies: list[dict]):
        self._bodies = bodies
        self.calls: list[tuple] = []

    def post(self, url, *, headers, json, timeout):
        idx = len(self.calls)
        self.calls.append((url, headers, json, timeout))
        body = self._bodies[idx] if idx < len(self._bodies) else self._bodies[-1]
        return FakeResponse(200, body)


def settings() -> HybridProviderSettings:
    return HybridProviderSettings(enabled=True, provider="google")


def arbiter(session: FakeSession, *, ledger_usd: float = 0.05, max_retries: int = 1) -> GoogleClaimEquivalenceArbiter:
    return GoogleClaimEquivalenceArbiter(
        api_key="test-key",
        model=settings().primary_model,
        settings=settings(),
        ledger=DollarBudgetLedger(ledger_usd),
        session=session,
        max_retries=max_retries,
    )


def test_build_request_never_includes_clip_or_video_identity():
    body = build_claim_equivalence_request("claim text", "realization text", max_output_tokens=200)
    prompt_text = body["contents"][0]["parts"][0]["text"]
    assert "clip_id" not in prompt_text
    assert "source_asset_id" not in prompt_text
    assert "claim text" in prompt_text
    assert "realization text" in prompt_text


def test_prompt_explicitly_instructs_number_negation_entity_causal_safety():
    body = build_claim_equivalence_request("claim", "realization", max_output_tokens=200)
    prompt_text = body["contents"][0]["parts"][0]["text"].lower()
    assert "number" in prompt_text or "quantity" in prompt_text
    assert "negated" in prompt_text
    assert "diagnosis" in prompt_text or "entity" in prompt_text
    assert "cause-and-effect" in prompt_text or "causal" in prompt_text
    assert "uncertain" in prompt_text


def test_parse_response_happy_path():
    body = gemini_response(decision_json(covered=True, confidence=0.85))
    covered, confidence, reason, output_tokens = parse_claim_equivalence_response(body)
    assert covered is True
    assert confidence == 0.85
    assert output_tokens == 40


def test_parse_response_raises_on_truncation():
    with pytest.raises(ClaimEquivalenceUnreliableResponseError, match="MAX_TOKENS"):
        parse_claim_equivalence_response(truncated_response())


def test_claim_covered_happy_path_true():
    session = FakeSession([gemini_response(decision_json(covered=True))])
    covered, confidence, reason = arbiter(session).claim_covered("claim", "realization")
    assert covered is True
    assert 0.0 <= confidence <= 1.0
    assert len(session.calls) == 1


def test_claim_covered_happy_path_false():
    session = FakeSession([gemini_response(decision_json(covered=False, reason="number differs"))])
    covered, _confidence, reason = arbiter(session).claim_covered("claim", "realization")
    assert covered is False
    assert reason


def test_claim_covered_retries_once_after_truncation_then_succeeds():
    session = FakeSession([truncated_response(), gemini_response(decision_json(covered=True))])
    covered, _confidence, _reason = arbiter(session, max_retries=1).claim_covered("claim", "realization")
    assert covered is True
    assert len(session.calls) == 2


def test_claim_covered_gives_up_after_exhausting_retries():
    session = FakeSession([truncated_response(), truncated_response()])
    with pytest.raises(ClaimEquivalenceUnreliableResponseError):
        arbiter(session, max_retries=1).claim_covered("claim", "realization")
    assert len(session.calls) == 2


def test_claim_covered_raises_when_ledger_cannot_afford_even_the_reserve():
    session = FakeSession([gemini_response(decision_json(covered=True))])
    with pytest.raises(RuntimeError, match="dollar budget exhausted"):
        arbiter(session, ledger_usd=0.0000001).claim_covered("claim", "realization")
    assert session.calls == []


def test_claim_covered_releases_unused_reservation_after_a_cheaper_real_call():
    session = FakeSession([gemini_response(decision_json(covered=True), output_tokens=5)])
    live_ledger = DollarBudgetLedger(0.05)
    covered, _confidence, _reason = GoogleClaimEquivalenceArbiter(
        api_key="test-key", model=settings().primary_model, settings=settings(),
        ledger=live_ledger, session=session,
    ).claim_covered("claim", "realization")
    assert covered is True
    assert live_ledger.reserved_usd < 0.0006


def test_claim_covered_rejects_model_outside_provider_policy():
    session = FakeSession([gemini_response(decision_json(covered=True))])
    bad = GoogleClaimEquivalenceArbiter(
        api_key="test-key", model="gemini-not-approved", settings=settings(),
        ledger=DollarBudgetLedger(0.05), session=session,
    )
    with pytest.raises(ValueError, match="not approved"):
        bad.claim_covered("claim", "realization")


def test_claim_covered_requires_api_key():
    with pytest.raises(ValueError, match="API key"):
        GoogleClaimEquivalenceArbiter(
            api_key="", model=settings().primary_model, settings=settings(),
            ledger=DollarBudgetLedger(0.05), session=FakeSession([]),
        ).claim_covered("claim", "realization")


def test_claim_covered_requires_transport_enabled():
    disabled = HybridProviderSettings(enabled=False, provider="google")
    with pytest.raises(RuntimeError, match="disabled"):
        GoogleClaimEquivalenceArbiter(
            api_key="test-key", model=disabled.primary_model, settings=disabled,
            ledger=DollarBudgetLedger(0.05), session=FakeSession([]),
        ).claim_covered("claim", "realization")
