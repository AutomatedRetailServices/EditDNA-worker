"""Provider-reliability coverage for the semantic-equivalence Gemini transport.

Mirrors test_cutsell_unified_selection_google.py's approach: these tests never
exercise real Selection/grouping logic, only the request/response contract,
pair_index validation, and the retry/cost-ledger interactions that the
unified_selection_google.py truncation lesson (RAW #119) says must be covered
for any new Gemini transport in this codebase.
"""
from __future__ import annotations

import json

import pytest
import requests

from cutsell_worker.hybrid_google_transport import DollarBudgetLedger
from cutsell_worker.hybrid_provider_settings import HybridProviderSettings
from cutsell_worker.semantic_idea_equivalence import IdeaEquivalencePair, IdeaEquivalenceRequest
from cutsell_worker.semantic_idea_equivalence_google import (
    GoogleSemanticEquivalenceArbiter,
    SemanticEquivalenceUnreliableResponseError,
    build_semantic_equivalence_request,
    output_token_reserve,
    parse_semantic_equivalence_response,
)


def _request(count: int) -> IdeaEquivalenceRequest:
    return IdeaEquivalenceRequest(pairs=tuple(
        IdeaEquivalencePair(left_text=f"left delivery {i}", right_text=f"right delivery {i}")
        for i in range(count)
    ))


def decisions_json(count: int, *, index_offset: int = 0) -> str:
    return json.dumps({
        "decisions": [
            {
                "pair_index": i + index_offset,
                "same_idea": i % 2 == 0,
                "confidence": 0.8,
                "reason": "shared topic, different wording",
            }
            for i in range(count)
        ]
    })


def gemini_response(text: str, *, finish_reason: str = "STOP", output_tokens: int = 60) -> dict:
    return {
        "candidates": [{
            "finishReason": finish_reason,
            "content": {"parts": [{"text": text}]},
        }],
        "usageMetadata": {"candidatesTokenCount": output_tokens},
    }


def truncated_response(*, finish_reason: str = "MAX_TOKENS") -> dict:
    text = '{"decisions":[{"pair_index":0,"same_idea":true,"confidence":0.9,"reas'
    return gemini_response(text, finish_reason=finish_reason)


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


def arbiter(session: FakeSession, *, ledger_usd: float = 0.05, max_retries: int = 1) -> GoogleSemanticEquivalenceArbiter:
    return GoogleSemanticEquivalenceArbiter(
        api_key="test-key",
        model=settings().primary_model,
        settings=settings(),
        ledger=DollarBudgetLedger(ledger_usd),
        session=session,
        max_retries=max_retries,
    )


def test_build_request_never_includes_clip_or_video_identity():
    request = _request(2)
    body = build_semantic_equivalence_request(request, max_output_tokens=400)
    prompt_text = body["contents"][0]["parts"][0]["text"]
    # No clip_id/timestamp/source keys anywhere in the payload -- the arbiter
    # sees only left_text/right_text/pair_index.
    assert "clip_id" not in prompt_text
    assert "source_asset_id" not in prompt_text
    assert '"pair_index":0' in prompt_text.replace(" ", "")


def test_parse_response_happy_path():
    body = gemini_response(decisions_json(2))
    decisions, output_tokens, finish_reason = parse_semantic_equivalence_response(body)
    assert len(decisions) == 2
    assert finish_reason == "STOP"
    assert output_tokens == 60


def test_parse_response_raises_on_truncation():
    body = truncated_response()
    with pytest.raises(SemanticEquivalenceUnreliableResponseError, match="MAX_TOKENS"):
        parse_semantic_equivalence_response(body)


def test_check_happy_path_returns_all_decisions_in_order():
    session = FakeSession([gemini_response(decisions_json(3))])
    result = arbiter(session).check(_request(3))
    assert result.available is True
    assert [d.pair_index for d in result.decisions] == [0, 1, 2]
    assert result.decisions[0].same_idea is True
    assert result.decisions[1].same_idea is False


def test_check_raises_on_pair_index_mismatch():
    # Every candidate_index is off by one -- this is exactly the class of
    # silent misalignment unified_selection_google.py guards against for
    # candidate_index; the semantic-equivalence transport must too.
    session = FakeSession([gemini_response(decisions_json(2, index_offset=1))])
    with pytest.raises(SemanticEquivalenceUnreliableResponseError, match="pair_index mismatch"):
        arbiter(session).check(_request(2))


def test_check_raises_on_decision_count_mismatch():
    session = FakeSession([gemini_response(decisions_json(1))])
    with pytest.raises(SemanticEquivalenceUnreliableResponseError, match="count mismatch"):
        arbiter(session).check(_request(2))


def test_check_retries_once_after_truncation_then_succeeds():
    session = FakeSession([truncated_response(), gemini_response(decisions_json(2))])
    result = arbiter(session, max_retries=1).check(_request(2))
    assert result.available is True
    assert len(session.calls) == 2


def test_check_gives_up_after_exhausting_retries():
    session = FakeSession([truncated_response(), truncated_response()])
    with pytest.raises(SemanticEquivalenceUnreliableResponseError):
        arbiter(session, max_retries=1).check(_request(2))
    assert len(session.calls) == 2


def test_check_empty_pairs_short_circuits_without_a_call():
    session = FakeSession([])
    result = arbiter(session).check(IdeaEquivalenceRequest(pairs=()))
    assert result.available is True
    assert result.decisions == ()
    assert session.calls == []


def test_check_raises_when_ledger_cannot_afford_even_the_reserve():
    session = FakeSession([gemini_response(decisions_json(2))])
    # A near-zero ledger cannot afford even the minimum reserve.
    with pytest.raises(RuntimeError, match="dollar budget exhausted"):
        arbiter(session, ledger_usd=0.0000001).check(_request(2))
    assert session.calls == []


def test_check_releases_unused_reservation_after_a_cheaper_real_call():
    session = FakeSession([gemini_response(decisions_json(2), output_tokens=10)])
    live_ledger = DollarBudgetLedger(0.05)
    result = GoogleSemanticEquivalenceArbiter(
        api_key="test-key",
        model=settings().primary_model,
        settings=settings(),
        ledger=live_ledger,
        session=session,
    ).check(_request(2))
    assert result.available is True
    # The reserved worst-case cost should have been mostly released back once
    # the real (small) usage came back.
    assert live_ledger.reserved_usd < 0.002


def test_check_rejects_model_outside_provider_policy():
    session = FakeSession([gemini_response(decisions_json(1))])
    bad = GoogleSemanticEquivalenceArbiter(
        api_key="test-key",
        model="gemini-not-approved",
        settings=settings(),
        ledger=DollarBudgetLedger(0.05),
        session=session,
    )
    with pytest.raises(ValueError, match="not approved"):
        bad.check(_request(1))


def test_output_token_reserve_scales_with_pair_count_but_respects_ceiling():
    small = output_token_reserve(1, ceiling=1500)
    large = output_token_reserve(14, ceiling=1500)
    assert small < large
    assert large <= 1500
    assert output_token_reserve(1000, ceiling=1500) == 1500
