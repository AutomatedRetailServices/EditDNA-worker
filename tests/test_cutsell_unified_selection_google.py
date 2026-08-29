"""Provider-reliability coverage for the Unified Selection Gemini transport.

RAW #119 saw Gemini return a MAX_TOKENS-truncated, unparseable response with
no retry available, so the reasoner failed open on the first hiccup even
though selection_reasoner_status had been "applied" moments earlier (RAW
#118). This file targets exactly that class of failure: truncated/malformed
provider responses, the output token budget that made truncation more likely
than it needed to be, and the retry policy added to recover from a transient
one. No editorial Selection rule is exercised or asserted on here -- these
tests only construct trivial two/three-candidate drafts to drive the
transport, never real story content.
"""
import json

import pytest

from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.hybrid_google_transport import DollarBudgetLedger
from cutsell_worker.hybrid_payload import estimate_tokens_from_chars
from cutsell_worker.hybrid_provider_settings import HybridProviderSettings
from cutsell_worker.unified_selection_google import (
    GoogleUnifiedSelectionReasoner,
    UnifiedSelectionUnreliableResponseError,
    build_unified_selection_payload,
    output_token_reserve,
    parse_unified_selection_response,
)


def clip(i: int) -> DraftClip:
    text = f"Independent story beat number {i} with some unique audience-facing detail."
    return DraftClip(
        clip_id=f"c{i}",
        source_asset_id="src",
        source_order=i,
        start=float(i * 5),
        end=float(i * 5 + 4),
        text=text,
        caption_text=text,
    )


def draft(candidate_count: int) -> DraftTimeline:
    clips = tuple(clip(i) for i in range(candidate_count))
    return DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id="p",
        strategy=EditStrategy.STORYTELLING,
        selected=clips,
        alternates=(),
        discarded=(),
    )


def decisions_json(candidate_count: int) -> str:
    return json.dumps({
        "decisions": [
            {
                "action": "select",
                "relation": "independent",
                "confidence": 1.0,
                "family_index": i,
                "reason_code": "independent_story_coverage",
            }
            for i in range(candidate_count)
        ]
    })


def gemini_response(text: str, *, finish_reason: str = "STOP", output_tokens: int = 100) -> dict:
    return {
        "candidates": [{
            "finishReason": finish_reason,
            "content": {"parts": [{"text": text}]},
        }],
        "usageMetadata": {"candidatesTokenCount": output_tokens},
    }


def truncated_response(*, finish_reason: str = "MAX_TOKENS") -> dict:
    # A realistic truncation: valid JSON prefix, cut off mid-string with no
    # closing delimiters -- exactly the shape json.loads chokes on.
    text = (
        '{"decisions":[{"action":"select","relation":"independent",'
        '"confidence":1.0,"family_index":0,"reason_code":"in'
    )
    return gemini_response(text, finish_reason=finish_reason)


class FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
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


def make_reasoner(fake: FakeSession, ledger: DollarBudgetLedger | None = None) -> GoogleUnifiedSelectionReasoner:
    settings = HybridProviderSettings(enabled=True)
    return GoogleUnifiedSelectionReasoner(
        api_key="fake-key",
        model=settings.primary_model,
        settings=settings,
        ledger=ledger or DollarBudgetLedger(1.0),
        session=fake,
    )


# --- output token budget -----------------------------------------------

def test_output_token_reserve_exceeds_the_old_flat_heuristic_that_truncated_in_raw119():
    old_flat_heuristic = max(640, 36 * 32)  # the exact formula RAW #119 ran with
    assert output_token_reserve(32, ceiling=4096) > old_flat_heuristic


def test_output_token_reserve_respects_an_explicit_ceiling():
    assert output_token_reserve(10_000, ceiling=4096) == 4096


def test_output_token_reserve_has_a_floor_for_tiny_candidate_counts():
    assert output_token_reserve(0, ceiling=4096) == 640
    assert output_token_reserve(1, ceiling=4096) == 640


# --- parser: truncated/malformed responses never look like a result -----

def test_parse_raises_unreliable_error_on_truncated_json_and_names_finish_reason():
    with pytest.raises(UnifiedSelectionUnreliableResponseError, match="MAX_TOKENS"):
        parse_unified_selection_response(truncated_response())


def test_parse_raises_unreliable_error_when_candidates_missing():
    with pytest.raises(UnifiedSelectionUnreliableResponseError, match="candidates"):
        parse_unified_selection_response({"candidates": []})


def test_parse_raises_unreliable_error_when_content_missing():
    with pytest.raises(UnifiedSelectionUnreliableResponseError, match="content"):
        parse_unified_selection_response({"candidates": [{"finishReason": "STOP"}]})


def test_parse_raises_unreliable_error_when_decisions_key_missing():
    raw = gemini_response(json.dumps({"not_decisions": []}))
    with pytest.raises(UnifiedSelectionUnreliableResponseError, match="decisions"):
        parse_unified_selection_response(raw)


def test_parse_succeeds_and_reports_finish_reason_on_a_clean_response():
    raw = gemini_response(decisions_json(2), finish_reason="STOP")
    decisions, output_tokens, finish_reason = parse_unified_selection_response(raw)
    assert len(decisions) == 2
    assert finish_reason == "STOP"
    assert output_tokens == 100


# --- reasoner: retry recovers from a truncated first attempt -------------

def test_reason_retries_once_after_a_truncated_first_attempt_and_succeeds():
    fake = FakeSession([truncated_response(), gemini_response(decisions_json(2))])
    reasoner = make_reasoner(fake)

    plan = reasoner.reason(draft(2))

    assert len(fake.calls) == 2
    assert len(plan.decisions) == 2
    assert all(d.action == "select" for d in plan.decisions)
    # retry asks for a larger response budget in case truncation was the cause
    first_budget = fake.calls[0][2]["generationConfig"]["maxOutputTokens"]
    second_budget = fake.calls[1][2]["generationConfig"]["maxOutputTokens"]
    assert second_budget > first_budget


def test_reason_raises_after_exhausting_retries_rather_than_returning_a_partial_result():
    fake = FakeSession([truncated_response(), truncated_response()])
    reasoner = make_reasoner(fake)

    with pytest.raises(UnifiedSelectionUnreliableResponseError):
        reasoner.reason(draft(2))

    # exactly one retry (max_retries defaults to 1) -- not an unbounded loop,
    # and the failure is a real raised exception, never a plan built from the
    # truncated response.
    assert len(fake.calls) == 2


def test_reason_retries_on_decision_count_mismatch_not_only_on_parse_failure():
    # A response that parses cleanly but is short one decision is exactly as
    # untrustworthy as a truncated one and must be treated the same way.
    short = gemini_response(decisions_json(1))  # draft below has 2 candidates
    good = gemini_response(decisions_json(2))
    fake = FakeSession([short, good])
    reasoner = make_reasoner(fake)

    plan = reasoner.reason(draft(2))

    assert len(fake.calls) == 2
    assert len(plan.decisions) == 2


def test_failed_first_attempt_releases_its_ledger_reservation_before_retrying():
    settings = HybridProviderSettings(enabled=True)
    d = draft(2)
    payload = build_unified_selection_payload(d)
    input_tokens = estimate_tokens_from_chars(len(json.dumps(payload, ensure_ascii=False)))
    first_reserve = output_token_reserve(2, ceiling=4096)
    retry_reserve = min(4096, max(first_reserve, int(first_reserve * 1.5)))
    retry_cost = settings.estimate_cost_usd(input_tokens=input_tokens, output_tokens=retry_reserve, escalation=False)

    # Sized so there is only ever enough budget for ONE reservation at a
    # time (at the larger, bumped retry size) -- if the failed first
    # attempt's reservation were not released, the retry's own reservation
    # would not fit and reason() would raise a budget error instead of
    # succeeding.
    ledger = DollarBudgetLedger(retry_cost * 1.2)
    fake = FakeSession([truncated_response(), gemini_response(decisions_json(2))])
    reasoner = GoogleUnifiedSelectionReasoner(
        api_key="fake-key", model=settings.primary_model, settings=settings,
        ledger=ledger, session=fake,
    )

    plan = reasoner.reason(d)

    assert len(plan.decisions) == 2


# --- non-retryable preflight failures never spend a retry ----------------

def test_missing_api_key_raises_before_any_http_call_and_is_never_retried():
    fake = FakeSession([gemini_response(decisions_json(2))])
    settings = HybridProviderSettings(enabled=True)
    reasoner = GoogleUnifiedSelectionReasoner(
        api_key="", model=settings.primary_model, settings=settings,
        ledger=DollarBudgetLedger(1.0), session=fake,
    )
    with pytest.raises(ValueError, match="Gemini API key required"):
        reasoner.reason(draft(2))
    assert len(fake.calls) == 0


def test_disallowed_model_raises_before_any_http_call_and_is_never_retried():
    fake = FakeSession([gemini_response(decisions_json(2))])
    settings = HybridProviderSettings(enabled=True)
    reasoner = GoogleUnifiedSelectionReasoner(
        api_key="fake-key", model="not-an-approved-model", settings=settings,
        ledger=DollarBudgetLedger(1.0), session=fake,
    )
    with pytest.raises(ValueError, match="not approved"):
        reasoner.reason(draft(2))
    assert len(fake.calls) == 0
