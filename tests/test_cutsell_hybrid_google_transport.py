import json

import pytest

from cutsell_worker.hybrid_google_transport import (
    DollarBudgetLedger,
    GoogleGeminiTransport,
    _compact_output_token_ceiling,
)
from cutsell_worker.hybrid_provider_settings import HybridProviderSettings


class FakeResponse:
    def __init__(self, decisions=None, output_tokens=41):
        self._decisions = decisions or [
            {"clip_id": "a", "label": "winner", "confidence": 0.97}
        ]
        self._output_tokens = output_tokens

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "candidates": [{
                "content": {"parts": [{"text": json.dumps({"decisions": self._decisions})}]}
            }],
            "usageMetadata": {"candidatesTokenCount": self._output_tokens},
        }


class FakeSession:
    def __init__(self, response_factory=None):
        self.calls = []
        self.response_factory = response_factory

    def post(self, url, *, headers, json, timeout):
        self.calls.append((url, headers, json, timeout))
        if self.response_factory:
            return self.response_factory(json)
        return FakeResponse()


def compact_payload(candidate_count=1):
    return {
        "task": "classify_recording_process_within_single_creator_session",
        "session_id": "hs_test",
        "source_asset_id": "src",
        "candidates": [
            {"clip_id": f"c{i}" if candidate_count > 1 else "a", "text": f"candidate speech {i}"}
            for i in range(candidate_count)
        ],
    }


def test_compact_output_ceiling_scales_with_candidates_and_respects_caller_cap():
    assert _compact_output_token_ceiling(compact_payload(1), 500) == 160
    assert _compact_output_token_ceiling(compact_payload(6), 500) == 192
    assert _compact_output_token_ceiling(compact_payload(12), 500) == 336
    assert _compact_output_token_ceiling(compact_payload(14), 500) == 384
    assert _compact_output_token_ceiling(compact_payload(12), 250) == 250
    assert _compact_output_token_ceiling({"candidates": []}, 500) == 500


def test_disabled_settings_block_before_http():
    fake = FakeSession()
    transport = GoogleGeminiTransport(
        api_key="fake",
        model="gemini-3.5-flash-lite",
        settings=HybridProviderSettings(enabled=False),
        ledger=DollarBudgetLedger(2.0),
        session=fake,
    )
    with pytest.raises(RuntimeError, match="disabled"):
        transport(compact_payload(), 500)
    assert fake.calls == []


def test_enabled_transport_uses_dynamic_ceiling_and_reconciles_to_actual_usage():
    fake = FakeSession()
    ledger = DollarBudgetLedger(2.0)
    settings = HybridProviderSettings(enabled=True)
    transport = GoogleGeminiTransport(
        api_key="fake",
        model=settings.primary_model,
        settings=settings,
        ledger=ledger,
        session=fake,
    )
    payload = compact_payload()
    result = transport(payload, 500)
    assert result["decisions"][0]["clip_id"] == "a"
    assert result["output_tokens"] == 41
    assert len(fake.calls) == 1
    assert fake.calls[0][2]["generationConfig"]["maxOutputTokens"] == 160

    from cutsell_worker.hybrid_payload import estimate_tokens_from_chars
    input_tokens = estimate_tokens_from_chars(len(json.dumps(payload, ensure_ascii=False)))
    actual_cost = settings.estimate_cost_usd(input_tokens=input_tokens, output_tokens=41)
    assert ledger.reserved_usd == pytest.approx(actual_cost)
    assert ledger.reserved_usd < settings.estimate_cost_usd(input_tokens=input_tokens, output_tokens=160)


def test_twelve_candidate_call_uses_336_token_worst_case_not_legacy_500():
    decisions = [
        {"clip_id": f"c{i}", "label": "keep", "confidence": 0.97}
        for i in range(12)
    ]
    fake = FakeSession(lambda _: FakeResponse(decisions=decisions, output_tokens=190))
    settings = HybridProviderSettings(enabled=True)
    ledger = DollarBudgetLedger(2.0)
    transport = GoogleGeminiTransport(
        api_key="fake",
        model=settings.primary_model,
        settings=settings,
        ledger=ledger,
        session=fake,
    )
    result = transport(compact_payload(12), 500)
    assert len(result["decisions"]) == 12
    assert fake.calls[0][2]["generationConfig"]["maxOutputTokens"] == 336


def test_multiple_twelve_candidate_calls_fit_same_hard_edit_cap_when_actual_usage_is_compact():
    decisions = [
        {"clip_id": f"c{i}", "label": "keep", "confidence": 0.97}
        for i in range(12)
    ]
    fake = FakeSession(lambda _: FakeResponse(decisions=decisions, output_tokens=190))
    settings = HybridProviderSettings(enabled=True, max_cost_per_edit_usd=0.0075)
    ledger = DollarBudgetLedger(settings.max_cost_per_edit_usd)
    transport = GoogleGeminiTransport(
        api_key="fake",
        model=settings.primary_model,
        settings=settings,
        ledger=ledger,
        session=fake,
    )

    for _ in range(6):
        transport(compact_payload(12), 500)
    assert len(fake.calls) == 6
    assert 0 < ledger.reserved_usd <= settings.max_cost_per_edit_usd


def test_actual_usage_reconciliation_allows_multiple_small_calls_under_same_hard_cap():
    fake = FakeSession()
    settings = HybridProviderSettings(enabled=True, max_cost_per_edit_usd=0.0075)
    ledger = DollarBudgetLedger(settings.max_cost_per_edit_usd)
    transport = GoogleGeminiTransport(
        api_key="fake",
        model=settings.primary_model,
        settings=settings,
        ledger=ledger,
        session=fake,
    )
    for _ in range(12):
        transport(compact_payload(), 500)
    assert len(fake.calls) == 12
    assert 0 < ledger.reserved_usd <= settings.max_cost_per_edit_usd


def test_zero_budget_blocks_before_http():
    fake = FakeSession()
    settings = HybridProviderSettings(enabled=True)
    transport = GoogleGeminiTransport(
        api_key="fake",
        model=settings.primary_model,
        settings=settings,
        ledger=DollarBudgetLedger(0.0),
        session=fake,
    )
    with pytest.raises(RuntimeError, match="budget exhausted"):
        transport(compact_payload(), 500)
    assert fake.calls == []


def test_unapproved_model_is_rejected_at_construction():
    settings = HybridProviderSettings(enabled=True)
    with pytest.raises(ValueError, match="not approved"):
        GoogleGeminiTransport(
            api_key="fake",
            model="some-random-model",
            settings=settings,
            ledger=DollarBudgetLedger(2.0),
            session=FakeSession(),
        )


def test_escalation_model_still_fits_per_session_cap_with_compact_payload():
    fake = FakeSession()
    settings = HybridProviderSettings(enabled=True)
    ledger = DollarBudgetLedger(2.0)
    transport = GoogleGeminiTransport(
        api_key="fake",
        model=settings.escalation_model,
        settings=settings,
        ledger=ledger,
        escalation=True,
        session=fake,
    )
    transport(compact_payload(), 500)
    assert ledger.reserved_usd < settings.max_cost_per_session_usd
