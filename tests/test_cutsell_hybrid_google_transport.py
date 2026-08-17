import json

import pytest

from cutsell_worker.hybrid_google_transport import DollarBudgetLedger, GoogleGeminiTransport
from cutsell_worker.hybrid_provider_settings import HybridProviderSettings


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "candidates": [{
                "content": {"parts": [{"text": json.dumps({"decisions": [
                    {"clip_id": "a", "label": "winner", "confidence": 0.97, "reason_code": "complete_take"}
                ]})}]}
            }],
            "usageMetadata": {"candidatesTokenCount": 41},
        }


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append((url, headers, json, timeout))
        return FakeResponse()


def compact_payload():
    return {
        "task": "classify_best_take_within_single_bounded_creator_session",
        "session_id": "hs_test",
        "source_asset_id": "src",
        "candidates": [{"clip_id": "a", "text": "complete product take"}],
    }


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


def test_enabled_transport_uses_mock_only_and_reconciles_to_actual_usage():
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

    from cutsell_worker.hybrid_payload import estimate_tokens_from_chars
    input_tokens = estimate_tokens_from_chars(len(json.dumps(payload, ensure_ascii=False)))
    actual_cost = settings.estimate_cost_usd(input_tokens=input_tokens, output_tokens=41)
    assert ledger.reserved_usd == pytest.approx(actual_cost)
    assert ledger.reserved_usd < settings.estimate_cost_usd(input_tokens=input_tokens, output_tokens=500)


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

    # Run #28 exhausted after about four chunks because every successful call retained
    # a worst-case 500-output-token reservation. Real 41-token responses should settle
    # to actual usage, allowing many more chunks while preserving the same $0.0075 cap.
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
