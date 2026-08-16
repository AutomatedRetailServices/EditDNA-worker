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


def test_enabled_transport_uses_mock_only_and_reserves_tiny_budget():
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
    result = transport(compact_payload(), 500)
    assert result["decisions"][0]["clip_id"] == "a"
    assert result["output_tokens"] == 41
    assert len(fake.calls) == 1
    assert ledger.reserved_usd > 0
    assert ledger.reserved_usd < 0.003


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
