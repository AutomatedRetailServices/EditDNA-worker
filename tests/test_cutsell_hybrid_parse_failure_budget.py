import json

import pytest

from cutsell_worker.hybrid_google_transport import DollarBudgetLedger, GoogleGeminiTransport
from cutsell_worker.hybrid_provider_settings import HybridProviderSettings


class MalformedStructuredResponse:
    def __init__(self, output_tokens=210):
        self.output_tokens = output_tokens

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "candidates": [{"content": {"parts": [{"text": '{"decisions": ['}]}}],
            "usageMetadata": {"candidatesTokenCount": self.output_tokens},
        }


class ValidStructuredResponse:
    def __init__(self, candidate_count=6, output_tokens=205):
        self.candidate_count = candidate_count
        self.output_tokens = output_tokens

    def raise_for_status(self):
        return None

    def json(self):
        decisions = [
            {"clip_id": f"c{i}", "label": "keep", "confidence": 0.8}
            for i in range(self.candidate_count)
        ]
        return {
            "candidates": [{"content": {"parts": [{"text": json.dumps({"decisions": decisions})}]}}],
            "usageMetadata": {"candidatesTokenCount": self.output_tokens},
        }


class SequencedSession:
    def __init__(self):
        self.calls = 0

    def post(self, url, *, headers, json, timeout):
        self.calls += 1
        if self.calls <= 2:
            return MalformedStructuredResponse(output_tokens=210)
        return ValidStructuredResponse(candidate_count=6, output_tokens=205)


def payload():
    return {
        "task": "classify_recording_process_within_single_creator_session",
        "session_id": "hs_parse_budget",
        "source_asset_id": "src",
        "candidates": [
            {"clip_id": f"c{i}", "text": f"candidate speech {i}"}
            for i in range(6)
        ],
    }


def test_structured_parse_failure_reconciles_reported_usage_and_does_not_starve_later_calls():
    settings = HybridProviderSettings(enabled=True, max_cost_per_edit_usd=0.0075)
    ledger = DollarBudgetLedger(settings.max_cost_per_edit_usd)
    session = SequencedSession()
    transport = GoogleGeminiTransport(
        api_key="fake",
        model=settings.primary_model,
        settings=settings,
        ledger=ledger,
        session=session,
    )

    for _ in range(2):
        with pytest.raises(ValueError, match="invalid JSON"):
            transport(payload(), 500)

    # Run32 showed parse failures exhausting the reservation and turning later chunks
    # into RuntimeError. Reported Gemini usage must now be reconciled so a valid later
    # six-candidate request still has budget to run.
    result = transport(payload(), 500)
    assert len(result["decisions"]) == 6
    assert session.calls == 3
    assert 0 < ledger.reserved_usd < settings.max_cost_per_edit_usd
