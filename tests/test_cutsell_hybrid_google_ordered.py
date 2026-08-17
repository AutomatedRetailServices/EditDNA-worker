import json

import pytest

from cutsell_worker.hybrid_google_transport import DollarBudgetLedger, GoogleGeminiTransport
from cutsell_worker.hybrid_provider_settings import HybridProviderSettings


class FakeResponse:
    def __init__(self, decisions, output_tokens=120):
        self._decisions = decisions
        self._output_tokens = output_tokens

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [{"text": json.dumps({"decisions": self._decisions})}]},
            }],
            "usageMetadata": {"candidatesTokenCount": self._output_tokens},
        }


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append((url, headers, json, timeout))
        return self.response


def payload(count=6):
    return {
        "task": "classify_recording_process_within_single_creator_session",
        "session_id": "hs_ordered",
        "source_asset_id": "src",
        "candidates": [
            {"clip_id": f"c{i}", "text": f"candidate speech {i}"}
            for i in range(count)
        ],
    }


def test_transport_reattaches_clip_ids_to_ordered_decisions():
    decisions = [
        {"label": "keep", "confidence": 0.90 + (i / 100.0)}
        for i in range(6)
    ]
    fake = FakeSession(FakeResponse(decisions))
    settings = HybridProviderSettings(enabled=True)
    transport = GoogleGeminiTransport(
        api_key="fake",
        model=settings.primary_model,
        settings=settings,
        ledger=DollarBudgetLedger(2.0),
        session=fake,
    )

    result = transport(payload(6), 500)
    assert [item["clip_id"] for item in result["decisions"]] == [f"c{i}" for i in range(6)]
    assert [item["label"] for item in result["decisions"]] == ["keep"] * 6
    schema = fake.calls[0][2]["generationConfig"]["responseJsonSchema"]
    item = schema["properties"]["decisions"]["items"]
    assert "clip_id" not in item["properties"]
    assert schema["properties"]["decisions"]["minItems"] == 6
    assert schema["properties"]["decisions"]["maxItems"] == 6


def test_transport_rejects_ordered_decision_count_mismatch_fail_open_upstream():
    decisions = [{"label": "keep", "confidence": 0.95} for _ in range(5)]
    fake = FakeSession(FakeResponse(decisions))
    settings = HybridProviderSettings(enabled=True)
    transport = GoogleGeminiTransport(
        api_key="fake",
        model=settings.primary_model,
        settings=settings,
        ledger=DollarBudgetLedger(2.0),
        session=fake,
    )
    with pytest.raises(ValueError, match="ordered decision count mismatch"):
        transport(payload(6), 500)
