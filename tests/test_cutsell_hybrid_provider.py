from cutsell_worker.hybrid_editorial import EditorialCandidate, EditorialSession
from cutsell_worker.hybrid_provider import BudgetLedger, TransportEditorialJudge


def session():
    return EditorialSession(
        session_id="hs-1",
        source_asset_id="src-1",
        local_confidence=0.65,
        conflict_score=0.5,
        candidates=(
            EditorialCandidate("a", "first attempt", 0.0, 1.0, "alternate", 0.7),
            EditorialCandidate("b", "complete good attempt", 1.2, 3.0, "winner", 0.8),
        ),
    )


def test_transport_adapter_is_provider_neutral_and_strict():
    calls = []

    def fake_transport(payload, max_output_tokens):
        calls.append((payload, max_output_tokens))
        return {
            "decisions": [
                {"clip_id": "a", "label": "failed", "confidence": 0.95, "reason_code": "restart"},
                {"clip_id": "b", "label": "winner", "confidence": 0.97, "reason_code": "complete_retry"},
            ],
            "output_tokens": 42,
        }

    judge = TransportEditorialJudge("mock-vendor", "mock-model", fake_transport)
    result = judge.judge(session())
    assert len(calls) == 1
    assert calls[0][0]["session_id"] == "hs-1"
    assert calls[0][1] <= 500
    assert result.provider == "mock-vendor"
    assert result.model == "mock-model"
    assert result.available is True
    assert result.estimated_input_tokens > 0
    assert result.estimated_output_tokens == 42


def test_missing_decisions_fails_before_any_edit_application():
    def broken_transport(_payload, _max_output_tokens):
        return {"output_tokens": 5}

    judge = TransportEditorialJudge("mock", "bad", broken_transport)
    try:
        judge.judge(session())
    except ValueError as exc:
        assert "decisions" in str(exc)
    else:
        raise AssertionError("malformed response must fail")


def test_budget_ledger_refuses_call_and_token_overages():
    ledger = BudgetLedger(max_calls=2, max_estimated_input_tokens=100)
    assert ledger.reserve(40) is True
    assert ledger.reserve(50) is True
    assert ledger.calls == 2
    assert ledger.estimated_input_tokens == 90
    assert ledger.reserve(1) is False

    token_limited = BudgetLedger(max_calls=5, max_estimated_input_tokens=50)
    assert token_limited.reserve(40) is True
    assert token_limited.reserve(20) is False
