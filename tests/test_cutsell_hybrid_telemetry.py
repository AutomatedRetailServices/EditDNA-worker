from cutsell_worker.hybrid_telemetry import HybridDecisionEvent, HybridTelemetry, summarize_hybrid_events


def event(**overrides):
    base = dict(
        session_id="s1",
        candidate_count=3,
        local_confidence=0.71,
        conflict_score=0.55,
        gate_requested=True,
        provider_requested=False,
        provider_available=False,
        estimated_input_tokens=420,
        estimated_output_tokens=0,
        local_winner_clip_id="a",
        final_winner_clip_id="a",
    )
    base.update(overrides)
    return HybridDecisionEvent(**base)


def test_zero_cost_gate_event_does_not_claim_provider_request():
    item = event()
    assert item.gate_requested is True
    assert item.provider_requested is False
    assert item.winner_changed is False


def test_telemetry_counts_provider_use_and_winner_changes():
    telemetry = HybridTelemetry()
    telemetry.record(event())
    telemetry.record(event(
        session_id="s2",
        provider_requested=True,
        provider_available=True,
        estimated_input_tokens=500,
        estimated_output_tokens=80,
        final_winner_clip_id="b",
        provider="mock",
        model="mock-model",
    ))
    snapshot = telemetry.snapshot()
    assert snapshot["sessions"] == 2
    assert snapshot["gate_requested_sessions"] == 2
    assert snapshot["provider_requested_sessions"] == 1
    assert snapshot["provider_available_sessions"] == 1
    assert snapshot["winner_changed_sessions"] == 1
    assert snapshot["estimated_input_tokens"] == 920
    assert snapshot["estimated_output_tokens"] == 80
    assert snapshot["provider_request_rate"] == 0.5
    assert snapshot["winner_change_rate"] == 0.5


def test_empty_summary_is_safe():
    snapshot = summarize_hybrid_events(())
    assert snapshot["sessions"] == 0
    assert snapshot["gate_rate"] == 0.0
    assert snapshot["provider_request_rate"] == 0.0
    assert snapshot["winner_change_rate"] == 0.0
