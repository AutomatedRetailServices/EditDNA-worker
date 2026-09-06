from cutsell_worker.hybrid_editorial import EditorialCandidate, EditorialSession, HybridGatePolicy
from cutsell_worker.hybrid_payload import (
    HybridCostPolicy,
    build_compact_editorial_payload,
    estimate_tokens_from_chars,
    preflight_hybrid_call,
)


def _session(text: str = "This product is amazing") -> EditorialSession:
    return EditorialSession(
        session_id="hs_test",
        source_asset_id="src-1",
        local_confidence=0.68,
        conflict_score=0.42,
        candidates=(
            EditorialCandidate(
                clip_id="a",
                text=text,
                start=0.0,
                end=2.0,
                local_label="alternate",
                local_confidence=0.72,
                evidence=(("visual_fumble", 0.7), ("complete_idea", False)),
            ),
            EditorialCandidate(
                clip_id="b",
                text="This product is amazing and I use it every day",
                start=2.2,
                end=5.2,
                local_label="winner",
                local_confidence=0.83,
                evidence=(("visual_fumble", 0.1), ("complete_idea", True)),
            ),
        ),
    )


def test_payload_is_compact_and_never_contains_edit_commands():
    payload = build_compact_editorial_payload(_session())
    assert payload["task"] == "classify_best_take_within_single_bounded_creator_session"
    assert [item["clip_id"] for item in payload["candidates"]] == ["a", "b"]
    assert "start" not in payload["candidates"][0]
    assert "end" not in payload["candidates"][0]
    assert "timestamp" not in repr(payload["candidates"][0]).lower()


def test_long_transcript_is_truncated_before_provider_boundary():
    policy = HybridCostPolicy(max_chars_per_candidate=40, max_payload_chars=10_000)
    payload = build_compact_editorial_payload(_session("x" * 500), cost_policy=policy)
    assert len(payload["candidates"][0]["text"]) == 40


def test_candidate_budget_fails_before_any_future_network_call():
    base = _session()
    candidates = tuple(base.candidates[0] for _ in range(15))
    oversized = EditorialSession("s", "src", candidates, 0.5, 0.5)
    try:
        build_compact_editorial_payload(oversized)
    except ValueError as exc:
        assert "candidate budget" in str(exc)
    else:
        raise AssertionError("oversized payload must fail preflight")


def test_preflight_caps_output_and_exposes_estimated_input():
    result = preflight_hybrid_call(
        _session(),
        HybridGatePolicy(max_estimated_input_tokens=12_000, max_estimated_output_tokens=1_000),
        cost_policy=HybridCostPolicy(max_estimated_input_tokens=4_000, max_estimated_output_tokens=500),
    )
    assert result["allowed"] is True
    assert result["estimated_input_tokens"] > 0
    assert result["max_output_tokens"] == 500


def test_large_realistic_cleanup_window_compacts_instead_of_failing_preflight():
    candidates = tuple(
        EditorialCandidate(
            clip_id=f"c{index}",
            text=("long creator delivery with useful semantic detail and repeated context " * 30) + str(index),
            start=float(index * 3),
            end=float(index * 3 + 2.5),
            local_label="keep",
            local_confidence=0.5,
            evidence=(
                ("complete_idea", index % 2 == 0),
                ("strong_reset_count", 2),
                ("strong_break_count", 1),
                ("visual_fumble", 0.22),
                ("expression_naturalness", 0.81),
                ("gesture_naturalness", 0.77),
                ("delivery_energy", 0.74),
                ("distraction_risk", 0.08),
            ),
        )
        for index in range(10)
    )
    session = EditorialSession(
        session_id="large-cleanup",
        source_asset_id="src",
        candidates=candidates,
        local_confidence=0.5,
        conflict_score=0.5,
        task="classify_recording_process_within_single_creator_session",
        source_context=(
            ("summary", "S" * 3600),
            ("creator_intent", "I" * 500),
            ("main_topic", "T" * 500),
            ("product_or_subject", "P" * 500),
            ("story_logic", "L" * 900),
            ("edit_mode", "natural"),
            ("sales_intent", 0.7),
        ),
    )

    payload = build_compact_editorial_payload(session)
    payload_chars = len(repr(payload))

    assert payload_chars <= 12_000
    assert estimate_tokens_from_chars(payload_chars) <= 4_000
    assert [item["clip_id"] for item in payload["candidates"]] == [f"c{i}" for i in range(10)]
    assert all(item["text"] for item in payload["candidates"])
    assert "full message/story" in " ".join(payload["rules"])
    assert payload["source_context"]["summary"]
