from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_composite_best_take import (
    _choose_composite_replacements,
    _delete_strong_prefix_prior_restarts,
    _restore_performance_only_unique_deliveries,
    _split_groups_for_composite,
)


def take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete,
    )


def test_complete_performance_failed_delivery_with_unique_tail_is_restored():
    winner = take(
        "winner",
        20.0,
        32.0,
        "I had bumps behind my ear and on my neck that looked like an allergy and seemed hormonal.",
    )
    candidate = take(
        "candidate",
        34.0,
        43.0,
        "Another symptom was bumps behind my ear and on my neck like an allergy. They came in seasons.",
    )
    semantic = {"winner": ("winner", 0.93), "candidate": ("failed", 0.90)}
    decisions = {
        "candidate": {
            "clip_id": "candidate",
            "applied_delete": True,
            "delete_basis": "semantic_failed_plus_local_performance",
            "reason_code": "",
            "local_failure_reasons": ["dense_physical_reset:6", "visual_fumble:0.85"],
        }
    }

    restored, rows = _restore_performance_only_unique_deliveries(
        (winner,),
        (candidate,),
        semantic,
        decisions,
    )

    assert restored == {"candidate"}
    assert rows[0]["peer_clip_id"] == "winner"
    assert rows[0]["reason"] == "restore_complete_performance_failed_delivery_with_unique_content"


def test_lexical_failure_is_not_restored_as_performance_only():
    winner = take("winner", 20.0, 30.0, "I had a symptom behind my ear and on my neck.")
    candidate = take("candidate", 32.0, 42.0, "I had a symptom behind my ear and on my neck by seasons.")
    semantic = {"winner": ("winner", 0.95), "candidate": ("failed", 0.91)}
    decisions = {
        "candidate": {
            "clip_id": "candidate",
            "applied_delete": True,
            "delete_basis": "semantic_failed_plus_local_performance",
            "reason_code": "repeated_phrase_restart",
            "local_failure_reasons": ["visual_fumble:0.88"],
        }
    }

    restored, rows = _restore_performance_only_unique_deliveries(
        (winner,),
        (candidate,),
        semantic,
        decisions,
    )

    assert restored == set()
    assert rows == []


def test_strong_prefix_incomplete_restart_can_yield_at_roughly_44_percent_coverage():
    prior = take(
        "prior",
        100.0,
        119.0,
        "This is my experience. I am the only one in my family with this cancer. Science shows only 5-10% is hereditary and lifestyle matters, so take care.",
        complete=True,
    )
    restart = take(
        "restart",
        123.0,
        137.0,
        "I am the first one in my family with this cancer. Nobody else has this thyroid condition. Science supports that only 5-10% of the",
        complete=False,
    )

    deleted, rows = _delete_strong_prefix_prior_restarts((prior, restart), {})

    assert deleted == {"restart"}
    assert rows[0]["prior_complete_clip_id"] == "prior"
    assert rows[0]["coverage"] >= 0.40
    assert rows[0]["prefix_ratio"] >= 0.60


def test_strong_prefix_fallback_does_not_delete_new_continuation():
    prior = take(
        "prior",
        100.0,
        119.0,
        "I am the only one in my family with this cancer and only 5-10% is hereditary.",
        complete=True,
    )
    continuation = take(
        "continuation",
        123.0,
        138.0,
        "After surgery my doctor changed the medication and I started tracking a completely different recovery symptom",
        complete=False,
    )

    deleted, rows = _delete_strong_prefix_prior_restarts((prior, continuation), {})

    assert deleted == set()
    assert rows == []


def test_two_complementary_restores_can_replace_one_monolithic_winner():
    setup = take(
        "setup",
        20.0,
        26.0,
        "I also had bumps. It looked like a rash, almost an allergy.",
    )
    winner = take(
        "winner",
        27.0,
        39.0,
        "I also had bumps behind my ear and on my neck that looked like an allergy and seemed hormonal.",
    )
    detail = take(
        "detail",
        42.0,
        51.0,
        "Another symptom was bumps like an allergy behind my ear and on my neck. It happened in seasons.",
    )
    semantic = {
        "setup": ("alternate", 0.82),
        "winner": ("winner", 0.93),
        "detail": ("failed", 0.90),
    }
    restored_rows = [
        {"clip_id": "setup", "peer_clip_id": "winner"},
        {"clip_id": "detail", "peer_clip_id": "winner"},
    ]

    suppressed, split_ids, rows = _choose_composite_replacements(
        (setup, winner, detail),
        semantic,
        restored_rows,
    )

    assert suppressed == {"winner"}
    assert split_ids == {"setup", "detail"}
    assert rows[0]["composite_clip_ids"] == ["setup", "detail"]


def test_composite_never_drops_winner_critical_number_missing_from_pair():
    setup = take("setup", 20.0, 26.0, "I also had bumps that looked like an allergy.")
    winner = take("winner", 27.0, 39.0, "I had a 3 centimeter lump plus bumps behind my ear and neck that looked like an allergy.")
    detail = take("detail", 42.0, 51.0, "The bumps were behind my ear and neck and came in seasons.")
    semantic = {
        "setup": ("alternate", 0.82),
        "winner": ("winner", 0.93),
        "detail": ("failed", 0.90),
    }
    restored_rows = [
        {"clip_id": "setup", "peer_clip_id": "winner"},
        {"clip_id": "detail", "peer_clip_id": "winner"},
    ]

    suppressed, split_ids, rows = _choose_composite_replacements(
        (setup, winner, detail),
        semantic,
        restored_rows,
    )

    assert suppressed == set()
    assert split_ids == set()
    assert rows == []


def test_composite_members_are_split_out_of_exclusive_retry_group():
    groups = (("setup", "detail"), ("later",))
    result = _split_groups_for_composite(groups, {"setup", "detail"}, ("setup", "detail", "later"))
    assert result == (("setup",), ("detail",), ("later",))


def test_pipeline_bindings_point_at_final_composite_authorities():
    from cutsell_worker import hybrid_session_cleanup, pipeline, session_boundaries

    assert pipeline.apply_hybrid_session_cleanup is hybrid_session_cleanup.apply_hybrid_session_cleanup
    assert getattr(pipeline.apply_hybrid_session_cleanup, "_cutsell_hybrid_composite_best_take", False) is True
    assert pipeline.safe_group_takes_by_sessions is session_boundaries.safe_group_takes_by_sessions
    assert getattr(pipeline.safe_group_takes_by_sessions, "_cutsell_hybrid_composite_group_split", False) is True
