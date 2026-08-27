from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_complementary_delivery_guard import (
    _delete_unavailable_prior_restarts,
    _restore_complementary_cross_group_deletions,
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


def test_complete_cross_group_alternate_with_unique_tail_is_restored():
    candidate = take(
        "candidate",
        20.0,
        26.0,
        "I also had bumps. It looked like a rash, almost an allergy.",
        complete=True,
    )
    winner = take(
        "winner",
        27.0,
        38.0,
        "I also had bumps behind my ear and on my neck that looked like an allergy.",
        complete=True,
    )
    semantic = {
        "candidate": ("alternate", 0.82),
        "winner": ("winner", 0.90),
    }

    restored, rows = _restore_complementary_cross_group_deletions(
        (winner,),
        (candidate,),
        semantic,
        {"candidate"},
    )

    assert restored == {"candidate"}
    assert rows[0]["reason"] == "restore_complete_complementary_delivery_with_unique_tail"
    assert rows[0]["unique_content_tokens"]


def test_exact_cross_group_retry_without_unique_tail_stays_deleted():
    candidate = take(
        "candidate",
        20.0,
        26.0,
        "I had bumps behind my ear and on my neck.",
        complete=True,
    )
    winner = take(
        "winner",
        27.0,
        36.0,
        "I had bumps behind my ear and on my neck.",
        complete=True,
    )
    semantic = {
        "candidate": ("alternate", 0.90),
        "winner": ("winner", 0.95),
    }

    restored, rows = _restore_complementary_cross_group_deletions(
        (winner,),
        (candidate,),
        semantic,
        {"candidate"},
    )

    assert restored == set()
    assert rows == []


def test_undecided_incomplete_immediate_restart_yields_to_prior_complete():
    prior = take(
        "prior",
        100.0,
        119.0,
        "This is my experience. I am the only one in my family with this cancer and only 5-10% is hereditary.",
        complete=True,
    )
    restart = take(
        "restart",
        123.0,
        137.0,
        "I am the first in my family with this cancer. Science says only 5-10% of the",
        complete=False,
    )

    deleted, rows = _delete_unavailable_prior_restarts((prior, restart), {})

    assert deleted == {"restart"}
    assert rows[0]["prior_complete_clip_id"] == "prior"
    assert rows[0]["reason"] == "hybrid_unavailable_incomplete_restart_yields_to_prior_complete_delivery"


def test_incomplete_continuation_with_new_opening_is_not_misread_as_restart():
    prior = take(
        "prior",
        100.0,
        118.0,
        "I had stomach problems and the doctor diagnosed gastritis after an endoscopy.",
        complete=True,
    )
    continuation = take(
        "continuation",
        122.0,
        135.0,
        "Three months later I changed my diet and started tracking a completely different symptom",
        complete=False,
    )

    deleted, rows = _delete_unavailable_prior_restarts((prior, continuation), {})

    assert deleted == set()
    assert rows == []


def test_pipeline_binding_points_at_current_final_hybrid_wrapper():
    from cutsell_worker import hybrid_session_cleanup, pipeline

    assert pipeline.apply_hybrid_session_cleanup is hybrid_session_cleanup.apply_hybrid_session_cleanup
    # The complementary guard remains nested in the wrapper chain; the public pipeline
    # binding must point at whichever final Hybrid authority was installed last.
    assert getattr(
        pipeline.apply_hybrid_session_cleanup,
        "_cutsell_hybrid_composite_best_take",
        False,
    ) is True
