from cutsell_worker.contracts import CandidateTake
from cutsell_worker.product_handling_failure import apply_product_handling_failure_cleanup
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def take(clip_id, text, start, end):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
    )


def event(kind, start, end, confidence=1.0):
    return TemporalEvent("src", start, end, kind, confidence, kind)


def context(events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="",
            dominant_style="talking_head",
            creator_intent="recording",
            events=tuple(events),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def handling_break_context(start=0.0):
    return context((
        event("hand_motion_reset_candidate", start + 1.0, start + 1.1, 0.98),
        event("hand_motion_reset_candidate", start + 1.4, start + 1.5, 0.94),
        event("facial_expression_shift_candidate", start + 1.5, start + 1.6, 0.86),
        event("body_reset_candidate", start + 1.6, start + 1.7, 0.88),
    ))


def test_product_drop_like_break_before_same_idea_retry_is_removed():
    failed = take("failed", "bonder seal makes this so easy", 0.0, 3.0)
    retry = take("retry", "the bonder seal makes this so easy to apply", 5.0, 9.0)

    kept, removed, diagnostics = apply_product_handling_failure_cleanup(
        (failed, retry), handling_break_context()
    )

    assert failed in removed
    assert retry in kept
    assert diagnostics[0]["reason"] == "product_handling_fumble_with_face_reaction_near_retry"


def test_product_drop_after_prior_same_idea_take_is_removed():
    prior = take("prior", "you got your bond and seal and then your", 0.0, 3.6)
    failed = take("failed", "bond and seal and then your", 9.0, 11.7)

    kept, removed, diagnostics = apply_product_handling_failure_cleanup(
        (prior, failed), handling_break_context(start=9.0)
    )

    assert prior in kept
    assert failed in removed
    assert diagnostics[0]["reason"] == "product_handling_fumble_with_face_reaction_near_retry"


def test_normal_product_gesture_without_face_reaction_survives():
    valid = take("valid", "look at the seal right here", 0.0, 3.0)
    retry = take("retry", "look at the seal right here on the edge", 5.0, 9.0)
    ctx = context((
        event("hand_motion_reset_candidate", 1.0, 1.1, 0.98),
        event("hand_motion_reset_candidate", 1.4, 1.5, 0.94),
        event("body_reset_candidate", 1.6, 1.7, 0.88),
    ))

    kept, removed, diagnostics = apply_product_handling_failure_cleanup((valid, retry), ctx)

    assert valid in kept
    assert removed == ()
    assert diagnostics == ()


def test_face_reaction_without_retry_survives():
    valid = take("valid", "bonder seal makes this so easy", 0.0, 3.0)

    kept, removed, diagnostics = apply_product_handling_failure_cleanup(
        (valid,), handling_break_context()
    )

    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_retry_without_dense_hand_break_survives():
    valid = take("valid", "bonder seal makes this so easy", 0.0, 3.0)
    retry = take("retry", "the bonder seal makes this so easy to apply", 5.0, 9.0)
    ctx = context((
        event("facial_expression_shift_candidate", 1.5, 1.6, 0.86),
        event("body_reset_candidate", 1.6, 1.7, 0.88),
    ))

    kept, removed, diagnostics = apply_product_handling_failure_cleanup((valid, retry), ctx)

    assert valid in kept
    assert retry in kept
    assert removed == ()
    assert diagnostics == ()


def test_next_take_reset_does_not_contaminate_full_take_boundary():
    full = take(
        "full",
        "the popular crop black denim jeans are back in stock anything with pockets is a win for me",
        0.0,
        6.0,
    )
    retry = take("retry", "the popular crop black jeans", 8.0, 11.0)
    ctx = context((
        event("hand_motion_reset_candidate", 4.0, 4.1, 0.98),
        event("body_reset_candidate", 4.1, 4.2, 0.90),
        # These belong to the next boundary. The old +350 ms tail window counted
        # them against ``full`` and created a false product-fumble deletion.
        event("hand_motion_reset_candidate", 6.20, 6.25, 0.98),
        event("facial_expression_shift_candidate", 6.22, 6.27, 0.86),
    ))

    kept, removed, diagnostics = apply_product_handling_failure_cleanup((full, retry), ctx)

    assert full in kept
    assert removed == ()
    assert diagnostics == ()
