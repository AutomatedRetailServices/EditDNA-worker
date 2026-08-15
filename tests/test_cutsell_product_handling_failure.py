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


def handling_break_context():
    return context((
        event("hand_motion_reset_candidate", 1.0, 1.1, 0.98),
        event("hand_motion_reset_candidate", 1.4, 1.5, 0.94),
        event("facial_expression_shift_candidate", 1.5, 1.6, 0.86),
        event("body_reset_candidate", 1.6, 1.7, 0.88),
    ))


def test_product_drop_like_break_before_same_idea_retry_is_removed():
    failed = take("failed", "bonder seal makes this so easy", 0.0, 3.0)
    retry = take("retry", "the bonder seal makes this so easy to apply", 5.0, 9.0)

    kept, removed, diagnostics = apply_product_handling_failure_cleanup(
        (failed, retry), handling_break_context()
    )

    assert failed in removed
    assert retry in kept
    assert diagnostics[0]["reason"] == "product_handling_fumble_with_face_reaction_before_retry"


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
