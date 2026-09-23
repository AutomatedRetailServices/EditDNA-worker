from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.restart_questions import apply_short_restart_question_cleanup
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


def strong_break_context():
    return context((
        event("hand_motion_reset_candidate", 126.50, 126.60, 1.0),
        event("facial_expression_shift_candidate", 126.50, 126.60, 0.73),
        event("hand_motion_reset_candidate", 127.30, 127.40, 1.0),
    ))


def test_what_again_before_again_restart_is_removed_with_multimodal_break():
    question = take("q", "What again?", 127.18, 127.98)
    restart = take("r", "Again this product. I'll link it below.", 129.50, 133.12)

    kept, removed, diagnostics = apply_short_restart_question_cleanup(
        (question, restart), strong_break_context()
    )

    assert kept == (restart,)
    assert removed == (question,)
    assert diagnostics[0]["reason"] == "short_restart_question_with_multimodal_reset"


def test_what_again_survives_without_multimodal_break():
    question = take("q", "What again?", 10.0, 11.0)
    restart = take("r", "Again this product is amazing", 12.0, 14.0)

    kept, removed, diagnostics = apply_short_restart_question_cleanup(
        (question, restart), context(())
    )

    assert kept == (question, restart)
    assert removed == ()
    assert diagnostics == ()


def test_what_again_survives_when_following_take_is_not_restart():
    question = take("q", "What again?", 10.0, 11.0)
    answer = take("a", "The bottle comes with two pumps", 12.0, 14.0)

    kept, removed, diagnostics = apply_short_restart_question_cleanup(
        (question, answer), strong_break_context()
    )

    assert kept == (question, answer)
    assert removed == ()
    assert diagnostics == ()


def test_intentional_again_question_longer_than_microtake_survives():
    question = take("q", "Again, what makes this formula different for dry skin?", 10.0, 13.0)
    restart = take("r", "Again this formula uses ceramides", 14.0, 16.0)

    kept, removed, diagnostics = apply_short_restart_question_cleanup(
        (question, restart), strong_break_context()
    )

    assert kept == (question, restart)
    assert removed == ()
    assert diagnostics == ()
