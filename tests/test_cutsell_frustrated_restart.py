from cutsell_worker.contracts import CandidateTake
from cutsell_worker.frustrated_restart import apply_soft_frustration_restart_cleanup
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def take(clip_id, text, start=0.0, end=4.0):
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


def break_context():
    return context((
        event("hand_motion_reset_candidate", 0.8, 0.9, 1.0),
        event("facial_expression_shift_candidate", 1.6, 1.7, 0.82),
    ))


def test_repeated_product_name_plus_oh_my_god_is_removed_with_visual_break():
    failed = take(
        "failed",
        "Phone holder election suction phone holder election. Oh my god",
    )
    kept, removed, diagnostics = apply_soft_frustration_restart_cleanup((failed,), break_context())

    assert kept == ()
    assert removed == (failed,)
    assert diagnostics[0]["reason"] == "repeated_restart_with_soft_frustration_and_visual_break"


def test_single_content_word_restart_echo_plus_oh_my_god_is_removed_with_visual_break():
    failed = take(
        "failed",
        "election suction phone holder election oh my god",
        end=4.3,
    )
    kept, removed, diagnostics = apply_soft_frustration_restart_cleanup((failed,), break_context())

    assert kept == ()
    assert removed == (failed,)
    assert diagnostics[0]["reason"] == "repeated_restart_with_soft_frustration_and_visual_break"


def test_oh_my_god_alone_survives_even_with_visual_break():
    valid = take("valid", "oh my god this is actually amazing")
    kept, removed, diagnostics = apply_soft_frustration_restart_cleanup((valid,), break_context())

    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_repeated_phrase_plus_oh_my_god_survives_without_visual_break():
    valid = take("valid", "look at this bag look at this bag oh my god")
    kept, removed, diagnostics = apply_soft_frustration_restart_cleanup((valid,), context(()))

    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_repeated_content_word_without_visual_break_survives():
    valid = take("valid", "electric suction phone holder electric oh my god")
    kept, removed, diagnostics = apply_soft_frustration_restart_cleanup((valid,), context(()))

    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_function_word_repetition_does_not_create_restart_evidence():
    valid = take("valid", "this is the one and the one oh my god", end=4.2)
    kept, removed, diagnostics = apply_soft_frustration_restart_cleanup((valid,), break_context())

    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_long_creator_story_with_repeated_content_word_is_not_removed_by_single_token_rule():
    valid = take(
        "valid",
        "this holder changed my desk setup and the holder is honestly so useful oh my god",
        end=8.5,
    )
    kept, removed, diagnostics = apply_soft_frustration_restart_cleanup((valid,), break_context())

    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_oh_my_goodness_reaction_alone_survives():
    valid = take("valid", "oh my goodness gracious these are beautiful")
    kept, removed, diagnostics = apply_soft_frustration_restart_cleanup((valid,), break_context())

    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()
