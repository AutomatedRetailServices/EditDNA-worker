from cutsell_worker.contracts import CandidateTake
from cutsell_worker.micro_self_talk import apply_micro_self_talk_cleanup
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def take(clip_id, text, start, end):
    return CandidateTake(clip_id=clip_id, source_asset_id="src", source_order=0, start=start, end=end, text=text)


def event(kind, start, end, confidence=1.0):
    return TemporalEvent("src", start, end, kind, confidence, kind)


def context(events):
    return WholeVideoContext(
        sources=(SourceVideoContext(source_asset_id="src", summary="", dominant_style="talking_head", creator_intent="recording", events=tuple(events)),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def break_context():
    return context((
        event("hand_motion_reset_candidate", 4.2, 4.3, 0.94),
        event("facial_expression_shift_candidate", 4.4, 4.5, 0.82),
    ))


def dense_reset_context():
    return context((
        event("hand_motion_reset_candidate", 3.9, 4.0, 1.0),
        event("hand_motion_reset_candidate", 4.2, 4.3, 1.0),
        event("hand_motion_reset_candidate", 4.5, 4.6, 1.0),
        event("body_reset_candidate", 4.8, 4.9, 1.0),
    ))


def test_short_expletive_between_attempts_is_removed_with_visual_break():
    before = take("before", "this serum helps seal the cuticle", 0.0, 3.0)
    reaction = take("reaction", "oh shit", 4.0, 5.0)
    after = take("after", "this serum helps seal the cuticle and reduce frizz", 6.0, 10.0)
    kept, removed, diagnostics = apply_micro_self_talk_cleanup((before, reaction, after), break_context())
    assert reaction in removed
    assert before in kept and after in kept
    assert diagnostics[0]["reason"] == "micro_self_talk_inside_retry_window_with_visual_break"


def test_short_expletive_between_attempts_is_removed_with_dense_physical_reset_even_without_face_break():
    before = take("before", "this serum helps seal the cuticle", 0.0, 3.0)
    reaction = take("reaction", "oh shit", 4.0, 5.0)
    after = take("after", "this serum helps seal the cuticle and reduce frizz", 6.0, 10.0)
    kept, removed, diagnostics = apply_micro_self_talk_cleanup((before, reaction, after), dense_reset_context())
    assert reaction in removed
    assert before in kept and after in kept
    assert diagnostics[0]["reason"] == "micro_self_talk_inside_retry_window_with_dense_physical_reset"


def test_this_is_crap_between_attempts_is_removed_with_visual_break():
    before = take("before", "the popular crop black jeans", 0.0, 3.0)
    reaction = take("reaction", "this is crap", 4.0, 5.0)
    after = take("after", "the popular cropped black jeans are back", 6.0, 10.0)
    kept, removed, diagnostics = apply_micro_self_talk_cleanup((before, reaction, after), break_context())
    assert reaction in removed
    assert diagnostics[0]["reason"] == "micro_self_talk_inside_retry_window_with_visual_break"


def test_isolated_profanity_reaction_survives_even_with_dense_reset():
    reaction = take("reaction", "oh shit", 0.0, 1.0)
    kept, removed, diagnostics = apply_micro_self_talk_cleanup((reaction,), dense_reset_context())
    assert kept == (reaction,)
    assert removed == ()
    assert diagnostics == ()


def test_reaction_between_attempts_survives_without_visual_or_dense_reset():
    before = take("before", "look at this bag", 0.0, 2.0)
    reaction = take("reaction", "oh shit", 3.0, 4.0)
    after = take("after", "look at this bag in natural light", 5.0, 8.0)
    kept, removed, diagnostics = apply_micro_self_talk_cleanup((before, reaction, after), context(()))
    assert reaction in kept
    assert removed == ()
    assert diagnostics == ()
