from cutsell_worker.contracts import CandidateTake
from cutsell_worker.micro_restart_cleanup import apply_micro_restart_cleanup
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def take(clip_id, text, start=0.0, end=3.0):
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


def break_context(start=0.0, end=3.0):
    return context((
        event("body_reset_candidate", start + 0.4, start + 0.5, 1.0),
        event("hand_motion_reset_candidate", start + 1.0, start + 1.1, 1.0),
        event("facial_expression_shift_candidate", start + 1.4, start + 1.5, 0.84),
    ))


def test_adjacent_content_word_restart_is_removed_with_visual_break():
    failed = take("failed", "the popular croc croc")
    kept, removed, diagnostics = apply_micro_restart_cleanup((failed,), break_context())
    assert kept == ()
    assert removed == (failed,)
    assert diagnostics[0]["reason"] == "adjacent_content_word_restart_with_visual_break"


def test_repeated_multiword_restart_is_removed_with_visual_break():
    failed = take("failed", "crop popular crop popular crop popular", end=4.0)
    kept, removed, diagnostics = apply_micro_restart_cleanup((failed,), break_context())
    assert kept == ()
    assert removed == (failed,)
    assert diagnostics[0]["reason"] == "repeated_phrase_restart_with_visual_break"


def test_explicit_whole_sentence_recording_meta_is_removed():
    failed = take("failed", "the popular crop black jeans okay now we whole sentence okay", end=5.0)
    kept, removed, diagnostics = apply_micro_restart_cleanup((failed,), context(()))
    assert kept == ()
    assert removed == (failed,)
    assert diagnostics[0]["reason"] == "explicit_delivery_process_meta"
    assert diagnostics[0]["confidence"] == 0.97


def test_negative_self_evaluation_needs_visual_break():
    failed = take("failed", "this is crap")
    kept, removed, diagnostics = apply_micro_restart_cleanup((failed,), break_context())
    assert kept == ()
    assert removed == (failed,)
    assert diagnostics[0]["reason"] == "negative_self_evaluation_with_visual_break"

    valid = take("valid", "this is crap")
    kept, removed, diagnostics = apply_micro_restart_cleanup((valid,), context(()))
    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_intentional_emphasis_repetition_survives_even_with_visual_break():
    valid = take("valid", "they are so so super cute")
    kept, removed, diagnostics = apply_micro_restart_cleanup((valid,), break_context())
    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_repeated_phrase_survives_without_visual_break():
    valid = take("valid", "look at this bag look at this bag", end=4.0)
    kept, removed, diagnostics = apply_micro_restart_cleanup((valid,), context(()))
    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_short_prefix_before_multiple_fuller_retries_is_removed_only_with_visual_break():
    short = take("short", "the popular crop", start=0.0, end=2.0)
    retry = take("retry", "the popular crop black denim jeans", start=4.0, end=8.0)
    full = take("full", "the popular crop black denim jeans are back in stock with pockets", start=10.0, end=16.0)
    kept, removed, diagnostics = apply_micro_restart_cleanup((short, retry, full), break_context(0.0, 2.0))
    assert short in removed
    assert any(item["reason"] == "short_prefix_before_fuller_retry_with_visual_break" for item in diagnostics)
    assert retry in kept
    assert full in kept


def test_short_valid_line_without_retry_sequence_survives():
    valid = take("valid", "the popular black jeans")
    kept, removed, diagnostics = apply_micro_restart_cleanup((valid,), break_context())
    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()
