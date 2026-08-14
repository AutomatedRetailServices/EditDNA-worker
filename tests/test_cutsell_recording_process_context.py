from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.recording_process_context import apply_recording_process_neighbors
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


def event(kind, start, end, confidence=0.9):
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


def test_visual_reset_plus_explicit_stop_removes_immediately_previous_failed_take():
    failed = take("failed", "using my tik tok shit", 30.16, 32.47)
    anchor = take("stop", "Damn it, okay stopped", 34.72, 37.25)
    ctx = context((
        event("facial_expression_shift_candidate", 33.39, 33.50, 0.75),
        event("body_reset_candidate", 34.09, 34.20, 1.0),
    ))

    kept, removed, diagnostics = apply_recording_process_neighbors((failed,), (anchor,), ctx)

    assert kept == ()
    assert removed == (failed,)
    assert diagnostics[0]["reason"] == "failed_take_before_explicit_stop_with_visual_reset"


def test_recording_meta_after_explicit_stop_is_removed_without_language_censorship():
    anchor = take("stop", "Damn it, okay stopped", 34.72, 37.25)
    meta = take("meta", "That better have been good", 37.89, 39.47)

    kept, removed, diagnostics = apply_recording_process_neighbors((meta,), (anchor,), context(()))

    assert kept == ()
    assert removed == (meta,)
    assert diagnostics[0]["reason"] == "recording_meta_after_explicit_stop"


def test_pre_stop_take_survives_without_two_visual_evidence_families():
    valid = take("valid", "this shampoo is actually amazing", 30.16, 32.47)
    anchor = take("stop", "okay stop", 34.72, 35.20)
    ctx = context((event("hand_motion_reset_candidate", 33.9, 34.0, 1.0),))

    kept, removed, diagnostics = apply_recording_process_neighbors((valid,), (anchor,), ctx)

    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_profanity_alone_is_never_a_recording_process_signal():
    valid = take("valid", "this shit actually works", 30.16, 32.47)
    unrelated_discard = take("other", "random fragment", 34.72, 35.20)

    kept, removed, diagnostics = apply_recording_process_neighbors(
        (valid,), (unrelated_discard,), context(())
    )

    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()
