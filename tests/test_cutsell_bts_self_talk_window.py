from cutsell_worker.contracts import CandidateTake
from cutsell_worker.micro_self_talk import apply_micro_self_talk_cleanup
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


def dense_context():
    events = (
        event("hand_motion_reset_candidate", 1.0, 1.1, 0.98),
        event("body_reset_candidate", 3.0, 3.1, 0.96),
        event("hand_motion_reset_candidate", 6.0, 6.1, 0.99),
        event("body_reset_candidate", 9.0, 9.1, 0.95),
        event("facial_expression_shift_candidate", 4.0, 4.1, 0.88),
        event("camera_disengagement_candidate", 8.0, 8.1, 0.86),
    )
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="",
            dominant_style="talking_head",
            creator_intent="recording",
            events=events,
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def test_bts_cluster_removes_process_self_talk_and_tiny_debris_but_keeps_clean_take():
    clips = (
        take("a", "Don't miss out", 0.0, 1.2),
        take("b", "My gosh, you're stupid", 1.3, 2.4),
        take("c", "But you", 2.45, 2.9),
        take("d", "If you want to join me then don't miss out on this", 3.1, 6.0),
        take("e", "Why do I say don't miss out on this?", 6.2, 7.8),
        take("f", "It's stupid", 8.1, 8.8),
        take("g", "Ugh", 9.0, 9.4),
        take("h", "I hate saying the links below", 9.8, 11.8),
        take("i", "I hate being salesy", 12.2, 13.7),
    )

    kept, removed, diagnostics = apply_micro_self_talk_cleanup(clips, dense_context())

    kept_ids = {x.clip_id for x in kept}
    removed_ids = {x.clip_id for x in removed}
    assert "d" in kept_ids
    assert {"b", "c", "e", "f", "g", "h", "i"}.issubset(removed_ids)
    assert all(item["reason"] == "corroborated_behind_the_scenes_self_talk_window" for item in diagnostics)


def test_process_phrase_alone_survives_without_cluster():
    valid = take("valid", "I hate being salesy but this product genuinely works", 0.0, 3.0)
    kept, removed, diagnostics = apply_micro_self_talk_cleanup((valid,), dense_context())

    assert kept == (valid,)
    assert removed == ()
    assert diagnostics == ()


def test_three_process_phrases_survive_without_dense_visual_breaks():
    clips = (
        take("a", "Why do I say that", 0.0, 1.0),
        take("b", "I hate saying the links below", 2.0, 3.5),
        take("c", "I hate being salesy", 4.0, 5.5),
    )
    empty = WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="",
            dominant_style="talking_head",
            creator_intent="recording",
            events=(),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )

    kept, removed, diagnostics = apply_micro_self_talk_cleanup(clips, empty)

    assert kept == clips
    assert removed == ()
    assert diagnostics == ()
