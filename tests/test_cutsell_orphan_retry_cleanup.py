from cutsell_worker.contracts import CandidateTake
from cutsell_worker.orphan_retry_cleanup import apply_orphan_retry_cleanup
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


def context(*events):
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


def event(kind, start, end, confidence=1.0):
    return TemporalEvent("src", start, end, kind, confidence, kind)


def test_middle_fragment_inside_failed_retry_envelope_is_removed():
    before = take("before", "If you have one of those universal", 0.0, 2.0)
    kept_take = take("kept", "Do you have one of those universal", 4.0, 6.0)
    after = take(
        "after",
        "If you have one of those universal frost buddies this is for you",
        8.0,
        12.0,
    )
    kept, removed, decisions = apply_orphan_retry_cleanup(
        (kept_take,),
        (before, after),
        context(event("hand_motion_reset_candidate", 4.8, 4.9, 1.0)),
    )
    assert kept == ()
    assert removed == (kept_take,)
    assert decisions[0].reason == "orphan_fragment_inside_failed_retry_envelope"


def test_one_sided_failed_retry_does_not_remove_kept_take():
    kept_take = take("kept", "Do you have one of those universal", 4.0, 6.0)
    after = take("after", "If you have one of those universal frost buddies", 8.0, 11.0)
    kept, removed, decisions = apply_orphan_retry_cleanup(
        (kept_take,),
        (after,),
        context(event("hand_motion_reset_candidate", 4.8, 4.9, 1.0)),
    )
    assert kept == (kept_take,)
    assert removed == ()
    assert decisions == ()


def test_distinct_failed_neighbors_do_not_remove_valid_middle_take():
    before = take("before", "This serum helps reduce frizz", 0.0, 2.0)
    kept_take = take("kept", "The bag has two hidden zipper pockets", 4.0, 7.0)
    after = take("after", "This serum also adds shine", 8.0, 11.0)
    kept, removed, decisions = apply_orphan_retry_cleanup(
        (kept_take,),
        (before, after),
        context(event("hand_motion_reset_candidate", 5.0, 5.1, 1.0)),
    )
    assert kept == (kept_take,)
    assert removed == ()
    assert decisions == ()
