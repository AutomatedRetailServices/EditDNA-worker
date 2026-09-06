from cutsell_worker.contracts import CandidateTake
from cutsell_worker.incomplete_retry_suffix import apply_incomplete_retry_suffix_cleanup
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def take(clip_id, text, start, end):
    return CandidateTake(clip_id, "src", 0, start, end, text)


def event(kind, start, end, confidence=1.0):
    return TemporalEvent("src", start, end, kind, confidence, kind)


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


def test_open_aux_fragment_before_fuller_retry_is_removed_with_end_reset():
    fragment = take("fragment", "Perfect what if I told you your kids are", 0.0, 3.2)
    full = take("full", "Perfect what if I told you your kids are going to love this little camera", 5.0, 9.0)
    kept, removed, decisions = apply_incomplete_retry_suffix_cleanup(
        (fragment, full),
        context(event("body_reset_candidate", 2.8, 3.0, 0.97)),
    )
    assert fragment in removed
    assert full in kept
    assert decisions[0].reason == "open_ended_retry_before_fuller_attempt_with_reset"


def test_valid_sentence_ending_in_is_survives_without_fuller_retry():
    valid = take("valid", "The reason I keep using it is", 0.0, 2.5)
    kept, removed, decisions = apply_incomplete_retry_suffix_cleanup(
        (valid,),
        context(event("body_reset_candidate", 2.2, 2.4, 0.99)),
    )
    assert kept == (valid,)
    assert removed == ()
    assert decisions == ()


def test_related_longer_content_without_reset_does_not_delete():
    fragment = take("fragment", "What if I told you your kids are", 0.0, 3.0)
    full = take("full", "What if I told you your kids are going to love this little camera", 5.0, 9.0)
    kept, removed, decisions = apply_incomplete_retry_suffix_cleanup((fragment, full), context())
    assert kept == (fragment, full)
    assert removed == ()
    assert decisions == ()


def test_distinct_nearby_longer_sentence_does_not_delete_open_ending():
    fragment = take("fragment", "The buttons on the back are", 0.0, 2.5)
    unrelated = take("unrelated", "The battery lasts all day and it comes with a strap", 4.0, 8.0)
    kept, removed, decisions = apply_incomplete_retry_suffix_cleanup(
        (fragment, unrelated),
        context(event("hand_motion_reset_candidate", 2.2, 2.4, 0.96)),
    )
    assert kept == (fragment, unrelated)
    assert removed == ()
    assert decisions == ()
