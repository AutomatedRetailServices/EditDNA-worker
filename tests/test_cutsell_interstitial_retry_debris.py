from cutsell_worker.contracts import CandidateTake
from cutsell_worker.interstitial_retry_debris import apply_interstitial_retry_debris_cleanup
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


def test_content_microtake_between_failed_and_full_retry_is_removed():
    failed = take("failed", "But 13 different scalloped", 0.0, 2.5)
    micro = take("micro", "Scalloped okay", 3.0, 4.4)
    full = take("full", "But 13 different scalloped G string underwear", 5.0, 8.5)
    kept, removed, decisions = apply_interstitial_retry_debris_cleanup(
        (micro, full),
        (failed,),
        context(event("hand_motion_reset_candidate", 3.6, 3.8, 0.98)),
    )
    assert micro in removed
    assert full in kept
    assert decisions[0].reason == "content_microtake_inside_retry_envelope_with_reset"


def test_plain_okay_between_retries_survives_without_content_word():
    failed = take("failed", "But 13 different scalloped", 0.0, 2.5)
    micro = take("micro", "Okay", 3.0, 3.6)
    full = take("full", "But 13 different scalloped G string underwear", 4.0, 8.0)
    kept, removed, decisions = apply_interstitial_retry_debris_cleanup(
        (micro, full),
        (failed,),
        context(event("body_reset_candidate", 3.2, 3.4, 0.99)),
    )
    assert kept == (micro, full)
    assert removed == ()
    assert decisions == ()


def test_content_microtake_survives_without_reset():
    failed = take("failed", "But 13 different scalloped", 0.0, 2.5)
    micro = take("micro", "Scalloped okay", 3.0, 4.4)
    full = take("full", "But 13 different scalloped G string underwear", 5.0, 8.5)
    kept, removed, decisions = apply_interstitial_retry_debris_cleanup((micro, full), (failed,), context())
    assert kept == (micro, full)
    assert removed == ()
    assert decisions == ()


def test_unrelated_microtake_does_not_delete():
    failed = take("failed", "This serum helps reduce frizz", 0.0, 2.5)
    micro = take("micro", "Scalloped okay", 3.0, 4.4)
    full = take("full", "This serum also helps add shine", 5.0, 8.5)
    kept, removed, decisions = apply_interstitial_retry_debris_cleanup(
        (micro, full),
        (failed,),
        context(event("hand_motion_reset_candidate", 3.6, 3.8, 0.98)),
    )
    assert kept == (micro, full)
    assert removed == ()
    assert decisions == ()
