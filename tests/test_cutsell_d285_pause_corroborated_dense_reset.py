"""D-285 (RAW #120 audit) -- an uncorroborated DENSE PHYSICAL RESET count
must never mark a complete realization deterministically unusable.

Real RAW #120 evidence: a 17.98s complete, both-references-preferred
realization was marked `deterministic_unusable` because 7 hand_motion_
reset_candidate events (all >= 0.92 confidence) fell inside it, with ZERO
audio_silence_interval evidence anywhere nearby -- continuous, unbroken
speech throughout. Per D-149's already-proven doctrine (real RAW #118
audit, perceptual_watch_listen.py), a hand/body reset candidate's
confidence is a pure kinematic magnitude score, not a probability the
movement is a genuine recording-process reset; a real reset/retry almost
always has a measured pause around it. The `dense_physical_reset` branch is
the one path in `_failed_local_evidence` with no independent break/
disengagement corroboration, so it is the one branch a natural gesture
during continuous speech can trip on its own. Fix: only PAUSE-CORROBORATED
resets count toward that branch's threshold. The `multimodal_reset_cluster`
branch (reset + independent break) is untouched and still fires on raw
counts -- it already has its own corroboration. Generic fixtures only.
"""
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_session_cleanup import (
    _failed_local_evidence,
    _performance_event_summary,
)
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext

TEXT = "This is my experience with the product and here is the full story from start to end."


def _take(clip_id="t", start=300.0, end=318.0, text=TEXT, source="src"):
    return CandidateTake(clip_id, source, 0, start, end, text)


def _event(kind, start, end, confidence=1.0, source="src"):
    return TemporalEvent(source_asset_id=source, start=start, end=end, kind=kind, confidence=confidence, description="")


def _context(*events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src", summary="", dominant_style="talking_head", creator_intent="",
            events=tuple(events), edit_mode="natural", sales_intent=0.0, main_topic="",
            product_or_subject="", story_logic="",
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def test_seven_uncorroborated_hand_motion_resets_no_longer_fail_the_take():
    """The exact real shape: 7 strong hand_motion_reset_candidate events,
    zero breaks, zero measured pauses anywhere nearby -- must fail open."""
    take = _take()
    events = [_event("hand_motion_reset_candidate", 302.0 + i, 302.07 + i, 0.95) for i in range(7)]
    failed, reasons = _failed_local_evidence(take, _context(*events))
    assert failed is False
    assert not any("dense_physical_reset" in r for r in reasons)


def test_pause_corroborated_dense_reset_still_fails():
    """Same 7 resets, but each one has a real measured pause nearby -- a
    genuine reset/retry signature -- must still be treated as evidence."""
    take = _take()
    events = []
    for i in range(7):
        center = 302.0 + i
        events.append(_event("hand_motion_reset_candidate", center, center + 0.07, 0.95))
        events.append(_event("audio_silence_interval", center - 0.2, center + 0.2, 1.0))
    failed, reasons = _failed_local_evidence(take, _context(*events))
    assert failed is True
    assert any("dense_physical_reset" in r for r in reasons)


def test_multimodal_reset_cluster_branch_is_unaffected_by_the_fix():
    """The independently-corroborated branch (reset + real break) must keep
    firing on raw counts -- it was never the uncorroborated path."""
    take = _take()
    events = [
        _event("hand_motion_reset_candidate", 303.0, 303.1, 0.95),
        _event("hand_motion_reset_candidate", 304.0, 304.1, 0.95),
        _event("camera_disengagement_candidate", 305.0, 305.4, 0.90),
    ]
    failed, reasons = _failed_local_evidence(take, _context(*events))
    assert failed is True
    assert any("multimodal_reset_cluster" in r for r in reasons)


def test_pause_corroborated_reset_count_is_visible_in_the_summary():
    take = _take()
    events = [
        _event("hand_motion_reset_candidate", 303.0, 303.1, 0.95),
        _event("audio_silence_interval", 302.8, 303.3, 1.0),
        _event("hand_motion_reset_candidate", 306.0, 306.1, 0.95),  # no nearby pause
    ]
    summary = _performance_event_summary(take, _context(*events))
    assert summary["strong_reset_count"] == 2
    assert summary["pause_corroborated_reset_count"] == 1


def test_negative_control_three_uncorroborated_resets_below_threshold():
    """Below the count floor regardless -- must never fail either way."""
    take = _take()
    events = [_event("hand_motion_reset_candidate", 302.0 + i, 302.07 + i, 0.95) for i in range(3)]
    failed, reasons = _failed_local_evidence(take, _context(*events))
    assert failed is False


def test_negative_control_pause_far_from_the_reset_does_not_corroborate():
    """A measured pause elsewhere in the take does not corroborate a reset
    it is not actually near -- proximity to the SPECIFIC event is required."""
    take = _take()
    events = [_event("hand_motion_reset_candidate", 302.0 + i, 302.07 + i, 0.95) for i in range(7)]
    events.append(_event("audio_silence_interval", 316.0, 317.0, 1.0))  # far outside proximity window
    failed, reasons = _failed_local_evidence(take, _context(*events))
    assert failed is False


def test_negative_control_no_context_fails_open():
    take = _take()
    failed, reasons = _failed_local_evidence(take, None)
    assert failed is False
