from cutsell_worker.clean_cut import evaluate_take
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.local_performance import LocalPerformanceTimeline
from cutsell_worker.performance_confirmation import confirm_local_performance_events, retry_similarity
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _take(clip_id, start, end, text):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
    )


def _context():
    return WholeVideoContext(
        sources=(SourceVideoContext("src", "", "talking_head", "explain"),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def _timeline(events):
    return LocalPerformanceTimeline(
        source_asset_id="src",
        observations=(),
        events=tuple(events),
        sampled_fps=12.0,
        source_fps=30.0,
        status=ProviderStatus("local", True, True, "applied"),
    )


def _event(kind, start, end, confidence=0.88):
    return TemporalEvent("src", start, end, kind, confidence, kind)


def test_retry_similarity_recognizes_reworded_same_attempt():
    assert retry_similarity(
        "This serum makes my skin look much smoother",
        "This serum really makes my skin look smoother",
    ) >= 0.58


def test_isolated_normal_movement_does_not_become_bad_take():
    first = _take("a", 0.0, 2.0, "I went to the airport yesterday")
    second = _take("b", 2.4, 4.2, "Then I realized my flight was delayed")
    context, diagnostics = confirm_local_performance_events(
        (first, second),
        (_timeline((_event("hand_motion_reset_candidate", 1.8, 1.9),)),),
        _context(),
    )
    assert diagnostics == ()
    assert not any(event.kind in {"wrong_take", "retry_setup"} for event in context.sources[0].events)


def test_reset_plus_disengagement_plus_retry_confirms_wrong_take():
    first = _take("a", 0.0, 2.0, "This serum makes my skin look much smoother")
    second = _take("b", 2.7, 4.8, "This serum really makes my skin look smoother")
    timeline = _timeline((
        _event("body_reset_candidate", 1.82, 1.94, 0.91),
        _event("camera_disengagement_candidate", 1.90, 2.03, 0.90),
    ))
    context, diagnostics = confirm_local_performance_events((first, second), (timeline,), _context())
    confirmed = [event for event in context.sources[0].events if event.kind == "wrong_take"]
    assert len(confirmed) == 1
    assert confirmed[0].start == first.start
    assert confirmed[0].end == first.end
    assert diagnostics[0]["retry_take_id"] == "b"
    decision = evaluate_take(first, context)
    assert decision.keep is False
    assert decision.reason.startswith("whole_video_bad_take:wrong_take")


def test_single_visual_family_with_retry_only_confirms_edge_setup():
    first = _take("a", 0.0, 2.0, "I found this after trying so many other products")
    second = _take("b", 2.6, 4.7, "I found this after I tried so many other products")
    timeline = _timeline((_event("body_reset_candidate", 1.86, 2.08, 0.90),))
    context, diagnostics = confirm_local_performance_events((first, second), (timeline,), _context())
    retry_events = [event for event in context.sources[0].events if event.kind == "retry_setup"]
    assert len(retry_events) == 1
    assert not any(event.kind == "wrong_take" for event in context.sources[0].events)
    assert diagnostics[0]["confirmed_kind"] == "retry_setup"
    # A single visual family must not delete a complete spoken take by itself.
    assert evaluate_take(first, context).keep is True
