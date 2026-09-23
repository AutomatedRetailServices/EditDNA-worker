"""D-149 (Gate 6 correction, real RAW #118 audit): a real audit against the
actual RAW #118 artifact found the 14 reported `reset_debris_at_edges_
source_evidence` findings were ALL `hand_motion_reset_candidate` -- frame
review showed these were normal expressive gesture while speaking, and
natural/microphone-repositioning hand movement, not recording-process
resets. D-145's explicit-marker/visual-candidate split (Gap C, first pass)
does not touch this finding class at all, since `hand_motion_reset_
candidate` was already, correctly, a visual candidate kind.

Root cause: `local_performance.py:detect_candidate_events` emits `hand_
motion_reset_candidate` from pure frame-to-frame wrist-displacement
kinematics (`0.52 + hand_delta*2.4 + b.motion`) -- its own module docstring
says this is deliberate ("abrupt changes are emitted as `*_candidate`
events so semantic/retry context remains authoritative"). That `confidence`
is a MOTION-MAGNITUDE score, not a genuine-reset probability: a large, fast,
natural gesture during animated speech scores just as high as an actual
stop-and-restart reset. `EDGE_DEBRIS_MIN_CONFIDENCE` cannot discriminate
between them.

Fix (in `perceptual_watch_listen.py`, the capability's OWN measurement/
classification -- not `local_performance.py`, which correctly stays a pure
measurement layer per its own docstring, and not BoundaryEngine, which
still never auto-trims on this routed label at all today): a visual/motion
candidate now also needs a REAL, already-measured source pause
(`audio_silence.AUDIO_SILENCE_EVENT_KIND`, the same `mp4_measured`-grade
signal `interior_dead_air_mp4` already trusts) within a bounded window of
its own timestamp to keep its hard FAIL -- continuous, unbroken speech
through the exact moment is the general, non-Video00-specific signature of
natural gesture, not a reset. No visual candidate is ever silently dropped
or converted to PASS; the ones without pause evidence become UNCERTAIN
(still reported, still routed, still visible), never hidden.
"""
from cutsell_worker import perceptual_watch_listen as pwl
from cutsell_worker.contracts import DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.render_plan import RenderSegment


def _draft(diagnostics):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(), alternates=(), discarded=(), diagnostics=diagnostics,
    )


def _seg(clip_id, start, end):
    return RenderSegment(clip_id=clip_id, source_asset_id="src", source_path="/x.mp4", start=start, end=end)


def _diag(events):
    return {"whole_video_context": {"sources": [{"source_asset_id": "src", "events": list(events)}]}}


def test_hand_motion_candidate_during_continuous_speech_downgrades_to_uncertain():
    """The core regression: no measured pause anywhere nearby -- the
    creator was talking continuously through this moment. This is the
    real RAW #118 shape (all 14 findings): must be UNCERTAIN, not FAIL."""
    diag = _diag([
        {"kind": "hand_motion_reset_candidate", "start": 10.05, "end": 10.30, "confidence": 0.93},
    ])
    report = pwl._reset_debris_at_edges(_draft(diag), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
    assert report.status == pwl.UNCERTAIN
    assert len(report.findings) == 1
    assert report.findings[0].severity == "UNCERTAIN"
    assert report.findings[0].detail["measured_pause_nearby"] is False
    # Still reported and routed -- never silently dropped.
    assert report.findings[0].routes_to == pwl.ROUTE_BOUNDARY


def test_hand_motion_candidate_with_a_real_pause_immediately_before_it_stays_fail():
    """Positive control: a measured pause sits right before the abrupt
    hand-motion event -- the classic stop-then-restart shape. Real evidence
    of a genuine reset, so this stays a hard FAIL."""
    diag = _diag([
        {"kind": "hand_motion_reset_candidate", "start": 10.05, "end": 10.30, "confidence": 0.93},
        {"kind": "audio_silence_interval", "start": 9.60, "end": 10.02, "confidence": 1.0},
    ])
    report = pwl._reset_debris_at_edges(_draft(diag), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
    assert report.status == pwl.EVALUATED_FAIL
    assert report.findings[0].severity == "FAIL"
    assert report.findings[0].detail["measured_pause_nearby"] is True


def test_hand_motion_candidate_with_a_pause_just_after_it_also_stays_fail():
    """Positive control: pause immediately AFTER the motion (stop-mid-
    gesture-then-pause is just as real a reset signature as pause-then-
    motion) -- the proximity check is symmetric, not directional."""
    diag = _diag([
        {"kind": "hand_motion_reset_candidate", "start": 10.05, "end": 10.30, "confidence": 0.93},
        {"kind": "audio_silence_interval", "start": 10.35, "end": 10.90, "confidence": 1.0},
    ])
    report = pwl._reset_debris_at_edges(_draft(diag), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
    assert report.status == pwl.EVALUATED_FAIL
    assert report.findings[0].detail["measured_pause_nearby"] is True


def test_a_distant_pause_elsewhere_in_the_take_does_not_count():
    """Negative control: a real pause DOES exist in the source, but nowhere
    near this specific event -- must not be credited as evidence for an
    unrelated moment."""
    diag = _diag([
        {"kind": "hand_motion_reset_candidate", "start": 10.05, "end": 10.30, "confidence": 0.93},
        {"kind": "audio_silence_interval", "start": 40.0, "end": 41.0, "confidence": 1.0},
    ])
    report = pwl._reset_debris_at_edges(_draft(diag), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
    assert report.status == pwl.UNCERTAIN
    assert report.findings[0].detail["measured_pause_nearby"] is False


def test_all_four_visual_candidate_kinds_require_pause_evidence_consistently():
    for kind in (
        "body_reset_candidate", "hand_motion_reset_candidate",
        "camera_disengagement_candidate", "facial_expression_shift_candidate",
    ):
        no_pause = _diag([{"kind": kind, "start": 10.05, "end": 10.30, "confidence": 0.93}])
        report = pwl._reset_debris_at_edges(_draft(no_pause), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
        assert report.status == pwl.UNCERTAIN, kind

        with_pause = _diag([
            {"kind": kind, "start": 10.05, "end": 10.30, "confidence": 0.93},
            {"kind": "audio_silence_interval", "start": 9.60, "end": 10.02, "confidence": 1.0},
        ])
        report_pause = pwl._reset_debris_at_edges(_draft(with_pause), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
        assert report_pause.status == pwl.EVALUATED_FAIL, kind


def test_explicit_markers_are_unaffected_by_the_pause_check_entirely():
    """Regression guard: D-145's explicit-marker handling (always UNCERTAIN,
    never checks pause evidence at all) must be completely untouched by this
    correction -- the pause check applies ONLY to visual/motion candidates."""
    diag = _diag([
        {"kind": "wrong_take", "start": 10.05, "end": 10.30, "confidence": 0.93},
        {"kind": "audio_silence_interval", "start": 40.0, "end": 41.0, "confidence": 1.0},
    ])
    report = pwl._reset_debris_at_edges(_draft(diag), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
    assert report.status == pwl.UNCERTAIN
    assert report.findings[0].severity == "UNCERTAIN"
    assert "measured_pause_nearby" not in report.findings[0].detail
