"""D-145 (Gate 6 / RAW #118 Gap C): `_reset_debris_at_edges` treats every
qualifying A-5 event kind identically -- FAIL, unconditionally -- although
two structurally different kinds are mixed into `_RESET_KINDS`:

- raw visual/motion RESET CANDIDATES (`body_reset_candidate`,
  `hand_motion_reset_candidate`, `camera_disengagement_candidate`,
  `facial_expression_shift_candidate`): specific, positive evidence of a
  physical artifact (a stumble, a hand/body reset, a camera bump) that, if
  it bleeds into the kept window, is genuinely likely to be visible debris;

- EXPLICIT recording-process-break MARKERS (`retry_setup`, `wrong_take`,
  `false_start`, `breaking_character`): this is the SAME evidence class
  `attempt_reconstruction.py`'s own `_EXPLICIT_ATTEMPT_BREAK_KINDS` already
  uses elsewhere as the reason a cut boundary was correctly placed AT that
  exact point. A marker of this kind sitting within the 0.35s edge window is
  at least as consistent with "this is exactly why Selection/Boundary cut
  here, and the timestamp just has ordinary measurement slop" as it is with
  "residue leaked into the render" -- and this capability never decodes a
  single rendered frame to tell the two apart (its own `note` already says
  so). Reporting it as an unconditional FAIL misrepresents an unverified
  inference as a confirmed defect and can send BoundaryEngine trimming
  material that was never actually visible.

Fix (this capability's own measurement/classification, not BoundaryEngine):
an explicit-marker finding downgrades to UNCERTAIN; a visual/motion
candidate finding stays FAIL, unchanged -- the blocking capability status
stays EVALUATED_FAIL whenever at least one FAIL-severity finding exists
(mirrors the existing `interior_dead_air_mp4` FAIL/UNCERTAIN split), so this
never silently drops or weakens genuine visual-artifact evidence.
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


def test_explicit_marker_at_edge_is_uncertain_not_an_unconditional_fail():
    diag = _diag([
        {"kind": "wrong_take", "start": 10.05, "end": 10.30, "confidence": 0.93},
    ])
    report = pwl._reset_debris_at_edges(_draft(diag), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
    assert report.status == pwl.UNCERTAIN
    assert len(report.findings) == 1
    assert report.findings[0].severity == "UNCERTAIN"
    assert report.findings[0].routes_to == pwl.ROUTE_BOUNDARY


def test_visual_motion_candidate_at_edge_with_a_measured_pause_nearby_remains_a_hard_fail():
    """Regression guard, updated for D-149: a visual/motion candidate
    co-occurring with a REAL measured pause (the genuine reset/retry
    signature) still remains a hard FAIL -- this fix narrows, never
    eliminates, the strong-evidence path. See
    test_cutsell_d149_reset_candidate_pause_discrimination.py for the
    full pause-vs-continuous-speech discrimination this correction adds."""
    diag = _diag([
        {"kind": "hand_motion_reset_candidate", "start": 10.05, "end": 10.30, "confidence": 0.93},
        {"kind": "audio_silence_interval", "start": 10.00, "end": 10.40, "confidence": 1.0},
    ])
    report = pwl._reset_debris_at_edges(_draft(diag), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
    assert report.status == pwl.EVALUATED_FAIL
    assert report.findings[0].severity == "FAIL"


def test_mixed_evidence_capability_status_stays_fail_when_any_hard_evidence_present():
    """A capability run carrying BOTH an explicit marker (now UNCERTAIN) and
    a visual candidate WITH a measured pause nearby (still FAIL) must still
    block as EVALUATED_FAIL -- the softening of one finding never silently
    drops the other."""
    diag = _diag([
        {"kind": "wrong_take", "start": 10.05, "end": 10.30, "confidence": 0.93},
        {"kind": "body_reset_candidate", "start": 14.80, "end": 15.05, "confidence": 0.95},
        {"kind": "audio_silence_interval", "start": 14.75, "end": 15.10, "confidence": 1.0},
    ])
    report = pwl._reset_debris_at_edges(_draft(diag), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
    assert report.status == pwl.EVALUATED_FAIL
    severities = {f.detail["event_kind"]: f.severity for f in report.findings}
    assert severities["wrong_take"] == "UNCERTAIN"
    assert severities["body_reset_candidate"] == "FAIL"


def test_all_four_explicit_marker_kinds_downgrade_consistently():
    for kind in ("retry_setup", "wrong_take", "false_start", "breaking_character"):
        diag = _diag([{"kind": kind, "start": 10.05, "end": 10.30, "confidence": 0.93}])
        report = pwl._reset_debris_at_edges(_draft(diag), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
        assert report.status == pwl.UNCERTAIN, kind
        assert report.findings[0].severity == "UNCERTAIN", kind


def test_below_confidence_floor_explicit_marker_still_excluded_entirely():
    """The existing EDGE_DEBRIS_MIN_CONFIDENCE floor still applies before
    the new severity split -- this fix adds a distinction, it does not
    remove the pre-existing confidence gate."""
    diag = _diag([{"kind": "wrong_take", "start": 10.05, "end": 10.30, "confidence": 0.50}])
    report = pwl._reset_debris_at_edges(_draft(diag), (_seg("a", 10.0, 15.0),), [(0.0, 5.0)])
    assert report.findings == ()
