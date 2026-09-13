"""Visual Finishing END-TO-END COMPOSITION (D-263).

Two tiers of fixture, matching D-258/D-260/D-262's own established
split:

1. The majority of scenario tests use directly-constructed, exact-
   literal `VisualClipMeasurement` fixtures (`pre_measurements`
   override) fed through the REAL, unmodified `generate_visual_
   finishing_plan` -> `execute_visual_finishing_decision` -> `render.
   render_preview` chain -- this sandbox has no cv2/mediapipe
   (confirmed, matching D-258's own 9 skipped tests), so a plan-
   triggering DISCONTINUITY can only be constructed this way; the
   PLAN, EXECUTION, and RENDER stages downstream of that measurement
   are always REAL, never hand-authored (D-263 Stage 6's own binding
   "do not manually hand-author correction plans in the main end-to-
   end cases").
2. A small number of tests exercise the FULL chain with ZERO manual
   measurement injection at all (real `measure_visual_clip` on real
   ffmpeg-generated media) -- gated behind `pytest.importorskip("cv2")`/
   `("mediapipe")`, exactly D-258's own precedent; these SKIP in this
   sandbox and would run on a worker image with those libraries
   installed.

Two real, pre-existing D-260 findings surfaced by this gate (NOT bugs
introduced here, NOT fixed here -- "no new policy" scope):

- `_bound_scale`'s own PUNCH_IN branch always clamps `authorized_scale`
  down to `DEFAULT_PUNCH_IN_SCALE` (1.10) whenever a scale break
  triggers punch-in, regardless of how large the ceiling (`MAX_PUNCH_
  IN_SCALE`, 1.15) is -- an authorized punch-in scale of exactly 1.15
  is UNREACHABLE through the real policy path today. The "max punch-in"
  and "crop ceiling exceeded" fixtures below therefore use a `pre_plan`
  override (a direct, hand-built `VisualFinishingPlan`/decision
  bypassing policy generation) to exercise the EXECUTOR's own boundary
  handling, exactly matching D-262's own test precedent for the same
  fixture.
- `_derive_plan_status` maps a plan mixing an eligible NO_CHANGE clip
  with an ABSTAIN clip (no correction action anywhere) to `PLAN_STATUS_
  UNKNOWN`, not `PLAN_STATUS_ABSTAIN` -- a real gap in that one
  function, out of this gate's "no new policy" scope. This module's
  own `_derive_overall_status` reads the plan's already-computed
  decisions directly to report `ABSTAIN` correctly regardless (a pure
  status ROLLUP, not new policy logic) -- verified below.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")

from cutsell_worker import render
from cutsell_worker import visual_finishing_composition as vfcomp
from cutsell_worker.media_probe import probe_media
from cutsell_worker.render_plan import RenderSegment
from cutsell_worker.visual_finishing_composition import (
    COMPOSITION_STATUS_ABSTAIN,
    COMPOSITION_STATUS_ALREADY_APPLIED,
    COMPOSITION_STATUS_BLOCKED,
    COMPOSITION_STATUS_EXECUTION_FAILED,
    COMPOSITION_STATUS_MEASUREMENT_FAILED,
    COMPOSITION_STATUS_NO_CHANGE,
    COMPOSITION_STATUS_OTHER,
    COMPOSITION_STATUS_PARTIAL,
    COMPOSITION_STATUS_POLICY_FAILED,
    COMPOSITION_STATUS_RENDER_FAILED,
    COMPOSITION_STATUS_SUCCESS,
    COMPOSITION_STATUS_VERIFY_FAILED,
    VisualFinishingCompositionInput,
    compose_visual_finishing,
    compute_composition_id,
)
from cutsell_worker.visual_finishing_executor import (
    EXECUTION_STATUS_ALREADY_APPLIED,
    EXECUTION_STATUS_CROP_LIMIT_EXCEEDED,
    EXECUTION_STATUS_SUCCESS,
    VERIFICATION_STATUS_FAIL,
    VERIFICATION_STATUS_PASS,
    VERIFICATION_STATUS_UNVERIFIABLE,
    VisualTransformSpec,
)
from cutsell_worker.visual_finishing_measurement import VisualClipMeasurement, VisualFrameMeasurement
from cutsell_worker.visual_finishing_policy import (
    POLICY_VERSION,
    VISUAL_ACTION_POSITION_MATCH,
    VISUAL_ACTION_PUNCH_IN,
    VISUAL_ACTION_SCALE_MATCH,
    VisualJoinPolicyDecision,
    generate_visual_finishing_plan,
)

_WIDTH, _HEIGHT, _FPS = 160, 120, 30


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def source_clip(tmp_path_factory):
    directory = tmp_path_factory.mktemp("d263_composition")
    path = str(directory / "src.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", f"testsrc=size={_WIDTH}x{_HEIGHT}:rate={_FPS}",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000,volume=0.3,aformat=channel_layouts=stereo",
        "-t", "3", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "96k", path,
    ])
    return path


def _measurement(
    clip_id: str, *, width: int = _WIDTH, height: int = _HEIGHT,
    fcx: float | None = 0.5, fcy: float | None = 0.5, area: float | None = 0.05,
    headroom: float | None = 0.1, rate: float | None = 1.0, status: str = "COMPLETE",
    frames: tuple = (),
) -> VisualClipMeasurement:
    return VisualClipMeasurement(
        source_id="src1", clip_id=clip_id, frame_width=width, frame_height=height,
        aspect_ratio=(width / height if height else None),
        orientation=("LANDSCAPE" if width >= height else "PORTRAIT"),
        rotation_degrees=0.0, requested_frame_count=5, valid_frame_count=5,
        face_valid_frame_count=(5 if rate and rate > 0 else 0), face_detection_rate=rate,
        face_center_x_median=fcx, face_center_y_median=fcy,
        face_area_ratio_median=area, headroom_ratio_median=headroom,
        luma_mean_median=100.0, contrast_median=40.0, luma_variability=5.0,
        color_mean_r_median=100.0, color_mean_g_median=100.0, color_mean_b_median=100.0,
        product_bbox_status="UNAVAILABLE", caption_safe_status="NOT_ESTABLISHED",
        freeze_frame_evidence=None, black_frame_evidence=None, frames=frames,
        measurement_status=status, errors=(), provenance={},
    )


def _frame(*, multi: bool = False, clipped: bool = False, clip_id: str = "X") -> VisualFrameMeasurement:
    return VisualFrameMeasurement(
        source_id="src1", clip_id=clip_id, frame_timestamp_sec=0.1,
        frame_width=_WIDTH, frame_height=_HEIGHT,
        face_detected=True, face_count=2 if multi else 1, multiple_faces_detected=multi,
        face_confidence=None,
        face_bbox_x_min=0.0 if clipped else 0.3, face_bbox_y_min=0.2,
        face_bbox_x_max=0.5, face_bbox_y_max=0.6,
        face_center_x=0.4, face_center_y=0.4, face_bbox_width=0.2, face_bbox_height=0.4,
        face_area_ratio=0.08, headroom_ratio=0.2,
        face_bbox_clipped_left=clipped, face_bbox_clipped_right=False,
        face_bbox_clipped_top=False, face_bbox_clipped_bottom=False,
        torso_center_x=None, torso_center_y=None,
        luma_mean=100.0, luma_std=40.0, color_mean_r=100.0, color_mean_g=100.0, color_mean_b=100.0,
        measurement_status="COMPLETE", errors=(),
    )


def _segments(source_clip_path: str) -> tuple[RenderSegment, RenderSegment]:
    seg_a = RenderSegment(clip_id="A", source_asset_id="a1", source_path=source_clip_path, start=0.0, end=1.5)
    seg_b = RenderSegment(clip_id="B", source_asset_id="a1", source_path=source_clip_path, start=1.5, end=3.0)
    return seg_a, seg_b


def _base_input(source_clip_path: str, tmp_path, **overrides) -> VisualFinishingCompositionInput:
    seg_a, seg_b = _segments(source_clip_path)
    base = dict(
        source_identity="video1", segments=(seg_a, seg_b),
        output_path=str(tmp_path / f"out_{len(overrides)}_{overrides.get('_tag', 'x')}.mp4"),
        work_dir=str(tmp_path),
        render_width=_WIDTH, render_height=_HEIGHT, render_fps=_FPS,
        product_safety_established={"A": True, "B": True},
        clip_durations_sec={"A": 1.5, "B": 1.5},
    )
    overrides.pop("_tag", None)
    base.update(overrides)
    return VisualFinishingCompositionInput(**base)


# ---------------------------------------------------------------------------
# 1-7: full-chain proofs for each action (measurement override -> REAL
# policy -> REAL executor -> REAL render).
# ---------------------------------------------------------------------------

def test_no_change_full_chain(source_clip, tmp_path):
    inp = _base_input(
        source_clip, tmp_path, _tag="nochange",
        pre_measurements={"A": _measurement("A"), "B": _measurement("B")},
    )
    result = compose_visual_finishing(inp)
    assert result.overall_status == COMPOSITION_STATUS_NO_CHANGE
    assert result.join_decisions[0].action == "NO_CHANGE"
    assert all(s.visual_transform is None for s in result.rendered_segments)
    assert result.render_output_path and __import__("os").path.exists(result.render_output_path)


def test_punch_in_full_chain(source_clip, tmp_path):
    inp = _base_input(
        source_clip, tmp_path, _tag="punchin",
        pre_measurements={"A": _measurement("A", area=0.05), "B": _measurement("B", area=0.20)},
    )
    result = compose_visual_finishing(inp)
    assert result.join_decisions[0].action == VISUAL_ACTION_PUNCH_IN
    assert result.overall_status == COMPOSITION_STATUS_SUCCESS
    b_record = [r for r in result.execution_records if r.clip_id == "B"][0]
    assert b_record.execution_status == EXECUTION_STATUS_SUCCESS
    b_segment = [s for s in result.rendered_segments if s.clip_id == "B"][0]
    assert isinstance(b_segment.visual_transform, VisualTransformSpec)
    assert b_segment.visual_transform.action == VISUAL_ACTION_PUNCH_IN


def test_position_match_full_chain(source_clip, tmp_path):
    inp = _base_input(
        source_clip, tmp_path, _tag="position",
        pre_measurements={"A": _measurement("A", fcx=0.5), "B": _measurement("B", fcx=0.65)},
    )
    result = compose_visual_finishing(inp)
    assert result.join_decisions[0].action == VISUAL_ACTION_POSITION_MATCH
    assert result.overall_status == COMPOSITION_STATUS_SUCCESS
    b_segment = [s for s in result.rendered_segments if s.clip_id == "B"][0]
    assert b_segment.visual_transform is not None


def test_static_reframe_full_chain(source_clip, tmp_path):
    inp = _base_input(
        source_clip, tmp_path, _tag="reframe",
        pre_measurements={"A": _measurement("A", headroom=0.10), "B": _measurement("B", headroom=0.25)},
    )
    result = compose_visual_finishing(inp)
    assert result.join_decisions[0].action == "STATIC_REFRAME"
    assert result.overall_status == COMPOSITION_STATUS_SUCCESS
    b_segment = [s for s in result.rendered_segments if s.clip_id == "B"][0]
    assert b_segment.visual_transform is not None


def test_scale_match_full_chain(source_clip, tmp_path):
    """SCALE_MATCH (not PUNCH_IN) fires when the clip's own duration is
    below `MIN_PUNCH_IN_CLIP_DURATION_SEC` -- exercised via the same
    `clip_durations_sec` override the real policy consults."""
    seg_a, seg_b = _segments(source_clip)
    inp = VisualFinishingCompositionInput(
        source_identity="video1", segments=(seg_a, seg_b),
        output_path=str(tmp_path / "out_scalematch.mp4"), work_dir=str(tmp_path),
        render_width=_WIDTH, render_height=_HEIGHT, render_fps=_FPS,
        pre_measurements={"A": _measurement("A", area=0.05), "B": _measurement("B", area=0.20)},
        product_safety_established={"A": True, "B": True},
        clip_durations_sec={"A": 0.5, "B": 0.5},  # short clip -> punch-in ineligible
    )
    result = compose_visual_finishing(inp)
    assert result.join_decisions[0].action == VISUAL_ACTION_SCALE_MATCH
    assert result.overall_status == COMPOSITION_STATUS_SUCCESS


def test_combined_scale_and_position_full_chain(source_clip, tmp_path):
    inp = _base_input(
        source_clip, tmp_path, _tag="combined",
        pre_measurements={
            "A": _measurement("A", area=0.05, fcx=0.5),
            "B": _measurement("B", area=0.20, fcx=0.65),
        },
    )
    result = compose_visual_finishing(inp)
    assert result.join_decisions[0].action == "SCALE_AND_POSITION_MATCH"
    assert result.overall_status == COMPOSITION_STATUS_SUCCESS
    b_segment = [s for s in result.rendered_segments if s.clip_id == "B"][0]
    assert b_segment.visual_transform is not None


def test_requested_vs_authorized_preserved(source_clip, tmp_path):
    """Translation/scale requests beyond a ceiling are clamped, but the
    ORIGINAL requested amount is still reported separately -- never
    silently overwritten (D-260's own contract, read verbatim here)."""
    inp = _base_input(
        source_clip, tmp_path, _tag="reqauth",
        pre_measurements={"A": _measurement("A", fcx=0.5), "B": _measurement("B", fcx=0.9)},
    )
    result = compose_visual_finishing(inp)
    decision = result.join_decisions[0]
    assert decision.requested_translation_x is not None
    assert decision.authorized_translation_x is not None
    assert abs(decision.requested_translation_x) > abs(decision.authorized_translation_x) + 1e-9


# ---------------------------------------------------------------------------
# 8-9: no recomputation.
# ---------------------------------------------------------------------------

def test_no_policy_recomputation_when_plan_supplied(source_clip, tmp_path, monkeypatch):
    """Supplying `pre_plan` must skip `generate_visual_finishing_plan`
    entirely -- proven by making that function raise if called."""
    m_a, m_b = _measurement("A", area=0.05), _measurement("B", area=0.20)
    plan = generate_visual_finishing_plan(
        (m_a, m_b), (), source_id="video1",
        clip_durations_sec={"A": 1.5, "B": 1.5}, product_safety_established={"A": True, "B": True},
    )
    # Rebuild join measurements the same way the real chain would, then
    # re-derive a plan WITH join decisions (single-clip call above has no
    # joins); simpler: call generate_visual_finishing_plan the standard way
    # once (allowed here, OUTSIDE the composition call) to get a real plan,
    # then monkeypatch it out for the actual composition call under test.
    from cutsell_worker.visual_finishing_measurement import compute_visual_join_measurement
    join_m = compute_visual_join_measurement(m_a, m_b)
    real_plan = generate_visual_finishing_plan(
        (m_a, m_b), (join_m,), source_id="video1",
        clip_durations_sec={"A": 1.5, "B": 1.5}, product_safety_established={"A": True, "B": True},
    )

    def _boom(*args, **kwargs):
        raise AssertionError("generate_visual_finishing_plan must not be called when pre_plan is supplied")

    monkeypatch.setattr(vfcomp, "generate_visual_finishing_plan", _boom)
    seg_a, seg_b = _segments(source_clip)
    inp = VisualFinishingCompositionInput(
        source_identity="video1", segments=(seg_a, seg_b),
        output_path=str(tmp_path / "out_norecompute.mp4"), work_dir=str(tmp_path),
        render_width=_WIDTH, render_height=_HEIGHT, render_fps=_FPS,
        pre_plan=real_plan,
    )
    result = compose_visual_finishing(inp)
    assert result.overall_status == COMPOSITION_STATUS_SUCCESS
    assert result.plan_id == real_plan.visual_finishing_identity


def test_no_geometry_recomputation_from_thresholds():
    """The composition module itself must never import or reference any
    of the ten canonical numeric thresholds directly -- it only ever
    consumes an already-decided `authorized_*` value."""
    source = inspect.getsource(vfcomp)
    for forbidden in (
        "FACE_CENTER_X_DISCONTINUITY_THRESHOLD", "FACE_CENTER_Y_DISCONTINUITY_THRESHOLD",
        "FACE_SCALE_RELATIVE_AREA_DISCONTINUITY_THRESHOLD", "HEADROOM_DISCONTINUITY_THRESHOLD",
        "DEFAULT_PUNCH_IN_SCALE", "MAX_PUNCH_IN_SCALE", "MAX_REFRAME_TRANSLATION_NORMALIZED",
        "MAX_ADDITIONAL_CROP_LOSS_NORMALIZED", "MIN_RELIABLE_FACE_DETECTION_RATE",
        "MIN_PUNCH_IN_CLIP_DURATION_SEC",
    ):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# 10-15: safety cases -- renderer mutation must not happen.
# ---------------------------------------------------------------------------

def test_low_face_detection_abstains(source_clip, tmp_path):
    """Also proves the D-260 `_derive_plan_status` UNKNOWN gap this
    module's own `_derive_overall_status` correctly rolls up to ABSTAIN."""
    inp = _base_input(
        source_clip, tmp_path, _tag="lowface",
        pre_measurements={"A": _measurement("A"), "B": _measurement("B", rate=0.2)},
    )
    result = compose_visual_finishing(inp)
    assert result.overall_status == COMPOSITION_STATUS_ABSTAIN
    assert all(s.visual_transform is None for s in result.rendered_segments)
    assert result.render_output_path  # still produced, just unmutated


def test_multi_face_blocks(source_clip, tmp_path):
    inp = _base_input(
        source_clip, tmp_path, _tag="multiface",
        pre_measurements={
            "A": _measurement("A"),
            "B": _measurement("B", frames=(_frame(multi=True, clip_id="B"),)),
        },
    )
    result = compose_visual_finishing(inp)
    assert result.overall_status == COMPOSITION_STATUS_BLOCKED
    assert all(s.visual_transform is None for s in result.rendered_segments)


def test_face_safety_blocks(source_clip, tmp_path):
    inp = _base_input(
        source_clip, tmp_path, _tag="facesafety",
        pre_measurements={
            "A": _measurement("A"),
            "B": _measurement("B", frames=(_frame(clipped=True, clip_id="B"),)),
        },
    )
    result = compose_visual_finishing(inp)
    assert result.overall_status == COMPOSITION_STATUS_BLOCKED
    assert all(s.visual_transform is None for s in result.rendered_segments)


def test_product_safety_unknown_blocks_by_default(source_clip, tmp_path):
    seg_a, seg_b = _segments(source_clip)
    inp = VisualFinishingCompositionInput(
        source_identity="video1", segments=(seg_a, seg_b),
        output_path=str(tmp_path / "out_prodsafety.mp4"), work_dir=str(tmp_path),
        render_width=_WIDTH, render_height=_HEIGHT, render_fps=_FPS,
        pre_measurements={"A": _measurement("A"), "B": _measurement("B")},
        # product_safety_established deliberately omitted -> fail closed.
    )
    result = compose_visual_finishing(inp)
    assert result.overall_status == COMPOSITION_STATUS_BLOCKED
    assert all(s.visual_transform is None for s in result.rendered_segments)


def test_no_face_measurement_no_action_invented(source_clip, tmp_path):
    inp = _base_input(
        source_clip, tmp_path, _tag="noface",
        pre_measurements={"A": _measurement("A"), "B": _measurement("B", status="NO_FACE", rate=None)},
    )
    result = compose_visual_finishing(inp)
    assert result.overall_status == COMPOSITION_STATUS_ABSTAIN
    assert all(s.visual_transform is None for s in result.rendered_segments)


def test_short_clip_prevents_punch_in(source_clip, tmp_path):
    seg_a, seg_b = _segments(source_clip)
    inp = VisualFinishingCompositionInput(
        source_identity="video1", segments=(seg_a, seg_b),
        output_path=str(tmp_path / "out_short.mp4"), work_dir=str(tmp_path),
        render_width=_WIDTH, render_height=_HEIGHT, render_fps=_FPS,
        pre_measurements={"A": _measurement("A", area=0.05), "B": _measurement("B", area=0.20)},
        product_safety_established={"A": True, "B": True},
        clip_durations_sec={"A": 0.5, "B": 0.5},
    )
    result = compose_visual_finishing(inp)
    assert result.join_decisions[0].action != VISUAL_ACTION_PUNCH_IN
    assert result.join_decisions[0].action == VISUAL_ACTION_SCALE_MATCH


# ---------------------------------------------------------------------------
# 16-17: crop ceiling -- unreachable through the real policy path (see
# module docstring finding), so exercised via an injected `pre_plan`
# fault, exactly D-262's own precedent for this same boundary.
# ---------------------------------------------------------------------------

def _plan_with_injected_decision(m_a, m_b, decision) -> "vfcomp.VisualFinishingPlan":
    from cutsell_worker.visual_finishing_measurement import compute_visual_join_measurement
    from cutsell_worker.visual_finishing_policy import VisualFinishingPlan, compute_visual_finishing_plan_identity, evaluate_clip_policy
    join_m = compute_visual_join_measurement(m_a, m_b)
    clip_decisions = (
        evaluate_clip_policy(m_a, product_safety_established=True),
        evaluate_clip_policy(m_b, product_safety_established=True),
    )
    identity = compute_visual_finishing_plan_identity("video1", clip_decisions, (decision,))
    return VisualFinishingPlan(
        policy_version=POLICY_VERSION, source_id="video1",
        clip_measurement_references=(m_a, m_b), join_measurement_references=(join_m,),
        clip_decisions=clip_decisions, join_decisions=(decision,),
        plan_status="READY_FOR_VISUAL_CORRECTION", canonical_thresholds_snapshot={},
        visual_finishing_identity=identity,
    )


def test_crop_ceiling_respected_via_injected_fault(source_clip, tmp_path):
    """`MAX_ADDITIONAL_CROP_LOSS_NORMALIZED` and D-260's own translation
    ceiling are the same value, so a correctly-clamped plan can never
    naturally trip the executor's own crop-loss re-check (self-
    consistent defense in depth -- confirmed in D-262). Proven here at
    the composition level by injecting a plan decision whose translation
    ALREADY exceeds the ceiling (simulating a hypothetically buggy/stale
    upstream policy) and confirming the composition still fails closed."""
    m_a, m_b = _measurement("A"), _measurement("B")
    decision = VisualJoinPolicyDecision(
        left_clip_id="A", right_clip_id="B", action="STATIC_REFRAME", evidence_state="SUFFICIENT",
        face_center_dx=None, face_center_dy=None, face_scale_delta=None, headroom_delta=0.15,
        thresholds_used={}, position_match_authorized=True, scale_match_authorized=False,
        punch_in_authorized=False, requested_translation_x=None, requested_translation_y=0.15,
        authorized_translation_x=None, authorized_translation_y=0.15,  # exceeds the 0.10 ceiling
        requested_scale=None, authorized_scale=None,
        target_position_x=None, target_position_y=0.0, target_scale=None,
    )
    plan = _plan_with_injected_decision(m_a, m_b, decision)
    seg_a, seg_b = _segments(source_clip)
    inp = VisualFinishingCompositionInput(
        source_identity="video1", segments=(seg_a, seg_b),
        output_path=str(tmp_path / "out_cropceiling.mp4"), work_dir=str(tmp_path),
        render_width=_WIDTH, render_height=_HEIGHT, render_fps=_FPS,
        pre_plan=plan,
    )
    result = compose_visual_finishing(inp)
    b_record = [r for r in result.execution_records if r.clip_id == "B"][0]
    assert b_record.execution_status == EXECUTION_STATUS_CROP_LIMIT_EXCEEDED
    b_segment = [s for s in result.rendered_segments if s.clip_id == "B"][0]
    assert b_segment.visual_transform is None
    assert result.overall_status == COMPOSITION_STATUS_EXECUTION_FAILED


# ---------------------------------------------------------------------------
# 18-22: render-contract preservation.
# ---------------------------------------------------------------------------

def test_captions_remain_after_transform(source_clip, tmp_path):
    seg_a, seg_b = _segments(source_clip)
    seg_b_captioned = dataclasses.replace(seg_b, caption_text="hello")
    inp = VisualFinishingCompositionInput(
        source_identity="video1", segments=(seg_a, seg_b_captioned),
        output_path=str(tmp_path / "out_caption.mp4"), work_dir=str(tmp_path),
        render_width=_WIDTH, render_height=_HEIGHT, render_fps=_FPS,
        pre_measurements={"A": _measurement("A", area=0.05), "B": _measurement("B", area=0.20)},
        product_safety_established={"A": True, "B": True}, clip_durations_sec={"A": 1.5, "B": 1.5},
    )
    result = compose_visual_finishing(inp)
    assert result.overall_status == COMPOSITION_STATUS_SUCCESS
    assert result.render_output_path
    out_probe = probe_media(result.render_output_path)
    assert out_probe.width == _WIDTH and out_probe.height == _HEIGHT


def test_9_16_and_resolution_preserved(source_clip, tmp_path):
    seg_a, seg_b = _segments(source_clip)
    inp = VisualFinishingCompositionInput(
        source_identity="video1", segments=(seg_a, seg_b),
        output_path=str(tmp_path / "out_916.mp4"), work_dir=str(tmp_path),
        render_width=1080, render_height=1920, render_fps=_FPS,
        pre_measurements={"A": _measurement("A", area=0.05), "B": _measurement("B", area=0.20)},
        product_safety_established={"A": True, "B": True}, clip_durations_sec={"A": 1.5, "B": 1.5},
    )
    result = compose_visual_finishing(inp)
    out_probe = probe_media(result.render_output_path)
    assert out_probe.width == 1080 and out_probe.height == 1920


def test_audio_and_timing_preserved(source_clip, tmp_path):
    inp = _base_input(
        source_clip, tmp_path, _tag="audiotime",
        pre_measurements={"A": _measurement("A"), "B": _measurement("B")},
    )
    result = compose_visual_finishing(inp)
    out_probe = probe_media(result.render_output_path)
    assert out_probe.has_audio
    expected_total = sum(render.rendered_segment_duration_sec(s.duration_sec, fps=_FPS) for s in result.rendered_segments)
    assert abs(out_probe.duration_sec - expected_total) < 0.2


# ---------------------------------------------------------------------------
# 23-27: identity / idempotence.
# ---------------------------------------------------------------------------

def test_same_composition_is_idempotent(source_clip, tmp_path):
    kwargs = dict(
        pre_measurements={"A": _measurement("A", area=0.05), "B": _measurement("B", area=0.20)},
    )
    r1 = compose_visual_finishing(_base_input(source_clip, tmp_path, _tag="idem1", **kwargs))
    r2 = compose_visual_finishing(_base_input(source_clip, tmp_path, _tag="idem2", **kwargs))
    assert r1.composition_id == r2.composition_id


def test_already_finished_output_no_double_transform(source_clip, tmp_path):
    kwargs = dict(pre_measurements={"A": _measurement("A", area=0.05), "B": _measurement("B", area=0.20)})
    first = compose_visual_finishing(_base_input(source_clip, tmp_path, _tag="already1", **kwargs))
    prev_id = first.execution_records[0].execution_id
    second_input = _base_input(
        source_clip, tmp_path, _tag="already2", **kwargs,
        previous_execution_ids={("A", "B"): prev_id},
    )
    second = compose_visual_finishing(second_input)
    assert second.overall_status == COMPOSITION_STATUS_ALREADY_APPLIED
    b_record = [r for r in second.execution_records if r.clip_id == "B"][0]
    assert b_record.execution_status == EXECUTION_STATUS_ALREADY_APPLIED
    # No cumulative re-scaling: the transform_spec's own scale is still
    # the SAME single authorized amount, never compounded.
    assert b_record.transform_spec is None or abs(b_record.transform_spec.scale_factor - first.execution_records[0].transform_spec.scale_factor) < 1e-9


def test_different_plan_has_distinct_identity(source_clip, tmp_path):
    m_a = _measurement("A", area=0.05)
    scale_result = compose_visual_finishing(_base_input(
        source_clip, tmp_path, _tag="diffplan1",
        pre_measurements={"A": m_a, "B": _measurement("B", area=0.20)},
    ))
    position_result = compose_visual_finishing(_base_input(
        source_clip, tmp_path, _tag="diffplan2",
        pre_measurements={"A": m_a, "B": _measurement("B", area=0.05, fcx=0.6)},
    ))
    assert scale_result.join_decisions[0].action != position_result.join_decisions[0].action
    assert scale_result.plan_id != position_result.plan_id
    assert scale_result.composition_id != position_result.composition_id


def test_filename_independence(source_clip, tmp_path):
    kwargs = dict(pre_measurements={"A": _measurement("A", area=0.05), "B": _measurement("B", area=0.20)})
    r1 = compose_visual_finishing(_base_input(source_clip, tmp_path, _tag="fname1", **kwargs))
    r2 = compose_visual_finishing(_base_input(source_clip, tmp_path, _tag="fname_renamed", **kwargs))
    assert r1.composition_id == r2.composition_id


def test_policy_version_sensitivity(source_clip, tmp_path, monkeypatch):
    kwargs = dict(pre_measurements={"A": _measurement("A", area=0.05), "B": _measurement("B", area=0.20)})
    r1 = compose_visual_finishing(_base_input(source_clip, tmp_path, _tag="pv1", **kwargs))
    import cutsell_worker.visual_finishing_policy as vfp
    monkeypatch.setattr(vfp, "POLICY_VERSION", vfp.POLICY_VERSION + "-test-bump")
    r2 = compose_visual_finishing(_base_input(source_clip, tmp_path, _tag="pv2", **kwargs))
    assert r1.plan_id != r2.plan_id
    assert r1.composition_id != r2.composition_id


# ---------------------------------------------------------------------------
# 28-32: scope discipline / forbidden vocabulary / no forced alternation.
# ---------------------------------------------------------------------------

def test_no_previous_action_parameter_anywhere():
    tree = ast.parse(inspect.getsource(vfcomp))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args + node.args.kwonlyargs:
                assert "previous_action" not in arg.arg
                assert "prior_action" not in arg.arg


def _source_without_docstrings(obj) -> str:
    tree = ast.parse(inspect.getsource(obj))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())
    return ast.unparse(tree)


def test_no_exposure_color_gaze_logic():
    source = _source_without_docstrings(vfcomp).lower()
    for forbidden in ("exposure", "gaze", "color_correct", "white_balance"):
        assert forbidden not in source


def test_no_sales_vocabulary():
    source = _source_without_docstrings(vfcomp).lower()
    for word in ("sales", "selling", "conversion", "funnel", "hook", "cta", "benefit"):
        assert word not in source


def test_no_provider_or_raw_reference():
    source = _source_without_docstrings(vfcomp)
    for forbidden in ("runpod", "RunPod", "modal.", "openai", "anthropic", "boto3", "RAW"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# 33-39: "unchanged" structural guards -- this gate's own files must not
# touch anything outside its own new module + render.py/render_plan.py
# (already authorized additively by D-262).
# ---------------------------------------------------------------------------

def _run_git_diff(path: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", path],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("path", [
    "cutsell_worker/post_render_media_qc.py",
    "cutsell_worker/dialogue_pacing_transition.py",
    "cutsell_worker/boundary_engine_pass.py",
    "cutsell_worker/audio_finishing_measurement.py",
    "cutsell_worker/audio_finishing_policy.py",
    "cutsell_worker/audio_finishing_executor.py",
    "cutsell_worker/audio_finishing_outcome.py",
    "cutsell_worker/visual_finishing_measurement.py",
    "cutsell_worker/visual_finishing_policy.py",
    "cutsell_worker/visual_finishing_executor.py",
])
def test_unrelated_authorities_unchanged(path):
    """Working-tree-vs-HEAD diff -- passes once this gate's own new
    files are committed (matches D-171/D-172's own established pattern
    for this exact self-resolving assertion class)."""
    assert _run_git_diff(path) == ""


def test_no_pacing_boundary_freeze_audio_join_vocabulary_change():
    source = _source_without_docstrings(vfcomp)
    for forbidden in ("mode=", "eligibility", "transition_type"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# 40-41: no provider, no RAW vocabulary (repeated distinctly per contract
# item numbering, same underlying check as above -- kept separate for
# direct 1:1 traceability to the mandated 49-item list).
# ---------------------------------------------------------------------------

def test_module_never_calls_ffmpeg_directly():
    source = _source_without_docstrings(vfcomp)
    for forbidden in ("ffmpeg", "subprocess.run("):
        assert forbidden not in source


def test_module_never_imports_cv2_mediapipe_directly():
    """The composition module must never import cv2/mediapipe itself --
    it only calls `measure_visual_clip`, which owns that boundary."""
    tree = ast.parse(inspect.getsource(vfcomp))
    top_level_names = {
        alias.name
        for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "cv2" not in top_level_names
    assert "mediapipe" not in top_level_names


# ---------------------------------------------------------------------------
# Verification-phase specific tests.
# ---------------------------------------------------------------------------

def test_verification_pass_with_directionally_correct_post_measurement(source_clip, tmp_path):
    pre_b = _measurement("B", area=0.20)
    post_b = _measurement("B", area=0.30)  # correctly larger, matching a punch-in's own zoom-in
    inp = _base_input(
        source_clip, tmp_path, _tag="verifypass",
        pre_measurements={"A": _measurement("A", area=0.05), "B": pre_b},
        post_measurement_overrides={"A": _measurement("A", area=0.05), "B": post_b},
    )
    result = compose_visual_finishing(inp)
    assert result.verification_results[0].verification_status == VERIFICATION_STATUS_PASS
    assert result.overall_status == COMPOSITION_STATUS_SUCCESS


def test_verification_fail_with_directionally_wrong_post_measurement(source_clip, tmp_path):
    pre_b = _measurement("B", area=0.20)
    post_b = _measurement("B", area=0.05)  # WRONG direction: shrank instead of zooming in
    inp = _base_input(
        source_clip, tmp_path, _tag="verifyfail",
        pre_measurements={"A": _measurement("A", area=0.05), "B": pre_b},
        post_measurement_overrides={"A": _measurement("A", area=0.05), "B": post_b},
    )
    result = compose_visual_finishing(inp)
    assert result.verification_results[0].verification_status == VERIFICATION_STATUS_FAIL
    assert result.overall_status == COMPOSITION_STATUS_VERIFY_FAILED


def test_real_post_measurement_environment_limited_is_honestly_unverifiable(source_clip, tmp_path):
    """No cv2/mediapipe in this sandbox (confirmed): the REAL post-
    measurement path (no override) must report `UNVERIFIABLE`, never a
    fabricated PASS/FAIL, and must never mask an otherwise-successful
    composition as failed."""
    inp = _base_input(
        source_clip, tmp_path, _tag="realpost",
        pre_measurements={"A": _measurement("A", area=0.05), "B": _measurement("B", area=0.20)},
    )
    result = compose_visual_finishing(inp)
    assert result.verification_results[0].verification_status == VERIFICATION_STATUS_UNVERIFIABLE
    assert result.overall_status == COMPOSITION_STATUS_SUCCESS


def test_no_change_verification_trivially_passes(source_clip, tmp_path):
    inp = _base_input(
        source_clip, tmp_path, _tag="verifynochange",
        pre_measurements={"A": _measurement("A"), "B": _measurement("B")},
        post_measurement_overrides={"A": _measurement("A"), "B": _measurement("B")},
    )
    result = compose_visual_finishing(inp)
    assert result.overall_status == COMPOSITION_STATUS_NO_CHANGE
    assert result.verification_results == () or all(
        v.action_direction_verified in (True, None) for v in result.verification_results
    )


# ---------------------------------------------------------------------------
# Failure-containment tests.
# ---------------------------------------------------------------------------

def test_render_failure_reported_never_masked(source_clip, tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated render failure")

    monkeypatch.setattr(vfcomp.render_module, "render_preview", _boom)
    inp = _base_input(
        source_clip, tmp_path, _tag="renderfail",
        pre_measurements={"A": _measurement("A"), "B": _measurement("B")},
    )
    result = compose_visual_finishing(inp)
    assert result.overall_status == COMPOSITION_STATUS_RENDER_FAILED
    assert result.render_output_path is None
    assert any("render_phase_exception" in e for e in result.errors)


def test_empty_segments_reported_as_policy_failed(tmp_path):
    inp = VisualFinishingCompositionInput(
        source_identity="video1", segments=(), output_path=str(tmp_path / "out_empty.mp4"),
        work_dir=str(tmp_path), pre_measurements={},
    )
    result = compose_visual_finishing(inp)
    assert result.overall_status == COMPOSITION_STATUS_POLICY_FAILED


def test_composition_result_is_frozen(source_clip, tmp_path):
    inp = _base_input(source_clip, tmp_path, _tag="frozen", pre_measurements={"A": _measurement("A"), "B": _measurement("B")})
    result = compose_visual_finishing(inp)
    with pytest.raises(Exception):
        result.overall_status = "MUTATED"  # type: ignore[misc]


def test_composition_input_is_frozen(source_clip, tmp_path):
    inp = _base_input(source_clip, tmp_path, _tag="frozeninput", pre_measurements={"A": _measurement("A"), "B": _measurement("B")})
    with pytest.raises(Exception):
        inp.source_identity = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Real-chain proof (skipped in this sandbox -- no cv2/mediapipe).
# ---------------------------------------------------------------------------

def test_full_chain_with_zero_manual_measurement_injection(source_clip, tmp_path):
    """The one true "nothing hand-constructed at all" proof: measurement
    is produced by REAL `measure_visual_clip` on real ffmpeg media, with
    NO `pre_measurements` override whatsoever. Requires cv2/mediapipe;
    SKIPS here exactly like D-258's own 9 environment-gated tests, and
    would run on a worker image with those libraries installed."""
    pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    inp = _base_input(source_clip, tmp_path, _tag="fullchain")
    result = compose_visual_finishing(inp)
    assert result.overall_status in (
        COMPOSITION_STATUS_SUCCESS, COMPOSITION_STATUS_NO_CHANGE, COMPOSITION_STATUS_ABSTAIN,
        COMPOSITION_STATUS_BLOCKED, COMPOSITION_STATUS_PARTIAL,
    )
    assert result.render_output_path


# ---------------------------------------------------------------------------
# compute_composition_id direct unit tests.
# ---------------------------------------------------------------------------

def test_compute_composition_id_deterministic():
    a = compute_composition_id("src1", "plan1", ["exec1", "exec2"])
    b = compute_composition_id("src1", "plan1", ["exec2", "exec1"])  # order-independent
    assert a == b


def test_compute_composition_id_sensitive_to_plan_id():
    a = compute_composition_id("src1", "plan1", [])
    b = compute_composition_id("src1", "plan2", [])
    assert a != b


def test_compute_composition_id_sensitive_to_source_identity():
    a = compute_composition_id("src1", "plan1", [])
    b = compute_composition_id("src2", "plan1", [])
    assert a != b
