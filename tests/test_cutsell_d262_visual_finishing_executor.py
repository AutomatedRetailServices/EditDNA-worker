"""Visual Finishing EXECUTOR FOUNDATION (D-262).

Two tiers of fixture, matching D-249/D-258's own established split:

1. Pure geometry/policy-routing tests use directly-constructed dataclass
   objects with EXACT literal numeric values -- no floating-point-noisy
   bbox/geometry reconstruction (`VisualClipMeasurement`/
   `VisualJoinPolicyDecision` built by hand, matching D-260's own test
   philosophy exactly: this module is a pure derivation layer over an
   already-decided policy action, so it deserves exact fixtures, not
   detector reconstructions).
2. The render.py integration tests use REAL, LOCALLY ffmpeg-generated
   synthetic media (`testsrc` lavfi sources, matching D-097.2/D-214's own
   precedent) and call `render._concat_render_command` directly, the same
   pattern D-214's own suite uses for `_concat_render_command_with_audio_
   windows`. No RAW, no downloaded content, no provider.
"""
from __future__ import annotations

import ast
import inspect
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")

from cutsell_worker import render
from cutsell_worker import visual_finishing_executor as vfe
from cutsell_worker.media_probe import probe_media
from cutsell_worker.render_plan import RenderSegment
from cutsell_worker.visual_finishing_executor import (
    EXECUTION_STATUS_ALREADY_APPLIED,
    EXECUTION_STATUS_CROP_LIMIT_EXCEEDED,
    EXECUTION_STATUS_FACE_SAFETY_BLOCKED,
    EXECUTION_STATUS_INVALID_GEOMETRY,
    EXECUTION_STATUS_NO_ACTION_NEEDED,
    EXECUTION_STATUS_OTHER,
    EXECUTION_STATUS_PLAN_NOT_EXECUTABLE,
    EXECUTION_STATUS_PRODUCT_SAFETY_BLOCKED,
    EXECUTION_STATUS_SOURCE_DIMENSIONS_UNAVAILABLE,
    EXECUTION_STATUS_SUCCESS,
    VERIFICATION_STATUS_FAIL,
    VERIFICATION_STATUS_PARTIAL,
    VERIFICATION_STATUS_PASS,
    VERIFICATION_STATUS_UNVERIFIABLE,
    VisualFinishingExecutionRecord,
    VisualFinishingVerificationResult,
    VisualTransformSpec,
    compute_crop_loss,
    compute_transform_geometry,
    compute_transformed_face_bbox,
    compute_visual_execution_id,
    execute_visual_finishing_decision,
    is_face_contained,
)
from cutsell_worker.visual_finishing_measurement import VisualClipMeasurement
from cutsell_worker.visual_finishing_policy import (
    DEFAULT_PUNCH_IN_SCALE,
    EVIDENCE_STATE_SUFFICIENT,
    MAX_ADDITIONAL_CROP_LOSS_NORMALIZED,
    MAX_PUNCH_IN_SCALE,
    MAX_REFRAME_TRANSLATION_NORMALIZED,
    VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE,
    VISUAL_ACTION_BLOCKED_FACE_SAFETY,
    VISUAL_ACTION_BLOCKED_MULTI_FACE,
    VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN,
    VISUAL_ACTION_NO_CHANGE,
    VISUAL_ACTION_POSITION_MATCH,
    VISUAL_ACTION_PUNCH_IN,
    VISUAL_ACTION_SCALE_AND_POSITION_MATCH,
    VISUAL_ACTION_SCALE_MATCH,
    VISUAL_ACTION_STATIC_REFRAME,
    VISUAL_ACTION_UNKNOWN,
    VisualJoinPolicyDecision,
)


def _measurement(width=1080, height=1920, fcx=0.5, fcy=0.5, **overrides) -> VisualClipMeasurement:
    base = dict(
        source_id="src1", clip_id="clip1", frame_width=width, frame_height=height,
        aspect_ratio=(width / height if height else None), orientation="PORTRAIT",
        rotation_degrees=0.0, requested_frame_count=5, valid_frame_count=5,
        face_valid_frame_count=5, face_detection_rate=1.0,
        face_center_x_median=fcx, face_center_y_median=fcy,
        face_area_ratio_median=0.1, headroom_ratio_median=0.1,
        luma_mean_median=100.0, contrast_median=40.0, luma_variability=5.0,
        color_mean_r_median=100.0, color_mean_g_median=100.0, color_mean_b_median=100.0,
        product_bbox_status="UNAVAILABLE", caption_safe_status="NOT_ESTABLISHED",
        freeze_frame_evidence=None, black_frame_evidence=None, frames=(),
        measurement_status="OK", errors=(), provenance={},
    )
    base.update(overrides)
    return VisualClipMeasurement(**base)


def _source_without_docstrings(obj) -> str:
    """Source of `obj` (a module, function, or class) with every docstring
    stripped -- avoids the false-positive class this suite has already hit
    twice in D-256/D-260: a module's own scope-discipline PROSE ("never
    calls ffmpeg", "no sales funnel") legitimately mentions the forbidden
    word while documenting that it does NOT do the forbidden thing."""
    tree = ast.parse(inspect.getsource(obj))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())
    return ast.unparse(tree)


def _decision(action: str, **overrides) -> VisualJoinPolicyDecision:
    base = dict(
        left_clip_id="clip0", right_clip_id="clip1", action=action, evidence_state=EVIDENCE_STATE_SUFFICIENT,
        face_center_dx=None, face_center_dy=None, face_scale_delta=None, headroom_delta=None,
        thresholds_used={}, position_match_authorized=False, scale_match_authorized=False,
        punch_in_authorized=False, requested_translation_x=None, requested_translation_y=None,
        authorized_translation_x=None, authorized_translation_y=None, requested_scale=None,
        authorized_scale=None, target_position_x=None, target_position_y=None, target_scale=None,
        safety_blockers=(), warnings=(), reasons=(), provenance={},
    )
    base.update(overrides)
    return VisualJoinPolicyDecision(**base)


# ---------------------------------------------------------------------------
# 1-6: compute_crop_loss.
# ---------------------------------------------------------------------------

def test_crop_loss_zero_when_no_translation():
    assert compute_crop_loss(None, None) == 0.0
    assert compute_crop_loss(0.0, 0.0) == 0.0


def test_crop_loss_x_only():
    assert compute_crop_loss(0.05, None) == 0.05


def test_crop_loss_y_only():
    assert compute_crop_loss(None, -0.08) == 0.08


def test_crop_loss_takes_max_magnitude_not_sum():
    assert compute_crop_loss(0.05, 0.08) == 0.08


def test_crop_loss_is_never_a_function_of_scale():
    """The module's own central unit-semantics finding: crop loss is the
    TRANSLATION component only, never derived from `authorized_scale`."""
    source = _source_without_docstrings(vfe.compute_crop_loss).lower()
    assert "scale" not in source


def test_crop_loss_pure_scale_action_is_zero():
    spec, crop_loss, err = compute_transform_geometry(
        VISUAL_ACTION_PUNCH_IN, source_width=1080, source_height=1920,
        authorized_scale=1.10, authorized_translation_x=None, authorized_translation_y=None,
    )
    assert err is None and crop_loss == 0.0


# ---------------------------------------------------------------------------
# 7-14: compute_transform_geometry exact math.
# ---------------------------------------------------------------------------

def test_punch_in_default_scale_exact_geometry():
    spec, crop_loss, err = compute_transform_geometry(
        VISUAL_ACTION_PUNCH_IN, source_width=1080, source_height=1920,
        authorized_scale=DEFAULT_PUNCH_IN_SCALE, authorized_translation_x=0.0, authorized_translation_y=0.0,
    )
    assert err is None
    assert crop_loss == 0.0
    assert abs(spec.scale_factor - DEFAULT_PUNCH_IN_SCALE) < 1e-9
    assert spec.crop_width == 1080 and spec.crop_height == 1920
    assert spec.scaled_width == round(1080 * DEFAULT_PUNCH_IN_SCALE)
    assert spec.scaled_height == round(1920 * DEFAULT_PUNCH_IN_SCALE)


def test_punch_in_max_scale_exact_geometry():
    spec, crop_loss, err = compute_transform_geometry(
        VISUAL_ACTION_PUNCH_IN, source_width=1080, source_height=1920,
        authorized_scale=MAX_PUNCH_IN_SCALE, authorized_translation_x=0.0, authorized_translation_y=0.0,
    )
    assert err is None
    assert abs(spec.scale_factor - MAX_PUNCH_IN_SCALE) < 1e-9


def test_executor_never_reclamps_an_over_max_scale_from_the_plan():
    """The executor translates whatever `authorized_scale` the D-260 plan
    already gives it -- it never re-applies `MAX_PUNCH_IN_SCALE` itself
    (that ceiling is D-260's own responsibility; D-261 Stage 9's "policy
    intent alone is not enough" governs SAFETY re-verification, not
    re-deriving an already-authorized numeric value)."""
    over_max = MAX_PUNCH_IN_SCALE + 0.50
    spec, crop_loss, err = compute_transform_geometry(
        VISUAL_ACTION_PUNCH_IN, source_width=1080, source_height=1920,
        authorized_scale=over_max, authorized_translation_x=0.0, authorized_translation_y=0.0,
    )
    assert err is None
    assert abs(spec.scale_factor - over_max) < 1e-9


def test_static_reframe_translation_only_zoom_x():
    spec, crop_loss, err = compute_transform_geometry(
        VISUAL_ACTION_STATIC_REFRAME, source_width=1080, source_height=1920,
        authorized_scale=None, authorized_translation_x=0.05, authorized_translation_y=0.0,
    )
    assert err is None
    assert abs(crop_loss - 0.05) < 1e-9
    assert abs(spec.scale_factor - (1.0 / (1.0 - 0.05))) < 1e-9


def test_static_reframe_translation_only_zoom_y():
    spec, crop_loss, err = compute_transform_geometry(
        VISUAL_ACTION_STATIC_REFRAME, source_width=1080, source_height=1920,
        authorized_scale=None, authorized_translation_x=0.0, authorized_translation_y=-0.07,
    )
    assert err is None
    assert abs(crop_loss - 0.07) < 1e-9
    assert abs(spec.scale_factor - (1.0 / (1.0 - 0.07))) < 1e-9


def test_position_match_uses_same_zoom_only_formula_as_static_reframe():
    spec_a, loss_a, _ = compute_transform_geometry(
        VISUAL_ACTION_STATIC_REFRAME, source_width=1080, source_height=1920,
        authorized_scale=None, authorized_translation_x=0.04, authorized_translation_y=0.0,
    )
    spec_b, loss_b, _ = compute_transform_geometry(
        VISUAL_ACTION_POSITION_MATCH, source_width=1080, source_height=1920,
        authorized_scale=None, authorized_translation_x=0.04, authorized_translation_y=0.0,
    )
    assert loss_a == loss_b
    assert abs(spec_a.scale_factor - spec_b.scale_factor) < 1e-9


def test_scale_match_pure_scale_no_translation():
    spec, crop_loss, err = compute_transform_geometry(
        VISUAL_ACTION_SCALE_MATCH, source_width=1080, source_height=1920,
        authorized_scale=1.05, authorized_translation_x=0.0, authorized_translation_y=0.0,
    )
    assert err is None
    assert crop_loss == 0.0
    assert abs(spec.scale_factor - 1.05) < 1e-9


def test_scale_and_position_match_combines_scale_and_translation():
    spec, crop_loss, err = compute_transform_geometry(
        VISUAL_ACTION_SCALE_AND_POSITION_MATCH, source_width=1080, source_height=1920,
        authorized_scale=1.05, authorized_translation_x=0.08, authorized_translation_y=0.0,
    )
    assert err is None
    assert abs(crop_loss - 0.08) < 1e-9
    translate_zoom = 1.0 / (1.0 - 0.08)
    # scale_factor is the LARGER of the two independent needs, never their sum.
    assert abs(spec.scale_factor - max(1.05, translate_zoom)) < 1e-9


def test_crop_loss_exactly_at_ceiling_is_not_exceeded():
    spec, crop_loss, err = compute_transform_geometry(
        VISUAL_ACTION_STATIC_REFRAME, source_width=1080, source_height=1920,
        authorized_scale=None, authorized_translation_x=MAX_REFRAME_TRANSLATION_NORMALIZED, authorized_translation_y=0.0,
    )
    assert err is None
    assert crop_loss == MAX_ADDITIONAL_CROP_LOSS_NORMALIZED
    assert crop_loss <= MAX_ADDITIONAL_CROP_LOSS_NORMALIZED


def test_focal_point_offsets_punch_in_crop_center():
    centered, _, _ = compute_transform_geometry(
        VISUAL_ACTION_PUNCH_IN, source_width=1080, source_height=1920,
        authorized_scale=1.10, authorized_translation_x=0.0, authorized_translation_y=0.0,
        focal_x=0.5, focal_y=0.5,
    )
    offset, _, _ = compute_transform_geometry(
        VISUAL_ACTION_PUNCH_IN, source_width=1080, source_height=1920,
        authorized_scale=1.10, authorized_translation_x=0.0, authorized_translation_y=0.0,
        focal_x=0.7, focal_y=0.5,
    )
    assert offset.crop_x > centered.crop_x


def test_invalid_source_dimensions_returns_none_spec_with_error():
    spec, crop_loss, err = compute_transform_geometry(
        VISUAL_ACTION_PUNCH_IN, source_width=0, source_height=1920,
        authorized_scale=1.10, authorized_translation_x=0.0, authorized_translation_y=0.0,
    )
    assert spec is None
    assert err == "invalid_source_dimensions"


def test_degenerate_translation_at_or_beyond_one_returns_error():
    spec, crop_loss, err = compute_transform_geometry(
        VISUAL_ACTION_STATIC_REFRAME, source_width=1080, source_height=1920,
        authorized_scale=None, authorized_translation_x=1.0, authorized_translation_y=0.0,
    )
    assert spec is None
    assert err == "translation_magnitude_invalid"


# ---------------------------------------------------------------------------
# 15-19: face bbox transform + containment.
# ---------------------------------------------------------------------------

def test_transformed_face_bbox_centered_stays_contained():
    spec, _, _ = compute_transform_geometry(
        VISUAL_ACTION_PUNCH_IN, source_width=1080, source_height=1920,
        authorized_scale=1.10, authorized_translation_x=0.0, authorized_translation_y=0.0,
    )
    transformed = compute_transformed_face_bbox(spec, (0.4, 0.4, 0.6, 0.6))
    assert is_face_contained(transformed)


def test_transformed_face_bbox_near_edge_pushed_out_of_bounds():
    spec, _, _ = compute_transform_geometry(
        VISUAL_ACTION_PUNCH_IN, source_width=1080, source_height=1920,
        authorized_scale=1.10, authorized_translation_x=0.0, authorized_translation_y=0.0,
    )
    transformed = compute_transformed_face_bbox(spec, (0.85, 0.4, 1.0, 0.6))
    assert not is_face_contained(transformed)


def test_is_face_contained_exact_boundary_is_contained():
    assert is_face_contained((0.0, 0.0, 1.0, 1.0)) is True


def test_is_face_contained_just_past_boundary_is_not():
    assert is_face_contained((0.0, -0.001, 1.0, 1.0)) is False
    assert is_face_contained((0.0, 0.0, 1.0001, 1.0)) is False


def test_transformed_bbox_uses_integer_scaled_dims_not_float_recompute():
    """`compute_transformed_face_bbox` must use `spec.scaled_width`/
    `scaled_height` (the SAME integer pixel dims the crop window was cut
    from), never re-derive `source * scale_factor` as a fresh float --
    keeps bbox verification geometrically consistent with the actual crop."""
    source = inspect.getsource(vfe.compute_transformed_face_bbox)
    assert "spec.scaled_width" in source and "spec.scaled_height" in source


# ---------------------------------------------------------------------------
# 20-31: execute_visual_finishing_decision routing (all 11 actions + unknown).
# ---------------------------------------------------------------------------

def test_no_change_routes_to_no_action_needed():
    rec = execute_visual_finishing_decision(_decision(VISUAL_ACTION_NO_CHANGE), _measurement(), plan_id="p1")
    assert rec.execution_status == EXECUTION_STATUS_NO_ACTION_NEEDED
    assert rec.transform_spec is None


@pytest.mark.parametrize("action", [
    VISUAL_ACTION_BLOCKED_MULTI_FACE, VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE, VISUAL_ACTION_UNKNOWN,
])
def test_no_selection_actions_route_to_plan_not_executable(action):
    rec = execute_visual_finishing_decision(_decision(action), _measurement(), plan_id="p1")
    assert rec.execution_status == EXECUTION_STATUS_PLAN_NOT_EXECUTABLE
    assert rec.transform_spec is None


def test_product_safety_unknown_routes_to_product_safety_blocked():
    rec = execute_visual_finishing_decision(
        _decision(VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN), _measurement(), plan_id="p1",
    )
    assert rec.execution_status == EXECUTION_STATUS_PRODUCT_SAFETY_BLOCKED
    assert rec.transform_spec is None


def test_blocked_face_safety_routes_to_face_safety_blocked():
    rec = execute_visual_finishing_decision(_decision(VISUAL_ACTION_BLOCKED_FACE_SAFETY), _measurement(), plan_id="p1")
    assert rec.execution_status == EXECUTION_STATUS_FACE_SAFETY_BLOCKED
    assert rec.transform_spec is None


def test_unrecognized_action_routes_to_other():
    rec = execute_visual_finishing_decision(_decision("SOME_BOGUS_ACTION"), _measurement(), plan_id="p1")
    assert rec.execution_status == EXECUTION_STATUS_OTHER
    assert rec.errors


def test_source_dimensions_unavailable_for_correction_action():
    rec = execute_visual_finishing_decision(
        _decision(VISUAL_ACTION_PUNCH_IN, authorized_scale=1.10, authorized_translation_x=0.0, authorized_translation_y=0.0),
        _measurement(width=None, height=None), plan_id="p1",
    )
    assert rec.execution_status == EXECUTION_STATUS_SOURCE_DIMENSIONS_UNAVAILABLE


def test_punch_in_success_produces_transform_spec():
    rec = execute_visual_finishing_decision(
        _decision(VISUAL_ACTION_PUNCH_IN, authorized_scale=1.10, authorized_translation_x=0.0, authorized_translation_y=0.0),
        _measurement(), plan_id="p1",
    )
    assert rec.execution_status == EXECUTION_STATUS_SUCCESS
    assert rec.transform_spec is not None
    assert abs(rec.transform_spec.scale_factor - 1.10) < 1e-9
    assert rec.renderer_operation == "SCALE_THEN_CROP"


def test_crop_limit_exceeded_when_translation_beyond_ceiling():
    rec = execute_visual_finishing_decision(
        _decision(VISUAL_ACTION_STATIC_REFRAME, authorized_translation_x=0.15, authorized_translation_y=0.0),
        _measurement(), plan_id="p1",
    )
    assert rec.execution_status == EXECUTION_STATUS_CROP_LIMIT_EXCEEDED
    assert rec.crop_loss == 0.15


def test_execution_time_face_safety_reverification_blocks_unsafe_transform():
    """D-261 Stage 9/10's doctrine: policy authorization alone is not
    enough -- the executor independently re-checks face containment
    against the REAL face bbox before ever returning SUCCESS."""
    rec = execute_visual_finishing_decision(
        _decision(VISUAL_ACTION_PUNCH_IN, authorized_scale=1.10, authorized_translation_x=0.0, authorized_translation_y=0.0),
        _measurement(), plan_id="p1", face_bbox=(0.85, 0.4, 1.0, 0.6),
    )
    assert rec.execution_status == EXECUTION_STATUS_FACE_SAFETY_BLOCKED
    assert rec.face_safety_status == "FAIL"


def test_no_face_bbox_supplied_still_succeeds_but_unverifiable():
    """Absent face evidence at execution time is reported honestly
    (`UNVERIFIABLE`), never silently assumed safe or invented."""
    rec = execute_visual_finishing_decision(
        _decision(VISUAL_ACTION_PUNCH_IN, authorized_scale=1.10, authorized_translation_x=0.0, authorized_translation_y=0.0),
        _measurement(), plan_id="p1",
    )
    assert rec.execution_status == EXECUTION_STATUS_SUCCESS
    assert rec.face_safety_status == "UNVERIFIABLE"


# ---------------------------------------------------------------------------
# 32-40: identity / idempotence.
# ---------------------------------------------------------------------------

def test_same_inputs_produce_same_execution_id():
    rec_a = execute_visual_finishing_decision(
        _decision(VISUAL_ACTION_SCALE_MATCH, authorized_scale=1.05, authorized_translation_x=0.0, authorized_translation_y=0.0),
        _measurement(), plan_id="p1", source_identity="abc123",
    )
    rec_b = execute_visual_finishing_decision(
        _decision(VISUAL_ACTION_SCALE_MATCH, authorized_scale=1.05, authorized_translation_x=0.0, authorized_translation_y=0.0),
        _measurement(), plan_id="p1", source_identity="abc123",
    )
    assert rec_a.execution_id == rec_b.execution_id


def test_different_plan_id_produces_different_execution_id():
    rec_a = execute_visual_finishing_decision(
        _decision(VISUAL_ACTION_SCALE_MATCH, authorized_scale=1.05, authorized_translation_x=0.0, authorized_translation_y=0.0),
        _measurement(), plan_id="p1",
    )
    rec_b = execute_visual_finishing_decision(
        _decision(VISUAL_ACTION_SCALE_MATCH, authorized_scale=1.05, authorized_translation_x=0.0, authorized_translation_y=0.0),
        _measurement(), plan_id="p2",
    )
    assert rec_a.execution_id != rec_b.execution_id


def test_execution_id_is_filename_independent():
    """Identity is a pure function of already-decided CONTENT, never a
    path -- mirrors `compute_visual_finishing_plan_identity`'s own
    filename-independence contract."""
    id_a = compute_visual_execution_id("content-hash-1", "p1", "clip1", VISUAL_ACTION_PUNCH_IN, None)
    id_b = compute_visual_execution_id("content-hash-1", "p1", "clip1", VISUAL_ACTION_PUNCH_IN, None)
    assert id_a == id_b


def test_execution_id_sensitive_to_policy_version():
    """A policy-version bump must change identity so a stale execution is
    never mistaken for `ALREADY_APPLIED` under a new policy. `POLICY_
    VERSION` is bound into `vfe`'s own namespace via `from ... import`, so
    the patch target is `vfe.POLICY_VERSION` (mirrors the real read site
    inside `compute_visual_execution_id`), not the origin module's
    attribute."""
    original = vfe.POLICY_VERSION
    try:
        id_before = compute_visual_execution_id("src", "p1", "clip1", VISUAL_ACTION_PUNCH_IN, None)
        vfe.POLICY_VERSION = original + "-test-bump"
        id_after = compute_visual_execution_id("src", "p1", "clip1", VISUAL_ACTION_PUNCH_IN, None)
    finally:
        vfe.POLICY_VERSION = original
    assert id_before != id_after


def test_previous_execution_id_match_produces_already_applied():
    decision = _decision(VISUAL_ACTION_PUNCH_IN, authorized_scale=1.10, authorized_translation_x=0.0, authorized_translation_y=0.0)
    first = execute_visual_finishing_decision(decision, _measurement(), plan_id="p1")
    second = execute_visual_finishing_decision(decision, _measurement(), plan_id="p1", previous_execution_id=first.execution_id)
    assert second.execution_status == EXECUTION_STATUS_ALREADY_APPLIED
    assert second.execution_id == first.execution_id


def test_previous_execution_id_mismatch_still_executes_normally():
    decision = _decision(VISUAL_ACTION_PUNCH_IN, authorized_scale=1.10, authorized_translation_x=0.0, authorized_translation_y=0.0)
    rec = execute_visual_finishing_decision(decision, _measurement(), plan_id="p1", previous_execution_id="not-a-real-id")
    assert rec.execution_status == EXECUTION_STATUS_SUCCESS


def test_execution_records_are_frozen():
    rec = execute_visual_finishing_decision(_decision(VISUAL_ACTION_NO_CHANGE), _measurement(), plan_id="p1")
    with pytest.raises(Exception):
        rec.execution_status = "MUTATED"  # type: ignore[misc]


def test_transform_spec_is_frozen():
    spec, _, _ = compute_transform_geometry(
        VISUAL_ACTION_PUNCH_IN, source_width=1080, source_height=1920,
        authorized_scale=1.10, authorized_translation_x=0.0, authorized_translation_y=0.0,
    )
    with pytest.raises(Exception):
        spec.scale_factor = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 41-48: scope discipline / forbidden-vocabulary / no-reinterpretation.
# ---------------------------------------------------------------------------

def test_module_never_calls_ffmpeg_or_subprocess():
    source = _source_without_docstrings(vfe)
    for forbidden in ("subprocess", "ffmpeg", "cv2.VideoCapture"):
        assert forbidden not in source


def test_module_has_no_exposure_color_or_gaze_logic():
    source = _source_without_docstrings(vfe).lower()
    for forbidden in ("exposure", "gaze", "color_correct", "white_balance"):
        assert forbidden.lower() not in source


_FORBIDDEN_SALES_WORDS = ("sales", "selling", "conversion", "funnel")


def test_no_sales_vocabulary_anywhere_in_executor_module():
    source = _source_without_docstrings(vfe).lower()
    for word in _FORBIDDEN_SALES_WORDS:
        assert word not in source


def test_no_provider_or_raw_reference():
    source = _source_without_docstrings(vfe)
    for forbidden in ("runpod", "RunPod", "modal.", "openai", "anthropic", "boto3"):
        assert forbidden not in source


def test_module_does_not_reimport_new_numeric_thresholds():
    """D-262 must read the ten canonical thresholds verbatim, never
    redeclare/extend them with a new NUMERIC policy value. Only checks
    module-level assignments whose VALUE is an int/float literal --
    string-vocabulary constants (`EXECUTION_STATUS_*`) and the private
    routing frozensets/dicts (`_CORRECTION_ACTIONS`, `_SCALE_ONLY_
    ACTIONS`, `_ACTION_TO_BLOCKED_STATUS`) are not numeric policy and are
    correctly excluded."""
    tree = ast.parse(inspect.getsource(vfe))
    # Module TOP-LEVEL assignments only (`tree.body`, not `ast.walk`) --
    # local variables inside function bodies (e.g. `base_scale` inside
    # `compute_transform_geometry`) are ordinary intermediate arithmetic,
    # never a new standing numeric policy constant.
    numeric_constant_names = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, (int, float))
        and not isinstance(node.value.value, bool)
    }
    assert numeric_constant_names == set()


def test_no_previous_action_parameter_anywhere():
    """NO FORCED PUNCH-IN ALTERNATION, structurally: no function in this
    module takes a "previous action" style parameter."""
    tree = ast.parse(inspect.getsource(vfe))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args + node.args.kwonlyargs:
                assert "previous_action" not in arg.arg
                assert "prior_action" not in arg.arg


def test_plan_decision_values_never_recomputed_only_translated():
    """The executor must consume `decision.authorized_scale`/
    `authorized_translation_x`/`_y` verbatim -- never re-derive them from
    `requested_*` fields (that re-evaluation is D-260's own authority)."""
    source = inspect.getsource(vfe.execute_visual_finishing_decision)
    assert "requested_scale" not in source
    assert "requested_translation" not in source


def test_verification_result_type_matches_stage2_field_list():
    fields = {f for f in VisualFinishingVerificationResult.__dataclass_fields__}
    expected = {
        "measurement_status", "face_detected", "face_bbox", "face_center_x", "face_center_y",
        "face_area_ratio", "headroom_ratio", "face_contained", "orientation", "width", "height",
        "duration_preserved", "action_direction_verified", "verification_status", "errors", "provenance",
    }
    assert expected.issubset(fields)


# ---------------------------------------------------------------------------
# 49-56: render.py / render_plan.py additive integration (REAL synthetic media).
# ---------------------------------------------------------------------------

_WIDTH, _HEIGHT, _FPS = 160, 120, 30


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def source_clip(tmp_path_factory):
    directory = tmp_path_factory.mktemp("d262_visual_executor")
    path = str(directory / "src.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", f"testsrc=size={_WIDTH}x{_HEIGHT}:rate={_FPS}",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000,volume=0.3,aformat=channel_layouts=stereo",
        "-t", "3", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "96k", path,
    ])
    return path


def test_render_segment_default_visual_transform_is_none():
    seg = RenderSegment(clip_id="c1", source_asset_id="a1", source_path="/x.mp4", start=0.0, end=1.0)
    assert seg.visual_transform is None


def test_no_change_path_byte_identical_to_pre_d262_filtergraph(source_clip, tmp_path):
    """`segment.visual_transform=None` (every live-produced segment today)
    must produce the EXACT SAME per-segment video chain as before D-262 --
    the `scale=...,pad=...,setsar=1,fps=...` head, no `crop=` clause."""
    segment = RenderSegment(clip_id="c1", source_asset_id="a1", source_path=source_clip, start=0.0, end=1.0)
    command = render._concat_render_command((segment,), tmp_path / "out.mp4", width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    filter_complex = command[command.index("-filter_complex") + 1]
    assert "crop=" not in filter_complex
    assert f"scale={_WIDTH}:{_HEIGHT}:force_original_aspect_ratio=decrease" in filter_complex


def test_visual_transform_inserts_scale_then_crop_before_fit_to_canvas(source_clip, tmp_path):
    spec = VisualTransformSpec(
        action=VISUAL_ACTION_PUNCH_IN, source_width=_WIDTH, source_height=_HEIGHT,
        scale_factor=1.10, scaled_width=round(_WIDTH * 1.10), scaled_height=round(_HEIGHT * 1.10),
        crop_x=8, crop_y=6, crop_width=_WIDTH, crop_height=_HEIGHT,
    )
    segment = RenderSegment(clip_id="c1", source_asset_id="a1", source_path=source_clip, start=0.0, end=1.0, visual_transform=spec)
    command = render._concat_render_command((segment,), tmp_path / "out.mp4", width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    filter_complex = command[command.index("-filter_complex") + 1]
    assert f"scale={spec.scaled_width}:{spec.scaled_height}" in filter_complex
    assert f"crop={spec.crop_width}:{spec.crop_height}:{spec.crop_x}:{spec.crop_y}" in filter_complex
    # the crop/pre-scale clause must appear BEFORE the existing fit-to-canvas scale.
    pre_scale_index = filter_complex.index(f"scale={spec.scaled_width}:{spec.scaled_height}")
    fit_scale_index = filter_complex.index(f"scale={_WIDTH}:{_HEIGHT}:force_original_aspect_ratio=decrease")
    assert pre_scale_index < fit_scale_index


def test_visual_transform_dimension_mismatch_fails_closed(source_clip, tmp_path):
    """A `VisualTransformSpec` whose OWN recorded `source_width`/
    `source_height` do not match the ACTUALLY PROBED source is never
    applied -- the renderer falls back to the unchanged fit-to-canvas
    path rather than cropping against the wrong geometry."""
    spec = VisualTransformSpec(
        action=VISUAL_ACTION_PUNCH_IN, source_width=9999, source_height=9999,
        scale_factor=1.10, scaled_width=10999, scaled_height=10999,
        crop_x=0, crop_y=0, crop_width=9999, crop_height=9999,
    )
    segment = RenderSegment(clip_id="c1", source_asset_id="a1", source_path=source_clip, start=0.0, end=1.0, visual_transform=spec)
    command = render._concat_render_command((segment,), tmp_path / "out.mp4", width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    filter_complex = command[command.index("-filter_complex") + 1]
    assert "crop=" not in filter_complex
    assert f"scale={_WIDTH}:{_HEIGHT}:force_original_aspect_ratio=decrease" in filter_complex


def test_visual_transform_command_still_executes_end_to_end(source_clip, tmp_path):
    """The generated command is not just syntactically plausible -- ffmpeg
    actually accepts and runs it, producing a playable output file."""
    probe = probe_media(source_clip)
    spec = VisualTransformSpec(
        action=VISUAL_ACTION_PUNCH_IN, source_width=probe.width, source_height=probe.height,
        scale_factor=1.10, scaled_width=round(probe.width * 1.10), scaled_height=round(probe.height * 1.10),
        crop_x=4, crop_y=3, crop_width=probe.width, crop_height=probe.height,
    )
    segment = RenderSegment(clip_id="c1", source_asset_id="a1", source_path=source_clip, start=0.0, end=1.0, visual_transform=spec)
    output = tmp_path / "out.mp4"
    command = render._concat_render_command((segment,), output, width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    subprocess.run(command, check=True, capture_output=True)
    assert output.exists() and output.stat().st_size > 0
    out_probe = probe_media(str(output))
    assert out_probe.width == _WIDTH and out_probe.height == _HEIGHT


def test_caption_present_still_ordered_after_transform(source_clip, tmp_path):
    """A caption filter, when present, must still apply AFTER the visual
    transform's own scale+crop (never before) -- the transform runs in
    SOURCE pixel space, captions in canvas space, matching D-261 Stage
    13's ordering."""
    spec = VisualTransformSpec(
        action=VISUAL_ACTION_PUNCH_IN, source_width=_WIDTH, source_height=_HEIGHT,
        scale_factor=1.10, scaled_width=round(_WIDTH * 1.10), scaled_height=round(_HEIGHT * 1.10),
        crop_x=8, crop_y=6, crop_width=_WIDTH, crop_height=_HEIGHT,
    )
    segment = RenderSegment(
        clip_id="c1", source_asset_id="a1", source_path=source_clip, start=0.0, end=1.0,
        visual_transform=spec, caption_text="hello",
    )
    command = render._concat_render_command((segment,), tmp_path / "out.mp4", width=_WIDTH, height=_HEIGHT, fps=_FPS, workdir=tmp_path)
    filter_complex = command[command.index("-filter_complex") + 1]
    crop_index = filter_complex.index("crop=")
    fit_scale_index = filter_complex.index(f"scale={_WIDTH}:{_HEIGHT}:force_original_aspect_ratio=decrease")
    assert crop_index < fit_scale_index


def test_render_plan_visual_transform_reference_is_type_checking_guarded():
    """`render_plan.py` must gain zero new hard runtime import -- the
    `VisualTransformSpec` reference stays `TYPE_CHECKING`-guarded, exactly
    D-214's own `audio_start`/`audio_end` discipline: the ONLY top-level
    statement referencing `visual_finishing_executor` is an
    `if TYPE_CHECKING:` block, never a bare `import`/`from` at module
    scope."""
    import cutsell_worker.render_plan as render_plan_module

    tree = ast.parse(inspect.getsource(render_plan_module))
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            names = ast.dump(stmt)
            assert "visual_finishing_executor" not in names, (
                "visual_finishing_executor must not be imported at module top level"
            )
        if isinstance(stmt, ast.If) and "visual_finishing_executor" in ast.dump(stmt):
            assert isinstance(stmt.test, ast.Name) and stmt.test.id == "TYPE_CHECKING"


def test_render_plan_imports_without_executor_module_preloaded():
    """`render_plan` must import cleanly on its own -- proves nothing at
    runtime actually needs `visual_finishing_executor` to be importable."""
    import importlib
    import sys

    sys.modules.pop("cutsell_worker.render_plan", None)
    sys.modules.pop("cutsell_worker.visual_finishing_executor", None)
    module = importlib.import_module("cutsell_worker.render_plan")
    assert "cutsell_worker.visual_finishing_executor" not in sys.modules
    assert module.RenderSegment(
        clip_id="c", source_asset_id="a", source_path="/x.mp4", start=0.0, end=1.0,
    ).visual_transform is None


def test_existing_renderer_test_suite_regressions_untouched():
    """D-262 must not change `_concat_render_command_with_audio_windows`
    (the D-214 test-only path) or `render_preview`'s own call signature."""
    source = inspect.getsource(render)
    assert "def _concat_render_command_with_audio_windows(" in source
    assert "def render_preview(" in source
