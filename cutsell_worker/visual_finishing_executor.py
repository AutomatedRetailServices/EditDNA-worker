"""Visual Finishing EXECUTOR FOUNDATION (D-262).

D-258 built real, factual VISUAL MEASUREMENT. D-260 built the POLICY +
PLAN layer against ten Product-Owner-approved V1 numeric values. D-261
designed -- but did not implement -- how a structured
`VisualFinishingPlan` becomes deterministic renderer geometry. This
module is the first gate that actually computes that geometry, on
synthetic media only -- no live pipeline integration.

    VISUAL MEASUREMENT (D-258) -> VISUAL POLICY + PLAN (D-260)
        -> VISUAL EXECUTOR (D-262, this module)
            -> RENDERER (render.py, unchanged authority, additive hook only)
                -> POST-RENDER VISUAL MEASUREMENT (D-258, re-used)

Policy decides. This module translates. The renderer executes -- it
never invents visual intent (D-261 Stage 15/D-249 §18.3's doctrine,
unchanged).

## Scope discipline (binding, D-262's own scope banner)

NO LIVE PIPELINE INTEGRATION. NO REAL USER MEDIA. NO EXPOSURE
CORRECTION. NO COLOR CORRECTION. NO GAZE LOGIC. NO SMART SALES FUNNEL.
NO NEW NUMERIC POLICY. NO THRESHOLD CHANGE. This module:

- never calls ffmpeg itself (geometry only -- pure arithmetic on
  already-real measurement/plan numbers);
- reads the ten canonical thresholds from `visual_finishing_policy.py`
  verbatim, never redeclaring or extending them;
- emits a `VisualTransformSpec` -- a symbolic, structured intent
  (pixel-space scale/crop geometry) -- never a raw ffmpeg filter
  string as canonical policy/execution state (`render.py` alone
  converts a spec into actual filter syntax, per D-261 Stage 13);
- fails closed by design: every blocked/abstained plan decision
  produces NO transform spec at all, with a specific, named
  `EXECUTION_STATUS_*` explaining why, never a silent no-op.

## Geometry model (this gate's own concrete design, grounded in D-260's
own established code, not invented from scratch)

All five non-`NO_CHANGE` actions share ONE pipeline: **scale the
source frame by a factor `S >= 1.0`, then crop a `source_width x
source_height` window back out of the scaled frame** (D-261 Stage 4/8's
own "scale then crop" ordering, applied uniformly). This works for
every action because:

- `PUNCH_IN`/`SCALE_MATCH`/`SCALE_AND_POSITION_MATCH` already carry an
  `authorized_scale >= 1.0` from D-260's own `_bound_scale` (never a
  scale-down -- D-261 Stage 7 confirmed V1 does not support scale-down).
- `STATIC_REFRAME`/`POSITION_MATCH` carry no scale at all, but a PURE
  translation, done via cropping, is mechanically inseparable from a
  small compensating zoom-back-up (cropping a same-size window off-
  center and feeding it through the renderer's own existing DOWNSCALE-
  ONLY `force_original_aspect_ratio=decrease` fit-to-canvas step would
  shrink the visible image, never restore it) -- so this module reserves
  **exactly the requested translation's own magnitude** as the minimal
  necessary zoom (`scale = 1 / (1 - crop_loss)`), reusing no new number
  (see `compute_crop_loss` below).

## The crop-loss / translation unit-semantics clarification (grounded
in D-260's own code, not this gate's invention)

D-260's own `_bound_translation` already treats `MAX_REFRAME_
TRANSLATION_NORMALIZED` and `MAX_ADDITIONAL_CROP_LOSS_NORMALIZED`
(both `0.10`) as literally the SAME clamped quantity. This module
therefore defines "crop loss" as the TRANSLATION-INDUCED component of
the crop window ONLY (`max(|tx|, |ty|)`) -- **never** as a function of
the separately-authorized scale factor, which has its own independent
ceiling (`MAX_PUNCH_IN_SCALE`). A pure `PUNCH_IN`/`SCALE_MATCH` (no
translation) therefore has `crop_loss == 0.0` by this definition; its
own zoom magnitude is bounded exclusively by the scale ceiling, exactly
as D-260's policy layer already enforces.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .visual_finishing_measurement import VisualClipMeasurement
from .visual_finishing_policy import (
    MAX_ADDITIONAL_CROP_LOSS_NORMALIZED,
    POLICY_VERSION,
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

# ---------------------------------------------------------------------------
# STAGE 3: failure/execution-status vocabulary.
# ---------------------------------------------------------------------------

EXECUTION_STATUS_SUCCESS = "SUCCESS"
EXECUTION_STATUS_NO_ACTION_NEEDED = "NO_ACTION_NEEDED"
EXECUTION_STATUS_PLAN_NOT_EXECUTABLE = "PLAN_NOT_EXECUTABLE"
EXECUTION_STATUS_FACE_SAFETY_BLOCKED = "FACE_SAFETY_BLOCKED"
EXECUTION_STATUS_PRODUCT_SAFETY_BLOCKED = "PRODUCT_SAFETY_BLOCKED"
EXECUTION_STATUS_CROP_LIMIT_EXCEEDED = "CROP_LIMIT_EXCEEDED"
EXECUTION_STATUS_INVALID_GEOMETRY = "INVALID_GEOMETRY"
EXECUTION_STATUS_SOURCE_DIMENSIONS_UNAVAILABLE = "SOURCE_DIMENSIONS_UNAVAILABLE"
EXECUTION_STATUS_RENDER_FAILURE = "RENDER_FAILURE"
EXECUTION_STATUS_POST_VERIFY_FAILED = "POST_VERIFY_FAILED"
EXECUTION_STATUS_ALREADY_APPLIED = "ALREADY_APPLIED"
EXECUTION_STATUS_OTHER = "OTHER"

VERIFICATION_STATUS_PASS = "PASS"
VERIFICATION_STATUS_FAIL = "FAIL"
VERIFICATION_STATUS_PARTIAL = "PARTIAL"
VERIFICATION_STATUS_UNVERIFIABLE = "UNVERIFIABLE"

_ACTION_TO_BLOCKED_STATUS = {
    VISUAL_ACTION_BLOCKED_FACE_SAFETY: EXECUTION_STATUS_FACE_SAFETY_BLOCKED,
    VISUAL_ACTION_BLOCKED_MULTI_FACE: EXECUTION_STATUS_PLAN_NOT_EXECUTABLE,
    VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN: EXECUTION_STATUS_PRODUCT_SAFETY_BLOCKED,
    VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE: EXECUTION_STATUS_PLAN_NOT_EXECUTABLE,
    VISUAL_ACTION_UNKNOWN: EXECUTION_STATUS_PLAN_NOT_EXECUTABLE,
}
_CORRECTION_ACTIONS = frozenset({
    VISUAL_ACTION_STATIC_REFRAME, VISUAL_ACTION_PUNCH_IN, VISUAL_ACTION_SCALE_MATCH,
    VISUAL_ACTION_POSITION_MATCH, VISUAL_ACTION_SCALE_AND_POSITION_MATCH,
})
_SCALE_ONLY_ACTIONS = frozenset({VISUAL_ACTION_PUNCH_IN, VISUAL_ACTION_SCALE_MATCH})


# ---------------------------------------------------------------------------
# STAGE 2/13: execution types.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VisualTransformSpec:
    """Symbolic, structured render intent -- pixel-space geometry only,
    never a raw ffmpeg filter string. `render.py` alone converts this
    into actual filter syntax (D-261 Stage 13)."""

    action: str
    source_width: int
    source_height: int
    scale_factor: float
    scaled_width: int
    scaled_height: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int


@dataclass(frozen=True)
class VisualFinishingExecutionRecord:
    execution_id: str
    plan_id: str
    clip_id: str | None
    action: str

    input_width: int | None
    input_height: int | None

    authorized_scale: float | None
    authorized_translation_x: float | None
    authorized_translation_y: float | None

    crop_x: int | None
    crop_y: int | None
    crop_width: int | None
    crop_height: int | None
    crop_loss: float | None

    face_safety_status: str
    product_safety_status: str
    crop_safety_status: str

    renderer_operation: str | None
    transform_spec: VisualTransformSpec | None

    execution_status: str
    errors: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VisualFinishingVerificationResult:
    measurement_status: str
    face_detected: bool | None
    face_bbox: tuple[float, float, float, float] | None
    face_center_x: float | None
    face_center_y: float | None
    face_area_ratio: float | None
    headroom_ratio: float | None
    face_contained: bool | None
    orientation: str | None
    width: int | None
    height: int | None
    duration_preserved: bool | None
    action_direction_verified: bool | None
    verification_status: str
    errors: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure geometry helpers.
# ---------------------------------------------------------------------------

def compute_crop_loss(
    authorized_translation_x: float | None, authorized_translation_y: float | None,
) -> float:
    """STAGE 9: the TRANSLATION-induced crop-window reduction only --
    never a function of any separately-authorized scale factor (see
    module docstring's unit-semantics note). Always `>= 0.0`."""
    tx = abs(authorized_translation_x) if authorized_translation_x is not None else 0.0
    ty = abs(authorized_translation_y) if authorized_translation_y is not None else 0.0
    return max(tx, ty)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def compute_transform_geometry(
    action: str,
    *,
    source_width: int,
    source_height: int,
    authorized_scale: float | None,
    authorized_translation_x: float | None,
    authorized_translation_y: float | None,
    focal_x: float = 0.5,
    focal_y: float = 0.5,
) -> tuple[VisualTransformSpec | None, float, str | None]:
    """STAGE 4-8: pure geometry computation for the five correction
    actions. Returns `(spec, crop_loss, error)` -- `spec` is `None` and
    `error` is set for `INVALID_GEOMETRY` (non-positive source
    dimensions, or a `NaN`/degenerate scale). `focal_x`/`focal_y`
    (normalized `[0,1]`, default frame-center) let a caller center a
    `PUNCH_IN`'s crop on the clip's own already-measured face center
    (a REAL, already-face-safety-cleared value) instead of blind
    geometric center -- STAGE 4's own "3. center crop around plan-
    authorized focal/target position" requirement."""
    if source_width <= 0 or source_height <= 0:
        return None, 0.0, "invalid_source_dimensions"

    crop_loss = compute_crop_loss(authorized_translation_x, authorized_translation_y)
    tx = authorized_translation_x or 0.0
    ty = authorized_translation_y or 0.0

    if action in _SCALE_ONLY_ACTIONS:
        base_scale = authorized_scale if authorized_scale is not None else 1.0
    elif action == VISUAL_ACTION_SCALE_AND_POSITION_MATCH:
        base_scale = authorized_scale if authorized_scale is not None else 1.0
    else:  # STATIC_REFRAME / POSITION_MATCH: no explicit scale authorized.
        base_scale = 1.0

    if crop_loss >= 1.0:
        return None, crop_loss, "translation_magnitude_invalid"
    translate_zoom = 1.0 / (1.0 - crop_loss)
    scale_factor = max(1.0, base_scale, translate_zoom)
    if not (scale_factor == scale_factor) or scale_factor <= 0:  # NaN/degenerate guard
        return None, crop_loss, "invalid_scale_factor"

    scaled_width = max(1, round(source_width * scale_factor))
    scaled_height = max(1, round(source_height * scale_factor))
    crop_width = min(source_width, scaled_width)
    crop_height = min(source_height, scaled_height)

    # Focal offset from geometric center, in scaled-frame pixels.
    focal_dx_px = (focal_x - 0.5) * source_width * scale_factor if action in _SCALE_ONLY_ACTIONS or action == VISUAL_ACTION_SCALE_AND_POSITION_MATCH else 0.0
    focal_dy_px = (focal_y - 0.5) * source_height * scale_factor if action in _SCALE_ONLY_ACTIONS or action == VISUAL_ACTION_SCALE_AND_POSITION_MATCH else 0.0

    center_x = scaled_width / 2.0 + tx * source_width * scale_factor + focal_dx_px
    center_y = scaled_height / 2.0 + ty * source_height * scale_factor + focal_dy_px
    crop_x = _clamp(round(center_x - crop_width / 2.0), 0, scaled_width - crop_width)
    crop_y = _clamp(round(center_y - crop_height / 2.0), 0, scaled_height - crop_height)

    spec = VisualTransformSpec(
        action=action, source_width=source_width, source_height=source_height,
        scale_factor=scale_factor, scaled_width=scaled_width, scaled_height=scaled_height,
        crop_x=crop_x, crop_y=crop_y, crop_width=crop_width, crop_height=crop_height,
    )
    return spec, crop_loss, None


def compute_transformed_face_bbox(
    spec: VisualTransformSpec,
    face_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """STAGE 10: transform a SOURCE-normalized `(x_min, y_min, x_max,
    y_max)` bbox through the exact same scale+crop geometry the
    renderer is about to apply, returning the bbox in the OUTPUT crop's
    own normalized `[0,1]` space. Pure arithmetic; no detector re-run."""
    x_min, y_min, x_max, y_max = face_bbox

    def _to_output(x: float, y: float) -> tuple[float, float]:
        # Uses the SAME integer scaled_width/scaled_height the crop window
        # was cut from (not a re-derived `source * scale_factor` float) --
        # consistency with the actual pixel geometry the renderer applies.
        scaled_x = x * spec.scaled_width
        scaled_y = y * spec.scaled_height
        out_x = (scaled_x - spec.crop_x) / spec.crop_width
        out_y = (scaled_y - spec.crop_y) / spec.crop_height
        return out_x, out_y

    out_x_min, out_y_min = _to_output(x_min, y_min)
    out_x_max, out_y_max = _to_output(x_max, y_max)
    return out_x_min, out_y_min, out_x_max, out_y_max


def is_face_contained(transformed_bbox: tuple[float, float, float, float]) -> bool:
    """STAGE 10: literal `[0,1]` containment -- no invented safety margin."""
    x_min, y_min, x_max, y_max = transformed_bbox
    return x_min >= 0.0 and y_min >= 0.0 and x_max <= 1.0 and y_max <= 1.0


# ---------------------------------------------------------------------------
# STAGE 18: deterministic execution identity.
# ---------------------------------------------------------------------------

def compute_visual_execution_id(
    source_identity: str | None,
    plan_id: str,
    clip_id: str | None,
    action: str,
    transform_spec: VisualTransformSpec | None,
) -> str:
    """A pure, deterministic function of the execution's own already-
    decided content -- no mutable global state, no filename dependency
    beyond the caller-supplied CONTENT identity (never a path). Mirrors
    `compute_execution_id`/`compute_finishing_identity`/`compute_
    visual_finishing_plan_identity`'s own established pattern exactly."""
    payload = {
        "policy_version": POLICY_VERSION,
        "source_identity": source_identity,
        "plan_id": plan_id,
        "clip_id": clip_id,
        "action": action,
        "transform_spec": None if transform_spec is None else {
            "scale_factor": transform_spec.scale_factor,
            "crop_x": transform_spec.crop_x, "crop_y": transform_spec.crop_y,
            "crop_width": transform_spec.crop_width, "crop_height": transform_spec.crop_height,
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


# ---------------------------------------------------------------------------
# STAGE 1/7/10/11/12: the top-level executor entry point.
# ---------------------------------------------------------------------------

def execute_visual_finishing_decision(
    decision: VisualJoinPolicyDecision,
    clip_measurement: VisualClipMeasurement,
    *,
    plan_id: str,
    source_identity: str | None = None,
    face_bbox: tuple[float, float, float, float] | None = None,
    previous_execution_id: str | None = None,
) -> VisualFinishingExecutionRecord:
    """The single entry point. `clip_measurement` describes the ONE
    clip this call is transforming (the caller -- not this gate's own
    scope, per "NO LIVE PIPELINE INTEGRATION" -- decides which side of
    `decision` that clip is). `face_bbox`, when supplied, is a REAL
    `(x_min, y_min, x_max, y_max)` normalized bbox from that clip's own
    measurement (e.g. a representative sampled frame); when omitted,
    face-safety is reported `UNVERIFIABLE`-equivalent
    (`crop_safety_status`/`face_safety_status` reflect this honestly,
    never assumed safe). Never re-evaluates D-260's own policy
    decision -- only translates an already-authorized action into
    geometry, or declines to, per this module's own execution-time
    re-verification (D-261 Stage 9/10's "policy intent alone is not
    enough")."""
    action = decision.action
    clip_id = clip_measurement.clip_id
    width, height = clip_measurement.frame_width, clip_measurement.frame_height

    def _record(
        *, status: str, spec: VisualTransformSpec | None = None,
        crop_loss: float | None = None, face_status: str = "UNVERIFIABLE",
        product_status: str = "UNVERIFIABLE", crop_status: str = "UNVERIFIABLE",
        errors: tuple[str, ...] = (),
    ) -> VisualFinishingExecutionRecord:
        execution_id = compute_visual_execution_id(source_identity, plan_id, clip_id, action, spec)
        already = previous_execution_id is not None and previous_execution_id == execution_id
        final_status = EXECUTION_STATUS_ALREADY_APPLIED if already and status == EXECUTION_STATUS_SUCCESS else status
        return VisualFinishingExecutionRecord(
            execution_id=execution_id, plan_id=plan_id, clip_id=clip_id, action=action,
            input_width=width, input_height=height,
            authorized_scale=decision.authorized_scale,
            authorized_translation_x=decision.authorized_translation_x,
            authorized_translation_y=decision.authorized_translation_y,
            crop_x=None if spec is None else spec.crop_x,
            crop_y=None if spec is None else spec.crop_y,
            crop_width=None if spec is None else spec.crop_width,
            crop_height=None if spec is None else spec.crop_height,
            crop_loss=crop_loss,
            face_safety_status=face_status, product_safety_status=product_status,
            crop_safety_status=crop_status,
            renderer_operation=None if spec is None else "SCALE_THEN_CROP",
            transform_spec=spec, execution_status=final_status, errors=errors,
        )

    # STAGE 12: multi-face / abstain -- executor does nothing, no selection.
    if action == VISUAL_ACTION_BLOCKED_MULTI_FACE:
        return _record(status=EXECUTION_STATUS_PLAN_NOT_EXECUTABLE, product_status="N/A", crop_status="N/A", face_status="N/A")
    if action == VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE:
        return _record(status=EXECUTION_STATUS_PLAN_NOT_EXECUTABLE, product_status="N/A", crop_status="N/A", face_status="N/A")
    if action == VISUAL_ACTION_UNKNOWN:
        return _record(status=EXECUTION_STATUS_PLAN_NOT_EXECUTABLE, product_status="N/A", crop_status="N/A", face_status="N/A")

    # STAGE 11: product safety -- pure pass-through, no executor override.
    if action == VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN:
        return _record(status=EXECUTION_STATUS_PRODUCT_SAFETY_BLOCKED, product_status="BLOCKED", crop_status="N/A", face_status="N/A")

    if action == VISUAL_ACTION_BLOCKED_FACE_SAFETY:
        return _record(status=EXECUTION_STATUS_FACE_SAFETY_BLOCKED, face_status="BLOCKED", product_status="N/A", crop_status="N/A")

    if action == VISUAL_ACTION_NO_CHANGE:
        return _record(status=EXECUTION_STATUS_NO_ACTION_NEEDED, face_status="N/A", product_status="N/A", crop_status="N/A")

    if action not in _CORRECTION_ACTIONS:
        return _record(status=EXECUTION_STATUS_OTHER, errors=(f"unrecognized_action:{action}",))

    if width is None or height is None or width <= 0 or height <= 0:
        return _record(status=EXECUTION_STATUS_SOURCE_DIMENSIONS_UNAVAILABLE, face_status="N/A", product_status="N/A", crop_status="N/A")

    focal_x = clip_measurement.face_center_x_median if clip_measurement.face_center_x_median is not None else 0.5
    focal_y = clip_measurement.face_center_y_median if clip_measurement.face_center_y_median is not None else 0.5

    spec, crop_loss, geometry_error = compute_transform_geometry(
        action, source_width=width, source_height=height,
        authorized_scale=decision.authorized_scale,
        authorized_translation_x=decision.authorized_translation_x,
        authorized_translation_y=decision.authorized_translation_y,
        focal_x=focal_x, focal_y=focal_y,
    )
    if geometry_error is not None or spec is None:
        return _record(status=EXECUTION_STATUS_INVALID_GEOMETRY, crop_loss=crop_loss,
                        errors=(geometry_error or "invalid_geometry",),
                        face_status="N/A", product_status="N/A", crop_status="N/A")

    # STAGE 9: execution-time crop-loss re-verification -- independent
    # of whatever the plan already authorized.
    if crop_loss > MAX_ADDITIONAL_CROP_LOSS_NORMALIZED:
        return _record(status=EXECUTION_STATUS_CROP_LIMIT_EXCEEDED, spec=spec, crop_loss=crop_loss,
                        crop_status="EXCEEDED", face_status="N/A", product_status="PASS")

    # STAGE 10: execution-time face-safety re-verification.
    face_status = "UNVERIFIABLE"
    if face_bbox is not None:
        transformed = compute_transformed_face_bbox(spec, face_bbox)
        if is_face_contained(transformed):
            face_status = "PASS"
        else:
            return _record(status=EXECUTION_STATUS_FACE_SAFETY_BLOCKED, spec=spec, crop_loss=crop_loss,
                            crop_status="WITHIN_LIMIT", face_status="FAIL", product_status="PASS")

    return _record(
        status=EXECUTION_STATUS_SUCCESS, spec=spec, crop_loss=crop_loss,
        face_status=face_status, product_status="PASS", crop_status="WITHIN_LIMIT",
    )
