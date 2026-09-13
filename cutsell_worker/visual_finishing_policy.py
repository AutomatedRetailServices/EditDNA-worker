"""Visual Finishing POLICY CONTRACT + PLAN GENERATION (D-260).

D-258 built real, factual VISUAL MEASUREMENT
(`cutsell_worker/visual_finishing_measurement.py`). D-259 designed --
but did not implement -- the POLICY + PLAN layer on top of it. This
module is the first gate that actually encodes the Product-Owner-
approved V1 numeric policy (D-260's own directive) and generates a
structured, non-executing `VisualFinishingPlan`.

    VISUAL MEASUREMENT (D-258)
        -> VISUAL POLICY + PLAN (D-260, this module)
            -> FUTURE RENDERER EXECUTION (MISSING-FUTURE, D-261+)
                -> FUTURE POST-RENDER VISUAL QC (MISSING-FUTURE)

## Scope discipline (binding, D-260's own scope banner)

NO CROP EXECUTION. NO REFRAME EXECUTION. NO PUNCH-IN EXECUTION. NO
VIDEO MUTATION. NO EXPOSURE CORRECTION. NO COLOR CORRECTION. NO RENDER
CHANGE. NO PACING/BOUNDARY/FREEZE/AUDIO-FINISHING CHANGE. NO NEW
NUMERIC POLICY beyond the ten Product-Owner-approved values below. This
module:

- never calls ffmpeg, never opens a frame, never touches `render.py`;
- consumes only already-computed `VisualClipMeasurement`/
  `VisualJoinMeasurement` objects (D-258, unchanged, imported only for
  type references);
- emits SYMBOLIC, structured intent only (a requested/authorized
  translation and scale, both floats) -- never a raw ffmpeg filter
  string, matching `AudioFinishingPlan.authorized_whole_video_gain_db`'s
  own "authorize a number, never a filter string" discipline exactly;
- fails closed by design: `NO_CHANGE` is the default outcome; every
  other action requires evidence to cross one of the ten canonical
  thresholds below, and any face/multi-face/product-safety concern
  blocks a correction outright, regardless of how strong the
  continuity evidence is (Stage 25's priority ladder).

## The ten Product-Owner-approved V1 numeric values (D-260)

```
FACE_CENTER_X_DISCONTINUITY_THRESHOLD          = 0.025
FACE_CENTER_Y_DISCONTINUITY_THRESHOLD          = 0.025
FACE_SCALE_RELATIVE_AREA_DISCONTINUITY_THRESHOLD = 0.30
HEADROOM_DISCONTINUITY_THRESHOLD               = 0.06
DEFAULT_PUNCH_IN_SCALE                          = 1.10
MAX_PUNCH_IN_SCALE                              = 1.15
MAX_REFRAME_TRANSLATION_NORMALIZED              = 0.10
MAX_ADDITIONAL_CROP_LOSS_NORMALIZED             = 0.10
MIN_RELIABLE_FACE_DETECTION_RATE                = 0.75
MIN_PUNCH_IN_CLIP_DURATION_SEC                  = 1.5
```

**These are V1 PRODUCT POLICY for professional Talking Head UGC
editing -- explicitly NOT universal cinematography truths.** No
additional numeric threshold is introduced anywhere in this module.

## A load-bearing unit-semantics note (D-259 Stage 14's own warning,
confirmed real during this gate's implementation)

D-258's `VisualJoinMeasurement.face_area_ratio_delta` is an ABSOLUTE
difference between two clips' median `face_area_ratio` values (both
already tiny fractions of total frame area, e.g. 0.04). The Product-
Owner-approved `FACE_SCALE_RELATIVE_AREA_DISCONTINUITY_THRESHOLD = 0.30`
is a **relative** (30%) change, not an absolute area-ratio delta of
0.30 (which would almost never occur and would silently make the scale
check inert). This module therefore computes the RELATIVE scale change
itself, from the two clips' own `face_area_ratio_median` values
(`abs(right - left) / left`), never reusing D-258's absolute delta
field directly for this comparison -- exactly the "do not silently
reinterpret delta units" trap D-259 warned about.

## Product safety: the honest consequence of D-258's own contract

D-257/D-258 established that no product-bbox detector exists; D-258's
`VisualClipMeasurement.product_bbox_status` is therefore always
`"UNAVAILABLE"`. D-259 Stage 8 recommended fail-closed option B. This
module implements that literally: `product_safety_established` defaults
to `False` for every clip, and while `False`, EVERY correction-
authorizing action (`STATIC_REFRAME`/`PUNCH_IN`/`SCALE_MATCH`/
`POSITION_MATCH`/`SCALE_AND_POSITION_MATCH`) resolves to
`BLOCKED_PRODUCT_SAFETY_UNKNOWN` -- **which is the correct, safe, and
expected outcome for every real plan generated against today's actual
measurement inputs.** The parameter exists so a future gate, once real
product-bbox evidence exists, can flip it per-clip without touching this
module's contract again; this gate's own test suite exercises the
correction-authorizing paths by explicitly passing
`product_safety_established=True`, never by weakening the default.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .visual_finishing_measurement import (
    MEASUREMENT_STATUS_DECODE_ERROR,
    MEASUREMENT_STATUS_MEASUREMENT_ERROR,
    MEASUREMENT_STATUS_NO_FACE,
    MEASUREMENT_STATUS_UNAVAILABLE,
    CAPTION_SAFE_NOT_ESTABLISHED,
    VisualClipMeasurement,
    VisualJoinMeasurement,
)

POLICY_VERSION = "V1"

# ---------------------------------------------------------------------------
# STAGE 2 (D-260): the ten Product-Owner-approved canonical V1 values.
# ---------------------------------------------------------------------------

FACE_CENTER_X_DISCONTINUITY_THRESHOLD = 0.025
FACE_CENTER_Y_DISCONTINUITY_THRESHOLD = 0.025
FACE_SCALE_RELATIVE_AREA_DISCONTINUITY_THRESHOLD = 0.30
HEADROOM_DISCONTINUITY_THRESHOLD = 0.06
DEFAULT_PUNCH_IN_SCALE = 1.10
MAX_PUNCH_IN_SCALE = 1.15
MAX_REFRAME_TRANSLATION_NORMALIZED = 0.10
MAX_ADDITIONAL_CROP_LOSS_NORMALIZED = 0.10
MIN_RELIABLE_FACE_DETECTION_RATE = 0.75
MIN_PUNCH_IN_CLIP_DURATION_SEC = 1.5

# Not a policy threshold: a fixed, negligible floating-point-representation
# guard (orders of magnitude below any real measurement's own precision)
# so an input that is exactly AT one of the ten approved values above is
# never pushed across the boundary by IEEE-754 representation noise (e.g.
# `0.5 + 0.025 - 0.5` is not bit-exact `0.025`). This never changes which
# side of a threshold a real measurement falls on -- it only protects the
# literal exact-equality case the Product Owner's own thresholds define.
_THRESHOLD_EPSILON = 1e-9


def canonical_thresholds_snapshot() -> dict:
    """A plain-dict snapshot of the ten approved values, embedded verbatim
    into every generated plan's `provenance` -- so a plan is always
    self-describing about which numbers produced it, independent of
    whatever this module's own constants later become."""
    return {
        "FACE_CENTER_X_DISCONTINUITY_THRESHOLD": FACE_CENTER_X_DISCONTINUITY_THRESHOLD,
        "FACE_CENTER_Y_DISCONTINUITY_THRESHOLD": FACE_CENTER_Y_DISCONTINUITY_THRESHOLD,
        "FACE_SCALE_RELATIVE_AREA_DISCONTINUITY_THRESHOLD": FACE_SCALE_RELATIVE_AREA_DISCONTINUITY_THRESHOLD,
        "HEADROOM_DISCONTINUITY_THRESHOLD": HEADROOM_DISCONTINUITY_THRESHOLD,
        "DEFAULT_PUNCH_IN_SCALE": DEFAULT_PUNCH_IN_SCALE,
        "MAX_PUNCH_IN_SCALE": MAX_PUNCH_IN_SCALE,
        "MAX_REFRAME_TRANSLATION_NORMALIZED": MAX_REFRAME_TRANSLATION_NORMALIZED,
        "MAX_ADDITIONAL_CROP_LOSS_NORMALIZED": MAX_ADDITIONAL_CROP_LOSS_NORMALIZED,
        "MIN_RELIABLE_FACE_DETECTION_RATE": MIN_RELIABLE_FACE_DETECTION_RATE,
        "MIN_PUNCH_IN_CLIP_DURATION_SEC": MIN_PUNCH_IN_CLIP_DURATION_SEC,
    }


# ---------------------------------------------------------------------------
# STAGE 3: action vocabulary.
# ---------------------------------------------------------------------------

VISUAL_ACTION_NO_CHANGE = "NO_CHANGE"
VISUAL_ACTION_STATIC_REFRAME = "STATIC_REFRAME"
VISUAL_ACTION_PUNCH_IN = "PUNCH_IN"
VISUAL_ACTION_SCALE_MATCH = "SCALE_MATCH"
VISUAL_ACTION_POSITION_MATCH = "POSITION_MATCH"
VISUAL_ACTION_SCALE_AND_POSITION_MATCH = "SCALE_AND_POSITION_MATCH"
VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE = "ABSTAIN_INSUFFICIENT_EVIDENCE"
VISUAL_ACTION_BLOCKED_FACE_SAFETY = "BLOCKED_FACE_SAFETY"
VISUAL_ACTION_BLOCKED_MULTI_FACE = "BLOCKED_MULTI_FACE"
VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN = "BLOCKED_PRODUCT_SAFETY_UNKNOWN"
VISUAL_ACTION_UNKNOWN = "UNKNOWN"

_BLOCKED_ACTIONS = frozenset({
    VISUAL_ACTION_BLOCKED_FACE_SAFETY,
    VISUAL_ACTION_BLOCKED_MULTI_FACE,
    VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN,
})
_CORRECTION_ACTIONS = frozenset({
    VISUAL_ACTION_STATIC_REFRAME,
    VISUAL_ACTION_PUNCH_IN,
    VISUAL_ACTION_SCALE_MATCH,
    VISUAL_ACTION_POSITION_MATCH,
    VISUAL_ACTION_SCALE_AND_POSITION_MATCH,
})

# ---------------------------------------------------------------------------
# STAGE 4: plan-status vocabulary.
# ---------------------------------------------------------------------------

PLAN_STATUS_READY_NO_CHANGE = "READY_NO_CHANGE"
PLAN_STATUS_READY_FOR_VISUAL_CORRECTION = "READY_FOR_VISUAL_CORRECTION"
PLAN_STATUS_PARTIAL = "PARTIAL"
PLAN_STATUS_ABSTAIN = "ABSTAIN"
PLAN_STATUS_BLOCKED = "BLOCKED"
PLAN_STATUS_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Evidence-state vocabulary.
# ---------------------------------------------------------------------------

EVIDENCE_STATE_SUFFICIENT = "SUFFICIENT"
EVIDENCE_STATE_INSUFFICIENT = "INSUFFICIENT"

# ---------------------------------------------------------------------------
# STAGE 30 (D-259) / Stage 30 (D-260): closed reasons/warnings/safety-
# blocker vocabulary. This is the Smart Sales Funnel firewall's actual
# mechanism -- none of these codes, nor any future one added beside
# them, may express a commercial concept. A structural test enforces
# this by scanning the module's own source for the literal forbidden
# words themselves.
# ---------------------------------------------------------------------------

REASON_NO_DISCONTINUITY_EVIDENCE = "NO_DISCONTINUITY_EVIDENCE"
REASON_FACE_POSITION_DISCONTINUITY = "FACE_POSITION_DISCONTINUITY"
REASON_FACE_SCALE_DISCONTINUITY = "FACE_SCALE_DISCONTINUITY"
REASON_HEADROOM_DISCONTINUITY = "HEADROOM_DISCONTINUITY"
REASON_JUMP_CUT_CONCEALMENT_CANDIDATE = "JUMP_CUT_CONCEALMENT_CANDIDATE"
REASON_INSUFFICIENT_FACE_EVIDENCE = "INSUFFICIENT_FACE_EVIDENCE"
REASON_FACE_ALREADY_CLIPPED = "FACE_ALREADY_CLIPPED"
REASON_TRANSLATION_EXCEEDS_MAX = "TRANSLATION_EXCEEDS_MAX"
REASON_CROP_LOSS_EXCEEDS_MAX = "CROP_LOSS_EXCEEDS_MAX"
REASON_PUNCH_IN_CLIP_TOO_SHORT = "PUNCH_IN_CLIP_TOO_SHORT"
REASON_ORIENTATION_UNSUPPORTED_FOR_CORRECTION = "ORIENTATION_UNSUPPORTED_FOR_CORRECTION"

SAFETY_BLOCKER_MULTI_FACE_PRESENT = "MULTI_FACE_PRESENT"
SAFETY_BLOCKER_FACE_ALREADY_CLIPPED = "FACE_ALREADY_CLIPPED"
SAFETY_BLOCKER_PRODUCT_SAFETY_UNKNOWN = "PRODUCT_SAFETY_UNKNOWN"

WARNING_CAPTION_SAFE_REGION_NOT_ESTABLISHED = "CAPTION_SAFE_REGION_NOT_ESTABLISHED"
WARNING_LUMA_DIFFERENCE_DIAGNOSTIC_ONLY = "LUMA_DIFFERENCE_DIAGNOSTIC_ONLY"
WARNING_COLOR_DIFFERENCE_DIAGNOSTIC_ONLY = "COLOR_DIFFERENCE_DIAGNOSTIC_ONLY"


# ---------------------------------------------------------------------------
# STAGE 5/6/7: decision + plan types.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VisualClipPolicyDecision:
    clip_id: str | None
    action: str
    evidence_state: str

    face_detection_rate: float | None
    face_bbox: tuple[float, float, float, float] | None
    face_center: tuple[float, float] | None
    face_area_ratio: float | None
    headroom_ratio: float | None

    reframe_authorized: bool
    punch_in_authorized: bool

    requested_translation_x: float | None
    requested_translation_y: float | None
    authorized_translation_x: float | None
    authorized_translation_y: float | None
    requested_scale: float | None
    authorized_scale: float | None

    safety_blockers: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VisualJoinPolicyDecision:
    left_clip_id: str | None
    right_clip_id: str | None
    action: str
    evidence_state: str

    face_center_dx: float | None
    face_center_dy: float | None
    face_scale_delta: float | None  # RELATIVE change, see module docstring's unit-semantics note
    headroom_delta: float | None
    thresholds_used: dict

    position_match_authorized: bool
    scale_match_authorized: bool
    punch_in_authorized: bool

    requested_translation_x: float | None
    requested_translation_y: float | None
    authorized_translation_x: float | None
    authorized_translation_y: float | None
    requested_scale: float | None
    authorized_scale: float | None
    target_position_x: float | None
    target_position_y: float | None
    target_scale: float | None

    safety_blockers: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VisualFinishingPlan:
    policy_version: str
    source_id: str | None

    clip_measurement_references: tuple[VisualClipMeasurement, ...]
    join_measurement_references: tuple[VisualJoinMeasurement, ...]
    clip_decisions: tuple[VisualClipPolicyDecision, ...]
    join_decisions: tuple[VisualJoinPolicyDecision, ...]

    plan_status: str
    canonical_thresholds_snapshot: dict

    abstentions: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict = field(default_factory=dict)

    visual_finishing_identity: str = ""


# ---------------------------------------------------------------------------
# Clip-level policy.
# ---------------------------------------------------------------------------

def _representative_face_bbox(measurement: VisualClipMeasurement) -> tuple[float, float, float, float] | None:
    """A REAL bbox from an actual sampled frame -- never a synthesized/
    averaged bbox. Picks the face-valid frame whose own `face_area_
    ratio` is closest to the clip's median, so the reported bbox is
    representative without being fabricated."""
    if measurement.face_area_ratio_median is None:
        return None
    candidates = [f for f in measurement.frames if f.face_detected and f.face_area_ratio is not None]
    if not candidates:
        return None
    target = measurement.face_area_ratio_median
    best = min(candidates, key=lambda f: abs(f.face_area_ratio - target))
    return (best.face_bbox_x_min, best.face_bbox_y_min, best.face_bbox_x_max, best.face_bbox_y_max)


def _clip_has_multi_face_evidence(measurement: VisualClipMeasurement) -> bool:
    return any(
        f.multiple_faces_detected
        for f in measurement.frames
        if f.measurement_status != MEASUREMENT_STATUS_DECODE_ERROR
    )


def _clip_has_clipped_face_evidence(measurement: VisualClipMeasurement) -> bool:
    return any(
        f.face_detected and (
            f.face_bbox_clipped_left or f.face_bbox_clipped_right
            or f.face_bbox_clipped_top or f.face_bbox_clipped_bottom
        )
        for f in measurement.frames
    )


def evaluate_clip_policy(
    measurement: VisualClipMeasurement,
    *,
    product_safety_established: bool = False,
) -> VisualClipPolicyDecision:
    """STAGE 8/9/10/11: a clip's own intrinsic eligibility for ANY future
    join-triggered correction. Never decides WHICH correction -- that is
    `evaluate_join_policy`'s job, since every correction action in the
    V1 vocabulary is inherently a comparison against a neighbor."""
    warnings: list[str] = []
    if measurement.caption_safe_status == CAPTION_SAFE_NOT_ESTABLISHED:
        warnings.append(WARNING_CAPTION_SAFE_REGION_NOT_ESTABLISHED)

    face_bbox = _representative_face_bbox(measurement)
    face_center = (
        (measurement.face_center_x_median, measurement.face_center_y_median)
        if measurement.face_center_x_median is not None and measurement.face_center_y_median is not None
        else None
    )

    def _decision(action: str, evidence_state: str, *, reasons: tuple[str, ...],
                  safety_blockers: tuple[str, ...] = (), reframe_authorized: bool = False,
                  punch_in_authorized: bool = False) -> VisualClipPolicyDecision:
        return VisualClipPolicyDecision(
            clip_id=measurement.clip_id, action=action, evidence_state=evidence_state,
            face_detection_rate=measurement.face_detection_rate, face_bbox=face_bbox,
            face_center=face_center, face_area_ratio=measurement.face_area_ratio_median,
            headroom_ratio=measurement.headroom_ratio_median,
            reframe_authorized=reframe_authorized, punch_in_authorized=punch_in_authorized,
            requested_translation_x=None, requested_translation_y=None,
            authorized_translation_x=None, authorized_translation_y=None,
            requested_scale=None, authorized_scale=None,
            safety_blockers=safety_blockers, warnings=tuple(warnings), reasons=reasons,
        )

    # STAGE 8: insufficient/unavailable measurement evidence.
    if measurement.measurement_status in (
        MEASUREMENT_STATUS_UNAVAILABLE, MEASUREMENT_STATUS_DECODE_ERROR, MEASUREMENT_STATUS_MEASUREMENT_ERROR,
    ):
        return _decision(
            VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE, EVIDENCE_STATE_INSUFFICIENT,
            reasons=(REASON_INSUFFICIENT_FACE_EVIDENCE,),
        )
    if measurement.measurement_status == MEASUREMENT_STATUS_NO_FACE:
        return _decision(
            VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE, EVIDENCE_STATE_INSUFFICIENT,
            reasons=(REASON_INSUFFICIENT_FACE_EVIDENCE,),
        )
    if measurement.face_detection_rate is None or measurement.face_detection_rate < MIN_RELIABLE_FACE_DETECTION_RATE:
        return _decision(
            VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE, EVIDENCE_STATE_INSUFFICIENT,
            reasons=(REASON_INSUFFICIENT_FACE_EVIDENCE,),
        )

    # STAGE 10: multi-face fail-closed.
    if _clip_has_multi_face_evidence(measurement):
        return _decision(
            VISUAL_ACTION_BLOCKED_MULTI_FACE, EVIDENCE_STATE_SUFFICIENT,
            reasons=(), safety_blockers=(SAFETY_BLOCKER_MULTI_FACE_PRESENT,),
        )

    # STAGE 9: pre-existing face clipping -- never authorize a correction
    # that could worsen it.
    if _clip_has_clipped_face_evidence(measurement):
        return _decision(
            VISUAL_ACTION_BLOCKED_FACE_SAFETY, EVIDENCE_STATE_SUFFICIENT,
            reasons=(REASON_FACE_ALREADY_CLIPPED,), safety_blockers=(SAFETY_BLOCKER_FACE_ALREADY_CLIPPED,),
        )

    # STAGE 11: product safety -- fail closed by default (see module docstring).
    if not product_safety_established:
        return _decision(
            VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN, EVIDENCE_STATE_SUFFICIENT,
            reasons=(), safety_blockers=(SAFETY_BLOCKER_PRODUCT_SAFETY_UNKNOWN,),
        )

    # Eligible: this clip alone presents no intrinsic reason a future
    # join-triggered correction could not safely target it.
    return _decision(
        VISUAL_ACTION_NO_CHANGE, EVIDENCE_STATE_SUFFICIENT,
        reasons=(REASON_NO_DISCONTINUITY_EVIDENCE,),
        reframe_authorized=True, punch_in_authorized=True,
    )


# ---------------------------------------------------------------------------
# Join-level policy.
# ---------------------------------------------------------------------------

def _clamp_magnitude(value: float, limit: float) -> float:
    if value >= 0:
        return min(value, limit)
    return max(value, -limit)


def evaluate_join_policy(
    left_measurement: VisualClipMeasurement,
    right_measurement: VisualClipMeasurement,
    join_measurement: VisualJoinMeasurement,
    *,
    left_decision: VisualClipPolicyDecision,
    right_decision: VisualClipPolicyDecision,
    clip_duration_sec: float | None = None,
) -> VisualJoinPolicyDecision:
    """STAGE 13-19/25: the actual correction decision. Priority ladder
    (STAGE 25, binding, checked in this exact order): face safety ->
    multi-face safety -> product safety -> [evidence sufficiency] ->
    crop-bound feasibility -> continuity improvement."""
    thresholds_used = canonical_thresholds_snapshot()
    warnings: list[str] = list(left_decision.warnings) + [
        w for w in right_decision.warnings if w not in left_decision.warnings
    ]

    def _decision(action: str, evidence_state: str, *, reasons: tuple[str, ...] = (),
                  safety_blockers: tuple[str, ...] = (),
                  position_match_authorized: bool = False, scale_match_authorized: bool = False,
                  punch_in_authorized: bool = False,
                  requested_translation_x: float | None = None, requested_translation_y: float | None = None,
                  authorized_translation_x: float | None = None, authorized_translation_y: float | None = None,
                  requested_scale: float | None = None, authorized_scale: float | None = None,
                  target_position_x: float | None = None, target_position_y: float | None = None,
                  target_scale: float | None = None) -> VisualJoinPolicyDecision:
        return VisualJoinPolicyDecision(
            left_clip_id=left_measurement.clip_id, right_clip_id=right_measurement.clip_id,
            action=action, evidence_state=evidence_state,
            face_center_dx=join_measurement.face_center_dx, face_center_dy=join_measurement.face_center_dy,
            face_scale_delta=_relative_scale_delta(left_measurement, right_measurement),
            headroom_delta=join_measurement.headroom_delta, thresholds_used=thresholds_used,
            position_match_authorized=position_match_authorized, scale_match_authorized=scale_match_authorized,
            punch_in_authorized=punch_in_authorized,
            requested_translation_x=requested_translation_x, requested_translation_y=requested_translation_y,
            authorized_translation_x=authorized_translation_x, authorized_translation_y=authorized_translation_y,
            requested_scale=requested_scale, authorized_scale=authorized_scale,
            target_position_x=target_position_x, target_position_y=target_position_y, target_scale=target_scale,
            safety_blockers=safety_blockers, warnings=tuple(warnings), reasons=reasons,
        )

    # Priority 1-3: inherit any clip-level safety/evidence blocker from
    # either side -- a join can never be safer than its weakest clip.
    for decision in (left_decision, right_decision):
        if decision.action == VISUAL_ACTION_BLOCKED_FACE_SAFETY:
            return _decision(VISUAL_ACTION_BLOCKED_FACE_SAFETY, EVIDENCE_STATE_SUFFICIENT,
                              safety_blockers=(SAFETY_BLOCKER_FACE_ALREADY_CLIPPED,))
    for decision in (left_decision, right_decision):
        if decision.action == VISUAL_ACTION_BLOCKED_MULTI_FACE:
            return _decision(VISUAL_ACTION_BLOCKED_MULTI_FACE, EVIDENCE_STATE_SUFFICIENT,
                              safety_blockers=(SAFETY_BLOCKER_MULTI_FACE_PRESENT,))
    for decision in (left_decision, right_decision):
        if decision.action == VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN:
            return _decision(VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN, EVIDENCE_STATE_SUFFICIENT,
                              safety_blockers=(SAFETY_BLOCKER_PRODUCT_SAFETY_UNKNOWN,))
    for decision in (left_decision, right_decision):
        if decision.action == VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE:
            return _decision(VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE, EVIDENCE_STATE_INSUFFICIENT,
                              reasons=(REASON_INSUFFICIENT_FACE_EVIDENCE,))

    # Both clips eligible. Evaluate discontinuity evidence.
    dx = join_measurement.face_center_dx
    dy = join_measurement.face_center_dy
    scale_delta = _relative_scale_delta(left_measurement, right_measurement)
    headroom_delta = join_measurement.headroom_delta

    position_break = (
        dx is not None and abs(dx) > FACE_CENTER_X_DISCONTINUITY_THRESHOLD + _THRESHOLD_EPSILON
    ) or (
        dy is not None and abs(dy) > FACE_CENTER_Y_DISCONTINUITY_THRESHOLD + _THRESHOLD_EPSILON
    )
    scale_break = (
        scale_delta is not None
        and abs(scale_delta) > FACE_SCALE_RELATIVE_AREA_DISCONTINUITY_THRESHOLD + _THRESHOLD_EPSILON
    )
    headroom_break = (
        headroom_delta is not None
        and abs(headroom_delta) > HEADROOM_DISCONTINUITY_THRESHOLD + _THRESHOLD_EPSILON
    )

    if not (position_break or scale_break or headroom_break):
        return _decision(VISUAL_ACTION_NO_CHANGE, EVIDENCE_STATE_SUFFICIENT,
                          reasons=(REASON_NO_DISCONTINUITY_EVIDENCE,))

    reasons: list[str] = []
    if position_break:
        reasons.append(REASON_FACE_POSITION_DISCONTINUITY)
    if scale_break:
        reasons.append(REASON_FACE_SCALE_DISCONTINUITY)
    if headroom_break:
        reasons.append(REASON_HEADROOM_DISCONTINUITY)

    # STAGE 19/20: punch-in eligibility (duration gate) -- evaluated
    # once, reused by the scale-discontinuity branch below.
    punch_in_eligible = clip_duration_sec is not None and clip_duration_sec >= MIN_PUNCH_IN_CLIP_DURATION_SEC
    if scale_break and not punch_in_eligible:
        reasons.append(REASON_PUNCH_IN_CLIP_TOO_SHORT)

    if scale_break and position_break:
        requested_x, authorized_x = _bound_translation(dx)
        requested_y, authorized_y = _bound_translation(dy)
        requested_s, authorized_s = _bound_scale(scale_delta)
        return _decision(
            VISUAL_ACTION_SCALE_AND_POSITION_MATCH, EVIDENCE_STATE_SUFFICIENT, reasons=tuple(reasons),
            position_match_authorized=True, scale_match_authorized=True,
            requested_translation_x=requested_x, requested_translation_y=requested_y,
            authorized_translation_x=authorized_x, authorized_translation_y=authorized_y,
            requested_scale=requested_s, authorized_scale=authorized_s,
            target_position_x=0.0, target_position_y=0.0, target_scale=authorized_s,
        )

    if scale_break:
        if punch_in_eligible:
            requested_s, authorized_s = _bound_scale(scale_delta, ceiling=MAX_PUNCH_IN_SCALE)
            authorized_s = min(authorized_s, DEFAULT_PUNCH_IN_SCALE) if authorized_s > 1.0 else authorized_s
            reasons.append(REASON_JUMP_CUT_CONCEALMENT_CANDIDATE)
            return _decision(
                VISUAL_ACTION_PUNCH_IN, EVIDENCE_STATE_SUFFICIENT, reasons=tuple(reasons),
                punch_in_authorized=True, requested_scale=requested_s, authorized_scale=authorized_s,
                target_scale=authorized_s,
            )
        requested_s, authorized_s = _bound_scale(scale_delta)
        return _decision(
            VISUAL_ACTION_SCALE_MATCH, EVIDENCE_STATE_SUFFICIENT, reasons=tuple(reasons),
            scale_match_authorized=True, requested_scale=requested_s, authorized_scale=authorized_s,
            target_scale=authorized_s,
        )

    if position_break:
        requested_x, authorized_x = _bound_translation(dx)
        requested_y, authorized_y = _bound_translation(dy)
        return _decision(
            VISUAL_ACTION_POSITION_MATCH, EVIDENCE_STATE_SUFFICIENT, reasons=tuple(reasons),
            position_match_authorized=True,
            requested_translation_x=requested_x, requested_translation_y=requested_y,
            authorized_translation_x=authorized_x, authorized_translation_y=authorized_y,
            target_position_x=0.0, target_position_y=0.0,
        )

    # headroom_break only.
    requested_y, authorized_y = _bound_translation(headroom_delta)
    return _decision(
        VISUAL_ACTION_STATIC_REFRAME, EVIDENCE_STATE_SUFFICIENT, reasons=tuple(reasons),
        position_match_authorized=True,
        requested_translation_y=requested_y, authorized_translation_y=authorized_y,
        target_position_y=0.0,
    )


def _relative_scale_delta(left: VisualClipMeasurement, right: VisualClipMeasurement) -> float | None:
    """The RELATIVE face-scale change between two clips, per this
    module's own load-bearing unit-semantics note (see module
    docstring) -- never D-258's own absolute `face_area_ratio_delta`."""
    l = left.face_area_ratio_median
    r = right.face_area_ratio_median
    if l is None or r is None or l <= 0:
        return None
    return (r - l) / l


def _bound_translation(requested: float | None) -> tuple[float | None, float | None]:
    """STAGE 23/24: preserve requested vs authorized separately -- never
    silently pretend a full correction occurred. `MAX_ADDITIONAL_CROP_
    LOSS_NORMALIZED` and `MAX_REFRAME_TRANSLATION_NORMALIZED` share the
    same approved value (0.10) in V1; this module clamps translation to
    both ceilings identically as a documented symbolic simplification
    (no real crop-window geometry exists yet -- that is D-261's own
    scope) and would need to diverge only if a future execution-
    architecture gate defines a non-1:1 crop-loss formula."""
    if requested is None:
        return None, None
    limit = min(MAX_REFRAME_TRANSLATION_NORMALIZED, MAX_ADDITIONAL_CROP_LOSS_NORMALIZED)
    return requested, _clamp_magnitude(requested, limit)


def _bound_scale(requested_relative_delta: float | None, *, ceiling: float = MAX_PUNCH_IN_SCALE) -> tuple[float | None, float | None]:
    if requested_relative_delta is None:
        return None, None
    requested_scale = 1.0 + abs(requested_relative_delta)
    authorized_scale = min(requested_scale, ceiling)
    return requested_scale, authorized_scale


# ---------------------------------------------------------------------------
# STAGE 31: deterministic plan identity.
# ---------------------------------------------------------------------------

def compute_visual_finishing_plan_identity(
    source_id: str | None,
    clip_decisions: tuple[VisualClipPolicyDecision, ...],
    join_decisions: tuple[VisualJoinPolicyDecision, ...],
) -> str:
    """A pure, deterministic function of the plan's own already-decided
    content -- no mutable global state, no filename dependency. Mirrors
    D-251/D-252/D-253/D-256's own `compute_execution_id`/`compute_
    finishing_identity` pattern exactly (`json.dumps(sort_keys=True,
    default=str)` -> SHA-256 -> 24-hex-char digest)."""
    payload = {
        "policy_version": POLICY_VERSION,
        "source_id": source_id,
        "canonical_thresholds": canonical_thresholds_snapshot(),
        "clip_decisions": [
            {
                "clip_id": d.clip_id, "action": d.action,
                "authorized_translation_x": d.authorized_translation_x,
                "authorized_translation_y": d.authorized_translation_y,
                "authorized_scale": d.authorized_scale,
            }
            for d in clip_decisions
        ],
        "join_decisions": [
            {
                "left_clip_id": d.left_clip_id, "right_clip_id": d.right_clip_id, "action": d.action,
                "authorized_translation_x": d.authorized_translation_x,
                "authorized_translation_y": d.authorized_translation_y,
                "authorized_scale": d.authorized_scale,
            }
            for d in join_decisions
        ],
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


# ---------------------------------------------------------------------------
# STAGE 22: plan-status derivation.
# ---------------------------------------------------------------------------

def _derive_plan_status(
    clip_decisions: tuple[VisualClipPolicyDecision, ...],
    join_decisions: tuple[VisualJoinPolicyDecision, ...],
) -> str:
    all_actions = [d.action for d in clip_decisions] + [d.action for d in join_decisions]
    if not all_actions:
        return PLAN_STATUS_UNKNOWN
    if any(a in _BLOCKED_ACTIONS for a in all_actions):
        return PLAN_STATUS_BLOCKED
    if all(a == VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE for a in all_actions):
        return PLAN_STATUS_ABSTAIN
    if all(a == VISUAL_ACTION_NO_CHANGE for a in all_actions):
        return PLAN_STATUS_READY_NO_CHANGE
    correction_present = any(a in _CORRECTION_ACTIONS for a in all_actions)
    abstain_present = any(a == VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE for a in all_actions)
    if correction_present and not abstain_present:
        return PLAN_STATUS_READY_FOR_VISUAL_CORRECTION
    if correction_present and abstain_present:
        return PLAN_STATUS_PARTIAL
    return PLAN_STATUS_UNKNOWN


# ---------------------------------------------------------------------------
# STAGE 7: the top-level plan-generation entry point.
# ---------------------------------------------------------------------------

def generate_visual_finishing_plan(
    clip_measurements: tuple[VisualClipMeasurement, ...],
    join_measurements: tuple[VisualJoinMeasurement, ...],
    *,
    source_id: str | None = None,
    clip_durations_sec: dict | None = None,
    product_safety_established: dict | None = None,
    provenance: dict | None = None,
) -> VisualFinishingPlan:
    """The one entry point that turns real D-258 measurements into a
    structured `VisualFinishingPlan`. Deterministic, pure, zero video
    mutation, zero DSP, zero renderer call. `clip_durations_sec` and
    `product_safety_established` are optional per-clip-id dicts a
    caller (an already-decided Selection/Boundary/render-planning
    authority) supplies when it has that context; neither is inferred
    or fabricated here."""
    clip_durations_sec = clip_durations_sec or {}
    product_safety_established = product_safety_established or {}

    clip_decisions = tuple(
        evaluate_clip_policy(
            m, product_safety_established=bool(product_safety_established.get(m.clip_id, False)),
        )
        for m in clip_measurements
    )
    by_clip_id = {m.clip_id: m for m in clip_measurements}
    decision_by_clip_id = {d.clip_id: d for d in clip_decisions}

    join_decisions = []
    for jm in join_measurements:
        left_m = by_clip_id.get(jm.left_clip_id)
        right_m = by_clip_id.get(jm.right_clip_id)
        left_d = decision_by_clip_id.get(jm.left_clip_id)
        right_d = decision_by_clip_id.get(jm.right_clip_id)
        if left_m is None or right_m is None or left_d is None or right_d is None:
            join_decisions.append(VisualJoinPolicyDecision(
                left_clip_id=jm.left_clip_id, right_clip_id=jm.right_clip_id,
                action=VISUAL_ACTION_UNKNOWN, evidence_state=EVIDENCE_STATE_INSUFFICIENT,
                face_center_dx=None, face_center_dy=None, face_scale_delta=None, headroom_delta=None,
                thresholds_used=canonical_thresholds_snapshot(),
                position_match_authorized=False, scale_match_authorized=False, punch_in_authorized=False,
                requested_translation_x=None, requested_translation_y=None,
                authorized_translation_x=None, authorized_translation_y=None,
                requested_scale=None, authorized_scale=None,
                target_position_x=None, target_position_y=None, target_scale=None,
            ))
            continue
        clip_duration = clip_durations_sec.get(jm.left_clip_id)
        join_decisions.append(evaluate_join_policy(
            left_m, right_m, jm, left_decision=left_d, right_decision=right_d,
            clip_duration_sec=clip_duration,
        ))
    join_decisions = tuple(join_decisions)

    plan_status = _derive_plan_status(clip_decisions, join_decisions)
    abstentions = tuple(
        f"{d.clip_id}: {r}" for d in clip_decisions
        if d.action == VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE for r in d.reasons
    ) + tuple(
        f"{d.left_clip_id}->{d.right_clip_id}: {r}" for d in join_decisions
        if d.action == VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE for r in d.reasons
    )
    all_warnings = tuple(dict.fromkeys(
        w for d in clip_decisions for w in d.warnings
    )) + tuple(dict.fromkeys(
        w for d in join_decisions for w in d.warnings if w not in {x for cd in clip_decisions for x in cd.warnings}
    ))
    all_reasons = tuple(d.action for d in clip_decisions) + tuple(d.action for d in join_decisions)

    identity = compute_visual_finishing_plan_identity(source_id, clip_decisions, join_decisions)

    return VisualFinishingPlan(
        policy_version=POLICY_VERSION, source_id=source_id,
        clip_measurement_references=clip_measurements, join_measurement_references=join_measurements,
        clip_decisions=clip_decisions, join_decisions=join_decisions,
        plan_status=plan_status, canonical_thresholds_snapshot=canonical_thresholds_snapshot(),
        abstentions=abstentions, warnings=all_warnings, reasons=all_reasons,
        provenance=dict(provenance) if provenance else {},
        visual_finishing_identity=identity,
    )
