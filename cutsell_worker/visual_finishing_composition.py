"""Visual Finishing END-TO-END COMPOSITION (D-263).

D-258 built real VISUAL MEASUREMENT. D-260 built the POLICY + PLAN
layer. D-262 built the EXECUTOR that turns an authorized join decision
into deterministic renderer geometry, plus a minimal, dormant hook into
`render.py`. This module is the first gate that proves the WHOLE chain
works together as one deterministic system, on synthetic media only:

    VISUAL MEASUREMENT (D-258) -> VISUAL POLICY (D-260) ->
    VisualFinishingPlan -> VISUAL EXECUTOR (D-262) ->
    EXISTING RENDERER (render.py, unchanged) ->
    POST-RENDER VISUAL MEASUREMENT (D-258, re-used) -> VERIFICATION (new)

## Scope discipline (binding, D-263's own scope banner)

NO REAL RAW. NO PROVIDER. NO LIVE PIPELINE INTEGRATION (no production
call site is changed to route through this module -- it is a new,
additive, standalone orchestration owner, called only by this gate's
own tests). NO NEW VISUAL POLICY, NO NEW NUMERIC THRESHOLD (the ten
D-260 values are the only numbers anywhere in this chain). NO EXPOSURE/
COLOR CORRECTION, NO GAZE LOGIC, NO SMART SALES FUNNEL. NO PACING/
BOUNDARY/FREEZE/AUDIO-JOIN/AUDIO-FINISHING/QC-AUTHORITY CHANGE.

This module orchestrates the four existing, proven modules -- it never
duplicates their logic:

- measurement: always calls `visual_finishing_measurement.measure_
  visual_clip`/`compute_visual_join_measurement` for real, unless a
  caller explicitly supplies a pre-computed measurement override (an
  escape hatch documented at each field below -- exact literal
  fixtures for a deterministic policy divergence, or a stand-in for a
  post-render measurement this sandbox cannot produce without cv2/
  mediapipe installed, mirroring D-258's own 9 environment-gated
  tests).
- policy: always calls `visual_finishing_policy.generate_visual_
  finishing_plan` for real, unless a caller explicitly supplies a
  pre-built `VisualFinishingPlan` (the escape hatch idempotence/
  identity tests need -- STAGE 23/24/27 of this gate's own directive
  demand PROVING no-recomputation given a plan, which requires the
  ability to supply one).
- geometry/execution: always calls `visual_finishing_executor.execute_
  visual_finishing_decision` -- never recomputes crop/scale geometry,
  never re-derives a threshold comparison.
- render: always calls `render.render_preview` (the one LIVE renderer
  entry point) -- never builds its own ffmpeg command.

## The one new capability this gate adds: VERIFICATION

D-262 defined `VisualFinishingVerificationResult`'s TYPE but explicitly
left its computation "MISSING-FUTURE pending a future gate" -- this
one. `_verify_execution` below is new, not a duplication of anything:
it compares a PRE and POST `VisualClipMeasurement` (both produced by
the SAME unchanged D-258 measurement authority) against the executed
action's own intended DIRECTION only -- never an invented perceptual
magnitude threshold (this gate's own Stage 10 instruction).

## The correction-target convention (this gate's own design decision,
not previously specified)

Every `VisualJoinPolicyDecision` names a LEFT and RIGHT clip; this
module always applies an authorized correction to the RIGHT clip
(bringing the later clip's framing toward continuity with its already-
established left neighbor) -- a single, deterministic, documented
convention, matching D-262's own "the caller decides which side of
`decision` that clip is" delegation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

from . import render as render_module
from .render_plan import RenderSegment
from .visual_finishing_executor import (
    EXECUTION_STATUS_ALREADY_APPLIED,
    EXECUTION_STATUS_CROP_LIMIT_EXCEEDED,
    EXECUTION_STATUS_FACE_SAFETY_BLOCKED,
    EXECUTION_STATUS_INVALID_GEOMETRY,
    EXECUTION_STATUS_NO_ACTION_NEEDED,
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
    execute_visual_finishing_decision,
)
from .visual_finishing_measurement import (
    MEASUREMENT_STATUS_COMPLETE,
    MEASUREMENT_STATUS_PARTIAL,
    VisualClipMeasurement,
    VisualJoinMeasurement,
    compute_visual_join_measurement,
    default_sample_timestamps,
    measure_visual_clip,
)
from .visual_finishing_policy import (
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
    VisualFinishingPlan,
    generate_visual_finishing_plan,
)

COMPOSITION_VERSION = "visual_finishing_composition.v1"

# ---------------------------------------------------------------------------
# STAGE 4: bounded composition-status vocabulary.
# ---------------------------------------------------------------------------

COMPOSITION_STATUS_SUCCESS = "SUCCESS"
COMPOSITION_STATUS_NO_CHANGE = "NO_CHANGE"
COMPOSITION_STATUS_PARTIAL = "PARTIAL"
COMPOSITION_STATUS_ABSTAIN = "ABSTAIN"
COMPOSITION_STATUS_BLOCKED = "BLOCKED"
COMPOSITION_STATUS_MEASUREMENT_FAILED = "MEASUREMENT_FAILED"
COMPOSITION_STATUS_POLICY_FAILED = "POLICY_FAILED"
COMPOSITION_STATUS_EXECUTION_FAILED = "EXECUTION_FAILED"
COMPOSITION_STATUS_RENDER_FAILED = "RENDER_FAILED"
COMPOSITION_STATUS_VERIFY_FAILED = "VERIFY_FAILED"
COMPOSITION_STATUS_ALREADY_APPLIED = "ALREADY_APPLIED"
COMPOSITION_STATUS_OTHER = "OTHER"

_EXECUTION_BLOCKED_STATUSES = frozenset({
    EXECUTION_STATUS_PLAN_NOT_EXECUTABLE, EXECUTION_STATUS_FACE_SAFETY_BLOCKED,
    EXECUTION_STATUS_PRODUCT_SAFETY_BLOCKED, EXECUTION_STATUS_CROP_LIMIT_EXCEEDED,
    EXECUTION_STATUS_INVALID_GEOMETRY, EXECUTION_STATUS_SOURCE_DIMENSIONS_UNAVAILABLE,
})
_CORRECTION_ACTIONS = frozenset({
    VISUAL_ACTION_PUNCH_IN, VISUAL_ACTION_SCALE_MATCH, VISUAL_ACTION_STATIC_REFRAME,
    VISUAL_ACTION_POSITION_MATCH, VISUAL_ACTION_SCALE_AND_POSITION_MATCH,
})


# ---------------------------------------------------------------------------
# STAGE 2: composition input.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VisualFinishingCompositionInput:
    """Every override field is OPTIONAL and defaults to `None`/empty --
    the default path always calls the real measurement/policy modules
    (STAGE 5/6's own binding requirement). Overrides exist ONLY for
    cases this gate's own directive explicitly authorizes hand-
    construction for: exact-literal policy-divergence fixtures (this
    sandbox has no cv2/mediapipe, matching D-258's own 9 skipped
    tests), idempotence/identity proofs that need a KNOWN plan, and a
    post-render measurement stand-in when the real one is
    environment-unavailable."""

    source_identity: str | None
    # Renderer-compatible BASELINE segments (no `visual_transform` set
    # yet -- this module assigns it after execution, never before).
    # `segments[i]` corresponds 1:1, in order, to `clip_order[i]`.
    segments: tuple[RenderSegment, ...]
    output_path: str
    work_dir: str

    render_width: int = 1080
    render_height: int = 1920
    render_fps: int = 30

    # clip_id -> VisualClipMeasurement override (skips measure_visual_clip
    # for that one clip only; any clip absent from this mapping is
    # measured for real).
    pre_measurements: Mapping[str, VisualClipMeasurement] | None = None
    # Full override: when set, measurement AND policy phases are both
    # skipped entirely and this plan is trusted verbatim (STAGE 6's own
    # "no policy recomputation" contract).
    pre_plan: VisualFinishingPlan | None = None

    product_safety_established: Mapping[str, bool] | None = None
    clip_durations_sec: Mapping[str, float] | None = None
    # (left_clip_id, right_clip_id) -> a prior execution_id, for the
    # ALREADY_APPLIED idempotence proof.
    previous_execution_ids: Mapping[tuple[str, str], str] | None = None
    # clip_id -> a REAL (x_min, y_min, x_max, y_max) face bbox for the
    # executor's own execution-time face-safety re-verification.
    face_bboxes: Mapping[str, tuple[float, float, float, float]] | None = None

    run_post_measurement: bool = True
    # clip_id -> VisualClipMeasurement stand-in for the POST-render
    # measurement (only used when `run_post_measurement` is True and a
    # real measurement is unavailable/undesired for that clip; the
    # REAL rendered file is always produced regardless of this field).
    post_measurement_overrides: Mapping[str, VisualClipMeasurement] | None = None

    provenance: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# STAGE 3: composition result.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VisualFinishingCompositionResult:
    composition_id: str
    policy_version: str
    plan_id: str | None
    source_identity: str | None

    pre_measurements: tuple[VisualClipMeasurement, ...]
    join_measurements: tuple[VisualJoinMeasurement, ...]
    clip_decisions: tuple
    join_decisions: tuple

    execution_records: tuple[VisualFinishingExecutionRecord, ...]

    render_output_path: str | None
    rendered_segments: tuple[RenderSegment, ...]

    post_measurements: tuple[VisualClipMeasurement, ...]
    verification_results: tuple[VisualFinishingVerificationResult, ...]

    overall_status: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# STAGE 21: deterministic composition identity.
# ---------------------------------------------------------------------------

def compute_composition_id(
    source_identity: str | None,
    plan_id: str | None,
    execution_ids: Sequence[str],
) -> str:
    """A pure function of already-decided content: source identity
    (never a path), the plan's own identity (already policy-version-
    sensitive, per `compute_visual_finishing_plan_identity`), and the
    sorted execution ids. Mirrors `compute_execution_id`/`compute_
    visual_execution_id`'s established pattern exactly."""
    payload = {
        "composition_version": COMPOSITION_VERSION,
        "source_identity": source_identity,
        "plan_id": plan_id,
        "execution_ids": sorted(execution_ids),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


# ---------------------------------------------------------------------------
# STAGE 5: measurement phase.
# ---------------------------------------------------------------------------

def _measure_segment(segment: RenderSegment, source_identity: str | None) -> VisualClipMeasurement:
    """Real measurement of exactly this segment's own trimmed window
    inside its (possibly shared) source file -- absolute timestamps
    computed the SAME way `default_sample_timestamps` derives them for
    a standalone clip, just offset by `segment.start`."""
    duration = max(0.0, float(segment.end) - float(segment.start))
    relative = default_sample_timestamps(duration)
    timestamps = tuple(float(segment.start) + t for t in relative)
    return measure_visual_clip(
        segment.source_path, timestamps=timestamps,
        source_id=source_identity, clip_id=segment.clip_id,
    )


def _collect_measurements(
    composition_input: VisualFinishingCompositionInput,
) -> tuple[VisualClipMeasurement, ...]:
    overrides = composition_input.pre_measurements or {}
    measurements = []
    for segment in composition_input.segments:
        override = overrides.get(segment.clip_id)
        if override is not None:
            measurements.append(override)
            continue
        measurements.append(_measure_segment(segment, composition_input.source_identity))
    return tuple(measurements)


def _collect_join_measurements(
    clip_measurements: tuple[VisualClipMeasurement, ...],
) -> tuple[VisualJoinMeasurement, ...]:
    """STAGE 5: adjacent-pair join measurement -- a pure function of two
    already-built clip measurements, always computed for real regardless
    of any override (no cv2/mediapipe dependency exists in this
    function at all)."""
    return tuple(
        compute_visual_join_measurement(clip_measurements[i], clip_measurements[i + 1])
        for i in range(len(clip_measurements) - 1)
    )


# ---------------------------------------------------------------------------
# STAGE 7: execution phase.
# ---------------------------------------------------------------------------

def _execute_plan(
    plan: VisualFinishingPlan,
    composition_input: VisualFinishingCompositionInput,
) -> tuple[VisualFinishingExecutionRecord, ...]:
    by_clip_id = {m.clip_id: m for m in plan.clip_measurement_references}
    previous_ids = composition_input.previous_execution_ids or {}
    face_bboxes = composition_input.face_bboxes or {}
    records = []
    for decision in plan.join_decisions:
        target_clip_id = decision.right_clip_id
        clip_measurement = by_clip_id.get(target_clip_id)
        if clip_measurement is None:
            continue
        records.append(execute_visual_finishing_decision(
            decision, clip_measurement,
            plan_id=plan.visual_finishing_identity,
            source_identity=composition_input.source_identity,
            face_bbox=face_bboxes.get(target_clip_id),
            previous_execution_id=previous_ids.get((decision.left_clip_id, decision.right_clip_id)),
        ))
    return tuple(records)


def _apply_transforms(
    segments: tuple[RenderSegment, ...],
    execution_records: tuple[VisualFinishingExecutionRecord, ...],
) -> tuple[RenderSegment, ...]:
    """STAGE 8: attach a `VisualTransformSpec` to exactly the segments
    whose own execution ended in `SUCCESS` -- every other segment's
    `visual_transform` stays `None` (fail-closed by construction, never
    an explicit clearing step)."""
    transform_by_clip_id = {
        record.clip_id: record.transform_spec
        for record in execution_records
        if record.execution_status == EXECUTION_STATUS_SUCCESS and record.transform_spec is not None
    }
    return tuple(
        replace(segment, visual_transform=transform_by_clip_id[segment.clip_id])
        if segment.clip_id in transform_by_clip_id else segment
        for segment in segments
    )


# ---------------------------------------------------------------------------
# STAGE 10: verification phase (the one genuinely NEW capability).
# ---------------------------------------------------------------------------

def _direction_ok(delta: float | None, expected_sign: float) -> bool | None:
    """STAGE 10: literal sign/direction comparison only -- no invented
    perceptual magnitude threshold. Returns `None` (unverifiable) when
    either operand is missing."""
    if delta is None or expected_sign == 0:
        return None
    if expected_sign > 0:
        return delta > 0
    return delta < 0


def _verify_execution(
    record: VisualFinishingExecutionRecord,
    pre_measurement: VisualClipMeasurement | None,
    post_measurement: VisualClipMeasurement | None,
) -> VisualFinishingVerificationResult:
    """STAGE 10: per-action factual direction verification, comparing a
    PRE and POST `VisualClipMeasurement` -- both produced by the SAME,
    unchanged D-258 measurement authority. Never re-derives a policy
    threshold; never invents a perceptual pass/fail band."""
    if post_measurement is None:
        return VisualFinishingVerificationResult(
            measurement_status="UNAVAILABLE", face_detected=None, face_bbox=None,
            face_center_x=None, face_center_y=None, face_area_ratio=None, headroom_ratio=None,
            face_contained=None, orientation=None, width=None, height=None,
            duration_preserved=None, action_direction_verified=None,
            verification_status=VERIFICATION_STATUS_UNVERIFIABLE,
            errors=("post_measurement_unavailable",),
        )

    face_bbox = None
    if all(v is not None for v in (
        post_measurement.face_center_x_median, post_measurement.face_center_y_median,
    )):
        face_bbox = None  # STAGE 10: no per-frame bbox aggregated at clip level; median center only.

    action = record.action
    direction: bool | None = None
    if action == "NO_ACTION_NEEDED" or record.execution_status == EXECUTION_STATUS_NO_ACTION_NEEDED:
        direction = True  # nothing was supposed to move -- trivially satisfied.
    elif pre_measurement is not None:
        if action in (VISUAL_ACTION_PUNCH_IN, VISUAL_ACTION_SCALE_MATCH, VISUAL_ACTION_SCALE_AND_POSITION_MATCH):
            pre_area = pre_measurement.face_area_ratio_median
            post_area = post_measurement.face_area_ratio_median
            area_delta = None if pre_area is None or post_area is None else post_area - pre_area
            # A punch-in/scale-match always makes the corrected clip's
            # own face appear LARGER in its own frame (that is what a
            # zoom-in does) -- direction is always "increase", regardless
            # of which side of the join it corrects toward.
            direction = _direction_ok(area_delta, +1.0)
        if action in (VISUAL_ACTION_POSITION_MATCH, VISUAL_ACTION_STATIC_REFRAME, VISUAL_ACTION_SCALE_AND_POSITION_MATCH):
            tx = record.authorized_translation_x
            pre_x = pre_measurement.face_center_x_median
            post_x = post_measurement.face_center_x_median
            x_delta = None if pre_x is None or post_x is None else post_x - pre_x
            x_ok = True if not tx else _direction_ok(x_delta, 1.0 if tx > 0 else -1.0)
            ty = record.authorized_translation_y
            pre_y = pre_measurement.face_center_y_median
            post_y = post_measurement.face_center_y_median
            y_delta = None if pre_y is None or post_y is None else post_y - pre_y
            y_ok = True if not ty else _direction_ok(y_delta, 1.0 if ty > 0 else -1.0)
            combined = None
            if x_ok is not None and y_ok is not None:
                combined = bool(x_ok) and bool(y_ok)
            direction = combined if direction is None else (direction and combined if combined is not None else direction)

    verification_status = VERIFICATION_STATUS_UNVERIFIABLE
    if direction is True:
        verification_status = VERIFICATION_STATUS_PASS
    elif direction is False:
        verification_status = VERIFICATION_STATUS_FAIL

    return VisualFinishingVerificationResult(
        measurement_status=post_measurement.measurement_status,
        face_detected=post_measurement.face_valid_frame_count > 0,
        face_bbox=face_bbox,
        face_center_x=post_measurement.face_center_x_median,
        face_center_y=post_measurement.face_center_y_median,
        face_area_ratio=post_measurement.face_area_ratio_median,
        headroom_ratio=post_measurement.headroom_ratio_median,
        face_contained=None,
        orientation=post_measurement.orientation,
        width=post_measurement.frame_width,
        height=post_measurement.frame_height,
        duration_preserved=None,
        action_direction_verified=direction,
        verification_status=verification_status,
        errors=(),
    )


# ---------------------------------------------------------------------------
# STAGE 9: post-render measurement phase.
# ---------------------------------------------------------------------------

def _post_measure_segments(
    output_path: str,
    rendered_segments: tuple[RenderSegment, ...],
    *,
    fps: int,
    source_identity: str | None,
    overrides: Mapping[str, VisualClipMeasurement] | None,
) -> tuple[VisualClipMeasurement, ...]:
    """STAGE 9: re-measures each segment's own window WITHIN the single
    rendered output file, using the exact same per-segment output
    duration the renderer itself computed (`rendered_segment_duration_
    sec`) -- so the sampled window matches what that segment actually
    occupies on the final timeline, not a re-guessed one."""
    overrides = overrides or {}
    results = []
    cursor = 0.0
    for segment in rendered_segments:
        duration = render_module.rendered_segment_duration_sec(segment.duration_sec, fps=fps)
        override = overrides.get(segment.clip_id)
        if override is not None:
            results.append(override)
        else:
            relative = default_sample_timestamps(duration)
            timestamps = tuple(cursor + t for t in relative)
            results.append(measure_visual_clip(
                output_path, timestamps=timestamps,
                source_id=source_identity, clip_id=segment.clip_id,
            ))
        cursor += duration
    return tuple(results)


# ---------------------------------------------------------------------------
# STAGE 1: the single top-level composition entry point.
# ---------------------------------------------------------------------------

def compose_visual_finishing(
    composition_input: VisualFinishingCompositionInput,
) -> VisualFinishingCompositionResult:
    """The single orchestration entry point. Calls the four existing,
    proven modules in order; never recomputes any of their logic."""
    warnings: list[str] = []
    errors: list[str] = []

    # --- STAGE 5/6: measurement + policy (unless a full plan override). ---
    if composition_input.pre_plan is not None:
        plan = composition_input.pre_plan
        clip_measurements = plan.clip_measurement_references
        join_measurements = plan.join_measurement_references
    else:
        try:
            clip_measurements = _collect_measurements(composition_input)
            join_measurements = _collect_join_measurements(clip_measurements)
        except Exception as exc:  # STAGE 25: bounded, never an unbounded crash.
            return VisualFinishingCompositionResult(
                composition_id=compute_composition_id(composition_input.source_identity, None, ()),
                policy_version=POLICY_VERSION, plan_id=None, source_identity=composition_input.source_identity,
                pre_measurements=(), join_measurements=(), clip_decisions=(), join_decisions=(),
                execution_records=(), render_output_path=None, rendered_segments=(),
                post_measurements=(), verification_results=(),
                overall_status=COMPOSITION_STATUS_MEASUREMENT_FAILED,
                warnings=(), errors=(f"measurement_phase_exception:{exc.__class__.__name__}",),
                provenance=dict(composition_input.provenance),
            )
        try:
            plan = generate_visual_finishing_plan(
                clip_measurements, join_measurements,
                source_id=composition_input.source_identity,
                clip_durations_sec=dict(composition_input.clip_durations_sec or {}),
                product_safety_established=dict(composition_input.product_safety_established or {}),
                provenance=dict(composition_input.provenance),
            )
        except Exception as exc:
            return VisualFinishingCompositionResult(
                composition_id=compute_composition_id(composition_input.source_identity, None, ()),
                policy_version=POLICY_VERSION, plan_id=None, source_identity=composition_input.source_identity,
                pre_measurements=clip_measurements, join_measurements=join_measurements,
                clip_decisions=(), join_decisions=(),
                execution_records=(), render_output_path=None, rendered_segments=(),
                post_measurements=(), verification_results=(),
                overall_status=COMPOSITION_STATUS_POLICY_FAILED,
                warnings=(), errors=(f"policy_phase_exception:{exc.__class__.__name__}",),
                provenance=dict(composition_input.provenance),
            )

    # A genuinely EMPTY plan (zero clip/join decisions -- e.g. zero
    # segments supplied) is a structural failure. `plan.plan_status ==
    # PLAN_STATUS_UNKNOWN` alone is NOT sufficient grounds for this: D-260's
    # own `_derive_plan_status` (pre-existing, unmodified, out of this
    # gate's scope) also returns `PLAN_STATUS_UNKNOWN` for a genuinely
    # mixed NO_CHANGE + ABSTAIN plan with no correction present -- a real
    # gap in that ladder, not a structural failure. This composition layer
    # derives its OWN richer status from the plan's already-computed
    # decisions instead of re-trusting that one collapsed field (see
    # `_derive_overall_status` below) -- no new policy logic, purely an
    # honest status ROLLUP of decisions D-260 already made.
    if not plan.clip_decisions and not plan.join_decisions:
        return VisualFinishingCompositionResult(
            composition_id=compute_composition_id(composition_input.source_identity, plan.visual_finishing_identity, ()),
            policy_version=plan.policy_version, plan_id=plan.visual_finishing_identity,
            source_identity=composition_input.source_identity,
            pre_measurements=clip_measurements, join_measurements=join_measurements,
            clip_decisions=plan.clip_decisions, join_decisions=plan.join_decisions,
            execution_records=(), render_output_path=None, rendered_segments=(),
            post_measurements=(), verification_results=(),
            overall_status=COMPOSITION_STATUS_POLICY_FAILED,
            warnings=tuple(plan.warnings), errors=("empty_plan_no_decisions",),
            provenance=dict(composition_input.provenance),
        )

    # --- STAGE 7: execution (never re-evaluates the plan's own decision). ---
    try:
        execution_records = _execute_plan(plan, composition_input)
    except Exception as exc:
        return VisualFinishingCompositionResult(
            composition_id=compute_composition_id(composition_input.source_identity, plan.visual_finishing_identity, ()),
            policy_version=plan.policy_version, plan_id=plan.visual_finishing_identity,
            source_identity=composition_input.source_identity,
            pre_measurements=clip_measurements, join_measurements=join_measurements,
            clip_decisions=plan.clip_decisions, join_decisions=plan.join_decisions,
            execution_records=(), render_output_path=None, rendered_segments=(),
            post_measurements=(), verification_results=(),
            overall_status=COMPOSITION_STATUS_EXECUTION_FAILED,
            warnings=tuple(plan.warnings), errors=(f"execution_phase_exception:{exc.__class__.__name__}",),
            provenance=dict(composition_input.provenance),
        )

    rendered_segments = _apply_transforms(composition_input.segments, execution_records)

    # --- STAGE 8: render (the existing, unchanged renderer). ---
    try:
        render_output_path = render_module.render_preview(
            rendered_segments, composition_input.output_path,
            width=composition_input.render_width, height=composition_input.render_height,
            fps=composition_input.render_fps,
        )
    except Exception as exc:
        composition_id = compute_composition_id(
            composition_input.source_identity, plan.visual_finishing_identity,
            [r.execution_id for r in execution_records],
        )
        return VisualFinishingCompositionResult(
            composition_id=composition_id,
            policy_version=plan.policy_version, plan_id=plan.visual_finishing_identity,
            source_identity=composition_input.source_identity,
            pre_measurements=clip_measurements, join_measurements=join_measurements,
            clip_decisions=plan.clip_decisions, join_decisions=plan.join_decisions,
            execution_records=execution_records, render_output_path=None, rendered_segments=rendered_segments,
            post_measurements=(), verification_results=(),
            overall_status=COMPOSITION_STATUS_RENDER_FAILED,
            warnings=tuple(plan.warnings), errors=(f"render_phase_exception:{exc.__class__.__name__}",),
            provenance=dict(composition_input.provenance),
        )

    # --- STAGE 9/10: post-measurement + verification. ---
    post_measurements: tuple[VisualClipMeasurement, ...] = ()
    verification_results: tuple[VisualFinishingVerificationResult, ...] = ()
    if composition_input.run_post_measurement:
        try:
            post_measurements = _post_measure_segments(
                render_output_path, rendered_segments, fps=composition_input.render_fps,
                source_identity=composition_input.source_identity,
                overrides=composition_input.post_measurement_overrides,
            )
        except Exception as exc:
            errors.append(f"post_measurement_exception:{exc.__class__.__name__}")
            post_measurements = ()

        pre_by_clip_id = {m.clip_id: m for m in clip_measurements}
        post_by_clip_id = {m.clip_id: m for m in post_measurements}
        verification_results = tuple(
            _verify_execution(record, pre_by_clip_id.get(record.clip_id), post_by_clip_id.get(record.clip_id))
            for record in execution_records
        )

    composition_id = compute_composition_id(
        composition_input.source_identity, plan.visual_finishing_identity,
        [r.execution_id for r in execution_records],
    )

    overall_status = _derive_overall_status(plan, execution_records, verification_results)

    return VisualFinishingCompositionResult(
        composition_id=composition_id,
        policy_version=plan.policy_version, plan_id=plan.visual_finishing_identity,
        source_identity=composition_input.source_identity,
        pre_measurements=clip_measurements, join_measurements=join_measurements,
        clip_decisions=plan.clip_decisions, join_decisions=plan.join_decisions,
        execution_records=execution_records, render_output_path=render_output_path,
        rendered_segments=rendered_segments,
        post_measurements=post_measurements, verification_results=verification_results,
        overall_status=overall_status,
        warnings=tuple(plan.warnings), errors=tuple(errors),
        provenance=dict(composition_input.provenance),
    )


# ---------------------------------------------------------------------------
# STAGE 4: overall-status derivation -- pure, deterministic, checked in a
# fixed priority order (mirrors D-260's own `_derive_plan_status` ladder
# discipline).
# ---------------------------------------------------------------------------

_BLOCKED_DECISION_ACTIONS = frozenset({
    VISUAL_ACTION_BLOCKED_MULTI_FACE, VISUAL_ACTION_BLOCKED_FACE_SAFETY,
    VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN,
})


def _derive_overall_status(
    plan: VisualFinishingPlan,
    execution_records: tuple[VisualFinishingExecutionRecord, ...],
    verification_results: tuple[VisualFinishingVerificationResult, ...],
) -> str:
    """Reads the plan's own already-computed `clip_decisions`/`join_
    decisions` actions directly, rather than solely trusting the single
    collapsed `plan.plan_status` field -- D-260's own `_derive_plan_
    status` (pre-existing, unmodified) maps BOTH a genuinely blocked/
    abstained plan AND a mixed NO_CHANGE + ABSTAIN plan (no correction
    present) to different things (`BLOCKED`/`ABSTAIN` vs. falling
    through to `UNKNOWN` for the latter) -- an existing gap in that one
    ladder, not something this gate's own "no new policy" scope may
    touch. This is a pure ROLLUP of decisions D-260 already made -- no
    new threshold, no new comparison, no re-evaluation of any evidence."""
    all_actions = [d.action for d in plan.clip_decisions] + [d.action for d in plan.join_decisions]

    if any(a in _BLOCKED_DECISION_ACTIONS for a in all_actions):
        return COMPOSITION_STATUS_BLOCKED

    # A genuine FAIL verdict from real (or stand-in) verification always
    # wins over an otherwise-successful execution -- never masked.
    if any(v.verification_status == VERIFICATION_STATUS_FAIL for v in verification_results):
        return COMPOSITION_STATUS_VERIFY_FAILED

    correction_present = any(a in _CORRECTION_ACTIONS for a in all_actions)
    abstain_present = any(a == VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE for a in all_actions)
    unknown_present = any(a == VISUAL_ACTION_UNKNOWN for a in all_actions)

    if not correction_present:
        if abstain_present or unknown_present:
            return COMPOSITION_STATUS_ABSTAIN
        return COMPOSITION_STATUS_NO_CHANGE

    correction_records = [r for r in execution_records if r.action in _CORRECTION_ACTIONS]
    already_applied = [r for r in correction_records if r.execution_status == EXECUTION_STATUS_ALREADY_APPLIED]
    succeeded = [r for r in correction_records if r.execution_status == EXECUTION_STATUS_SUCCESS]
    blocked = [r for r in correction_records if r.execution_status in _EXECUTION_BLOCKED_STATUSES]

    if abstain_present:
        return COMPOSITION_STATUS_PARTIAL
    if already_applied and not succeeded and not blocked:
        return COMPOSITION_STATUS_ALREADY_APPLIED
    if succeeded and blocked:
        return COMPOSITION_STATUS_PARTIAL
    if succeeded or already_applied:
        return COMPOSITION_STATUS_SUCCESS
    if blocked:
        return COMPOSITION_STATUS_EXECUTION_FAILED
    return COMPOSITION_STATUS_OTHER
