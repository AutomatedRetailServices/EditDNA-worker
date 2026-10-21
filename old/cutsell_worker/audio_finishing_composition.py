"""End-to-End Audio Finishing Composition (D-253).

D-247 built MEASUREMENT. D-249 built POLICY + PLAN. D-251 built the
LEVEL-2 (whole-video) executor. D-252 built the LEVEL-1 (adjacent-take)
executor. Nothing before this gate has ever run those two executors
TOGETHER, through a real render, in one deterministic chain. This module
is that chain -- a bounded, offline, test/tooling-only ORCHESTRATION
layer. It contains no new policy, no new numeric threshold, and no new
DSP: every real decision and every real ffmpeg invocation happens inside
the already-existing D-247/D-249/D-251/D-252 authorities this module
only calls, in order.

    SYNTHETIC SOURCE SEGMENTS
        -> D-247 MEASUREMENT (caller's responsibility, produces the plan)
        -> D-249 POLICY / PLAN                     (caller's responsibility)
        -> D-252 ADJACENT GAIN APPLICATION          (this module calls it)
        -> EXISTING RENDERER (render.render_preview) (this module calls it)
        -> D-251 WHOLE-VIDEO EXECUTOR                (this module calls it)
        -> D-247 POST-MEASUREMENT (via D-251's own verification)
        -> D-251 VERIFICATION

No stage collapses into another: Level 1 only ever touches
`RenderSegment.audio_volume` before any ffmpeg runs; the renderer only
ever renders; Level 2 only ever applies its own already-authorized
whole-video gain/limiter to the REAL rendered file; verification only
ever re-measures the REAL final file. This module's only job is calling
them in the right order and refusing to continue past a real failure --
never inventing a fallback measurement, a fallback gain, or a fallback
"looks fine" verdict.

## Not wired into production, on purpose

This module is never imported by `process_universal_clean_cut_sources`,
`pipeline.py`, `flow_b.py`, any GitHub workflow, or any Modal/RunPod entry
point (confirmed by the same structural "never imported by render.py or
pipeline" test convention D-251/D-252 already use). `render.py` remains
the sole renderer; no finishing policy is added to it.

## Where the "final file" comes from when Level 2 does nothing

When the whole-video plan resolves to `NO_ACTION_NEEDED` or
`PLAN_NOT_EXECUTABLE` (D-251), no separate whole-video output file is
ever written -- the render's own output (post-Level-1, pre-Level-2) IS
the deliverable. This module still runs D-251's own `_verify_execution`
against that file (via a synthetic pass-through `AudioFinishingExecutionRecord`
pointing at the render output) so the composition's final verification
always reflects the REAL final file, never skipped just because Level 2
had nothing to add.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field

from . import render as render_module
from .audio_finishing_executor import (
    ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED,
    EXECUTION_STATUS_SUCCESS,
    VERIFICATION_STATUS_PASS,
    VERIFICATION_STATUS_TECHNICAL_FAILURE,
    AudioFinishingExecutionRecord,
    ExecutionVerificationResult,
    SegmentGainAdjustmentResult,
    _verify_execution,
    apply_adjacent_take_adjustments,
    execute_audio_finishing_plan,
)
from .audio_finishing_policy import AudioFinishingPlan
from .render_plan import RenderSegment

# ---------------------------------------------------------------------------
# STAGE 11: bounded composition-status vocabulary.
# ---------------------------------------------------------------------------

COMPOSITION_STATUS_SUCCESS = "SUCCESS"
COMPOSITION_STATUS_NO_CHANGE = "NO_CHANGE"
COMPOSITION_STATUS_LEVEL1_BLOCKED = "LEVEL1_BLOCKED"
COMPOSITION_STATUS_RENDER_FAILED = "RENDER_FAILED"
COMPOSITION_STATUS_LEVEL2_FAILED = "LEVEL2_FAILED"
COMPOSITION_STATUS_VERIFY_FAILED = "VERIFY_FAILED"
COMPOSITION_STATUS_PARTIAL = "PARTIAL"
COMPOSITION_STATUS_OTHER = "OTHER"

# D-251 execution statuses that represent a genuine LEVEL-2 DSP/measurement
# failure (as opposed to a deliberate, correct "nothing to do" outcome).
_LEVEL2_FAILURE_STATUSES = frozenset({
    "MEASUREMENT_REFERENCE_MISSING", "INVALID_GAIN", "PEAK_SAFETY_UNVERIFIED",
    "FFMPEG_FAILURE", "POST_VERIFY_OUT_OF_POLICY", "OUTPUT_MISSING", "OTHER",
})


@dataclass(frozen=True)
class CompositionInput:
    """STAGE 2: the composition's own input contract -- everything it
    needs is passed explicitly, nothing is read from hidden global state."""

    segments: tuple[RenderSegment, ...]
    plan: AudioFinishingPlan
    output_dir: str
    render_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CompositionRecord:
    """STAGE 9: the end-to-end provenance record. Every sub-stage's own
    real result object is embedded verbatim (never re-summarized into a
    lossy string) -- no raw ffmpeg stderr appears anywhere as the
    canonical contract; each embedded record already carries its own
    structured `provenance` for that."""

    composition_id: str
    input_segment_ids: tuple[str, ...]
    policy_version: str
    plan_status: str

    level1_results: tuple[SegmentGainAdjustmentResult, ...]
    render_output_path: str | None
    whole_video_execution_record: AudioFinishingExecutionRecord | None
    post_measurement: object | None  # the real AudioFinishingMeasurement embedded in `verification`
    verification: ExecutionVerificationResult | None

    composition_status: str
    errors: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict = field(default_factory=dict)


def compute_composition_id(plan: AudioFinishingPlan, segments: tuple[RenderSegment, ...]) -> str:
    """STAGE 10/14: a pure, deterministic function of the plan's identity
    and the exact segment states being composed -- no mutable global
    state, matching D-251's `compute_execution_id` and D-252's
    `compute_adjustment_application_id` pattern exactly. Two calls with
    the same plan and the same segment states always agree; any segment
    (or plan) difference always changes the id."""
    payload = {
        "policy_version": plan.policy_version,
        "plan_status": plan.plan_status,
        "whole_video_state": plan.whole_video_state,
        "whole_video_integrated_loudness_lufs": plan.whole_video_integrated_loudness_lufs,
        "requested_whole_video_gain_db": plan.requested_whole_video_gain_db,
        "authorized_whole_video_gain_db": plan.authorized_whole_video_gain_db,
        "limiter_authorized": plan.limiter_authorized,
        "adjacent_adjustments": [
            (a.left_segment_id, a.right_segment_id, a.gain_state, a.authorized_correction_db, a.direction)
            for a in plan.adjacent_take_adjustments
        ],
        "segments": [
            (s.clip_id, s.source_path, s.start, s.end, s.audio_volume) for s in segments
        ],
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def _verify_final_output(plan: AudioFinishingPlan, final_output_path: str, level2_record: AudioFinishingExecutionRecord) -> ExecutionVerificationResult:
    """Re-measure and verify the REAL final deliverable file, reusing
    D-251's own `_verify_execution` verbatim, even when Level 2 itself
    wrote no separate output (a synthetic pass-through record pointing at
    the real render output is used so the same, single verification
    logic runs regardless of which stage actually produced the final
    file)."""
    record_for_verification = dataclasses.replace(level2_record, output_path=final_output_path)
    return _verify_execution(plan, record_for_verification)


def run_audio_finishing_composition(
    composition_input: CompositionInput,
    *,
    already_applied_level1_ids: frozenset[str] = frozenset(),
    existing_level2_record: AudioFinishingExecutionRecord | None = None,
) -> CompositionRecord:
    """STAGE 1/2-8: the one end-to-end entry point. Deterministic, no
    policy recomputation anywhere -- every gain value applied anywhere in
    this chain comes from `composition_input.plan`, already computed by
    D-249 before this function is ever called."""
    plan = composition_input.plan
    segments = composition_input.segments
    composition_id = compute_composition_id(plan, segments)
    input_segment_ids = tuple(s.clip_id for s in segments)
    common_kwargs = dict(
        composition_id=composition_id, input_segment_ids=input_segment_ids,
        policy_version=plan.policy_version, plan_status=plan.plan_status,
    )

    # ---- STAGE 3: LEVEL 1 (adjacent-take) -- always before render. -----
    if not segments:
        return CompositionRecord(
            **common_kwargs, level1_results=(), render_output_path=None,
            whole_video_execution_record=None, post_measurement=None, verification=None,
            composition_status=COMPOSITION_STATUS_LEVEL1_BLOCKED,
            errors=("no segments supplied -- nothing to compose",), provenance={},
        )

    try:
        corrected_segments, level1_results = apply_adjacent_take_adjustments(
            plan, segments, already_applied_ids=already_applied_level1_ids,
        )
    except Exception as exc:  # pragma: no cover - defensive; apply_adjacent_take_adjustments never raises normally
        return CompositionRecord(
            **common_kwargs, level1_results=(), render_output_path=None,
            whole_video_execution_record=None, post_measurement=None, verification=None,
            composition_status=COMPOSITION_STATUS_LEVEL1_BLOCKED,
            errors=(f"unexpected exception during Level-1 application: {exc}",), provenance={},
        )

    # ---- STAGE 4: RENDER -- the existing, unmodified live renderer. ----
    os.makedirs(composition_input.output_dir, exist_ok=True)
    render_output_path = os.path.join(composition_input.output_dir, f"{composition_id}_render.mp4")
    try:
        render_module.render_preview(corrected_segments, render_output_path, **composition_input.render_kwargs)
    except Exception as exc:
        return CompositionRecord(
            **common_kwargs, level1_results=level1_results, render_output_path=None,
            whole_video_execution_record=None, post_measurement=None, verification=None,
            composition_status=COMPOSITION_STATUS_RENDER_FAILED,
            errors=(f"render failed: {exc}",), provenance={},
        )
    if not os.path.exists(render_output_path):
        return CompositionRecord(
            **common_kwargs, level1_results=level1_results, render_output_path=None,
            whole_video_execution_record=None, post_measurement=None, verification=None,
            composition_status=COMPOSITION_STATUS_RENDER_FAILED,
            errors=("render reported success but no output file exists",), provenance={},
        )

    # ---- STAGE 5/6: LEVEL 2 (whole-video gain + limiter if authorized).
    whole_video_output_path = os.path.join(composition_input.output_dir, f"{composition_id}_finished.mp4")
    level2_record, level2_verification = execute_audio_finishing_plan(
        plan, render_output_path, whole_video_output_path, existing_record=existing_level2_record,
    )

    if level2_record.execution_status in _LEVEL2_FAILURE_STATUSES:
        return CompositionRecord(
            **common_kwargs, level1_results=level1_results, render_output_path=render_output_path,
            whole_video_execution_record=level2_record, post_measurement=None, verification=None,
            composition_status=COMPOSITION_STATUS_LEVEL2_FAILED,
            errors=level2_record.errors, provenance={},
        )

    # ---- STAGE 7/8: POST-MEASUREMENT + VERIFICATION of the REAL final
    # deliverable, whichever stage actually produced it.
    if level2_record.execution_status == EXECUTION_STATUS_SUCCESS:
        final_output_path = level2_record.output_path
        verification = level2_verification
        if verification is None:
            # D-251's own idempotence shortcut (a matching `existing_record`
            # already succeeded) returns `(existing_record, None)` -- no
            # new DSP ran, but the real final file still exists and must
            # still be verified for real, never silently skipped.
            verification = _verify_final_output(plan, final_output_path, level2_record)
    else:
        # NO_ACTION_NEEDED / PLAN_NOT_EXECUTABLE -- no separate Level-2
        # file exists; the render's own output IS the deliverable, still
        # verified for real (never skipped).
        final_output_path = render_output_path
        verification = _verify_final_output(plan, final_output_path, level2_record)

    level1_applied = any(r.execution_status == ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED for r in level1_results)
    level2_applied = level2_record.execution_status == EXECUTION_STATUS_SUCCESS

    if verification.verification_status == VERIFICATION_STATUS_TECHNICAL_FAILURE:
        composition_status = COMPOSITION_STATUS_VERIFY_FAILED
    elif verification.verification_status == VERIFICATION_STATUS_PASS:
        composition_status = COMPOSITION_STATUS_SUCCESS if (level1_applied or level2_applied) else COMPOSITION_STATUS_NO_CHANGE
    else:
        # VERIFICATION_STATUS_POLICY_OUT_OF_RANGE or VERIFICATION_STATUS_PARTIAL
        composition_status = COMPOSITION_STATUS_PARTIAL

    return CompositionRecord(
        **common_kwargs, level1_results=level1_results, render_output_path=render_output_path,
        whole_video_execution_record=level2_record, post_measurement=verification, verification=verification,
        composition_status=composition_status, errors=(), provenance={"final_output_path": final_output_path},
    )
