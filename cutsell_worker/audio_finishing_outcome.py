"""Audio Finishing OUTCOME / PRODUCT-STATE CONTRACT (D-256).

D-247 built MEASUREMENT. D-249 built POLICY + PLAN. D-251/D-252 built the
whole-video and adjacent-take EXECUTOR. D-253 built the COMPOSITION
record tying Level 1 + Level 2 together. D-254C proved the whole chain
against real Video00 media and produced an honest, non-fabricated
"safe but out of policy" result for an extreme (-32.2 LUFS) source.
D-255 designed -- but did not implement -- the product-facing meaning of
that result: a source can be extreme enough that the canonical ±6dB
envelope cannot fully correct it, and that is a valid, safe V1 outcome,
not a defect.

This module is the first gate that turns the existing, already-real
measurement/plan/execution/verification records into a single,
deterministic, typed OUTCOME the rest of the product (API responses,
analytics, support tooling, a future rescue pipeline) can read without
ever re-deriving policy logic itself or collapsing four independent
questions into one misleading boolean:

    1. EXECUTION  -- did the DSP step itself run and succeed?
    2. POLICY     -- did the output actually land inside the canonical
                     loudness/peak band?
    3. EXPORT     -- is the file safe to deliver regardless of #2?
    4. RESCUE     -- would a (not-yet-built) more aggressive correction
                     pipeline help this source? (always False in V1 --
                     no such pipeline exists; this axis is reserved.)

## Scope discipline (binding, D-256 STAGE 1 / D-091 scope banner)

NO DSP CHANGE. NO NUMERIC/THRESHOLD/POLICY CHANGE. NO RAW. NO PROVIDER
CALL. This module:

- reads `AudioFinishingPlan` (D-249), `AudioFinishingExecutionRecord` +
  `ExecutionVerificationResult` (D-251), and an optional technical QC
  status (D-inherited `post_render_media_qc`/`live_render_qc` result);
- NEVER calls ffmpeg, `measure_audio`, `evaluate_peak_safety`,
  `evaluate_whole_video_loudness`, or `evaluate_adjacent_take_continuity`
  itself -- it is a pure function of already-computed, already-real
  records;
- NEVER imports or references any of the six canonical numeric
  constants directly (POLICY_VERSION, the loudness target/tolerance,
  the ±6dB envelope, the true-peak ceiling, the mismatch threshold, the
  minimum reliable window) -- it reads `plan.policy_version`,
  `plan.target_loudness_lufs`, `plan.true_peak_ceiling_dbtp`, etc. off
  the already-decided plan, never re-declares or re-derives a number;
- does not touch `render.py`, any renderer/QC/Pacing/Audio-Join/Freeze
  authority, or any DSP filter-chain code (verified structurally by this
  gate's own test suite -- see `tests/test_cutsell_d256_audio_finishing_outcome.py`).

## Design basis

D-255's four-axis canonical status model is the direct design contract
for this module: `EXECUTION_STATUS` (existing D-251 vocabulary, reused
verbatim), `POLICY_COMPLETE` (new), `EXPORT_ALLOWED` (new, independent
of policy completeness), `SOURCE_RESCUE_REQUIRED` (new, reserved,
always False in V1 -- no rescue pipeline exists to require).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .audio_finishing_executor import (
    EXECUTION_STATUS_FFMPEG_FAILURE,
    EXECUTION_STATUS_INVALID_GAIN,
    EXECUTION_STATUS_MEASUREMENT_REFERENCE_MISSING,
    EXECUTION_STATUS_NO_ACTION_NEEDED,
    EXECUTION_STATUS_OTHER,
    EXECUTION_STATUS_OUTPUT_MISSING,
    EXECUTION_STATUS_PEAK_SAFETY_UNVERIFIED,
    EXECUTION_STATUS_PLAN_NOT_EXECUTABLE,
    EXECUTION_STATUS_POST_VERIFY_OUT_OF_POLICY,
    EXECUTION_STATUS_SUCCESS,
    VERIFICATION_STATUS_PASS,
    VERIFICATION_STATUS_TECHNICAL_FAILURE,
    AudioFinishingExecutionRecord,
    ExecutionVerificationResult,
)
from .audio_finishing_measurement import (
    MEASUREMENT_STATUS_COMPLETE,
    AudioFinishingMeasurement,
)
from .audio_finishing_policy import (
    GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE,
    GAIN_STATE_BLOCKED_CLIPPING,
    GAIN_STATE_BLOCKED_PEAK_RISK,
    GAIN_STATE_BLOCKED_SILENCE,
    GAIN_STATE_CORRECTION_ALLOWED,
    GAIN_STATE_CORRECTION_LIMITED,
    GAIN_STATE_NO_CHANGE_NEEDED,
    AudioFinishingPlan,
)

OUTCOME_VERSION = "audio_finishing_outcome.v1"

# ---------------------------------------------------------------------------
# STAGE 3: source classification -- derived ONLY from the existing ±6dB
# envelope already decided by `audio_finishing_policy.py`. No new
# threshold is introduced here.
# ---------------------------------------------------------------------------

SOURCE_CLASS_NORMAL_CORRECTABLE = "NORMAL_CORRECTABLE"
SOURCE_CLASS_EXTREME_UNDER_LEVEL = "EXTREME_UNDER_LEVEL"
SOURCE_CLASS_EXTREME_OVER_LEVEL = "EXTREME_OVER_LEVEL"
SOURCE_CLASS_ABSTAINED = "ABSTAINED"
SOURCE_CLASS_BLOCKED_SAFETY = "BLOCKED_SAFETY"
SOURCE_CLASS_UNKNOWN = "UNKNOWN"

_BLOCKED_GAIN_STATES = (
    GAIN_STATE_BLOCKED_PEAK_RISK,
    GAIN_STATE_BLOCKED_SILENCE,
    GAIN_STATE_BLOCKED_CLIPPING,
)

# ---------------------------------------------------------------------------
# STAGE 4: product state (six states; SOURCE_RESCUE_REQUIRED is reserved
# and never produced by this V1 module).
# ---------------------------------------------------------------------------

PRODUCT_STATE_COMPLETE = "AUDIO_FINISHING_COMPLETE"
PRODUCT_STATE_PARTIAL_TOO_QUIET = "AUDIO_FINISHING_PARTIAL_SOURCE_TOO_QUIET"
PRODUCT_STATE_PARTIAL_TOO_LOUD = "AUDIO_FINISHING_PARTIAL_SOURCE_TOO_LOUD"
PRODUCT_STATE_ABSTAINED = "AUDIO_FINISHING_ABSTAINED"
PRODUCT_STATE_BLOCKED_SAFETY = "AUDIO_FINISHING_BLOCKED_SAFETY"
PRODUCT_STATE_UNKNOWN = "AUDIO_FINISHING_UNKNOWN"
# Reserved for a future rescue pipeline. NEVER assigned by this module in
# V1 -- there is no rescue pipeline to require. Kept here so the product
# vocabulary is stable the day one exists, per D-255 Stage 5/D-256 Stage 4.
PRODUCT_STATE_SOURCE_RESCUE_REQUIRED = "AUDIO_FINISHING_SOURCE_RESCUE_REQUIRED"

# ---------------------------------------------------------------------------
# STAGE 4 (cont'd): execution-state axis. Reuses D-251's own
# EXECUTION_STATUS_* vocabulary as its input; this axis is the coarser,
# product-facing summary of it.
# ---------------------------------------------------------------------------

EXECUTION_STATE_SUCCEEDED = "EXECUTION_SUCCEEDED"
EXECUTION_STATE_NO_ACTION_NEEDED = "EXECUTION_NO_ACTION_NEEDED"
EXECUTION_STATE_FAILED = "EXECUTION_FAILED"
EXECUTION_STATE_NOT_RUN = "EXECUTION_NOT_RUN"

_EXECUTION_STATE_MAP = {
    EXECUTION_STATUS_SUCCESS: EXECUTION_STATE_SUCCEEDED,
    EXECUTION_STATUS_NO_ACTION_NEEDED: EXECUTION_STATE_NO_ACTION_NEEDED,
    # ABSTAIN/BLOCKED plans correctly produce PLAN_NOT_EXECUTABLE: no
    # ffmpeg call ever occurs. That is a deliberate, correct non-action,
    # never a defect -- it must map to NOT_RUN, never FAILED.
    EXECUTION_STATUS_PLAN_NOT_EXECUTABLE: EXECUTION_STATE_NOT_RUN,
    EXECUTION_STATUS_MEASUREMENT_REFERENCE_MISSING: EXECUTION_STATE_FAILED,
    EXECUTION_STATUS_INVALID_GAIN: EXECUTION_STATE_FAILED,
    EXECUTION_STATUS_PEAK_SAFETY_UNVERIFIED: EXECUTION_STATE_FAILED,
    EXECUTION_STATUS_FFMPEG_FAILURE: EXECUTION_STATE_FAILED,
    EXECUTION_STATUS_POST_VERIFY_OUT_OF_POLICY: EXECUTION_STATE_FAILED,
    EXECUTION_STATUS_OUTPUT_MISSING: EXECUTION_STATE_FAILED,
    EXECUTION_STATUS_OTHER: EXECUTION_STATE_FAILED,
}

# ---------------------------------------------------------------------------
# STAGE 15: warnings contract -- stable machine states only, no UI copy.
# ---------------------------------------------------------------------------

WARNING_SOURCE_TOO_QUIET = "SOURCE_TOO_QUIET_FOR_FULL_AUTOMATIC_FINISHING"
WARNING_SOURCE_TOO_LOUD = "SOURCE_TOO_LOUD_FOR_FULL_AUTOMATIC_FINISHING"
WARNING_FINISHING_PARTIAL = "FINISHING_PARTIAL"
WARNING_FINISHING_ABSTAINED = "FINISHING_ABSTAINED"
WARNING_FINISHING_BLOCKED_SAFETY = "FINISHING_BLOCKED_SAFETY"
WARNING_MEASUREMENT_INCOMPLETE = "MEASUREMENT_INCOMPLETE"
WARNING_PEAK_SAFETY_UNVERIFIED = "PEAK_SAFETY_UNVERIFIED"
# Double-finishing-firewall warnings (STAGE 17/19).
WARNING_ALREADY_FINISHED = "ALREADY_FINISHED_SAME_POLICY"
WARNING_FINISHED_OUTPUT_AS_SOURCE = "FINISHED_OUTPUT_SUPPLIED_AS_NEW_SOURCE"

# ---------------------------------------------------------------------------
# STAGE 19: re-finishing decision vocabulary (double-finishing firewall).
# ---------------------------------------------------------------------------

REFINISH_DECISION_NEW_SOURCE = "NEW_SOURCE"
REFINISH_DECISION_ALREADY_FINISHED_SAME_POLICY = "SAME_SOURCE_SAME_POLICY_ALREADY_FINISHED"
REFINISH_DECISION_SAME_SOURCE_NEW_POLICY_VERSION = "SAME_SOURCE_NEW_POLICY_VERSION"
REFINISH_DECISION_FINISHED_OUTPUT_AS_NEW_SOURCE = "FINISHED_OUTPUT_SUPPLIED_AS_NEW_SOURCE"


def classify_source(plan: AudioFinishingPlan) -> str:
    """STAGE 3. Pure function of `plan.whole_video_state` (and, for the
    CORRECTION_LIMITED case, the sign of the requested gain the policy
    layer already computed) -- introduces no new threshold. Mirrors
    D-255's audited `CORRECTION_LIMITED` semantic exactly: a positive
    requested gain that got clamped means the source was too quiet for
    full automatic correction (EXTREME_UNDER_LEVEL); a negative one
    means it was too loud (EXTREME_OVER_LEVEL)."""
    state = plan.whole_video_state
    if state in (GAIN_STATE_NO_CHANGE_NEEDED, GAIN_STATE_CORRECTION_ALLOWED):
        return SOURCE_CLASS_NORMAL_CORRECTABLE
    if state == GAIN_STATE_CORRECTION_LIMITED:
        requested = plan.requested_whole_video_gain_db
        if requested is not None and requested > 0:
            return SOURCE_CLASS_EXTREME_UNDER_LEVEL
        if requested is not None and requested < 0:
            return SOURCE_CLASS_EXTREME_OVER_LEVEL
        return SOURCE_CLASS_UNKNOWN
    if state == GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE:
        return SOURCE_CLASS_ABSTAINED
    if state in _BLOCKED_GAIN_STATES:
        return SOURCE_CLASS_BLOCKED_SAFETY
    return SOURCE_CLASS_UNKNOWN


def classify_execution(execution_record: AudioFinishingExecutionRecord | None) -> str:
    """STAGE 4. `None` (no execution attempted at all, e.g. an
    ABSTAIN/BLOCKED plan that never reached the executor) is NOT_RUN,
    matching `execute_audio_finishing_plan`'s own PLAN_NOT_EXECUTABLE
    mapping -- the two paths a caller might take (skip calling the
    executor entirely, or call it and receive PLAN_NOT_EXECUTABLE back)
    must agree."""
    if execution_record is None:
        return EXECUTION_STATE_NOT_RUN
    return _EXECUTION_STATE_MAP.get(execution_record.execution_status, EXECUTION_STATE_FAILED)


def compute_policy_complete(
    execution_state: str,
    verification: ExecutionVerificationResult | None,
    plan: AudioFinishingPlan | None,
) -> bool:
    """STAGE 5. Never fabricated: a verdict is only ever read off a real
    `ExecutionVerificationResult`, except for the one case where
    `execute_audio_finishing_plan` correctly never produces one --
    NO_ACTION_NEEDED, where no DSP ran because the pre-measurement was
    already inside the canonical band. That case reuses the plan's own
    already-real judgment (`GAIN_STATE_NO_CHANGE_NEEDED`) rather than
    treating a legitimately-absent verification as a failure."""
    if execution_state == EXECUTION_STATE_SUCCEEDED:
        return verification is not None and verification.verification_status == VERIFICATION_STATUS_PASS
    if execution_state == EXECUTION_STATE_NO_ACTION_NEEDED:
        return plan is not None and plan.whole_video_state == GAIN_STATE_NO_CHANGE_NEEDED
    return False


def compute_export_allowed(
    execution_state: str,
    verification: ExecutionVerificationResult | None,
    technical_qc_status: str | None,
) -> bool:
    """STAGE 6. Deliberately independent of `policy_complete`: a source
    that is safely, verifiably out of policy (D-254C's real
    -32.2->-26.2 LUFS result) is still exportable -- Audio Finishing
    policy incompleteness is never itself a reason to withhold the file.
    Only a genuine technical defect (failed technical QC, a
    TECHNICAL_FAILURE verification, or the DSP step itself failing)
    blocks export."""
    if technical_qc_status is not None and technical_qc_status != "PASS":
        return False
    if verification is not None and verification.verification_status == VERIFICATION_STATUS_TECHNICAL_FAILURE:
        return False
    if execution_state == EXECUTION_STATE_FAILED:
        return False
    return True


def compute_source_rescue_required() -> bool:
    """STAGE 7. Always False in V1 -- no rescue pipeline exists for this
    module to route to. Kept as an explicit function (rather than a bare
    constant inlined at call sites) so a future rescue-pipeline gate has
    exactly one place to change this axis's derivation."""
    return False


def compute_product_state(source_class: str, execution_state: str, policy_complete: bool) -> str:
    """STAGE 4. Ordering matters: ABSTAINED/BLOCKED_SAFETY are checked
    first because their own NOT_RUN execution state is definitionally
    correct (no DSP should run for them) and must not be swept into the
    generic FAILED/NOT_RUN guard below, which instead exists to catch a
    genuine defect (the DSP failed, or unexpectedly never ran) for a
    source that SHOULD have been actionable."""
    if source_class == SOURCE_CLASS_ABSTAINED:
        return PRODUCT_STATE_ABSTAINED
    if source_class == SOURCE_CLASS_BLOCKED_SAFETY:
        return PRODUCT_STATE_BLOCKED_SAFETY
    if execution_state in (EXECUTION_STATE_FAILED, EXECUTION_STATE_NOT_RUN):
        return PRODUCT_STATE_UNKNOWN
    if source_class == SOURCE_CLASS_EXTREME_UNDER_LEVEL:
        return PRODUCT_STATE_PARTIAL_TOO_QUIET
    if source_class == SOURCE_CLASS_EXTREME_OVER_LEVEL:
        return PRODUCT_STATE_PARTIAL_TOO_LOUD
    if source_class == SOURCE_CLASS_NORMAL_CORRECTABLE:
        return PRODUCT_STATE_COMPLETE if policy_complete else PRODUCT_STATE_UNKNOWN
    return PRODUCT_STATE_UNKNOWN


def compute_warnings(
    source_class: str,
    measurement: AudioFinishingMeasurement | None,
    verification: ExecutionVerificationResult | None,
) -> tuple[str, ...]:
    """STAGE 15. Stable machine states only -- no UI copy. Order is
    deterministic (source-class-driven warnings first, then measurement/
    verification-derived ones) so the same inputs always yield the same
    tuple, byte-for-byte."""
    warnings: list[str] = []
    if source_class == SOURCE_CLASS_EXTREME_UNDER_LEVEL:
        warnings.append(WARNING_SOURCE_TOO_QUIET)
        warnings.append(WARNING_FINISHING_PARTIAL)
    elif source_class == SOURCE_CLASS_EXTREME_OVER_LEVEL:
        warnings.append(WARNING_SOURCE_TOO_LOUD)
        warnings.append(WARNING_FINISHING_PARTIAL)
    elif source_class == SOURCE_CLASS_ABSTAINED:
        warnings.append(WARNING_FINISHING_ABSTAINED)
    elif source_class == SOURCE_CLASS_BLOCKED_SAFETY:
        warnings.append(WARNING_FINISHING_BLOCKED_SAFETY)
    if measurement is not None and measurement.measurement_status != MEASUREMENT_STATUS_COMPLETE:
        warnings.append(WARNING_MEASUREMENT_INCOMPLETE)
    if verification is not None and verification.true_peak_within_ceiling is None:
        warnings.append(WARNING_PEAK_SAFETY_UNVERIFIED)
    return tuple(warnings)


# ---------------------------------------------------------------------------
# STAGE 18: finishing identity (the double-finishing firewall's key).
# ---------------------------------------------------------------------------

def compute_finishing_identity(
    source_sha256: str | None,
    plan: AudioFinishingPlan,
    execution_record: AudioFinishingExecutionRecord | None,
    output_sha256: str | None,
) -> str:
    """STAGE 18. A pure, deterministic function of the ORIGINAL source's
    content identity, the policy version, the exact authorized plan, the
    execution's own identity (when one ran), and the output's content
    identity when known -- never the filename or path alone, matching
    D-251/D-252/D-253's `compute_execution_id` / `compute_adjustment_
    application_id` / `compute_composition_id` pattern exactly. Two
    finishing attempts against byte-identical source content, the same
    policy version, and the same authorized plan always agree; any
    difference in any of those always changes the identity."""
    payload = {
        "source_sha256": source_sha256,
        "policy_version": plan.policy_version,
        "whole_video_state": plan.whole_video_state,
        "authorized_whole_video_gain_db": plan.authorized_whole_video_gain_db,
        "limiter_authorized": plan.limiter_authorized,
        "true_peak_ceiling_dbtp": plan.true_peak_ceiling_dbtp,
        "execution_id": execution_record.execution_id if execution_record is not None else None,
        "output_sha256": output_sha256,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def decide_refinishing(
    candidate_source_sha256: str | None,
    candidate_policy_version: str | None,
    previous_finishing_identity: str | None,
    previous_source_sha256: str | None,
    previous_policy_version: str | None,
    previous_output_sha256: str | None,
) -> str:
    """STAGE 19. Content-identity-based only -- never filename-based, per
    the directive's own binding constraint. `previous_*` describes a
    finishing outcome already on record (if any) for either this exact
    candidate source or an ancestor of it; `previous_output_sha256` is
    the prior finishing's OWN output, checked so that a caller who
    mistakenly re-submits an already-finished file as if it were a fresh
    source is still caught (fail-closed) even with no other record
    available.

    - No prior record at all, and the candidate isn't a known finished
      output -> NEW_SOURCE.
    - The candidate's own content hash matches a PRIOR FINISHING'S
      OUTPUT hash -> FINISHED_OUTPUT_SUPPLIED_AS_NEW_SOURCE (fail-closed:
      this is caught even when the candidate's "source" hash is the
      thing being compared, because a finished output handed back in as
      a new source IS, by content, that prior output).
    - Same source content + same policy version as a prior finishing
      -> SAME_SOURCE_SAME_POLICY_ALREADY_FINISHED (the firewall itself:
      the caller must not re-run DSP for this pair).
    - Same source content + a different policy version
      -> SAME_SOURCE_NEW_POLICY_VERSION (a deliberate, policy-version-
      gated re-finish is allowed; this is not double-finishing).
    """
    if (
        previous_output_sha256 is not None
        and candidate_source_sha256 is not None
        and candidate_source_sha256 == previous_output_sha256
    ):
        return REFINISH_DECISION_FINISHED_OUTPUT_AS_NEW_SOURCE
    if (
        previous_source_sha256 is not None
        and candidate_source_sha256 is not None
        and candidate_source_sha256 == previous_source_sha256
    ):
        if candidate_policy_version == previous_policy_version:
            return REFINISH_DECISION_ALREADY_FINISHED_SAME_POLICY
        return REFINISH_DECISION_SAME_SOURCE_NEW_POLICY_VERSION
    return REFINISH_DECISION_NEW_SOURCE


# ---------------------------------------------------------------------------
# STAGE 2: the outcome structure itself.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioFinishingOutcome:
    """The single, deterministic, typed Audio Finishing product outcome.
    Never collapses execution/policy/export/rescue into one boolean; a
    consumer that only reads `product_state` still gets a correct,
    non-misleading summary, but every finer axis remains independently
    inspectable."""

    outcome_version: str

    execution_status: str  # EXECUTION_STATE_* (this module's own axis)
    policy_status: str  # the underlying AudioFinishingPlan.plan_status (D-249, verbatim)
    product_state: str  # PRODUCT_STATE_*
    export_allowed: bool
    policy_complete: bool
    source_rescue_required: bool  # always False in V1 -- reserved axis

    technical_qc_status: str | None
    source_classification: str  # SOURCE_CLASS_*

    measurement_reference: AudioFinishingMeasurement | None
    plan_reference: AudioFinishingPlan
    execution_reference: AudioFinishingExecutionRecord | None
    verification_reference: ExecutionVerificationResult | None

    before_integrated_loudness_lufs: float | None
    after_integrated_loudness_lufs: float | None
    target_loudness_lufs: float
    true_peak_ceiling_dbtp: float
    requested_gain_db: float | None
    authorized_gain_db: float | None

    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    provenance: dict = field(default_factory=dict)

    finishing_identity: str = ""
    finishing_already_applied: bool = False


def build_audio_finishing_outcome(
    plan: AudioFinishingPlan,
    execution_record: AudioFinishingExecutionRecord | None = None,
    verification: ExecutionVerificationResult | None = None,
    technical_qc_status: str | None = None,
    source_sha256: str | None = None,
    output_sha256: str | None = None,
    previous_finishing_identity: str | None = None,
    previous_source_sha256: str | None = None,
    previous_policy_version: str | None = None,
    previous_output_sha256: str | None = None,
    provenance: dict | None = None,
) -> AudioFinishingOutcome:
    """The single owner-module entry point (STAGE 1's binding: this logic
    lives HERE, not in render.py, the measurement module, the DSP
    executor, or any existing QC authority). Ties together every helper
    above into one `AudioFinishingOutcome`. Pure: makes no DSP call, no
    ffmpeg invocation, no measurement call, no RAW/provider call -- every
    input is an already-computed, already-real record the caller
    supplies."""
    execution_state = classify_execution(execution_record)
    source_class = classify_source(plan)
    policy_complete = compute_policy_complete(execution_state, verification, plan)
    export_allowed = compute_export_allowed(execution_state, verification, technical_qc_status)
    source_rescue_required = compute_source_rescue_required()
    product_state = compute_product_state(source_class, execution_state, policy_complete)
    measurement = plan.measurement_reference
    warnings = compute_warnings(source_class, measurement, verification)

    finishing_identity = compute_finishing_identity(
        source_sha256=source_sha256,
        plan=plan,
        execution_record=execution_record,
        output_sha256=output_sha256,
    )
    refinish_decision = decide_refinishing(
        candidate_source_sha256=source_sha256,
        candidate_policy_version=plan.policy_version,
        previous_finishing_identity=previous_finishing_identity,
        previous_source_sha256=previous_source_sha256,
        previous_policy_version=previous_policy_version,
        previous_output_sha256=previous_output_sha256,
    )
    finishing_already_applied = refinish_decision == REFINISH_DECISION_ALREADY_FINISHED_SAME_POLICY
    if finishing_already_applied:
        warnings = warnings + (WARNING_ALREADY_FINISHED,)
    elif refinish_decision == REFINISH_DECISION_FINISHED_OUTPUT_AS_NEW_SOURCE:
        warnings = warnings + (WARNING_FINISHED_OUTPUT_AS_SOURCE,)

    before_lufs = measurement.integrated_loudness_lufs if measurement is not None else None
    after_lufs = verification.integrated_loudness_lufs if verification is not None else None

    merged_provenance = dict(provenance) if provenance else {}
    merged_provenance.setdefault("refinish_decision", refinish_decision)

    return AudioFinishingOutcome(
        outcome_version=OUTCOME_VERSION,
        execution_status=execution_state,
        policy_status=plan.plan_status,
        product_state=product_state,
        export_allowed=export_allowed,
        policy_complete=policy_complete,
        source_rescue_required=source_rescue_required,
        technical_qc_status=technical_qc_status,
        source_classification=source_class,
        measurement_reference=measurement,
        plan_reference=plan,
        execution_reference=execution_record,
        verification_reference=verification,
        before_integrated_loudness_lufs=before_lufs,
        after_integrated_loudness_lufs=after_lufs,
        target_loudness_lufs=plan.target_loudness_lufs,
        true_peak_ceiling_dbtp=plan.true_peak_ceiling_dbtp,
        requested_gain_db=plan.requested_whole_video_gain_db,
        authorized_gain_db=plan.authorized_whole_video_gain_db,
        reasons=tuple(plan.reasons),
        warnings=warnings,
        provenance=merged_provenance,
        finishing_identity=finishing_identity,
        finishing_already_applied=finishing_already_applied,
    )
