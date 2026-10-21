"""Audio Finishing POLICY + PLAN GENERATION -- V1 canonical (D-249).

D-247 built real MEASUREMENT. D-248 designed POLICY without canonizing any
number. This module is the first gate that actually ENCODES canonical V1
numeric policy and turns real `AudioFinishingMeasurement`s into a
structured `AudioFinishingPlan`.

    MEASUREMENT (D-247, unchanged)
        -> POLICY + PLAN (this module)
            -> FUTURE EXECUTOR (not built -- D-250+)
                -> POST-RENDER VERIFICATION (unchanged -- Stage 16)

## What IS built here

Pure Python decision logic over already-computed `AudioFinishingMeasurement`
objects (from `audio_finishing_measurement.py`). Nothing in this module
runs a subprocess, imports `subprocess`, or references `ffmpeg`/`ffprobe`
-- it cannot execute a correction even by accident, because it has no
mechanism to touch media at all. Every decision function returns a frozen
dataclass describing what SHOULD happen and why; nothing here applies a
gain, a limiter, a normalization pass, or any other DSP operation.

## Canonical V1 numeric policy (Product-Owner-approved, D-249)

These six values were `PRODUCT_OWNER_DECISION_REQUIRED` in D-248 and were
approved by the Product Owner for this gate. They are V1 PRODUCT POLICY
for professional Talking Head UGC / TikTok Shop / creator video --
explicitly NOT a universal audio-engineering truth, and NOT broadcast
mastering practice. A future product decision could revise them; nothing
below should be read as an audio-engineering constant.

- `TARGET_INTEGRATED_LOUDNESS_LUFS = -14.0`
- `LOUDNESS_TOLERANCE_LU = 1.0` (acceptable range: -15.0 to -13.0 LUFS)
- `ADJACENT_TAKE_MISMATCH_THRESHOLD_LU = 2.0`
- `MAX_AUTOMATIC_GAIN_CORRECTION_DB = 6.0`
- `TRUE_PEAK_CEILING_DBTP = -1.0`
- `MINIMUM_RELIABLE_LOUDNESS_WINDOW_SEC = 1.5`

No numeric value beyond these six is introduced by this module. Every
threshold-shaped decision below (peak-ceiling comparisons, the
correction/limited split, the abstain-on-short-window rule) is expressed
purely in terms of these six constants -- never a new invented number.

## What is honestly NOT built here, and why

No DSP. `generate_audio_finishing_plan` and every helper below return data
-- never a rendered file, never a modified sample, never an ffmpeg
invocation. `finishing_contract.py` (D-024)'s `FinishingProvider.finish()`
remains the (still unimplemented) future EXECUTOR that would eventually
consume an `AudioFinishingPlan` and turn it into real DSP; this module
does not implement that Protocol and does not call it. Post-render QC
(`post_render_media_qc.py`) is untouched -- this module exposes a plan for
additive diagnostics only (Stage 16), never a new PASS/FAIL input.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .audio_finishing_measurement import (
    CLIPPING_STATUS_CLIPPING_DETECTED,
    AudioFinishingMeasurement,
)

# ---------------------------------------------------------------------------
# STAGE 1: canonical policy owner -- the six Product-Owner-approved V1 values.
# ---------------------------------------------------------------------------

POLICY_VERSION = "V1"

TARGET_INTEGRATED_LOUDNESS_LUFS = -14.0
LOUDNESS_TOLERANCE_LU = 1.0
ADJACENT_TAKE_MISMATCH_THRESHOLD_LU = 2.0
MAX_AUTOMATIC_GAIN_CORRECTION_DB = 6.0
TRUE_PEAK_CEILING_DBTP = -1.0
MINIMUM_RELIABLE_LOUDNESS_WINDOW_SEC = 1.5

ACCEPTABLE_LOUDNESS_LOWER_BOUND_LUFS = TARGET_INTEGRATED_LOUDNESS_LUFS - LOUDNESS_TOLERANCE_LU
ACCEPTABLE_LOUDNESS_UPPER_BOUND_LUFS = TARGET_INTEGRATED_LOUDNESS_LUFS + LOUDNESS_TOLERANCE_LU

# ---------------------------------------------------------------------------
# STAGE 2: bounded policy-state vocabulary.
# ---------------------------------------------------------------------------

GAIN_STATE_NO_CHANGE_NEEDED = "NO_CHANGE_NEEDED"
GAIN_STATE_CORRECTION_ALLOWED = "CORRECTION_ALLOWED"
GAIN_STATE_CORRECTION_LIMITED = "CORRECTION_LIMITED"
GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE = "ABSTAIN_INSUFFICIENT_EVIDENCE"
GAIN_STATE_BLOCKED_PEAK_RISK = "BLOCKED_PEAK_RISK"
GAIN_STATE_BLOCKED_SILENCE = "BLOCKED_SILENCE"
GAIN_STATE_BLOCKED_CLIPPING = "BLOCKED_CLIPPING"
GAIN_STATE_UNKNOWN = "UNKNOWN"

PEAK_EVIDENCE_TRUE_PEAK = "TRUE_PEAK"
PEAK_EVIDENCE_FALLBACK_SAMPLE_PEAK = "FALLBACK_SAMPLE_PEAK"
PEAK_EVIDENCE_UNAVAILABLE = "UNAVAILABLE"

PLAN_STATUS_READY_NO_CHANGE = "READY_NO_CHANGE"
PLAN_STATUS_READY_FOR_CORRECTION = "READY_FOR_CORRECTION"
PLAN_STATUS_READY_WITH_LIMITER = "READY_WITH_LIMITER"
PLAN_STATUS_PARTIAL = "PARTIAL"
PLAN_STATUS_ABSTAIN = "ABSTAIN"
PLAN_STATUS_BLOCKED = "BLOCKED"
PLAN_STATUS_UNKNOWN = "UNKNOWN"

_BLOCKED_STATES = (GAIN_STATE_BLOCKED_PEAK_RISK, GAIN_STATE_BLOCKED_SILENCE, GAIN_STATE_BLOCKED_CLIPPING)
_CORRECTION_STATES = (GAIN_STATE_CORRECTION_ALLOWED, GAIN_STATE_CORRECTION_LIMITED)


# ---------------------------------------------------------------------------
# STAGE 6/7/12/13: peak safety (shared by whole-video and adjacent-take paths).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PeakSafetyDecision:
    peak_evidence_source: str
    existing_peak_dbfs: float | None
    predicted_post_gain_peak_dbfs: float | None
    limiter_needed: bool
    blocked: bool
    reason: str


def evaluate_peak_safety(measurement: AudioFinishingMeasurement, candidate_gain_db: float) -> PeakSafetyDecision:
    """Predict peak-safety consequence of applying `candidate_gain_db` to
    `measurement`. True peak is preferred; sample peak is used only as an
    explicitly-tagged fallback (never silently treated as true peak,
    per D-247/D-248's binding rule). Never mutates anything -- this is a
    pure prediction from already-real measured numbers.

    Negative or zero gain can never increase peak risk (reducing level
    cannot push a peak higher), so it is never blocked here regardless of
    existing peak evidence -- "negative gain may remain plan-able where
    safe" (D-249 Stage 7)."""
    if measurement.true_peak_dbfs is not None:
        evidence_source = PEAK_EVIDENCE_TRUE_PEAK
        existing_peak = measurement.true_peak_dbfs
    elif measurement.sample_peak_dbfs is not None:
        evidence_source = PEAK_EVIDENCE_FALLBACK_SAMPLE_PEAK
        existing_peak = measurement.sample_peak_dbfs
    else:
        evidence_source = PEAK_EVIDENCE_UNAVAILABLE
        existing_peak = None

    if candidate_gain_db <= 0.0:
        predicted = (existing_peak + candidate_gain_db) if existing_peak is not None else None
        return PeakSafetyDecision(
            peak_evidence_source=evidence_source, existing_peak_dbfs=existing_peak,
            predicted_post_gain_peak_dbfs=predicted, limiter_needed=False, blocked=False,
            reason="non-positive gain never increases peak risk",
        )

    if evidence_source == PEAK_EVIDENCE_UNAVAILABLE:
        return PeakSafetyDecision(
            peak_evidence_source=evidence_source, existing_peak_dbfs=None,
            predicted_post_gain_peak_dbfs=None, limiter_needed=False, blocked=True,
            reason="peak evidence unavailable (no true peak, no sample-peak fallback) -- "
                   "positive gain must not be authorized without peak safety evidence",
        )

    if existing_peak >= TRUE_PEAK_CEILING_DBTP:
        return PeakSafetyDecision(
            peak_evidence_source=evidence_source, existing_peak_dbfs=existing_peak,
            predicted_post_gain_peak_dbfs=existing_peak, limiter_needed=False, blocked=True,
            reason=f"existing peak ({existing_peak} dBFS via {evidence_source}) already at/above "
                   f"the {TRUE_PEAK_CEILING_DBTP} dBTP ceiling -- no further positive gain authorized",
        )

    predicted = existing_peak + candidate_gain_db
    if predicted >= TRUE_PEAK_CEILING_DBTP:
        return PeakSafetyDecision(
            peak_evidence_source=evidence_source, existing_peak_dbfs=existing_peak,
            predicted_post_gain_peak_dbfs=predicted, limiter_needed=True, blocked=False,
            reason=f"predicted post-gain peak ({predicted} dBFS) would reach the "
                   f"{TRUE_PEAK_CEILING_DBTP} dBTP ceiling -- limiter authorized as final safety, "
                   "not as the mechanism for the gain decision itself",
        )
    return PeakSafetyDecision(
        peak_evidence_source=evidence_source, existing_peak_dbfs=existing_peak,
        predicted_post_gain_peak_dbfs=predicted, limiter_needed=False, blocked=False,
        reason="predicted post-gain peak stays under the ceiling -- no limiter needed",
    )


# ---------------------------------------------------------------------------
# STAGE 4/5/6/7: whole-video loudness policy.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WholeVideoLoudnessDecision:
    gain_state: str
    integrated_loudness_lufs: float | None
    requested_gain_db: float | None
    authorized_gain_db: float | None
    peak_evidence_source: str
    limiter_needed: bool
    reasons: tuple[str, ...]


def _is_silent(measurement: AudioFinishingMeasurement) -> bool:
    """A KNOWN, confirmed silence condition -- distinct from an unknown/
    missing measurement (Stage 5's "silence firewall" vs Stage 19's
    "abstain on missing evidence"). True digital silence measures a real
    `-inf` integrated loudness (D-247 verified this, never fabricated);
    the reused `check_accidental_silence` result is a second, independent
    real signal for a long silent region."""
    if measurement.integrated_loudness_lufs == float("-inf"):
        return True
    if measurement.silence_result is not None and measurement.silence_result.status == "FAIL":
        return True
    return False


def _has_reliable_window(measurement: AudioFinishingMeasurement) -> bool:
    if measurement.duration_sec is None:
        return False
    return measurement.duration_sec >= MINIMUM_RELIABLE_LOUDNESS_WINDOW_SEC


def evaluate_whole_video_loudness(measurement: AudioFinishingMeasurement) -> WholeVideoLoudnessDecision:
    """STAGE 4-7: the whole-video Level-2 loudness/peak/clipping/silence
    decision for one `AudioFinishingMeasurement`. Pure function -- reads
    the measurement, returns a decision, touches nothing."""
    reasons: list[str] = []

    # STAGE 5: silence firewall -- checked first, wins over every other rule.
    if _is_silent(measurement):
        return WholeVideoLoudnessDecision(
            gain_state=GAIN_STATE_BLOCKED_SILENCE,
            integrated_loudness_lufs=measurement.integrated_loudness_lufs,
            requested_gain_db=None, authorized_gain_db=0.0,
            peak_evidence_source=PEAK_EVIDENCE_UNAVAILABLE, limiter_needed=False,
            reasons=("material is silent/near-silent -- never chase a loudness target on it",),
        )

    # STAGE 10/15: fail-closed on a window too short to trust the integrated read.
    if not _has_reliable_window(measurement) or measurement.integrated_loudness_lufs is None:
        return WholeVideoLoudnessDecision(
            gain_state=GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE,
            integrated_loudness_lufs=measurement.integrated_loudness_lufs,
            requested_gain_db=None, authorized_gain_db=None,
            peak_evidence_source=PEAK_EVIDENCE_UNAVAILABLE, limiter_needed=False,
            reasons=(
                f"duration {measurement.duration_sec!r}s below the "
                f"{MINIMUM_RELIABLE_LOUDNESS_WINDOW_SEC}s minimum reliable window, or loudness "
                "unmeasured -- abstaining rather than acting on an unstable/missing number",
            ),
        )

    measured = measurement.integrated_loudness_lufs
    if ACCEPTABLE_LOUDNESS_LOWER_BOUND_LUFS <= measured <= ACCEPTABLE_LOUDNESS_UPPER_BOUND_LUFS:
        return WholeVideoLoudnessDecision(
            gain_state=GAIN_STATE_NO_CHANGE_NEEDED,
            integrated_loudness_lufs=measured,
            requested_gain_db=0.0, authorized_gain_db=0.0,
            peak_evidence_source=PEAK_EVIDENCE_UNAVAILABLE, limiter_needed=False,
            reasons=(
                f"{measured} LUFS is within the accepted "
                f"[{ACCEPTABLE_LOUDNESS_LOWER_BOUND_LUFS}, {ACCEPTABLE_LOUDNESS_UPPER_BOUND_LUFS}] "
                "LUFS band -- no normalization applied to already-acceptable material",
            ),
        )

    # Requested gain preserved separately from authorized gain (D-249 Stage 4's binding rule):
    # never silently clamp and pretend the target will be reached.
    requested = TARGET_INTEGRATED_LOUDNESS_LUFS - measured
    if abs(requested) <= MAX_AUTOMATIC_GAIN_CORRECTION_DB:
        gain_state = GAIN_STATE_CORRECTION_ALLOWED
        authorized = requested
        reasons.append(
            f"requested {requested:+.2f} dB to reach {TARGET_INTEGRATED_LOUDNESS_LUFS} LUFS is "
            f"within the {MAX_AUTOMATIC_GAIN_CORRECTION_DB} dB automatic envelope"
        )
    else:
        gain_state = GAIN_STATE_CORRECTION_LIMITED
        authorized = MAX_AUTOMATIC_GAIN_CORRECTION_DB if requested > 0 else -MAX_AUTOMATIC_GAIN_CORRECTION_DB
        reasons.append(
            f"requested {requested:+.2f} dB exceeds the {MAX_AUTOMATIC_GAIN_CORRECTION_DB} dB "
            f"automatic envelope -- authorizing only {authorized:+.2f} dB, target will NOT be "
            "fully reached automatically"
        )

    # STAGE 7: clipping proxy blocks further POSITIVE gain only (negative gain stays safe/plan-able).
    if authorized > 0 and measurement.clipping_status == CLIPPING_STATUS_CLIPPING_DETECTED:
        return WholeVideoLoudnessDecision(
            gain_state=GAIN_STATE_BLOCKED_CLIPPING,
            integrated_loudness_lufs=measured,
            requested_gain_db=requested, authorized_gain_db=0.0,
            peak_evidence_source=PEAK_EVIDENCE_UNAVAILABLE, limiter_needed=False,
            reasons=reasons + [
                "clipping proxy indicates full-scale risk -- blocking further positive gain "
                "(this does NOT by itself declare the source categorically clipped, D-249 Stage 7)"
            ],
        )

    # STAGE 6: peak safety, only relevant for a positive move.
    peak_evidence_source = PEAK_EVIDENCE_UNAVAILABLE
    limiter_needed = False
    if authorized > 0:
        peak = evaluate_peak_safety(measurement, authorized)
        peak_evidence_source = peak.peak_evidence_source
        reasons.append(peak.reason)
        if peak.blocked:
            return WholeVideoLoudnessDecision(
                gain_state=GAIN_STATE_BLOCKED_PEAK_RISK,
                integrated_loudness_lufs=measured,
                requested_gain_db=requested, authorized_gain_db=0.0,
                peak_evidence_source=peak_evidence_source, limiter_needed=False,
                reasons=tuple(reasons),
            )
        limiter_needed = peak.limiter_needed

    return WholeVideoLoudnessDecision(
        gain_state=gain_state,
        integrated_loudness_lufs=measured,
        requested_gain_db=requested, authorized_gain_db=authorized,
        peak_evidence_source=peak_evidence_source, limiter_needed=limiter_needed,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# STAGE 8/9/10/11: adjacent-take gain-continuity policy.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdjacentTakeAdjustment:
    left_segment_id: str | None
    right_segment_id: str | None
    measured_delta_lu: float | None
    gain_state: str
    requested_correction_db: float | None
    authorized_correction_db: float | None
    direction: str | None  # "RAISE_LEFT" | "RAISE_RIGHT" | None
    limiter_needed: bool
    reason: str


def evaluate_adjacent_take_continuity(
    left: AudioFinishingMeasurement,
    right: AudioFinishingMeasurement,
    *,
    left_segment_id: str | None = None,
    right_segment_id: str | None = None,
) -> AdjacentTakeAdjustment:
    """STAGE 8/9: bounded, evidence-gated join-boundary continuity check.

    This function takes only the two bounded-window measurements it is
    given -- it never infers same-speaker/same-take-family/same-recording-
    intent from anything (STAGE 9's natural-dynamics firewall / no
    semantic guessing): the caller is the one asserting these two windows
    are the relevant adjacent takes at a join. This is a bounded threshold
    rule using ONLY `ADJACENT_TAKE_MISMATCH_THRESHOLD_LU` and
    `MAX_AUTOMATIC_GAIN_CORRECTION_DB` -- never a learned/heuristic
    classifier.

    STAGE 9's "do not normalize both independently" is honored structurally:
    this function proposes moving only the QUIETER side toward the louder
    one (a relative correction), never an absolute per-side target."""
    if not _has_reliable_window(left) or not _has_reliable_window(right):
        return AdjacentTakeAdjustment(
            left_segment_id=left_segment_id, right_segment_id=right_segment_id,
            measured_delta_lu=None, gain_state=GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE,
            requested_correction_db=None, authorized_correction_db=None, direction=None,
            limiter_needed=False,
            reason=f"at least one window is below the {MINIMUM_RELIABLE_LOUDNESS_WINDOW_SEC}s "
                   "minimum reliable duration -- abstaining rather than stretching the window",
        )
    if left.integrated_loudness_lufs is None or right.integrated_loudness_lufs is None:
        return AdjacentTakeAdjustment(
            left_segment_id=left_segment_id, right_segment_id=right_segment_id,
            measured_delta_lu=None, gain_state=GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE,
            requested_correction_db=None, authorized_correction_db=None, direction=None,
            limiter_needed=False,
            reason="loudness unmeasured on at least one side -- abstaining",
        )

    delta = right.integrated_loudness_lufs - left.integrated_loudness_lufs
    abs_delta = abs(delta)
    if abs_delta <= ADJACENT_TAKE_MISMATCH_THRESHOLD_LU:
        return AdjacentTakeAdjustment(
            left_segment_id=left_segment_id, right_segment_id=right_segment_id,
            measured_delta_lu=delta, gain_state=GAIN_STATE_NO_CHANGE_NEEDED,
            requested_correction_db=0.0, authorized_correction_db=0.0, direction=None,
            limiter_needed=False,
            reason=f"{abs_delta:.2f} LU delta is within the "
                   f"{ADJACENT_TAKE_MISMATCH_THRESHOLD_LU} LU natural-variation threshold",
        )

    if delta > 0:  # right is louder -> left is the quieter side
        direction = "RAISE_LEFT"
        raised_measurement, raised_id = left, left_segment_id
    else:  # left is louder -> right is the quieter side
        direction = "RAISE_RIGHT"
        raised_measurement, raised_id = right, right_segment_id

    requested = abs_delta
    if requested <= MAX_AUTOMATIC_GAIN_CORRECTION_DB:
        gain_state = GAIN_STATE_CORRECTION_ALLOWED
        authorized = requested
        reason = (
            f"{requested:.2f} LU adjacent mismatch exceeds the "
            f"{ADJACENT_TAKE_MISMATCH_THRESHOLD_LU} LU threshold; raising {raised_id!r} by "
            f"{authorized:.2f} dB toward the other take (relative correction, not independent "
            "normalization of either side)"
        )
    else:
        gain_state = GAIN_STATE_CORRECTION_LIMITED
        authorized = MAX_AUTOMATIC_GAIN_CORRECTION_DB
        reason = (
            f"{requested:.2f} LU adjacent mismatch exceeds the "
            f"{MAX_AUTOMATIC_GAIN_CORRECTION_DB} dB automatic envelope -- authorizing only "
            f"{authorized:.2f} dB toward {raised_id!r}, full continuity will NOT be reached "
            "automatically"
        )

    limiter_needed = False
    if raised_measurement.clipping_status == CLIPPING_STATUS_CLIPPING_DETECTED:
        gain_state, authorized = GAIN_STATE_BLOCKED_CLIPPING, 0.0
        reason = f"clipping proxy on {raised_id!r} blocks further positive gain at this join"
    else:
        peak = evaluate_peak_safety(raised_measurement, authorized)
        if peak.blocked:
            gain_state, authorized = GAIN_STATE_BLOCKED_PEAK_RISK, 0.0
            reason = f"peak safety blocks raising {raised_id!r}: {peak.reason}"
        else:
            limiter_needed = peak.limiter_needed

    return AdjacentTakeAdjustment(
        left_segment_id=left_segment_id, right_segment_id=right_segment_id,
        measured_delta_lu=delta, gain_state=gain_state,
        requested_correction_db=requested, authorized_correction_db=authorized,
        direction=direction, limiter_needed=limiter_needed, reason=reason,
    )


# ---------------------------------------------------------------------------
# STAGE 3/14/18: the AudioFinishingPlan contract + plan generation.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioFinishingPlan:
    """MEASUREMENT -> POLICY -> **PLAN** -> future executor -> post-render
    verification. This object AUTHORIZES a future correction; it never
    executes one. No field is a raw ffmpeg filter string -- a future
    executor consumes this structured intent and is itself responsible
    for turning it into whatever DSP implements it (mirrors
    `render_plan.RenderSegment` separating "what to render" from "the
    ffmpeg command that renders it")."""

    policy_version: str
    measurement_reference: AudioFinishingMeasurement

    whole_video_state: str
    whole_video_integrated_loudness_lufs: float | None
    target_loudness_lufs: float
    loudness_tolerance_lu: float
    requested_whole_video_gain_db: float | None
    authorized_whole_video_gain_db: float | None

    adjacent_take_adjustments: tuple[AdjacentTakeAdjustment, ...]

    limiter_authorized: bool
    true_peak_ceiling_dbtp: float
    peak_evidence_source: str

    abstentions: tuple[str, ...]
    reasons: tuple[str, ...]
    provenance: dict = field(default_factory=dict)

    plan_status: str = PLAN_STATUS_UNKNOWN


def _derive_plan_status(
    whole_video_state: str, adjacent_adjustments: tuple[AdjacentTakeAdjustment, ...], limiter_authorized: bool,
) -> str:
    """STAGE 14: summarize the plan without hiding per-reason detail (the
    detail lives in `reasons`/`abstentions`/each `AdjacentTakeAdjustment`;
    this is only the top-level roll-up)."""
    all_states = [whole_video_state] + [a.gain_state for a in adjacent_adjustments]

    if any(state in _BLOCKED_STATES for state in all_states):
        return PLAN_STATUS_BLOCKED
    if any(state == GAIN_STATE_UNKNOWN for state in all_states):
        return PLAN_STATUS_UNKNOWN
    if all(state == GAIN_STATE_NO_CHANGE_NEEDED for state in all_states):
        return PLAN_STATUS_READY_NO_CHANGE
    abstain_present = any(state == GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE for state in all_states)
    correction_present = any(state in _CORRECTION_STATES for state in all_states)
    if correction_present:
        if limiter_authorized:
            return PLAN_STATUS_READY_WITH_LIMITER
        if any(state == GAIN_STATE_CORRECTION_LIMITED for state in all_states) or abstain_present:
            return PLAN_STATUS_PARTIAL
        return PLAN_STATUS_READY_FOR_CORRECTION
    if abstain_present:
        return PLAN_STATUS_ABSTAIN
    return PLAN_STATUS_UNKNOWN


def generate_audio_finishing_plan(
    whole_video_measurement: AudioFinishingMeasurement,
    *,
    adjacent_pairs: tuple[tuple[AudioFinishingMeasurement, AudioFinishingMeasurement, str | None, str | None], ...] = (),
    provenance: dict | None = None,
) -> AudioFinishingPlan:
    """STAGE 3/18: the one entry point that turns real measurement into a
    structured `AudioFinishingPlan`. Deterministic, pure, zero media
    mutation, zero DSP. `adjacent_pairs` is a tuple of
    `(left_measurement, right_measurement, left_segment_id, right_segment_id)`
    -- the caller (an already-decided Selection/Boundary authority) is
    responsible for saying which windows are adjacent at a join; this
    function never infers that itself (Stage 9)."""
    whole = evaluate_whole_video_loudness(whole_video_measurement)

    adjustments = tuple(
        evaluate_adjacent_take_continuity(
            left, right, left_segment_id=left_id, right_segment_id=right_id,
        )
        for left, right, left_id, right_id in adjacent_pairs
    )

    limiter_authorized = whole.limiter_needed or any(a.limiter_needed for a in adjustments)

    abstentions = tuple(
        f"whole_video: {reason}" for reason in whole.reasons
        if whole.gain_state == GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE
    ) + tuple(
        f"adjacent[{a.left_segment_id!r}/{a.right_segment_id!r}]: {a.reason}"
        for a in adjustments
        if a.gain_state == GAIN_STATE_ABSTAIN_INSUFFICIENT_EVIDENCE
    )

    reasons = tuple(f"whole_video: {r}" for r in whole.reasons) + tuple(
        f"adjacent[{a.left_segment_id!r}/{a.right_segment_id!r}]: {a.reason}" for a in adjustments
    )

    plan_status = _derive_plan_status(whole.gain_state, adjustments, limiter_authorized)

    return AudioFinishingPlan(
        policy_version=POLICY_VERSION,
        measurement_reference=whole_video_measurement,
        whole_video_state=whole.gain_state,
        whole_video_integrated_loudness_lufs=whole.integrated_loudness_lufs,
        target_loudness_lufs=TARGET_INTEGRATED_LOUDNESS_LUFS,
        loudness_tolerance_lu=LOUDNESS_TOLERANCE_LU,
        requested_whole_video_gain_db=whole.requested_gain_db,
        authorized_whole_video_gain_db=whole.authorized_gain_db,
        adjacent_take_adjustments=adjustments,
        limiter_authorized=limiter_authorized,
        true_peak_ceiling_dbtp=TRUE_PEAK_CEILING_DBTP,
        peak_evidence_source=whole.peak_evidence_source,
        abstentions=abstentions,
        reasons=reasons,
        provenance=dict(provenance or {}),
        plan_status=plan_status,
    )
