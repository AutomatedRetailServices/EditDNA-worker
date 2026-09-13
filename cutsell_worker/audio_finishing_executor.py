"""Audio Finishing WHOLE-VIDEO EXECUTOR FOUNDATION (D-251).

D-247 built MEASUREMENT. D-249 built POLICY + PLAN generation. D-250
designed -- but did not implement -- the EXECUTION layer. This module is
the first gate that actually runs ffmpeg to apply a plan's already-
decided whole-video gain/limiter intent, on synthetic media only, fully
standalone from the live render pipeline.

    MEASUREMENT (D-247) -> POLICY + PLAN (D-249) -> **EXECUTOR (this
    module)** -> POST-EXECUTION MEASUREMENT (D-247, reused) -> VERIFICATION

## Scope: whole-video stage only

This module executes ONLY `AudioFinishingPlan.authorized_whole_video_gain_db`
and `AudioFinishingPlan.limiter_authorized` -- the Level-2 whole-video
pass from D-250's selected V1 strategy. `adjacent_take_adjustments`
(Level 1, the per-segment `RenderSegment.audio_volume` integration point)
is explicitly NOT executed here -- D-250 Stage 22 scoped that to its own
later gate, since it is a different integration point (inside the live
render's own segment/concat machinery, not a standalone post-render
pass). Nothing in this module reads or acts on
`plan.adjacent_take_adjustments`.

## What IS built here

- A real ffmpeg invocation applying EXACTLY the plan's already-authorized
  `volume=<db>dB` gain and, only when `limiter_authorized`, an `alimiter`
  safety stage after it (D-250 Stage 2/5/7/21's selected design). The
  executor never recomputes a gain from measurement -- it reads the
  number the policy layer already decided and applies that number,
  nothing else.
- A real re-measurement of the actual output file via
  `audio_finishing_measurement.measure_audio` (D-247) -- the sole
  measurement authority; this module never parses ffmpeg's own stderr
  into a policy/verification value.
- A structured `AudioFinishingExecutionRecord` and
  `ExecutionVerificationResult` -- typed, frozen, no raw ffmpeg command
  line anywhere in either.
- A deterministic, content/plan-derived `execution_id` (D-250 Stage 9's
  idempotence contract): identical (input path, policy version, whole-
  video state, authorized gain, limiter authorization, peak ceiling)
  always yields the same id; a caller-supplied `existing_record` whose id
  matches and whose prior run already succeeded is honored as a no-op --
  no mutable global state, no hidden cache.

## What is honestly NOT built here, and why

- **Adjacent-take execution** (Level 1) -- out of scope this gate (see
  above); `plan.adjacent_take_adjustments` is read only far enough to
  confirm it is never mutated or executed by this module (see the test
  suite's own structural guard).
- **Live pipeline integration** -- this module is never imported by
  `render.py`, `flow_b.py`, `pipeline.py`, any GitHub workflow, or any
  Modal/RunPod entry point. It is reachable only by direct, explicit call
  -- a standalone foundation, not a wired capability.
- **A second, longer retry loop** -- D-250 Stage 15 designed (not
  implemented) a bounded one-retry contract; this gate deliberately does
  not implement automatic retry (per this gate's own explicit
  instruction) -- `execute_audio_finishing_plan` runs the plan exactly
  once and reports whatever `ExecutionVerificationResult` results. A
  future gate may add the retry loop around this same function.
- **Denoise/hum/compressor/any loudnorm-style normalization** -- none of
  these are referenced, imported, or invoked anywhere in this module.
- **True-peak-verified limiting** -- `alimiter` (ffmpeg's own sample-peak-
  domain limiter, confirmed via `ffmpeg -h filter=alimiter` this session:
  its `limit` option is a linear SAMPLE amplitude, 0.0625-1, with no
  documented true-peak/oversampling mode) is applied with `level=0`
  (auto-level compensation explicitly disabled, so it is a fixed ceiling,
  not a second, undocumented gain decision) -- but this module never
  claims -1.0 dBTP compliance from the limiter's configuration alone.
  Compliance is only ever asserted from the real post-execution
  measurement (`ExecutionVerificationResult.true_peak_within_ceiling`).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field, replace as dataclass_replace

from .audio_finishing_measurement import (
    MEASUREMENT_STATUS_MEASUREMENT_ERROR,
    MEASUREMENT_STATUS_UNAVAILABLE,
    AudioFinishingMeasurement,
    measure_audio,
)
from .audio_finishing_policy import (
    ACCEPTABLE_LOUDNESS_LOWER_BOUND_LUFS,
    ACCEPTABLE_LOUDNESS_UPPER_BOUND_LUFS,
    GAIN_STATE_CORRECTION_ALLOWED,
    GAIN_STATE_CORRECTION_LIMITED,
    GAIN_STATE_NO_CHANGE_NEEDED,
    MAX_AUTOMATIC_GAIN_CORRECTION_DB,
    PEAK_EVIDENCE_FALLBACK_SAMPLE_PEAK,
    PEAK_EVIDENCE_TRUE_PEAK,
    PEAK_EVIDENCE_UNAVAILABLE,
    PLAN_STATUS_ABSTAIN,
    PLAN_STATUS_BLOCKED,
    PLAN_STATUS_UNKNOWN,
    TRUE_PEAK_CEILING_DBTP,
    AdjacentTakeAdjustment,
    AudioFinishingPlan,
)
from .media_probe import probe_media
from .render_plan import RenderSegment

_FFMPEG = "ffmpeg"
_DEFAULT_TIMEOUT_SEC = 120.0
_GAIN_EPSILON_DB = 1e-9

# ---------------------------------------------------------------------------
# STAGE 12: execution failure vocabulary.
# ---------------------------------------------------------------------------

EXECUTION_STATUS_SUCCESS = "SUCCESS"
EXECUTION_STATUS_NO_ACTION_NEEDED = "NO_ACTION_NEEDED"
EXECUTION_STATUS_PLAN_NOT_EXECUTABLE = "PLAN_NOT_EXECUTABLE"
EXECUTION_STATUS_MEASUREMENT_REFERENCE_MISSING = "MEASUREMENT_REFERENCE_MISSING"
EXECUTION_STATUS_INVALID_GAIN = "INVALID_GAIN"
EXECUTION_STATUS_PEAK_SAFETY_UNVERIFIED = "PEAK_SAFETY_UNVERIFIED"
EXECUTION_STATUS_FFMPEG_FAILURE = "FFMPEG_FAILURE"
EXECUTION_STATUS_POST_VERIFY_OUT_OF_POLICY = "POST_VERIFY_OUT_OF_POLICY"
EXECUTION_STATUS_OUTPUT_MISSING = "OUTPUT_MISSING"
EXECUTION_STATUS_OTHER = "OTHER"

_NON_EXECUTABLE_PLAN_STATUSES = (PLAN_STATUS_ABSTAIN, PLAN_STATUS_BLOCKED, PLAN_STATUS_UNKNOWN)

VERIFICATION_STATUS_PASS = "PASS"
VERIFICATION_STATUS_POLICY_OUT_OF_RANGE = "POLICY_OUT_OF_RANGE"
VERIFICATION_STATUS_TECHNICAL_FAILURE = "TECHNICAL_FAILURE"
VERIFICATION_STATUS_PARTIAL = "PARTIAL"


def _run(args: list[str], *, timeout_sec: float = _DEFAULT_TIMEOUT_SEC) -> tuple[int, str]:
    """Matches `post_render_media_qc._run`/`audio_finishing_measurement._run`'s
    exact pattern verbatim: never raises on a nonzero exit, callers decide."""
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout_sec, check=False,
    )
    return proc.returncode, proc.stdout.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# STAGE 2/3: typed, frozen execution + verification records.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioFinishingExecutionRecord:
    execution_id: str
    policy_version: str
    plan_status: str
    input_path: str
    output_path: str | None

    requested_whole_video_gain_db: float | None
    authorized_whole_video_gain_db: float | None
    limiter_authorized: bool
    true_peak_ceiling_dbtp: float
    peak_evidence_source: str

    filters_applied: tuple[str, ...]
    execution_status: str
    ffmpeg_return_code: int | None
    errors: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionVerificationResult:
    measurement_status: str
    integrated_loudness_lufs: float | None
    true_peak_dbfs: float | None
    sample_peak_dbfs: float | None
    peak_evidence_source: str

    audio_present: bool
    duration_preserved: bool | None  # None: observational only, see module docstring / D-251 Stage 16
    duration_delta_sec: float | None
    sample_rate_expected: bool
    channel_count_expected: bool

    loudness_in_target_range: bool | None
    true_peak_within_ceiling: bool | None

    verification_status: str
    errors: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# STAGE 5/6/7: the deterministic ffmpeg filter chain.
# ---------------------------------------------------------------------------

def _linear_gain_ratio_for_ceiling(ceiling_dbtp: float) -> float:
    return 10.0 ** (ceiling_dbtp / 20.0)


def _build_audio_filter_chain(gain_db: float | None, limiter_authorized: bool, ceiling_dbtp: float) -> list[str]:
    """Builds the exact, deterministic filter list -- never `loudnorm`,
    never a compressor/denoiser/hum filter, never a recomputed gain.
    `aformat` at the end matches this codebase's existing per-segment
    standardization (`render.py`'s own `aformat=sample_fmts=fltp:
    sample_rates=48000:channel_layouts=stereo`, D-246's trace) so a mono
    or non-48k source never receives an unintended perceived-level change
    purely from format standardization (D-249/D-250's own invariant)."""
    parts: list[str] = []
    if gain_db is not None and abs(gain_db) > _GAIN_EPSILON_DB:
        parts.append(f"volume={gain_db:.6f}dB")
    if limiter_authorized:
        limit = _linear_gain_ratio_for_ceiling(ceiling_dbtp)
        # level=0: auto-level compensation explicitly disabled -- alimiter
        # must be a fixed ceiling, never a second, undocumented gain
        # decision layered on top of the policy's own authorized number.
        parts.append(f"alimiter=limit={limit:.6f}:level=0")
    parts.append("aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo")
    return parts


def _has_video_stream(path: str) -> bool:
    probe = probe_media(path)
    return probe.width > 0 and probe.height > 0


def _audio_codec_args(output_path: str) -> list[str]:
    """Project-standard delivery codec (`render.py`'s own `-c:a aac -b:a
    160k -ar 48000`, D-246's trace) for an mp4/mov-style container; for
    any other container (e.g. a synthetic .wav test fixture) ffmpeg's own
    default codec for that container is used instead of forcing an
    incompatible one -- no new export format is invented either way."""
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".mp4", ".mov", ".m4v"):
        return ["-c:a", "aac", "-b:a", "160k"]
    return []


def _ffmpeg_execute(input_path: str, output_path: str, audio_filter_parts: list[str]) -> tuple[int, str, list[str]]:
    has_video = _has_video_stream(input_path)
    args = [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", input_path]
    if has_video:
        args += ["-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy"]
    args += ["-af", ",".join(audio_filter_parts)]
    args += _audio_codec_args(output_path)
    if has_video:
        args += ["-movflags", "+faststart"]
    args += [output_path]
    returncode, output = _run(args)
    return returncode, output, args


# ---------------------------------------------------------------------------
# STAGE 10: deterministic, content/plan-derived idempotence id.
# ---------------------------------------------------------------------------

def compute_execution_id(plan: AudioFinishingPlan, input_path: str) -> str:
    """A pure function of the plan's whole-video-relevant fields and the
    input path -- no mutable global state. The SAME input+plan pair
    always yields the SAME id; a different plan (even for the same
    input) always yields a different one."""
    payload = {
        "input_path": input_path,
        "policy_version": plan.policy_version,
        "whole_video_state": plan.whole_video_state,
        "authorized_whole_video_gain_db": plan.authorized_whole_video_gain_db,
        "limiter_authorized": plan.limiter_authorized,
        "true_peak_ceiling_dbtp": plan.true_peak_ceiling_dbtp,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


# ---------------------------------------------------------------------------
# STAGE 4/5/6/13: the one entry point.
# ---------------------------------------------------------------------------

def execute_audio_finishing_plan(
    plan: AudioFinishingPlan,
    input_path: str,
    output_path: str,
    *,
    existing_record: AudioFinishingExecutionRecord | None = None,
) -> tuple[AudioFinishingExecutionRecord, ExecutionVerificationResult | None]:
    """Execute ONLY the whole-video gain/limiter portion of `plan` against
    the real file at `input_path`, writing a SEPARATE file at
    `output_path` (never mutating `input_path`). Runs the plan's DSP
    exactly once -- no automatic retry (D-251's own explicit scope).

    Returns `(execution_record, verification_result)`; `verification_result`
    is `None` whenever no DSP actually ran (a non-executable plan, a
    no-action plan, or a defense-in-depth rejection) -- verification only
    ever follows a real execution attempt."""
    execution_id = compute_execution_id(plan, input_path)

    if (
        existing_record is not None
        and existing_record.execution_id == execution_id
        and existing_record.execution_status == EXECUTION_STATUS_SUCCESS
        and existing_record.output_path is not None
        and os.path.exists(existing_record.output_path)
    ):
        return existing_record, None

    common_kwargs = dict(
        execution_id=execution_id,
        policy_version=plan.policy_version,
        plan_status=plan.plan_status,
        input_path=input_path,
        requested_whole_video_gain_db=plan.requested_whole_video_gain_db,
        authorized_whole_video_gain_db=plan.authorized_whole_video_gain_db,
        limiter_authorized=plan.limiter_authorized,
        true_peak_ceiling_dbtp=plan.true_peak_ceiling_dbtp,
        peak_evidence_source=plan.peak_evidence_source,
    )

    if plan.measurement_reference is None:
        return AudioFinishingExecutionRecord(
            **common_kwargs, output_path=None, filters_applied=(),
            execution_status=EXECUTION_STATUS_MEASUREMENT_REFERENCE_MISSING,
            ffmpeg_return_code=None,
            errors=("plan.measurement_reference is missing -- refusing to execute",),
            provenance={},
        ), None

    # STAGE 4: reject action outright on a non-executable top-level plan status.
    if plan.plan_status in _NON_EXECUTABLE_PLAN_STATUSES:
        return AudioFinishingExecutionRecord(
            **common_kwargs, output_path=None, filters_applied=(),
            execution_status=EXECUTION_STATUS_PLAN_NOT_EXECUTABLE,
            ffmpeg_return_code=None,
            errors=(f"plan_status={plan.plan_status!r} is not executable",),
            provenance={},
        ), None

    gain_db = plan.authorized_whole_video_gain_db
    # Defense in depth: never trust a gain magnitude the policy layer
    # (whose own ±6dB envelope, D-249) could not have produced -- a
    # tampered/hand-built plan must not silently execute.
    if gain_db is not None and abs(gain_db) > MAX_AUTOMATIC_GAIN_CORRECTION_DB + 1e-6:
        return AudioFinishingExecutionRecord(
            **common_kwargs, output_path=None, filters_applied=(),
            execution_status=EXECUTION_STATUS_INVALID_GAIN,
            ffmpeg_return_code=None,
            errors=(f"authorized_whole_video_gain_db={gain_db!r} exceeds the "
                    f"{MAX_AUTOMATIC_GAIN_CORRECTION_DB} dB envelope -- refusing to execute",),
            provenance={},
        ), None

    # Defense in depth: a positive gain must never execute without real
    # peak evidence, even if some other bug let it reach this far (D-249's
    # own evaluate_peak_safety already prevents this combination upstream).
    if gain_db is not None and gain_db > 0 and plan.peak_evidence_source == PEAK_EVIDENCE_UNAVAILABLE:
        return AudioFinishingExecutionRecord(
            **common_kwargs, output_path=None, filters_applied=(),
            execution_status=EXECUTION_STATUS_PEAK_SAFETY_UNVERIFIED,
            ffmpeg_return_code=None,
            errors=("positive gain requested with no peak evidence -- refusing to execute",),
            provenance={},
        ), None

    # STAGE 4: no needless transcode when nothing is actually authorized
    # for THIS executor's scope (whole-video gain/limiter only).
    has_gain = gain_db is not None and abs(gain_db) > _GAIN_EPSILON_DB
    if not has_gain and not plan.limiter_authorized:
        return AudioFinishingExecutionRecord(
            **common_kwargs, output_path=None, filters_applied=(),
            execution_status=EXECUTION_STATUS_NO_ACTION_NEEDED,
            ffmpeg_return_code=None,
            errors=(), provenance={"reason": "no whole-video gain or limiter authorized"},
        ), None

    filter_parts = _build_audio_filter_chain(gain_db, plan.limiter_authorized, plan.true_peak_ceiling_dbtp)

    try:
        returncode, ffmpeg_output, args = _ffmpeg_execute(input_path, output_path, filter_parts)
    except subprocess.CalledProcessError as exc:
        # `_has_video_stream`'s own ffprobe call (media_probe.probe_media)
        # raises hard on a malformed/nonexistent input (its own documented
        # convention, D-096) -- this IS an ffmpeg/ffprobe-family failure on
        # the real input, not an unanticipated internal error.
        return AudioFinishingExecutionRecord(
            **common_kwargs, output_path=None, filters_applied=(),
            execution_status=EXECUTION_STATUS_FFMPEG_FAILURE,
            ffmpeg_return_code=exc.returncode,
            errors=(str(exc)[:2000],),
            provenance={},
        ), None
    except Exception as exc:  # pragma: no cover - defensive, matches _run-never-raises convention
        return AudioFinishingExecutionRecord(
            **common_kwargs, output_path=None, filters_applied=tuple(filter_parts),
            execution_status=EXECUTION_STATUS_OTHER,
            ffmpeg_return_code=None,
            errors=(f"unexpected exception invoking ffmpeg: {exc}",),
            provenance={},
        ), None

    if returncode != 0:
        return AudioFinishingExecutionRecord(
            **common_kwargs, output_path=None, filters_applied=tuple(filter_parts),
            execution_status=EXECUTION_STATUS_FFMPEG_FAILURE,
            ffmpeg_return_code=returncode,
            errors=(ffmpeg_output.strip()[:2000],),
            provenance={"ffmpeg_args": args},
        ), None

    if not os.path.exists(output_path):
        return AudioFinishingExecutionRecord(
            **common_kwargs, output_path=None, filters_applied=tuple(filter_parts),
            execution_status=EXECUTION_STATUS_OUTPUT_MISSING,
            ffmpeg_return_code=returncode,
            errors=("ffmpeg exited 0 but output file does not exist",),
            provenance={"ffmpeg_args": args},
        ), None

    record = AudioFinishingExecutionRecord(
        **common_kwargs, output_path=output_path, filters_applied=tuple(filter_parts),
        execution_status=EXECUTION_STATUS_SUCCESS,
        ffmpeg_return_code=returncode,
        errors=(), provenance={"ffmpeg_args": args},
    )

    verification = _verify_execution(plan, record)
    return record, verification


# ---------------------------------------------------------------------------
# STAGE 13/14/15/16: post-execution measurement + verification (sole
# measurement authority: D-247's measure_audio; never ffmpeg stderr).
# ---------------------------------------------------------------------------

def _verify_execution(plan: AudioFinishingPlan, record: AudioFinishingExecutionRecord) -> ExecutionVerificationResult:
    output_measurement: AudioFinishingMeasurement = measure_audio(record.output_path)

    input_duration = plan.measurement_reference.duration_sec
    output_duration = output_measurement.duration_sec
    duration_delta: float | None = None
    duration_preserved: bool | None = None
    if input_duration is not None and output_duration is not None:
        duration_delta = output_duration - input_duration
        # STAGE 16: no existing project-canonical AUDIO duration tolerance
        # was found (render.py's own frame-exact timing contracts govern
        # VIDEO segment timing, not a standalone audio-file delta) -- so
        # this stays observational (`duration_preserved=None`, the real
        # delta reported) rather than inventing a new numeric tolerance.

    audio_present = output_measurement.channel_count not in (None, 0)

    sample_rate_expected = output_measurement.sample_rate_hz == 48000
    channel_count_expected = output_measurement.channel_count == 2

    loudness = output_measurement.integrated_loudness_lufs
    loudness_in_range: bool | None = None
    if loudness is not None:
        loudness_in_range = ACCEPTABLE_LOUDNESS_LOWER_BOUND_LUFS <= loudness <= ACCEPTABLE_LOUDNESS_UPPER_BOUND_LUFS

    true_peak = output_measurement.true_peak_dbfs
    sample_peak = output_measurement.sample_peak_dbfs
    if true_peak is not None:
        peak_evidence_source = PEAK_EVIDENCE_TRUE_PEAK
        peak_within_ceiling: bool | None = true_peak <= TRUE_PEAK_CEILING_DBTP
    elif sample_peak is not None:
        peak_evidence_source = PEAK_EVIDENCE_FALLBACK_SAMPLE_PEAK
        peak_within_ceiling = sample_peak <= TRUE_PEAK_CEILING_DBTP
    else:
        peak_evidence_source = PEAK_EVIDENCE_UNAVAILABLE
        peak_within_ceiling = None

    errors = list(output_measurement.measurement_errors)

    if output_measurement.measurement_status in (MEASUREMENT_STATUS_UNAVAILABLE, MEASUREMENT_STATUS_MEASUREMENT_ERROR) or not audio_present:
        verification_status = VERIFICATION_STATUS_TECHNICAL_FAILURE
    elif loudness_in_range is False or peak_within_ceiling is False:
        verification_status = VERIFICATION_STATUS_POLICY_OUT_OF_RANGE
    elif loudness_in_range is True and (peak_within_ceiling is not False):
        verification_status = VERIFICATION_STATUS_PASS
    else:
        verification_status = VERIFICATION_STATUS_PARTIAL

    return ExecutionVerificationResult(
        measurement_status=output_measurement.measurement_status,
        integrated_loudness_lufs=loudness,
        true_peak_dbfs=true_peak,
        sample_peak_dbfs=sample_peak,
        peak_evidence_source=peak_evidence_source,
        audio_present=audio_present,
        duration_preserved=duration_preserved,
        duration_delta_sec=duration_delta,
        sample_rate_expected=sample_rate_expected,
        channel_count_expected=channel_count_expected,
        loudness_in_target_range=loudness_in_range,
        true_peak_within_ceiling=peak_within_ceiling,
        verification_status=verification_status,
        errors=tuple(errors),
        provenance={"output_path": record.output_path},
    )


# ===========================================================================
# LEVEL 1 -- ADJACENT-TAKE GAIN EXECUTION FOUNDATION (D-252)
# ===========================================================================
"""
D-251 built the LEVEL-2 (whole-video) executor above. This section adds
the LEVEL-1 (adjacent-take continuity) executor -- one finishing
execution authority, per D-252's own "prefer one finishing execution
authority" instruction, rather than a second module.

## Scope: plan-authorized segment gain only

This executes ONLY `AudioFinishingPlan.adjacent_take_adjustments` against
already-built `RenderSegment` objects (`render_plan.py`, D-036/D-097.2's
canonical pre-render representation), applying each authorized
adjustment's `authorized_correction_db` to the EXISTING
`RenderSegment.audio_volume` linear-multiplier field -- the same field
`render.py`'s live `_segment_command`/`_concat_render_command` already
read (D-246's trace: `volume=<audio_volume>` is applied first in the
per-segment audio filter chain, BEFORE the 12ms click fade, so a flat
gain multiplier composes with the fade trivially and cannot reintroduce
a click, per D-250 Stage 11's own reasoning, now confirmed directly
against the live filter-chain code). Nothing here decides whether a
mismatch is meaningful, whether two segments share a speaker, or whether
correction is needed -- those are `evaluate_adjacent_take_continuity`'s
job (`audio_finishing_policy.py`, unchanged). This module only asks: is
this specific, already-authorized `authorized_correction_db` safe and
identifiable to apply, and if so, applies EXACTLY that number.

## STAGE 1 finding: segment identity is caller-supplied, not guaranteed

`AdjacentTakeAdjustment.left_segment_id`/`right_segment_id` are typed
`str | None` -- `generate_audio_finishing_plan`'s own `adjacent_pairs`
parameter accepts `None` for either identity (the policy layer does not
require the caller to supply real segment identity; the CALLER, an
already-decided Selection/Boundary authority, is documented to be
responsible for supplying it). This is a real, honest structural gap:
nothing downstream of `evaluate_adjacent_take_continuity` enforces that
an adjustment authorized for `direction="RAISE_LEFT"` actually carries a
non-`None` `left_segment_id`. This module does NOT paper over that gap
with positional/array-index inference (explicitly forbidden, D-252
Stage 3) -- it fails closed (`ADJACENT_EXECUTION_STATUS_IDENTITY_MISMATCH`)
whenever the identity a `direction` requires is missing, and reports
`ADJACENT_EXECUTION_STATUS_SEGMENT_NOT_FOUND` whenever a real, non-`None`
identity does not match any `RenderSegment.clip_id` in the segments the
caller supplied (this is also the structural cross-source-leakage guard:
an adjustment computed against a different render's segments finds no
matching `clip_id` here and touches nothing).

## STAGE 8 finding: a segment can appear in two adjacency relationships

`generate_audio_finishing_plan` evaluates each `adjacent_pairs` entry in
total isolation -- nothing in the policy layer prevents a middle segment
in a 3-segment chain (A-B, B-C) from receiving two independently-computed,
independently-authorized corrections (once as B in the A-B pair, once as
B in the B-C pair). No new numeric weighting/merge scheme is invented
here (forbidden, Stage 8): the chosen, documented rule is **conflict ->
abstain** -- when more than one ACTIONABLE adjustment (a real,
nonzero-authorized correction) targets the same resolved segment
identity, none of them are applied, and each is reported as
`ADJACENT_EXECUTION_STATUS_DUPLICATE_TARGET_CONFLICT`. A segment named by
one actionable adjustment and one merely-informational
`NO_CHANGE_NEEDED` adjustment is not a conflict -- the single real
correction still applies.

## STAGE 9 finding: composition with a pre-existing `audio_volume`

`RenderSegment.audio_volume` is a single scalar (no history of prior
gains). A finishing correction MULTIPLIES into whatever value is already
there (`new_audio_volume = existing_audio_volume * 10**(db/20)`) rather
than overwriting it -- gains compound multiplicatively in the linear
domain (equivalently, dB values add), so this is the physically correct
way to let an existing manual/editorial gain (e.g. `draft_edits.py`'s
`swap_take` layer, D-024) and an automatic finishing correction coexist,
never silently discarding one. Both the prior value and the correction
that was multiplied in are recorded on the result record.

## STAGE 18 finding: peak safety is already guaranteed upstream

`evaluate_adjacent_take_continuity` (`audio_finishing_policy.py`,
unchanged this gate) already calls `evaluate_peak_safety` on the raised
side and resolves `gain_state` to `BLOCKED_PEAK_RISK`/`BLOCKED_CLIPPING`
when unsafe -- this module consumes that `gain_state` as the sole
authority and adds NO new peak check, NO limiter, and NO new numeric
threshold at this stage (confirmed by a test constructing a
`BLOCKED_PEAK_RISK` adjustment and proving zero mutation results). The
limiter remains whole-video/final-safety only (D-250/D-251), unchanged.

## No DSP, no rendering, here

This section never invokes `ffmpeg`/`subprocess` and never touches a
media file -- it returns a NEW tuple of `RenderSegment` objects (frozen
dataclasses, `dataclasses.replace`d, never mutated in place) plus a
tuple of typed result records. `render.py`'s existing segment/concat
machinery, unchanged this gate, is what would eventually consume the
returned segments on some future, separately-authorized live-integration
gate -- this module does not call it.
"""

ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED = "ADJUSTMENT_APPLIED"
ADJACENT_EXECUTION_STATUS_NO_CHANGE = "NO_CHANGE"
ADJACENT_EXECUTION_STATUS_PLAN_NOT_AUTHORIZED = "PLAN_NOT_AUTHORIZED"
ADJACENT_EXECUTION_STATUS_SEGMENT_NOT_FOUND = "SEGMENT_NOT_FOUND"
ADJACENT_EXECUTION_STATUS_DUPLICATE_TARGET_CONFLICT = "DUPLICATE_TARGET_CONFLICT"
ADJACENT_EXECUTION_STATUS_IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
ADJACENT_EXECUTION_STATUS_INVALID_GAIN = "INVALID_GAIN"
ADJACENT_EXECUTION_STATUS_ALREADY_APPLIED = "ALREADY_APPLIED"
ADJACENT_EXECUTION_STATUS_OTHER = "OTHER"

_ADJACENT_ACTIONABLE_STATES = (GAIN_STATE_CORRECTION_ALLOWED, GAIN_STATE_CORRECTION_LIMITED)


@dataclass(frozen=True)
class SegmentGainAdjustmentResult:
    segment_id: str | None
    adjustment_application_id: str | None
    prior_audio_volume: float | None
    authorized_correction_db: float | None
    applied_linear_multiplier: float | None
    resulting_audio_volume: float | None
    execution_status: str
    reason: str
    provenance: dict = field(default_factory=dict)


def db_to_linear(gain_db: float) -> float:
    """The one, canonical dB-to-linear conversion (D-252 Stage 5): never
    reimplemented elsewhere, never a second formula."""
    return 10.0 ** (gain_db / 20.0)


def compute_adjustment_application_id(plan: AudioFinishingPlan, segment_id: str, authorized_correction_db: float, direction: str | None) -> str:
    """A pure, deterministic function of the plan's policy version, the
    resolved segment identity, the exact authorized correction, and the
    direction -- no mutable global state (matches D-251's
    `compute_execution_id` pattern exactly)."""
    payload = {
        "policy_version": plan.policy_version,
        "segment_id": segment_id,
        "authorized_correction_db": authorized_correction_db,
        "direction": direction,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def _resolve_target_segment_id(adjustment: AdjacentTakeAdjustment) -> str | None:
    if adjustment.direction == "RAISE_LEFT":
        return adjustment.left_segment_id
    if adjustment.direction == "RAISE_RIGHT":
        return adjustment.right_segment_id
    return None


def apply_adjacent_take_adjustments(
    plan: AudioFinishingPlan,
    segments: tuple[RenderSegment, ...],
    *,
    already_applied_ids: frozenset[str] = frozenset(),
) -> tuple[tuple[RenderSegment, ...], tuple[SegmentGainAdjustmentResult, ...]]:
    """STAGE 2-15/18: the one LEVEL-1 entry point. Pure, deterministic,
    immutable-input-oriented (Stage 15/20): `plan` and `segments` are
    never mutated; a NEW segments tuple and a tuple of per-adjustment
    result records are returned. Every segment not targeted by an
    APPLIED adjustment is returned as the exact same object (`is`
    identity preserved) it was passed in as -- the structural proof that
    no independent per-clip normalization ever happens here (Stage 7)."""
    segments_by_clip_id: dict[str, RenderSegment] = {segment.clip_id: segment for segment in segments}

    # STAGE 6: gate strictly per-adjustment `gain_state` -- never the
    # top-level `plan.plan_status` (Level 1 and Level 2 are independent
    # per the canonical two-level architecture; an unrelated whole-video
    # block must never suppress an otherwise-safe adjacent correction).
    actionable: list[tuple[AdjacentTakeAdjustment, str]] = []
    non_actionable_results: list[SegmentGainAdjustmentResult] = []

    for adjustment in plan.adjacent_take_adjustments:
        if adjustment.gain_state == GAIN_STATE_NO_CHANGE_NEEDED:
            non_actionable_results.append(SegmentGainAdjustmentResult(
                segment_id=_resolve_target_segment_id(adjustment), adjustment_application_id=None,
                prior_audio_volume=None, authorized_correction_db=adjustment.authorized_correction_db,
                applied_linear_multiplier=None, resulting_audio_volume=None,
                execution_status=ADJACENT_EXECUTION_STATUS_NO_CHANGE,
                reason=adjustment.reason, provenance={},
            ))
            continue
        if adjustment.gain_state not in _ADJACENT_ACTIONABLE_STATES:
            # ABSTAIN_INSUFFICIENT_EVIDENCE / BLOCKED_SILENCE /
            # BLOCKED_CLIPPING / BLOCKED_PEAK_RISK / UNKNOWN -- policy
            # already decided; this module performs no gain mutation and
            # invents no override.
            non_actionable_results.append(SegmentGainAdjustmentResult(
                segment_id=_resolve_target_segment_id(adjustment), adjustment_application_id=None,
                prior_audio_volume=None, authorized_correction_db=adjustment.authorized_correction_db,
                applied_linear_multiplier=None, resulting_audio_volume=None,
                execution_status=ADJACENT_EXECUTION_STATUS_PLAN_NOT_AUTHORIZED,
                reason=adjustment.reason, provenance={"gain_state": adjustment.gain_state},
            ))
            continue

        target_segment_id = _resolve_target_segment_id(adjustment)
        if target_segment_id is None:
            # STAGE 1/3: the plan authorized a correction but the
            # adjustment does not carry the identity needed to know WHICH
            # segment to touch -- never guess by position.
            non_actionable_results.append(SegmentGainAdjustmentResult(
                segment_id=None, adjustment_application_id=None,
                prior_audio_volume=None, authorized_correction_db=adjustment.authorized_correction_db,
                applied_linear_multiplier=None, resulting_audio_volume=None,
                execution_status=ADJACENT_EXECUTION_STATUS_IDENTITY_MISMATCH,
                reason="adjustment authorizes a correction but its direction's segment_id is missing",
                provenance={"direction": adjustment.direction},
            ))
            continue

        if adjustment.authorized_correction_db is None or abs(adjustment.authorized_correction_db) > MAX_AUTOMATIC_GAIN_CORRECTION_DB + 1e-6:
            non_actionable_results.append(SegmentGainAdjustmentResult(
                segment_id=target_segment_id, adjustment_application_id=None,
                prior_audio_volume=None, authorized_correction_db=adjustment.authorized_correction_db,
                applied_linear_multiplier=None, resulting_audio_volume=None,
                execution_status=ADJACENT_EXECUTION_STATUS_INVALID_GAIN,
                reason=f"authorized_correction_db={adjustment.authorized_correction_db!r} is missing or exceeds "
                       f"the {MAX_AUTOMATIC_GAIN_CORRECTION_DB} dB envelope -- refusing to apply",
                provenance={},
            ))
            continue

        actionable.append((adjustment, target_segment_id))

    # STAGE 8: conflict -> abstain when two+ actionable adjustments target
    # the same resolved segment identity.
    targets_seen: dict[str, list[AdjacentTakeAdjustment]] = {}
    for adjustment, target_segment_id in actionable:
        targets_seen.setdefault(target_segment_id, []).append(adjustment)

    conflict_results: list[SegmentGainAdjustmentResult] = []
    non_conflicting: list[tuple[AdjacentTakeAdjustment, str]] = []
    for target_segment_id, adjustments in targets_seen.items():
        if len(adjustments) > 1:
            for adjustment in adjustments:
                conflict_results.append(SegmentGainAdjustmentResult(
                    segment_id=target_segment_id, adjustment_application_id=None,
                    prior_audio_volume=None, authorized_correction_db=adjustment.authorized_correction_db,
                    applied_linear_multiplier=None, resulting_audio_volume=None,
                    execution_status=ADJACENT_EXECUTION_STATUS_DUPLICATE_TARGET_CONFLICT,
                    reason=f"{len(adjustments)} actionable adjustments target segment {target_segment_id!r} -- "
                           "neither applied (D-252 Stage 8: conflict -> abstain, no new merge/weighting invented)",
                    provenance={},
                ))
        else:
            non_conflicting.append((adjustments[0], target_segment_id))

    apply_results: list[SegmentGainAdjustmentResult] = []
    updated_segments: dict[str, RenderSegment] = {}

    for adjustment, target_segment_id in non_conflicting:
        segment = segments_by_clip_id.get(target_segment_id)
        if segment is None:
            # STAGE 3/19 cross-source-leakage guard: a segment_id naming a
            # clip not present in THIS call's `segments` never touches
            # anything -- structurally impossible to leak across sources.
            apply_results.append(SegmentGainAdjustmentResult(
                segment_id=target_segment_id, adjustment_application_id=None,
                prior_audio_volume=None, authorized_correction_db=adjustment.authorized_correction_db,
                applied_linear_multiplier=None, resulting_audio_volume=None,
                execution_status=ADJACENT_EXECUTION_STATUS_SEGMENT_NOT_FOUND,
                reason=f"no segment with clip_id={target_segment_id!r} in the supplied segments",
                provenance={},
            ))
            continue

        application_id = compute_adjustment_application_id(
            plan, target_segment_id, adjustment.authorized_correction_db, adjustment.direction,
        )
        if application_id in already_applied_ids:
            apply_results.append(SegmentGainAdjustmentResult(
                segment_id=target_segment_id, adjustment_application_id=application_id,
                prior_audio_volume=segment.audio_volume, authorized_correction_db=adjustment.authorized_correction_db,
                applied_linear_multiplier=None, resulting_audio_volume=segment.audio_volume,
                execution_status=ADJACENT_EXECUTION_STATUS_ALREADY_APPLIED,
                reason="this exact adjustment was already applied to this segment -- skipping to avoid double-gain",
                provenance={},
            ))
            continue

        linear = db_to_linear(adjustment.authorized_correction_db)
        prior_volume = segment.audio_volume
        # STAGE 9: multiplicative composition with whatever gain (manual
        # editorial or a prior finishing pass) already sits on this
        # segment -- never a silent overwrite.
        resulting_volume = prior_volume * linear
        updated_segments[target_segment_id] = dataclass_replace(segment, audio_volume=resulting_volume)

        apply_results.append(SegmentGainAdjustmentResult(
            segment_id=target_segment_id, adjustment_application_id=application_id,
            prior_audio_volume=prior_volume, authorized_correction_db=adjustment.authorized_correction_db,
            applied_linear_multiplier=linear, resulting_audio_volume=resulting_volume,
            execution_status=ADJACENT_EXECUTION_STATUS_ADJUSTMENT_APPLIED,
            reason=adjustment.reason,
            provenance={"direction": adjustment.direction, "policy_version": plan.policy_version},
        ))

    final_segments = tuple(
        updated_segments.get(segment.clip_id, segment) for segment in segments
    )
    all_results = tuple(non_actionable_results + conflict_results + apply_results)
    return final_segments, all_results
