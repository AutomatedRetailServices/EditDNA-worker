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
from dataclasses import dataclass, field

from .audio_finishing_measurement import (
    MEASUREMENT_STATUS_MEASUREMENT_ERROR,
    MEASUREMENT_STATUS_UNAVAILABLE,
    AudioFinishingMeasurement,
    measure_audio,
)
from .audio_finishing_policy import (
    ACCEPTABLE_LOUDNESS_LOWER_BOUND_LUFS,
    ACCEPTABLE_LOUDNESS_UPPER_BOUND_LUFS,
    MAX_AUTOMATIC_GAIN_CORRECTION_DB,
    PEAK_EVIDENCE_FALLBACK_SAMPLE_PEAK,
    PEAK_EVIDENCE_TRUE_PEAK,
    PEAK_EVIDENCE_UNAVAILABLE,
    PLAN_STATUS_ABSTAIN,
    PLAN_STATUS_BLOCKED,
    PLAN_STATUS_UNKNOWN,
    TRUE_PEAK_CEILING_DBTP,
    AudioFinishingPlan,
)
from .media_probe import probe_media

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
