"""D-274B: Rotation + VFR / Timeline Normalization Executor.

Chain this gate proves OFFLINE, with synthetic media only:

    original source
    -> D-271 profile (source_media_profile.probe_source_media_profile)
    -> D-272 policy  (source_format_policy.evaluate_source_format_policy)
    -> D-274A plan   (source_normalization_plan.build_source_normalization_plan)
    -> D-274B EXECUTOR (this module)
    -> normalized derived file
    -> D-271 RE-PROBE
    -> D-272 RE-EVALUATE
    -> ACCEPT only -> editorial pipeline

Separation of authority (Stage 1): PROFILE -> POLICY -> PLAN -> EXECUTOR ->
VERIFICATION. This module never re-decides whether rotation/VFR/timeline
normalization is needed -- the `SourceNormalizationPlan` handed in is the
sole authority for WHAT to do; this module only knows HOW to do it. Logic
here is never duplicated into worker_job.py/render.py/source_format_policy.py/
source_normalization_plan.py, and vice versa.

Supported actions this gate (Stage 3): NO_ACTION, ROTATE_90/180/270,
VFR_TO_CFR, TIMELINE_TO_ZERO. Anything else present on the plan
(HDR_PQ_TO_SDR_BT709, HDR_HLG_TO_SDR_BT709, HEVC_TO_H264,
TEN_BIT_TO_EIGHT_BIT, PIXEL_FORMAT_TO_YUV420P) fails BEFORE any ffmpeg
subprocess runs, with `NORMALIZATION_UNSUPPORTED_ACTION` -- no partial
normalization, no pretending.

Execution safety here MIRRORS (never imports) `render.py`'s own D-266
pattern: structured typed failures, bounded stderr, deterministic command
fingerprint, job-local temp output + atomic `os.replace` promotion,
best-effort cleanup that never masks the primary failure, `shell=False`
argv-list subprocess invocation, no user data interpolated into filter
syntax. See docs/CUTSELL_DECISIONS.md D-266 for the original design and
D-274B for this module's own entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import output_format_qc as ofq
from . import production_runtime_capability as prc
from . import source_format_policy as sfp
from . import source_media_profile as smp
from . import source_normalization_plan as snp
from .render_delivery import compute_output_sha256

# =============================================================================
# Stage 22 -- normalization timeout seam. Mirrors D-266's own original
# `TIMEOUT_POLICY_PENDING_PRODUCT_OWNER` design (the seam D-266A later
# resolved for RENDER_FFMPEG_TIMEOUT_SEC=1200.0). D-274B's own audit found
# NO existing canonical normalization/media-operation timeout to reuse, and
# Stage 22 explicitly forbids silently reusing RENDER_FFMPEG_TIMEOUT_SEC.
# This seam stays `None` until a Product Owner decision activates a number
# (the same shape D-266A itself followed for the renderer). Callers/tests
# inject a concrete `timeout_sec` explicitly; the module-level default is
# never silently substituted with an invented number.
# =============================================================================
NORMALIZATION_FFMPEG_TIMEOUT_SEC: float | None = None  # TIMEOUT_POLICY_PENDING_PRODUCT_OWNER
PRODUCT_OWNER_NORMALIZATION_TIMEOUT_REQUIRED = "PRODUCT_OWNER_NORMALIZATION_TIMEOUT_REQUIRED"

_STDERR_EXCERPT_MAX_CHARS = 2000  # reuses post_render_media_qc.py's own [:2000] bound

# --- Stage 3 -- supported vs. explicitly-unsupported actions ---------------
_ROTATION_FILTER: dict[str, str] = {
    # Empirically verified (D-274B forensic, asymmetric-fixture proof):
    # transpose=1 is a genuine 90-degree CLOCKWISE rotation, transpose=2 is
    # genuine COUNTERCLOCKWISE, and transpose=1 chained twice is a genuine
    # 180-degree rotation (dims unchanged) -- consistent with D-271's own
    # rotation_degrees convention (clockwise decoder correction).
    snp.ACTION_ROTATE_90: "transpose=1",
    snp.ACTION_ROTATE_180: "transpose=1,transpose=1",
    snp.ACTION_ROTATE_270: "transpose=2",
}

# D-274C Stage 8: ACTION_HEVC_TO_H264 is now IMPLEMENTED and removed from
# this set -- the standard command-construction path already decodes
# whatever codec the source carries and encodes to the plan's own target
# (libx264), so no HEVC-specific filter/flag was ever needed once the
# pre-existing unsupported-action gate stopped blocking it.
#
# D-274D Stage 2: ACTION_HDR_PQ_TO_SDR_BT709, ACTION_HDR_HLG_TO_SDR_BT709,
# ACTION_TEN_BIT_TO_EIGHT_BIT, and ACTION_PIXEL_FORMAT_TO_YUV420P are now
# ALL implemented and removed too -- this set is therefore EMPTY as of
# this gate (every currently-defined D-274A action has an executor). It
# stays as a real, live frozenset (not deleted) as the correct
# extensibility point for any FUTURE action D-274A might add that this
# executor does not yet implement (Stage 37's own general principle).
#
# Dolby Vision and HDR_OTHER never reach this set at all -- D-274A's own
# plan builder already marks them `unsupported=True` at the PLAN level
# (never an executable action), so the pre-existing `if not plan.is_
# executable: reject` check (before this set is even consulted) already
# blocks them with zero ffmpeg subprocess calls. No Dolby-Vision-specific
# or HDR_OTHER-specific code was added or is needed here (Stage 3/4's own
# firewall requirement satisfied for free, same pattern as D-274C's own
# HEVC+HDR/HEVC+10-bit firewall).
_UNSUPPORTED_ACTIONS: frozenset[str] = frozenset()

# D-274D Stage 2/8: the two HDR tone-map actions this gate implements.
_HDR_TONEMAP_ACTIONS: frozenset[str] = frozenset({
    snp.ACTION_HDR_PQ_TO_SDR_BT709,
    snp.ACTION_HDR_HLG_TO_SDR_BT709,
})

# D-274D Stage 8/9: the real, empirically-verified ffmpeg tone-map chain
# (this sandbox's own zscale+tonemap+zscale filters, proven against real
# PQ- and HLG-tagged synthetic fixtures via the actual D-271 profiler and
# D-272 policy re-evaluation -- see docs/CUTSELL_DECISIONS.md D-274D).
# Deliberately NOT a plain `format=yuv420p` (Stage 8's own explicit
# "that is the original defect"): decode -> linearize in the SOURCE
# transfer domain (zscale transfer=linear) -> convert to float RGB for
# the tonemap filter's own required pixel format -> convert primaries to
# BT.709 (still linear) -> apply the actual tone-mapping algorithm
# (Hable, a well-established filmic curve) -> convert back to BT.709
# transfer/matrix/TV range -> final yuv420p. `zscale`'s own `transfer=
# linear` step reads the INPUT's real transfer characteristics from its
# own metadata (Stage 5/6: PQ vs HLG each has its own, distinct transfer
# tag, both genuinely present in D-271's own probed `color_transfer`
# field) -- this executor never guesses or infers a transfer function.
#
# `npl` (nominal peak luminance, Stage 8's own "signal peak") is a
# disclosed, provisional numeric default (100 nits) -- the same
# "well-established default, not an invented arbitrary business number"
# treatment D-274B's own CRF-18 precedent used, since no real mastering-
# display/content-light-level metadata is probed by D-271 today to
# supply a source-specific value. Flagged for Product Owner confirmation
# in the decision log if per-source precision ever matters.
_TONEMAP_NOMINAL_PEAK_LUMINANCE_DEFAULT = 100.0


def _hdr_tonemap_filter_segment() -> str:
    return (
        f"zscale=transfer=linear:npl={_TONEMAP_NOMINAL_PEAK_LUMINANCE_DEFAULT}"
        ",format=gbrpf32le"
        ",zscale=primaries=bt709"
        ",tonemap=tonemap=hable:desat=0"
        ",zscale=transfer=bt709:matrix=bt709:range=tv"
        ",format=yuv420p"
    )

# --- Stage 23 -- structured failure category vocabulary ---------------------
# The four generically-applicable categories D-274A already pre-declared
# (Stage 23's own "create/reuse typed failures") are reused directly by
# value; the two D-274A did NOT pre-declare are new, locally-scoped
# constants -- types/vocabulary for a future executor stayed in D-274A's
# own module, but THIS gate's genuinely-new categories belong here, not
# retrofitted into source_normalization_plan.py (Stage 1 separation).
FAILURE_FFMPEG_FAILED = snp.NORMALIZATION_FFMPEG_FAILED
FAILURE_TIMEOUT = snp.NORMALIZATION_TIMEOUT
FAILURE_OUTPUT_MISSING = snp.NORMALIZATION_OUTPUT_MISSING
FAILURE_OUTPUT_EMPTY = snp.NORMALIZATION_OUTPUT_EMPTY
FAILURE_UNSUPPORTED_ACTION = "NORMALIZATION_UNSUPPORTED_ACTION"
FAILURE_ATOMIC_PROMOTION_FAILED = "NORMALIZATION_ATOMIC_PROMOTION_FAILED"
FAILURE_TIMEOUT_POLICY_REQUIRED = "NORMALIZATION_TIMEOUT_POLICY_REQUIRED"
FAILURE_INVALID_INPUT = "NORMALIZATION_INVALID_INPUT"
FAILURE_SECOND_PASS_REJECTED = "NORMALIZATION_SECOND_PASS_REJECTED"
# D-274C Stage 21: reuses D-274A's own pre-declared (previously unused)
# NORMALIZATION_CODEC_UNAVAILABLE -- Stage 4's own "verify canonical H264
# encoder availability... no fallback encoder silently introduced" check.
FAILURE_CODEC_UNAVAILABLE = snp.NORMALIZATION_CODEC_UNAVAILABLE
# D-274D Stage 9/10: mirrors FAILURE_CODEC_UNAVAILABLE's own pattern for
# the tonemap capability pre-check -- a new, locally-scoped category
# (this gate's genuinely new failure mode), never retrofitted into
# source_normalization_plan.py's own vocabulary module (Stage 1
# separation, same discipline D-274C already followed for codec).
FAILURE_HDR_CAPABILITY_UNAVAILABLE = "NORMALIZATION_HDR_CAPABILITY_UNAVAILABLE"


@dataclass(frozen=True)
class NormalizationExecutionFailure:
    """Stage 23: bounded, structured evidence for one normalization
    execution failure. Never carries raw command argv/env -- only a
    deterministic fingerprint (Stage 25) -- and stderr is bounded
    (Stage 24's own "no secret leakage" reused verbatim from D-266)."""

    error_category: str
    return_code: int | None
    command_fingerprint: str
    stderr_excerpt: str
    timed_out: bool
    timeout_sec: float | None
    plan_identity: str


class NormalizationExecutionError(RuntimeError):
    def __init__(self, failure: NormalizationExecutionFailure) -> None:
        super().__init__(f"normalization_execution_failed:{failure.error_category}")
        self.failure = failure


@dataclass(frozen=True)
class NormalizationExecutionResult:
    """The full, bounded outcome of one `execute_source_normalization` call.
    `outcome` is one of D-274A's own vocabulary (NORMALIZATION_SUCCEEDED /
    NORMALIZATION_FAILED / NORMALIZATION_UNSUPPORTED /
    NORMALIZATION_VERIFICATION_FAILED) -- never a new one invented here."""

    outcome: str
    normalized_path: str | None
    normalized_reference: "snp.NormalizedSourceReference | None"
    normalized_profile: "smp.SourceMediaProfile | None"
    verification: "snp.NormalizationVerificationResult | None"
    diagnostics: dict
    failure: NormalizationExecutionFailure | None


def _command_fingerprint(command: list[str]) -> str:
    """Stage 25: reuses render.py's own `_command_fingerprint` convention
    (SHA-256 of `json.dumps(list(command))`, `[:24]`) rather than inventing
    a new scheme -- deliberately MIRRORED, not imported (Stage 1)."""
    normalized = json.dumps(list(command))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _bounded_excerpt(text: str | bytes | None) -> str:
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text.strip()[:_STDERR_EXCERPT_MAX_CHARS]


def _job_local_temp_output_path(destination: Path, execution_id: str) -> Path:
    """Stage 26: mirrors render.py's own `_job_local_temp_output_path` --
    unique, job-local, same-directory temp name so `os.replace` is atomic."""
    return destination.with_name(f".{destination.name}.{execution_id}.normalizing{destination.suffix}")


def _cleanup_temp_output(temp_output: Path) -> None:
    """Stage 28: best-effort cleanup; never raises, never masks the
    primary failure the caller is already propagating."""
    try:
        if temp_output.exists():
            temp_output.unlink()
    except OSError:
        pass


# D-274D Stage 23/39: bounded, best-effort luma evidence via ffmpeg's own
# `signalstats` filter. Time-bounded (`-t 1`) rather than frame-bounded
# (`-frames:v N`): empirically, `-frames:v` only limits ENCODING/muxing
# at the output stage, not how many frames the filter graph itself
# processes before that limit applies -- `signalstats`+`metadata=print`
# still emits one stderr block per decoded frame regardless, so `-frames:
# v 1` does not actually bound cost. `-t 1` genuinely stops decode after
# one second of source time, keeping this diagnostic's cost flat and
# small regardless of total source duration (D-274B's own "cheap,
# diagnostic-only" discipline).
_LUMA_PROBE_TIMEOUT_SEC = 30.0  # reuses D-271's own _FFPROBE_TIMEOUT_SEC value


def _measure_luma_summary(path: Path) -> dict | None:
    """Diagnostic-only, NEVER fatal, NEVER gates outcome (Stage 39's own
    "before/after A/V relation, measured, no invented tolerance" applied
    to luminance rather than duration). Returns the LAST frame's own
    YMIN/YAVG/YMAX from the first second of decode, or `None` on any
    probe failure -- a caller must never treat `None` as evidence of
    anything, only as "measurement unavailable this run."""
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "info", "-y",
                "-t", "1", "-i", str(path),
                "-vf", "signalstats,metadata=print",
                "-f", "null", "-",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=_LUMA_PROBE_TIMEOUT_SEC,
        )
    except Exception:  # noqa: BLE001 -- diagnostic-only, never fatal
        return None

    stderr = completed.stderr or ""
    result: dict = {}
    for field in ("YMIN", "YAVG", "YMAX"):
        matches = re.findall(rf"lavfi\.signalstats\.{field}=([0-9.eE+-]+)", stderr)
        if matches:
            try:
                result[field.lower()] = float(matches[-1])
            except ValueError:
                pass
    return result or None


def _unsupported_actions_present(plan: "snp.SourceNormalizationPlan") -> tuple[str, ...]:
    """Stage 37: inspect every action field on the plan for anything beyond
    this gate's supported vocabulary -- BEFORE any ffmpeg invocation."""
    candidate_actions = (
        plan.hdr_action,
        plan.codec_action,
        plan.bit_depth_action,
        plan.pixel_format_action,
        plan.container_action,
        plan.audio_action,
    )
    return tuple(a for a in candidate_actions if a in _UNSUPPORTED_ACTIONS)


def _build_filter_chain(plan: "snp.SourceNormalizationPlan") -> tuple[list[str], list[str], bool, bool]:
    """Builds the ONE-PASS (Stage 17) video/audio filter chains for the
    supported actions this plan actually requests. Returns
    (video_filters, audio_filters, needs_audio_reencode, needs_bt709_tagging).

    D-274D Stage 16 ordering: rotation -> HDR tonemap -> VFR fps ->
    timeline setpts. Rotation (`transpose`) and the timing filters
    (`fps`, `setpts`) operate on frame geometry/timestamps only and are
    provably independent of pixel color values, so their relative order
    around the tonemap step never changes correctness; tonemap is placed
    directly after rotation and before any timing filter purely so the
    filter chain reads as "fix orientation, fix color, fix timing" in one
    pass, matching the plan's own field declaration order.

    `needs_bt709_tagging` (D-274D Stage 8): `True` only when this plan's
    `hdr_action` is one of `_HDR_TONEMAP_ACTIONS` -- signals the caller to
    add explicit `-color_primaries/-color_trc/-colorspace bt709` output
    flags. Deliberately NEVER set for a non-HDR normalization (Stage 8's
    own "do not mislabel non-HDR outputs" -- a rotation-only or VFR-only
    output carries whatever color tags its own source already had; this
    executor does not invent or force BT.709 on media it never touched
    photometrically)."""
    video_filters: list[str] = []
    audio_filters: list[str] = []
    needs_audio_reencode = False
    needs_bt709_tagging = False

    rotation_filter = _ROTATION_FILTER.get(plan.rotation_action)
    if rotation_filter:
        video_filters.append(rotation_filter)

    if plan.hdr_action in _HDR_TONEMAP_ACTIONS:
        video_filters.append(_hdr_tonemap_filter_segment())
        needs_bt709_tagging = True

    if plan.frame_rate_action == snp.ACTION_VFR_TO_CFR:
        # Stage 10: the numeric target comes ONLY from the plan's own
        # target_fps (D-271's profile.effective_fps) -- never a hardcoded
        # 30, never re-derived here.
        if plan.target_fps is None or plan.target_fps <= 0:
            raise ValueError("plan requests VFR_TO_CFR but carries no usable target_fps")
        video_filters.append(f"fps={plan.target_fps}")

    if plan.timeline_action == snp.ACTION_TIMELINE_TO_ZERO:
        video_filters.append("setpts=PTS-STARTPTS")
        audio_filters.append("asetpts=PTS-STARTPTS")
        # Stage 14: asetpts requires decoding/re-encoding audio -- ffmpeg
        # filters cannot run on stream-copied audio. This is the one,
        # disclosed, narrow exception to "preserve audio content" -- only
        # timestamps are intentionally touched; sample rate/channels/gain
        # are never forced (no -ar/-ac/loudnorm anywhere in this module).
        needs_audio_reencode = True

    return video_filters, audio_filters, needs_audio_reencode, needs_bt709_tagging


def execute_source_normalization(
    source_path: str,
    plan: "snp.SourceNormalizationPlan",
    *,
    output_directory: str,
    timeout_sec: float | None = NORMALIZATION_FFMPEG_TIMEOUT_SEC,
    runtime_capability: "sfp.RuntimeCapabilityInput | None" = None,
    attempt_count: int = 0,
    codec_capability: "prc.ProductionRuntimeCapability | None" = None,
    tonemap_capability: "prc.ProductionRuntimeCapability | None" = None,
) -> NormalizationExecutionResult:
    """Stages 2/17/32-35: execute exactly the actions `plan` names, in one
    ffmpeg generation, then run the MANDATORY D-271 re-probe + D-272
    re-evaluation + D-274A verification loop on the actual output. Never
    re-decides whether normalization is needed (Stage 2); never runs a
    second pass (Stage 18/35).

    `codec_capability` (D-274C Stage 4, optional, defaults to `None` for
    full backward compatibility with every pre-D-274C caller/test): when
    the plan requests `ACTION_HEVC_TO_H264` and a `ProductionRuntime
    Capability` is supplied, its `h264_encoder_available` is checked
    BEFORE any ffmpeg call -- "no fallback encoder silently introduced."
    When omitted, this pre-check is skipped and ffmpeg's own nonzero-exit
    failure path remains the safety net (unchanged D-274B behavior).

    `tonemap_capability` (D-274D Stage 9/10, optional, defaults to `None`
    for the identical backward-compatibility reason): when the plan
    requests an HDR tonemap action and a `ProductionRuntimeCapability` is
    supplied, its `hdr_tonemap_usable` (zscale AND tonemap both listed)
    is checked BEFORE any ffmpeg call. This is a SEPARATE, distinctly-
    named parameter from `codec_capability` even though both currently
    accept the same `ProductionRuntimeCapability` type -- Stage 1's own
    "PROFILE -> POLICY -> PLAN -> EXECUTOR" separation-of-concerns
    discipline extends to keeping each gate's own capability check
    independently toggleable by its own caller (a caller may know H264
    capability but not yet know tonemap capability, or vice versa); a
    single shared parameter would force both checks to rise or fall
    together. When omitted, this pre-check is skipped and ffmpeg's own
    nonzero-exit failure path remains the safety net."""
    diagnostics: dict = {
        "plan_identity": plan.plan_identity,
        "actions_requested": {
            "rotation_action": plan.rotation_action,
            "frame_rate_action": plan.frame_rate_action,
            "timeline_action": plan.timeline_action,
            # D-274D Stage 39: the four fields this gate's own actions
            # live on -- codec_action was already executable pre-D-274D
            # (D-274C) but was never disclosed in this dict either; added
            # here alongside the new fields for one complete picture.
            "codec_action": plan.codec_action,
            "hdr_action": plan.hdr_action,
            "bit_depth_action": plan.bit_depth_action,
            "pixel_format_action": plan.pixel_format_action,
        },
        "target_fps": plan.target_fps,
    }

    # Stage 18/28 -- one-pass firewall. Purely stateless: the CALLER tracks
    # attempt state across the job and supplies it here.
    if not snp.is_normalization_attempt_allowed(attempt_count):
        diagnostics["execution_status"] = "REJECTED_SECOND_PASS"
        return NormalizationExecutionResult(
            outcome=snp.NORMALIZATION_FAILED,
            normalized_path=None, normalized_reference=None, normalized_profile=None,
            verification=None, diagnostics=diagnostics,
            failure=NormalizationExecutionFailure(
                error_category=FAILURE_SECOND_PASS_REJECTED,
                return_code=None, command_fingerprint="", stderr_excerpt="",
                timed_out=False, timeout_sec=timeout_sec, plan_identity=plan.plan_identity,
            ),
        )

    if not plan.is_executable:
        diagnostics["execution_status"] = "REJECTED_NOT_EXECUTABLE"
        return NormalizationExecutionResult(
            outcome=snp.NORMALIZATION_FAILED,
            normalized_path=None, normalized_reference=None, normalized_profile=None,
            verification=None, diagnostics=diagnostics,
            failure=NormalizationExecutionFailure(
                error_category=FAILURE_INVALID_INPUT,
                return_code=None, command_fingerprint="", stderr_excerpt="",
                timed_out=False, timeout_sec=timeout_sec, plan_identity=plan.plan_identity,
            ),
        )

    # Stage 37 -- unsupported-action pre-check, BEFORE any ffmpeg call.
    unsupported = _unsupported_actions_present(plan)
    if unsupported:
        diagnostics["execution_status"] = "REJECTED_UNSUPPORTED_ACTION"
        diagnostics["unsupported_actions"] = list(unsupported)
        return NormalizationExecutionResult(
            outcome=snp.NORMALIZATION_UNSUPPORTED,
            normalized_path=None, normalized_reference=None, normalized_profile=None,
            verification=None, diagnostics=diagnostics,
            failure=NormalizationExecutionFailure(
                error_category=FAILURE_UNSUPPORTED_ACTION,
                return_code=None, command_fingerprint="", stderr_excerpt="",
                timed_out=False, timeout_sec=timeout_sec, plan_identity=plan.plan_identity,
            ),
        )

    # D-274C Stage 4 -- H264 encoder availability pre-check, BEFORE ffmpeg.
    # Only consulted when the caller supplies `codec_capability` AND the
    # plan actually requests HEVC_TO_H264; a caller that never passes this
    # (every pre-D-274C test/call site) gets byte-identical behavior.
    if (
        plan.codec_action == snp.ACTION_HEVC_TO_H264
        and codec_capability is not None
        and not codec_capability.h264_encoder_available
    ):
        diagnostics["execution_status"] = "REJECTED_CODEC_UNAVAILABLE"
        return NormalizationExecutionResult(
            outcome=snp.NORMALIZATION_FAILED,
            normalized_path=None, normalized_reference=None, normalized_profile=None,
            verification=None, diagnostics=diagnostics,
            failure=NormalizationExecutionFailure(
                error_category=FAILURE_CODEC_UNAVAILABLE,
                return_code=None, command_fingerprint="", stderr_excerpt="",
                timed_out=False, timeout_sec=timeout_sec, plan_identity=plan.plan_identity,
            ),
        )

    # D-274D Stage 9/10 -- HDR tonemap capability pre-check, BEFORE ffmpeg.
    # Only consulted when the caller supplies `tonemap_capability` AND the
    # plan actually requests an HDR tonemap action; a caller that never
    # passes this (every pre-D-274D test/call site) gets byte-identical
    # behavior. Mirrors D-274C's own codec pre-check exactly (Stage 9's
    # own "do not assume availability" applied to the second capability
    # this executor now depends on).
    if (
        plan.hdr_action in _HDR_TONEMAP_ACTIONS
        and tonemap_capability is not None
        and not tonemap_capability.hdr_tonemap_usable
    ):
        diagnostics["execution_status"] = "REJECTED_HDR_CAPABILITY_UNAVAILABLE"
        return NormalizationExecutionResult(
            outcome=snp.NORMALIZATION_FAILED,
            normalized_path=None, normalized_reference=None, normalized_profile=None,
            verification=None, diagnostics=diagnostics,
            failure=NormalizationExecutionFailure(
                error_category=FAILURE_HDR_CAPABILITY_UNAVAILABLE,
                return_code=None, command_fingerprint="", stderr_excerpt="",
                timed_out=False, timeout_sec=timeout_sec, plan_identity=plan.plan_identity,
            ),
        )

    # Stage 22 -- timeout seam. No number is ever silently chosen.
    if timeout_sec is None:
        diagnostics["execution_status"] = "REJECTED_TIMEOUT_POLICY_REQUIRED"
        return NormalizationExecutionResult(
            outcome=PRODUCT_OWNER_NORMALIZATION_TIMEOUT_REQUIRED,
            normalized_path=None, normalized_reference=None, normalized_profile=None,
            verification=None, diagnostics=diagnostics,
            failure=NormalizationExecutionFailure(
                error_category=FAILURE_TIMEOUT_POLICY_REQUIRED,
                return_code=None, command_fingerprint="", stderr_excerpt="",
                timed_out=False, timeout_sec=None, plan_identity=plan.plan_identity,
            ),
        )

    source = Path(source_path)
    if not source.exists() or source.stat().st_size <= 0:
        diagnostics["execution_status"] = "REJECTED_INVALID_SOURCE"
        return NormalizationExecutionResult(
            outcome=snp.NORMALIZATION_FAILED,
            normalized_path=None, normalized_reference=None, normalized_profile=None,
            verification=None, diagnostics=diagnostics,
            failure=NormalizationExecutionFailure(
                error_category=FAILURE_INVALID_INPUT,
                return_code=None, command_fingerprint="", stderr_excerpt="",
                timed_out=False, timeout_sec=timeout_sec, plan_identity=plan.plan_identity,
            ),
        )

    # Stage 29 -- original immutability evidence (provenance only).
    original_sha256_before = compute_output_sha256(source)

    # Stage 15/41 -- "before" profile/duration evidence, best-effort. A
    # probe failure here must never block execution -- it is diagnostic
    # only, never authority.
    profile_before = None
    duration_before = None
    try:
        profile_before = smp.probe_source_media_profile(str(source))
        duration_before = profile_before.duration_sec
    except Exception:  # noqa: BLE001 -- diagnostics-only, never fatal
        pass
    diagnostics["duration_before_sec"] = duration_before

    # D-274D Stage 23/39 -- "before" luma evidence, HDR actions only (the
    # only case this gate genuinely changes pixel VALUES, not just
    # geometry/timing/container). Best-effort, diagnostic-only, bounded --
    # see `_measure_luma_summary`'s own docstring.
    if plan.hdr_action in _HDR_TONEMAP_ACTIONS:
        diagnostics["luma_before"] = _measure_luma_summary(source)

    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    execution_id = uuid.uuid4().hex[:16]
    final_output = out_dir / f"normalized_{plan.plan_identity}_{execution_id}.mp4"
    temp_output = _job_local_temp_output_path(final_output, execution_id)

    try:
        video_filters, audio_filters, needs_audio_reencode, needs_bt709_tagging = _build_filter_chain(plan)
        diagnostics["needs_bt709_tagging"] = needs_bt709_tagging
    except ValueError:
        diagnostics["execution_status"] = "REJECTED_INVALID_PLAN"
        return NormalizationExecutionResult(
            outcome=snp.NORMALIZATION_FAILED,
            normalized_path=None, normalized_reference=None, normalized_profile=None,
            verification=None, diagnostics=diagnostics,
            failure=NormalizationExecutionFailure(
                error_category=FAILURE_INVALID_INPUT,
                return_code=None, command_fingerprint="", stderr_excerpt="",
                timed_out=False, timeout_sec=timeout_sec, plan_identity=plan.plan_identity,
            ),
        )

    # --- Stage 8/24 -- command construction ---------------------------------
    # -noautorotate placed immediately before -i: defensive, empirically
    # confirmed (real ffmpeg -h full input option), guarantees ffmpeg never
    # ALSO applies its own metadata-driven rotation on top of the plan's
    # explicit transpose (Stage 8's own "no double autorotation").
    command: list[str] = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-noautorotate",
        "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0?",
    ]
    if video_filters:
        command += ["-vf", ",".join(video_filters)]
    if audio_filters:
        command += ["-af", ",".join(audio_filters)]
    command += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
    if needs_bt709_tagging:
        # D-274D Stage 8: explicit BT.709 output color-metadata tags --
        # ONLY when this execution actually ran the HDR tonemap filter
        # chain (Stage 8's own "do not mislabel non-HDR outputs"). This is
        # genuinely writable metadata (unlike rotation metadata, which
        # D-271's own forensic proved this ffmpeg build cannot write) --
        # empirically confirmed via real ffprobe re-probe of a tagged
        # fixture (docs/CUTSELL_DECISIONS.md D-274D).
        command += [
            "-color_primaries", "bt709", "-color_trc", "bt709",
            "-colorspace", "bt709", "-color_range", "tv",
        ]
    if needs_audio_reencode:
        # Stage 14: re-encode ONLY because asetpts requires it; omit
        # -ar/-ac so the encoder's own defaults preserve the original
        # sample rate/channel count -- never forced to 48k/stereo.
        command += ["-c:a", "aac"]
    else:
        command += ["-c:a", "copy"]
    command += [str(temp_output)]

    fingerprint = _command_fingerprint(command)
    diagnostics["command_fingerprint"] = fingerprint

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=timeout_sec, shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        _cleanup_temp_output(temp_output)
        diagnostics["execution_status"] = "FAILED_TIMEOUT"
        diagnostics["wall_time_sec"] = round(time.monotonic() - started, 3)
        return NormalizationExecutionResult(
            outcome=snp.NORMALIZATION_FAILED,
            normalized_path=None, normalized_reference=None, normalized_profile=None,
            verification=None, diagnostics=diagnostics,
            failure=NormalizationExecutionFailure(
                error_category=FAILURE_TIMEOUT,
                return_code=None, command_fingerprint=fingerprint,
                stderr_excerpt=_bounded_excerpt(exc.stderr),
                timed_out=True, timeout_sec=timeout_sec, plan_identity=plan.plan_identity,
            ),
        )
    wall_time_sec = time.monotonic() - started
    diagnostics["wall_time_sec"] = round(wall_time_sec, 3)
    diagnostics["return_code"] = completed.returncode

    if completed.returncode != 0:
        _cleanup_temp_output(temp_output)
        diagnostics["execution_status"] = "FAILED_FFMPEG_NONZERO"
        return NormalizationExecutionResult(
            outcome=snp.NORMALIZATION_FAILED,
            normalized_path=None, normalized_reference=None, normalized_profile=None,
            verification=None, diagnostics=diagnostics,
            failure=NormalizationExecutionFailure(
                error_category=FAILURE_FFMPEG_FAILED,
                return_code=completed.returncode, command_fingerprint=fingerprint,
                stderr_excerpt=_bounded_excerpt(completed.stderr),
                timed_out=False, timeout_sec=timeout_sec, plan_identity=plan.plan_identity,
            ),
        )

    # --- Stage 26/27 -- validate + atomic promotion -------------------------
    if not temp_output.exists():
        diagnostics["execution_status"] = "FAILED_OUTPUT_MISSING"
        return NormalizationExecutionResult(
            outcome=snp.NORMALIZATION_FAILED,
            normalized_path=None, normalized_reference=None, normalized_profile=None,
            verification=None, diagnostics=diagnostics,
            failure=NormalizationExecutionFailure(
                error_category=FAILURE_OUTPUT_MISSING,
                return_code=completed.returncode, command_fingerprint=fingerprint,
                stderr_excerpt=_bounded_excerpt(completed.stderr),
                timed_out=False, timeout_sec=timeout_sec, plan_identity=plan.plan_identity,
            ),
        )
    if temp_output.stat().st_size <= 0:
        _cleanup_temp_output(temp_output)
        diagnostics["execution_status"] = "FAILED_OUTPUT_EMPTY"
        return NormalizationExecutionResult(
            outcome=snp.NORMALIZATION_FAILED,
            normalized_path=None, normalized_reference=None, normalized_profile=None,
            verification=None, diagnostics=diagnostics,
            failure=NormalizationExecutionFailure(
                error_category=FAILURE_OUTPUT_EMPTY,
                return_code=completed.returncode, command_fingerprint=fingerprint,
                stderr_excerpt=_bounded_excerpt(completed.stderr),
                timed_out=False, timeout_sec=timeout_sec, plan_identity=plan.plan_identity,
            ),
        )

    try:
        os.replace(temp_output, final_output)
    except OSError as exc:
        _cleanup_temp_output(temp_output)
        diagnostics["execution_status"] = "FAILED_ATOMIC_PROMOTION"
        return NormalizationExecutionResult(
            outcome=snp.NORMALIZATION_FAILED,
            normalized_path=None, normalized_reference=None, normalized_profile=None,
            verification=None, diagnostics=diagnostics,
            failure=NormalizationExecutionFailure(
                error_category=FAILURE_ATOMIC_PROMOTION_FAILED,
                return_code=completed.returncode, command_fingerprint=fingerprint,
                stderr_excerpt=_bounded_excerpt(str(exc)),
                timed_out=False, timeout_sec=timeout_sec, plan_identity=plan.plan_identity,
            ),
        )

    # --- Stage 29 -- prove original bytes are untouched ---------------------
    original_sha256_after = compute_output_sha256(source)
    diagnostics["original_source_unchanged"] = (original_sha256_before == original_sha256_after)

    # --- Stage 30/31 -- normalized hash + reference (actual bytes only) -----
    normalized_output_sha256 = compute_output_sha256(final_output)
    reference = snp.NormalizedSourceReference(
        original_source_identity=plan.source_identity,
        normalization_plan_identity=plan.plan_identity,
        normalized_output_sha256=normalized_output_sha256,
    )
    diagnostics["normalized_output_sha256"] = normalized_output_sha256
    diagnostics["normalized_distinct_from_original"] = (normalized_output_sha256 != original_sha256_after)

    # --- Stage 32/33 -- MANDATORY re-probe + re-evaluate ---------------------
    normalized_profile = smp.probe_source_media_profile(str(final_output))
    normalized_decision = sfp.evaluate_source_format_policy(
        normalized_profile, runtime_capability=runtime_capability or sfp.RuntimeCapabilityInput(),
    )
    diagnostics["duration_after_sec"] = normalized_profile.duration_sec
    diagnostics["normalized_profile_summary"] = {
        "video_codec": normalized_profile.video_codec,
        "pixel_format": normalized_profile.pixel_format,
        "bit_depth": normalized_profile.bit_depth,
        "vfr_status": normalized_profile.vfr_status,
        "effective_fps": normalized_profile.effective_fps,
        "rotation_degrees": normalized_profile.rotation_degrees,
        "format_start_time": normalized_profile.format_start_time,
        "video_stream_start_time": normalized_profile.video_stream_start_time,
        "audio_stream_start_time": normalized_profile.audio_stream_start_time,
        "container_name": normalized_profile.container_name,
        # D-274D Stage 39: HDR/color-metadata evidence, both directions --
        # the exact D-271 fields the D-272 re-evaluation below actually
        # keys its ACCEPT/REJECT decision on for HDR/bit-depth/pixel
        # format, disclosed rather than left implicit in `final_d272_
        # decision` alone.
        "hdr_status": normalized_profile.hdr_status,
        "color_primaries": normalized_profile.color_primaries,
        "color_transfer": normalized_profile.color_transfer,
        "color_space": normalized_profile.color_space,
    }
    diagnostics["final_d272_decision"] = normalized_decision.decision

    # D-274D Stage 39 -- HDR-specific "before" disclosure (input side) and
    # "after" luma evidence (this execution's own output, HDR actions
    # only -- the before/after pair `_measure_luma_summary` produces is
    # only meaningful for a genuine value transformation, never for
    # rotation/VFR/timeline/HEVC-only executions).
    diagnostics["hdr_action"] = plan.hdr_action
    diagnostics["input_color_metadata"] = {
        "hdr_status": profile_before.hdr_status if profile_before is not None else None,
        "color_primaries": profile_before.color_primaries if profile_before is not None else None,
        "color_transfer": profile_before.color_transfer if profile_before is not None else None,
        "color_space": profile_before.color_space if profile_before is not None else None,
    }
    if plan.hdr_action in _HDR_TONEMAP_ACTIONS:
        diagnostics["luma_after"] = _measure_luma_summary(final_output)

    # --- Stage 15 -- A/V relation before/after, measured, no invented tolerance
    if duration_before is not None and normalized_profile.duration_sec is not None:
        diagnostics["duration_delta_sec"] = round(normalized_profile.duration_sec - duration_before, 6)

    # --- Stage 18/19/21/35 -- verification -----------------------------------
    verification = snp.verify_normalized_source(normalized_profile, normalized_decision)
    d272_outcome = snp.verification_outcome(verification)

    # D-274E Stage 19/20 -- D-272 ACCEPT alone is no longer sufficient: the
    # canonical output-format QC authority (`output_format_qc.py`, the ONE
    # place format verification logic lives -- Stage 1) must ALSO be
    # consulted against `NORMALIZED_SOURCE_CONTRACT_V1`. Only a genuine QC
    # `FAIL` (a proven contract VIOLATION, e.g. remaining HDR, wrong pixel
    # format, wrong container) turns an otherwise-ACCEPTed normalization
    # into `NORMALIZATION_VERIFICATION_FAILED` -- reusing D-274A's own
    # existing outcome category (Stage 20's own "never invent a new one").
    # A QC `PARTIAL`/`UNKNOWN` (missing, not violated, evidence -- e.g. a
    # rotation/VFR/timeline/HEVC-only normalization that never had reason
    # to write explicit BT709 color tags) does NOT retroactively fail an
    # otherwise-legitimate D-272 ACCEPT: D-272 remains the sole authority
    # for what "normalization is complete" means; this is an ADDITIVE
    # safety net for genuine violations, never a stricter re-litigation of
    # every already-passing D-274B/C/D normalization path (Stage 21's own
    # "no second normalization pass" implies this must never regress a
    # today-successful case, only catch a real one D-272 itself missed).
    format_qc_result = ofq.verify_output_format(normalized_profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    diagnostics["format_qc_status"] = format_qc_result.status
    diagnostics["format_qc_failed_checks"] = list(format_qc_result.failed_checks)
    diagnostics["format_qc_unknown_checks"] = list(format_qc_result.unknown_checks)

    if d272_outcome == snp.NORMALIZATION_SUCCEEDED and format_qc_result.status == ofq.STATUS_FAIL:
        outcome = snp.NORMALIZATION_VERIFICATION_FAILED
        diagnostics["execution_status"] = "FORMAT_QC_FAILED_AFTER_D272_ACCEPT"
    else:
        outcome = d272_outcome
        diagnostics["execution_status"] = "SUCCESS" if outcome == snp.NORMALIZATION_SUCCEEDED else "VERIFICATION_FAILED"
    diagnostics["verification_status"] = outcome

    return NormalizationExecutionResult(
        outcome=outcome,
        normalized_path=str(final_output),
        normalized_reference=reference,
        normalized_profile=normalized_profile,
        verification=verification,
        diagnostics=diagnostics,
        failure=None if outcome == snp.NORMALIZATION_SUCCEEDED else NormalizationExecutionFailure(
            error_category=snp.NORMALIZATION_VERIFICATION_FAILED,
            return_code=completed.returncode, command_fingerprint=fingerprint,
            stderr_excerpt="", timed_out=False, timeout_sec=timeout_sec,
            plan_identity=plan.plan_identity,
        ),
    )
