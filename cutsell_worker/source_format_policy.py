"""Source Format Policy / Early Media Gate -- D-272.

Post D-271 (source media probe + format classification foundation,
Verdict A: `SourceMediaProfile`/`classify_source_format` proven, wired
into no live call site). This module answers the ONE question D-271's
own classification stopped short of: **should CutSell spend expensive
editorial compute (ASR, GPU, semantic reasoning) on this source AS-IS,
does it need normalization first, or should it be rejected before any
of that work starts?**

`evaluate_source_format_policy` is a pure function of a `SourceMediaProfile`
(D-271, never re-probed here) plus an optional `RuntimeCapabilityInput`
(explicitly NOT the same thing as D-271's own `LocalFfmpegCapabilitySnapshot`
-- see that type's own docstring). It returns one immutable, frozen
`SourceFormatPolicyDecision`: `ACCEPT` / `NORMALIZE_REQUIRED` / `REJECT` /
`INSUFFICIENT_EVIDENCE`, with machine-readable reasons split into
`blocking_reasons` / `normalization_reasons` / `warnings` (Stage 30's own
severity distinction).

## What this module explicitly does NOT do (D-272's own binding scope)

No transcode, no HDR tonemap, no rotation pixel transform, no fps
conversion, no codec conversion, no resample policy change, no
filtergraph change. This is a read-only decision, never a mutation.

## Integration seam, deliberately NOT activated in this gate

`evaluate_source_for_editorial_entry` composes D-271's own
`probe_source_media_profile` with this module's `evaluate_source_format_
policy` into the ONE pure function a future call site (the earliest
identified seam: `cutsell_worker/worker_job.py::run_flow_b_job`, right
after its own existing per-source `probe_media` call and before
`process_local_sources` -- the ASR/semantic-reasoning entry point) would
call to gate a real source before expensive compute. This gate proves
that seam works correctly against real synthetic media -- it does NOT
wire it into `worker_job.py` itself. Rejecting or blocking a real user's
upload is an editorial/product-policy decision (CLAUDE.md's own D-091
escalation condition A: "product behavior, UX doctrine, workflow,
editorial policy, ... acceptance criteria"), not a decision this offline
implementation gate is authorized to activate silently. `worker_job.py`
is confirmed untouched by this gate's own regression firewall.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import source_media_profile as smp

# =============================================================================
# Stage 3 -- policy version, single canonical owner
# =============================================================================

SOURCE_FORMAT_POLICY_VERSION = 1

# =============================================================================
# Stage 2 -- decision vocabulary (bounded, never proliferated)
# =============================================================================

DECISION_ACCEPT = "ACCEPT"
DECISION_NORMALIZE_REQUIRED = "NORMALIZE_REQUIRED"
DECISION_REJECT = "REJECT"
DECISION_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

# =============================================================================
# Stage 30 -- reason severity vocabulary
# =============================================================================

REASON_SEVERITY_BLOCKING = "BLOCKING"
REASON_SEVERITY_NORMALIZATION = "NORMALIZATION"
REASON_SEVERITY_WARNING = "WARNING"

# =============================================================================
# Stage 29 -- reason codes
# =============================================================================

REASON_MISSING_VIDEO = "MISSING_VIDEO"
REASON_PROBE_FAILED = "PROBE_FAILED"
REASON_UNSUPPORTED_CONTAINER = "UNSUPPORTED_CONTAINER"
REASON_UNKNOWN_CONTAINER = "UNKNOWN_CONTAINER"
REASON_UNSUPPORTED_CODEC = "UNSUPPORTED_CODEC"
REASON_UNKNOWN_CODEC = "UNKNOWN_CODEC"
REASON_HEVC_RUNTIME_UNVERIFIED = "HEVC_RUNTIME_UNVERIFIED"
REASON_CODEC_RUNTIME_UNVERIFIED = "CODEC_RUNTIME_UNVERIFIED"
REASON_ROTATION_NORMALIZATION_REQUIRED = "ROTATION_NORMALIZATION_REQUIRED"
REASON_HDR_NORMALIZATION_REQUIRED = "HDR_NORMALIZATION_REQUIRED"
REASON_VFR_NORMALIZATION_REQUIRED = "VFR_NORMALIZATION_REQUIRED"
REASON_TEN_BIT_NORMALIZATION_REQUIRED = "TEN_BIT_NORMALIZATION_REQUIRED"
REASON_PIXEL_FORMAT_NORMALIZATION_REQUIRED = "PIXEL_FORMAT_NORMALIZATION_REQUIRED"
REASON_MULTIPLE_VIDEO_STREAMS = "MULTIPLE_VIDEO_STREAMS"
REASON_MULTIPLE_AUDIO_STREAMS = "MULTIPLE_AUDIO_STREAMS"
REASON_AUDIO_MISSING = "AUDIO_MISSING"
REASON_INVALID_DIMENSIONS = "INVALID_DIMENSIONS"
REASON_RESOURCE_RISK_HIGH_RESOLUTION = "RESOURCE_RISK_HIGH_RESOLUTION"
REASON_RESOURCE_RISK_LONG_DURATION = "RESOURCE_RISK_LONG_DURATION"
REASON_COLOR_METADATA_UNCERTAIN = "COLOR_METADATA_UNCERTAIN"

# Stage 22: no video codec is currently confirmed hard-unsupported at the
# renderer level -- D-270's own audit found "nothing at the codec level
# by the renderer itself" (only container-level MKV/AVI is rejected, at
# upload). This set is deliberately empty today: a real, structural seam
# for a FUTURE finding to populate, never fabricated ahead of evidence
# (Stage 33's own "no silent fallback" cuts both ways -- this module
# never invents a rejection either).
KNOWN_UNSUPPORTED_VIDEO_CODECS: frozenset[str] = frozenset()

# Codecs whose production decode capability D-270 found genuinely
# unverified, keyed to the specific `RuntimeCapabilityInput` field that
# can confirm each one. Any OTHER non-H.264 codec (VP9/ProRes/MPEG4) has
# no confirmation mechanism defined yet at all, so it is always
# capability-unverified (Stage 9's own "no optimistic assumption").
_CAPABILITY_GATED_CODECS = {
    smp.VIDEO_CODEC_HEVC: ("hevc_decode_confirmed", REASON_HEVC_RUNTIME_UNVERIFIED),
    smp.VIDEO_CODEC_AV1: ("av1_decode_confirmed", REASON_CODEC_RUNTIME_UNVERIFIED),
}

_TEN_BIT_PIX_FMT_SUFFIXES = ("10le", "10be", "12le", "12be")
_EIGHT_BIT_ACCEPTABLE_PIX_FMTS = {"yuv420p", "yuvj420p"}


@dataclass(frozen=True)
class RuntimeCapabilityInput:
    """D-272 Stage 28: the PRODUCTION runtime capability evidence this
    policy consumes -- deliberately a DIFFERENT type from D-271's own
    `source_media_profile.LocalFfmpegCapabilitySnapshot`, which is
    permanently labelled local-sandbox-only and must never be fed here
    as if it were production truth. Every field defaults to `False`
    (unconfirmed) -- no caller in this codebase constructs a non-default
    instance today (Stage 28's own "do not hardcode local sandbox
    capabilities as production capabilities"); a future gate with REAL
    verified production capability evidence is the only legitimate
    source of a `True` value here."""

    hevc_decode_confirmed: bool = False
    av1_decode_confirmed: bool = False


# =============================================================================
# Stage 34 -- user-facing error code foundation
# =============================================================================

USER_FACING_UNSUPPORTED_VIDEO_FORMAT = "UNSUPPORTED_VIDEO_FORMAT"
USER_FACING_VIDEO_REQUIRES_NORMALIZATION = "VIDEO_REQUIRES_NORMALIZATION"
USER_FACING_VIDEO_CORRUPT = "VIDEO_CORRUPT"
USER_FACING_VIDEO_STREAM_AMBIGUOUS = "VIDEO_STREAM_AMBIGUOUS"
USER_FACING_RUNTIME_CODEC_SUPPORT_UNVERIFIED = "RUNTIME_CODEC_SUPPORT_UNVERIFIED"


def _user_facing_error_code(decision: str, blocking: tuple[str, ...]) -> str | None:
    if decision == DECISION_ACCEPT:
        return None
    if REASON_PROBE_FAILED in blocking:
        return USER_FACING_VIDEO_CORRUPT
    if REASON_MULTIPLE_VIDEO_STREAMS in blocking or REASON_MULTIPLE_AUDIO_STREAMS in blocking:
        return USER_FACING_VIDEO_STREAM_AMBIGUOUS
    if REASON_HEVC_RUNTIME_UNVERIFIED in blocking or REASON_CODEC_RUNTIME_UNVERIFIED in blocking:
        return USER_FACING_RUNTIME_CODEC_SUPPORT_UNVERIFIED
    if decision == DECISION_NORMALIZE_REQUIRED:
        return USER_FACING_VIDEO_REQUIRES_NORMALIZATION
    # REJECT and INSUFFICIENT_EVIDENCE both fall back to the generic
    # "we cannot confirm this format is supported" code -- never a more
    # specific claim than the evidence actually supports.
    return USER_FACING_UNSUPPORTED_VIDEO_FORMAT


# =============================================================================
# Stage 1 -- the decision type itself
# =============================================================================

@dataclass(frozen=True)
class SourceFormatPolicyDecision:
    """D-272 Stage 1: one immutable policy verdict. Never mutates the
    `SourceMediaProfile` it was built from -- carries only its own,
    separately-derived reasons and a copy of the profile's own
    `probe_status`/`source_format_class` for observability (Stage 35)."""

    decision: str
    policy_version: int
    reason_codes: tuple[str, ...]
    blocking_reasons: tuple[str, ...]
    normalization_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    source_profile_status: str
    source_format_class: str
    user_facing_error_code: str | None

    @property
    def can_enter_editorial_pipeline(self) -> bool:
        """Stage 4/31: `True` only for `ACCEPT` -- every other decision
        blocks expensive editorial entry, no exceptions."""
        return self.decision == DECISION_ACCEPT

    @property
    def requires_normalization(self) -> bool:
        return self.decision == DECISION_NORMALIZE_REQUIRED

    @property
    def is_rejected(self) -> bool:
        return self.decision == DECISION_REJECT


def _decision(
    decision: str,
    *,
    blocking: tuple[str, ...] = (),
    normalization: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
    profile: "smp.SourceMediaProfile",
) -> SourceFormatPolicyDecision:
    reason_codes = tuple(blocking) + tuple(normalization) + tuple(warnings)
    return SourceFormatPolicyDecision(
        decision=decision,
        policy_version=SOURCE_FORMAT_POLICY_VERSION,
        reason_codes=reason_codes,
        blocking_reasons=tuple(blocking),
        normalization_reasons=tuple(normalization),
        warnings=tuple(warnings),
        source_profile_status=profile.probe_status,
        source_format_class=smp.classify_source_format(profile).source_format_class,
        user_facing_error_code=_user_facing_error_code(decision, tuple(blocking)),
    )


# =============================================================================
# Stage 4-27 -- the policy itself
# =============================================================================

def evaluate_source_format_policy(
    profile: "smp.SourceMediaProfile",
    *,
    runtime_capability: RuntimeCapabilityInput | None = None,
    max_pixel_count: int | None = None,
    max_duration_sec: float | None = None,
) -> SourceFormatPolicyDecision:
    """D-272: a pure function of `profile` (and optional, real-evidence-
    only capability/resource bounds) -- never re-probes, never invokes
    ffmpeg, never mutates `profile`. Conservative by design: every branch
    that cannot positively confirm safety returns `INSUFFICIENT_EVIDENCE`
    or `REJECT` rather than an optimistic `ACCEPT` (Stage 33's own "no
    silent 'let ffmpeg try'" instruction)."""
    capability = runtime_capability or RuntimeCapabilityInput()

    # Stage 23 -- probe failed -> REJECT, no expensive processing.
    if profile.probe_status == smp.PROBE_STATUS_FAILED:
        return _decision(DECISION_REJECT, blocking=(REASON_PROBE_FAILED,), profile=profile)

    # Stage 18 -- missing video -> REJECT. Closes D-270's missing-video P0
    # at the decision layer.
    if profile.video_presence == smp.VIDEO_MISSING:
        return _decision(DECISION_REJECT, blocking=(REASON_MISSING_VIDEO,), profile=profile)

    # Stage 24 -- invalid dimensions -> REJECT. No new minimum-quality
    # threshold; only a genuinely invalid (<=0) dimension blocks.
    if (profile.coded_width is not None and profile.coded_width <= 0) or (
        profile.coded_height is not None and profile.coded_height <= 0
    ):
        return _decision(DECISION_REJECT, blocking=(REASON_INVALID_DIMENSIONS,), profile=profile)

    # Stage 22 -- container: known-unsupported REJECTs, genuinely unknown
    # is INSUFFICIENT_EVIDENCE (distinct from unsupported, per Stage 22's
    # own explicit instruction).
    if profile.container_name in (smp.CONTAINER_MKV, smp.CONTAINER_AVI):
        return _decision(DECISION_REJECT, blocking=(REASON_UNSUPPORTED_CONTAINER,), profile=profile)
    if profile.container_name == smp.CONTAINER_UNKNOWN:
        return _decision(DECISION_INSUFFICIENT_EVIDENCE, blocking=(REASON_UNKNOWN_CONTAINER,), profile=profile)

    # Stage 21 -- unknown codec -> INSUFFICIENT_EVIDENCE, never "try and see".
    if profile.video_codec is None or profile.video_codec == smp.VIDEO_CODEC_UNKNOWN:
        return _decision(DECISION_INSUFFICIENT_EVIDENCE, blocking=(REASON_UNKNOWN_CODEC,), profile=profile)

    # Stage 22 -- known-unsupported codec (structurally present, no
    # member today -- see KNOWN_UNSUPPORTED_VIDEO_CODECS's own docstring).
    if profile.video_codec in KNOWN_UNSUPPORTED_VIDEO_CODECS:
        return _decision(DECISION_REJECT, blocking=(REASON_UNSUPPORTED_CODEC,), profile=profile)

    # Stage 8/9/28 -- H.264 has no capability gate (this pipeline's own
    # native format); every other codec is capability-gated, never
    # optimistically accepted from local-sandbox evidence alone.
    if profile.video_codec != smp.VIDEO_CODEC_H264:
        gate = _CAPABILITY_GATED_CODECS.get(profile.video_codec)
        if gate is not None:
            field_name, reason = gate
            if not getattr(capability, field_name, False):
                return _decision(DECISION_INSUFFICIENT_EVIDENCE, blocking=(reason,), profile=profile)
        else:
            # VP9/ProRes/MPEG4/etc. -- no confirmation mechanism defined
            # at all yet.
            return _decision(DECISION_INSUFFICIENT_EVIDENCE, blocking=(REASON_CODEC_RUNTIME_UNVERIFIED,), profile=profile)

    # Stage 19/20 -- multi-stream: no deterministic selection policy
    # exists yet -- never silently pick "the first one".
    if profile.video_stream_count > 1:
        return _decision(DECISION_INSUFFICIENT_EVIDENCE, blocking=(REASON_MULTIPLE_VIDEO_STREAMS,), profile=profile)
    if profile.audio_stream_count > 1:
        return _decision(DECISION_INSUFFICIENT_EVIDENCE, blocking=(REASON_MULTIPLE_AUDIO_STREAMS,), profile=profile)

    # From here, every remaining property is NORMALIZATION or WARNING
    # severity only (Stage 30) -- none of them alone blocks entry.
    normalization: list[str] = []
    warnings: list[str] = []

    # Stage 10 -- rotation (including a malformed-but-present tag,
    # conservatively treated as requiring normalization since we cannot
    # confirm it is safely zero).
    if profile.rotation_degrees not in (0, None) or (
        profile.rotation_degrees is None and profile.rotation_source not in (smp.ROTATION_SOURCE_NONE,)
    ):
        normalization.append(REASON_ROTATION_NORMALIZATION_REQUIRED)

    # Stage 11/12 -- HDR / uncertain color metadata.
    if profile.hdr_status in (
        smp.HDR_STATUS_HDR_PQ, smp.HDR_STATUS_HDR_HLG, smp.HDR_STATUS_HDR_DOLBY_VISION, smp.HDR_STATUS_HDR_OTHER,
    ):
        normalization.append(REASON_HDR_NORMALIZATION_REQUIRED)
    elif profile.hdr_status == smp.HDR_STATUS_UNKNOWN and (profile.color_primaries or "").strip().lower() == "bt2020":
        warnings.append(REASON_COLOR_METADATA_UNCERTAIN)

    # Stage 13/14 -- VFR (CFR is never blocked regardless of source fps).
    if profile.vfr_status in (smp.VFR_STATUS_LIKELY_VFR, smp.VFR_STATUS_VFR):
        normalization.append(REASON_VFR_NORMALIZATION_REQUIRED)

    # Stage 15 -- bit depth.
    if profile.bit_depth is not None and profile.bit_depth > 8:
        normalization.append(REASON_TEN_BIT_NORMALIZATION_REQUIRED)

    # Stage 16 -- pixel format (8-bit non-4:2:0 variants, e.g. yuv422p/
    # yuv444p -- never conflated with the bit-depth reason above).
    pix_fmt = (profile.pixel_format or "").strip().lower()
    if pix_fmt and pix_fmt not in _EIGHT_BIT_ACCEPTABLE_PIX_FMTS and not pix_fmt.endswith(_TEN_BIT_PIX_FMT_SUFFIXES):
        normalization.append(REASON_PIXEL_FORMAT_NORMALIZATION_REQUIRED)

    # Stage 17 -- missing audio is a warning only, never blocking (the
    # renderer already synthesizes silence -- D-270's own finding).
    if profile.audio_presence == smp.AUDIO_MISSING:
        warnings.append(REASON_AUDIO_MISSING)

    # Stage 25/27 -- resource risk, no invented thresholds (mirrors
    # source_media_profile.resource_risk_flags's own binding constraint).
    if (
        max_pixel_count is not None and profile.display_width and profile.display_height
        and profile.display_width * profile.display_height > max_pixel_count
    ):
        warnings.append(REASON_RESOURCE_RISK_HIGH_RESOLUTION)
    if max_duration_sec is not None and profile.duration_sec is not None and profile.duration_sec > max_duration_sec:
        warnings.append(REASON_RESOURCE_RISK_LONG_DURATION)

    decision = DECISION_NORMALIZE_REQUIRED if normalization else DECISION_ACCEPT
    return _decision(decision, normalization=tuple(normalization), warnings=tuple(warnings), profile=profile)


# =============================================================================
# Stage 31 -- the pure pipeline-entry helper
# =============================================================================

def can_enter_editorial_pipeline(decision: SourceFormatPolicyDecision) -> bool:
    """D-272 Stage 31: `True` if and only if `decision.decision ==
    ACCEPT`. `NORMALIZE_REQUIRED`/`REJECT`/`INSUFFICIENT_EVIDENCE` all
    return `False` -- no expensive editorial entry for any non-ACCEPT
    state, no exceptions."""
    return decision.can_enter_editorial_pipeline


# =============================================================================
# Stage 32 -- the integration seam (proven, deliberately NOT wired into
# any live job in this gate -- see module docstring)
# =============================================================================

def evaluate_source_for_editorial_entry(
    path: str,
    *,
    runtime_capability: RuntimeCapabilityInput | None = None,
    max_pixel_count: int | None = None,
    max_duration_sec: float | None = None,
    runner=None,
) -> SourceFormatPolicyDecision:
    """D-272 Stage 32: the ONE pure function a future call site (the
    earliest identified seam: `worker_job.py::run_flow_b_job`, right
    after its own existing per-source `probe_media` call and before
    `process_local_sources`'s ASR/semantic work) would call to gate a
    real source before expensive compute. Composes D-271's own
    `probe_source_media_profile` with this module's own `evaluate_
    source_format_policy` -- never invokes ASR, GPU, or any provider
    itself; a `REJECT`/`NORMALIZE_REQUIRED`/`INSUFFICIENT_EVIDENCE`
    result from this function costs exactly one bounded ffprobe call,
    nothing more. NOT called from `worker_job.py` by this gate itself --
    see this module's own docstring for why."""
    kwargs = {} if runner is None else {"runner": runner}
    profile = smp.probe_source_media_profile(path, **kwargs)
    return evaluate_source_format_policy(
        profile, runtime_capability=runtime_capability,
        max_pixel_count=max_pixel_count, max_duration_sec=max_duration_sec,
    )
