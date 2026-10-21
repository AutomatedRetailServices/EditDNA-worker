"""Output Format Technical QC -- D-274E.

Post D-274D. D-274D completed EXECUTION support for every currently-
defined D-274A normalization action (rotation, VFR, timeline, HEVC->H264,
HDR PQ/HLG->SDR, 10-bit->8-bit, broad-pixel-format->yuv420p). What was
still missing was ONE canonical technical QC authority that answers, for
any media artifact: "does this file actually match the required media
contract?" -- rather than each caller independently re-deriving that
answer from a raw `SourceMediaProfile`.

## Architecture

    MEDIA ARTIFACT
    -> D-271 SourceMediaProfile   (probe_source_media_profile)
    -> ExpectedMediaFormatContract (this module, one of two canonical instances)
    -> verify_output_format()      (this module, the ONE QC entry point)
    -> OutputFormatQCResult        (PASS / FAIL / PARTIAL / UNKNOWN)

This module is the ONE owner of output-format verification logic. It is
a pure function of an already-probed `SourceMediaProfile` plus a
declarative `ExpectedMediaFormatContract` -- it never itself calls
ffprobe/ffmpeg, never mutates media, never re-derives facts D-271 already
owns. `source_normalization_executor.py`, `render.py`, and
`worker_job.py` are never given their own separate copy of this logic
(Stage 1's own explicit prohibition).

## Two contracts, deliberately NOT assumed identical (Stage 2)

`NORMALIZED_SOURCE_CONTRACT_V1` describes what a D-274B/C/D-normalized
source artifact must look like (audio OPTIONAL -- D-273's own "no
mandatory source-layer audio normalization for V1" decision; width/
height/fixed-fps deliberately NOT constrained -- normalization never
resizes or forces a numeric frame rate).

`FINAL_RENDER_OUTPUT_CONTRACT_V1` describes the ACTUAL current renderer
output contract, audited from `render.py`'s own real ffmpeg commands
and empirically re-confirmed against a real `render_preview` output
(never invented): `-c:v libx264` (H264), `-c:a aac -ar 48000` with a
`sine`/`anullsrc` stereo layout (AAC/48000Hz/stereo, ALWAYS present --
`render.py`'s own `anullsrc` fallback guarantees an audio stream even
for a silent segment, so audio is REQUIRED here, unlike the normalized-
source contract), `fps=30` (`RENDER_FPS_DEFAULT`), a configurable output
canvas (`width`/`height`, default 1080x1920 -- `render_preview`'s own
defaults). Empirically confirmed pixel format is `yuv420p`/8-bit even
from a `yuv444p` source (ffmpeg's own filter-to-encoder negotiation, not
an explicit `-pix_fmt` flag in `render.py`) and rotation metadata is
always absent post-render (scale/pad bakes frames; no side-data survives
or is written) -- both proven, not assumed, via a real `render_preview`
call in this gate's own qualification.

**A real, honestly-disclosed gap** (Stage 23): `render.py` writes NO
explicit `-color_primaries`/`-color_trc`/`-colorspace`/`-color_range`
output flags anywhere. A real render probed with the real D-271 profiler
reports `color_primaries=None, color_transfer=None, color_space=None,
color_range=None, hdr_status=UNKNOWN` -- NOT `SDR`/`BT709` as the
canonical contract at the top of this gate's directive states. This
module does NOT silently soften the final-render contract to match that
gap (Stage 8's own "each contract must define REQUIRED/OPTIONAL/
IGNORED" is honest data, not a workaround): `FINAL_RENDER_OUTPUT_
CONTRACT_V1` still DECLARES the true canonical BT709/SDR requirement as
REQUIRED, so `verify_output_format` against a real, current render
output correctly and honestly reports overall `PARTIAL` (Stage 5's own
"artifact probed, but required evidence missing" -- the render does not
actively CONTRADICT the contract with a wrong tag, e.g. HLG/BT2020; it
simply carries no color tag at all, which is missing REQUIRED evidence,
not a proven violation) with `HDR_STATUS`/`COLOR_PRIMARIES`/`COLOR_
TRANSFER`/`COLOR_SPACE`/`COLOR_RANGE` all landing in `unknown_checks`.
No renderer code is changed by this gate (this gate's own explicit
"NO RENDERER BEHAVIOR CHANGE" banner) -- see D-274E's own decision-log
entry for the resulting Verdict B and the proposed remediation gate
(D-274E-A).

## Integration (Stage 19/20/22)

D-274D's own `execute_source_normalization` now ALSO requires this
module's `verify_output_format` to report `PASS` against `NORMALIZED_
SOURCE_CONTRACT_V1`, in addition to D-272 re-evaluation reaching
`ACCEPT` -- neither alone is sufficient (Stage 20's own "D-272 ACCEPT
alone is not enough"). A format-QC `FAIL` after a genuine D-272 `ACCEPT`
is reported as `NORMALIZATION_VERIFICATION_FAILED`, exactly the same
outcome D-274A's own `verify_normalized_source`/`verification_outcome`
already use for a D-272-still-blocked case -- this module does not
invent a new outcome category, it becomes a second, additive gate on the
same existing pass/fail path. No second normalization pass is ever
triggered by a format-QC failure (`MAX_NORMALIZATION_ATTEMPTS` stays 1,
unchanged).

A clean, pure seam for a FUTURE, separately-authorized gate to wire the
`FINAL_RENDER_OUTPUT_CONTRACT_V1` half of this module into the real
render/delivery path (`live_render_qc.py`/`export_job.py`) is exposed
(`verify_output_format(profile, FINAL_RENDER_OUTPUT_CONTRACT_V1)`) but
NOT itself called from any live render/delivery call site by this gate
(Stage 22's own "prefer build/prove first, activate later")."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import source_media_profile as smp

# =============================================================================
# Stage 24 -- QC version, single owner
# =============================================================================

OUTPUT_FORMAT_QC_VERSION = 1

# =============================================================================
# Artifact type vocabulary (Stage 2)
# =============================================================================

ARTIFACT_TYPE_NORMALIZED_SOURCE = "NORMALIZED_SOURCE"
ARTIFACT_TYPE_FINAL_RENDER = "FINAL_RENDER"

# =============================================================================
# Stage 5 -- overall status vocabulary
# =============================================================================

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_PARTIAL = "PARTIAL"
STATUS_UNKNOWN = "UNKNOWN"

# Stage 8 -- per-check requirement level (contract-declared, not QC-derived)
REQUIREMENT_REQUIRED = "REQUIRED"
REQUIREMENT_OPTIONAL = "OPTIONAL"
REQUIREMENT_IGNORED = "IGNORED"

# Per-check outcome (finer-grained than the overall STATUS_* vocabulary --
# never exposed as the overall `status`, only inside `passed_checks`/
# `failed_checks`/`unknown_checks`).
_CHECK_PASS = "PASS"
_CHECK_FAIL = "FAIL"
_CHECK_UNKNOWN = "UNKNOWN"

# =============================================================================
# Stage 7 -- check vocabulary
# =============================================================================

CHECK_CONTAINER = "CONTAINER"
CHECK_VIDEO_PRESENT = "VIDEO_PRESENT"
CHECK_VIDEO_STREAM_COUNT = "VIDEO_STREAM_COUNT"
CHECK_VIDEO_CODEC = "VIDEO_CODEC"
CHECK_PIXEL_FORMAT = "PIXEL_FORMAT"
CHECK_BIT_DEPTH = "BIT_DEPTH"
CHECK_WIDTH = "WIDTH"
CHECK_HEIGHT = "HEIGHT"
CHECK_FPS = "FPS"
CHECK_VFR_STATUS = "VFR_STATUS"
CHECK_ROTATION = "ROTATION"
CHECK_ORIENTATION = "ORIENTATION"
CHECK_HDR_STATUS = "HDR_STATUS"
CHECK_COLOR_PRIMARIES = "COLOR_PRIMARIES"
CHECK_COLOR_TRANSFER = "COLOR_TRANSFER"
CHECK_COLOR_SPACE = "COLOR_SPACE"
CHECK_COLOR_RANGE = "COLOR_RANGE"
CHECK_AUDIO_STREAM_COUNT = "AUDIO_STREAM_COUNT"
CHECK_AUDIO_CODEC = "AUDIO_CODEC"
CHECK_AUDIO_SAMPLE_RATE = "AUDIO_SAMPLE_RATE"
CHECK_AUDIO_CHANNELS = "AUDIO_CHANNELS"
CHECK_TIMELINE_START = "TIMELINE_START"

# Stage 9: this module's own bounded copy of the "8-bit-acceptable pixel
# format" synonym set `source_format_policy.py` already established
# (`_EIGHT_BIT_ACCEPTABLE_PIX_FMTS`) -- never imported (that name is
# private to that module; this gate's own per-module bounded-copy
# convention, same as D-271's own docstring precedent), never widened.
_YUV420P_SYNONYMS = frozenset({"yuv420p", "yuvj420p"})

# Audio-stream-count policy vocabulary (Stage 8/18)
AUDIO_POLICY_SINGLE_STREAM_OR_ABSENT = "SINGLE_STREAM_OR_ABSENT"
AUDIO_POLICY_SINGLE_STREAM_REQUIRED = "SINGLE_STREAM_REQUIRED"

# Frame-rate policy vocabulary (Stage 15)
FRAME_RATE_POLICY_PRESERVE_CFR = "PRESERVE_CFR_NORMALIZE_VFR_TO_SOURCE_RATE"
FRAME_RATE_POLICY_FIXED_FPS = "FIXED_FPS"


# =============================================================================
# Stage 2 -- the contract type itself
# =============================================================================

@dataclass(frozen=True)
class ExpectedMediaFormatContract:
    """Stage 2/25: one immutable, deterministic description of what an
    artifact of `artifact_type` must look like. Pure declarative data --
    no callables, no execution semantics -- so two contract instances
    with identical field values are also equal (`==`), giving this type
    a real, structural identity rather than a name alone (Stage 25's own
    "deterministic identity/version... do not use local paths")."""

    contract_id: str
    artifact_type: str
    contract_version: int

    container: str | None
    video_codec: str | None
    pixel_format: str | None
    bit_depth: int | None
    expected_video_stream_count: int | None

    rotation_expected_zero_or_absent: bool
    orientation_evidence_required: bool

    hdr_target: str | None
    color_primaries: str | None
    color_transfer: str | None
    color_space: str | None
    color_range: str | None

    frame_rate_policy: str
    fixed_fps: float | None

    expected_width: int | None
    expected_height: int | None

    timeline_start_required: bool

    audio_policy: str
    audio_codec: str | None
    audio_sample_rate_hz: int | None
    audio_channels: int | None

    required_checks: tuple[str, ...]
    optional_checks: tuple[str, ...]


# =============================================================================
# Stage 3 -- NORMALIZED_SOURCE_CONTRACT_V1 (D-273/D-274A canon, Stage 3)
# =============================================================================

NORMALIZED_SOURCE_CONTRACT_V1 = ExpectedMediaFormatContract(
    contract_id="NORMALIZED_SOURCE_V1",
    artifact_type=ARTIFACT_TYPE_NORMALIZED_SOURCE,
    contract_version=OUTPUT_FORMAT_QC_VERSION,
    container=smp.CONTAINER_MP4,
    video_codec=smp.VIDEO_CODEC_H264,
    pixel_format="yuv420p",
    bit_depth=8,
    expected_video_stream_count=1,
    rotation_expected_zero_or_absent=True,
    orientation_evidence_required=True,
    hdr_target=smp.HDR_STATUS_SDR,
    color_primaries="bt709",
    color_transfer="bt709",
    color_space="bt709",
    color_range="tv",
    # Stage 15: never a fixed numeric fps for normalized source -- only
    # "must be CFR" is enforced (VFR_TO_CFR normalizes; an already-CFR
    # source stays untouched at its OWN rate).
    frame_rate_policy=FRAME_RATE_POLICY_PRESERVE_CFR,
    fixed_fps=None,
    # Stage 4: normalization never resizes -- no canvas expectation.
    expected_width=None,
    expected_height=None,
    timeline_start_required=True,
    # Stage 3/17: audio is OPTIONAL for a normalized source -- D-273's own
    # "no mandatory source-layer audio normalization for V1" decision.
    audio_policy=AUDIO_POLICY_SINGLE_STREAM_OR_ABSENT,
    audio_codec=None,
    audio_sample_rate_hz=None,
    audio_channels=None,
    required_checks=(
        CHECK_CONTAINER, CHECK_VIDEO_PRESENT, CHECK_VIDEO_STREAM_COUNT, CHECK_VIDEO_CODEC,
        CHECK_PIXEL_FORMAT, CHECK_BIT_DEPTH, CHECK_ROTATION, CHECK_ORIENTATION,
        CHECK_HDR_STATUS, CHECK_COLOR_PRIMARIES, CHECK_COLOR_TRANSFER, CHECK_COLOR_SPACE,
        CHECK_COLOR_RANGE, CHECK_VFR_STATUS, CHECK_TIMELINE_START, CHECK_AUDIO_STREAM_COUNT,
    ),
    optional_checks=(CHECK_FPS, CHECK_AUDIO_CODEC, CHECK_AUDIO_SAMPLE_RATE, CHECK_AUDIO_CHANNELS),
)

# =============================================================================
# Stage 4/23 -- FINAL_RENDER_OUTPUT_CONTRACT_V1, audited from render.py's
# OWN real ffmpeg commands + empirically re-confirmed via a real
# render_preview() call in this gate's own qualification (see module
# docstring). NEVER invented -- every field below traces to a specific
# `render.py` fact:
#   - "-c:v", "libx264"                              -> video_codec H264
#   - (no -pix_fmt flag; empirically yuv420p/8-bit)  -> pixel_format/bit_depth
#   - "-c:a", "aac", "-ar", "48000", stereo anullsrc  -> audio_* (REQUIRED:
#     the anullsrc fallback guarantees an audio stream even when silent)
#   - RENDER_FPS_DEFAULT = 30                         -> fixed_fps
#   - render_preview(width=1080, height=1920, ...)    -> expected_width/height
#   - empirically: rotation_degrees=None post-render  -> rotation/orientation
#   - NO -color_primaries/-color_trc/-colorspace/     -> the real, disclosed
#     -color_range flags anywhere in render.py           gap (see docstring)
# =============================================================================

FINAL_RENDER_OUTPUT_CONTRACT_V1 = ExpectedMediaFormatContract(
    contract_id="FINAL_RENDER_OUTPUT_V1",
    artifact_type=ARTIFACT_TYPE_FINAL_RENDER,
    contract_version=OUTPUT_FORMAT_QC_VERSION,
    container=smp.CONTAINER_MP4,
    video_codec=smp.VIDEO_CODEC_H264,
    pixel_format="yuv420p",
    bit_depth=8,
    expected_video_stream_count=1,
    rotation_expected_zero_or_absent=True,
    orientation_evidence_required=True,
    hdr_target=smp.HDR_STATUS_SDR,
    color_primaries="bt709",
    color_transfer="bt709",
    color_space="bt709",
    color_range="tv",
    frame_rate_policy=FRAME_RATE_POLICY_FIXED_FPS,
    fixed_fps=30.0,
    expected_width=1080,
    expected_height=1920,
    timeline_start_required=True,
    # Stage 17: audio is REQUIRED for a final render -- render.py's own
    # anullsrc fallback guarantees a stream even for a silent segment.
    audio_policy=AUDIO_POLICY_SINGLE_STREAM_REQUIRED,
    audio_codec=smp.AUDIO_CODEC_AAC,
    audio_sample_rate_hz=48000,
    audio_channels=2,
    required_checks=(
        CHECK_CONTAINER, CHECK_VIDEO_PRESENT, CHECK_VIDEO_STREAM_COUNT, CHECK_VIDEO_CODEC,
        CHECK_PIXEL_FORMAT, CHECK_BIT_DEPTH, CHECK_WIDTH, CHECK_HEIGHT, CHECK_FPS,
        CHECK_VFR_STATUS, CHECK_ROTATION, CHECK_ORIENTATION, CHECK_TIMELINE_START,
        CHECK_AUDIO_STREAM_COUNT, CHECK_AUDIO_CODEC, CHECK_AUDIO_SAMPLE_RATE, CHECK_AUDIO_CHANNELS,
        # Stage 23: kept REQUIRED (the true canonical target), not
        # softened to match the disclosed current renderer gap -- a
        # contract that quietly matches whatever the code currently does
        # is not a contract. This is exactly why a real render today
        # currently reports overall PARTIAL, with these five landing in
        # `unknown_checks` (missing evidence, not a proven violation --
        # see D-274E decision log).
        CHECK_HDR_STATUS, CHECK_COLOR_PRIMARIES, CHECK_COLOR_TRANSFER, CHECK_COLOR_SPACE, CHECK_COLOR_RANGE,
    ),
    optional_checks=(),
)


# =============================================================================
# Stage 6 -- the result type itself
# =============================================================================

@dataclass(frozen=True)
class OutputFormatQCResult:
    """Stage 6: the full, bounded outcome of one `verify_output_format`
    call. Never carries raw media bytes (Stage 6's own "no media
    bytes") -- `observed_profile_summary` is a small dict of scalar
    facts only."""

    status: str
    artifact_type: str
    contract_version: int
    contract_id: str
    required_checks: tuple[str, ...]
    passed_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    unknown_checks: tuple[str, ...]
    warnings: tuple[str, ...]
    observed_profile_summary: dict
    expected_contract_summary: dict


# =============================================================================
# Stage 9-18 -- per-check evaluators. Each returns (_CHECK_PASS/_CHECK_FAIL/
# _CHECK_UNKNOWN, optional warning message). Pure functions of
# (profile, contract) -- Stage 32's own "operate from profile" (never a
# fresh ffprobe call).
# =============================================================================

def _norm(value: str | None) -> str | None:
    return value.strip().lower() if isinstance(value, str) else value


def _check_container(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    if contract.container is None:
        return _CHECK_UNKNOWN, "no container expectation declared"
    if profile.container_name == contract.container:
        return _CHECK_PASS, None
    return _CHECK_FAIL, f"container={profile.container_name!r} expected={contract.container!r}"


def _check_video_present(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    if profile.video_presence == smp.VIDEO_PRESENT:
        return _CHECK_PASS, None
    return _CHECK_FAIL, f"video_presence={profile.video_presence!r}"


def _check_video_stream_count(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    if contract.expected_video_stream_count is None:
        return _CHECK_UNKNOWN, "no video-stream-count expectation declared"
    if profile.video_stream_count == contract.expected_video_stream_count:
        return _CHECK_PASS, None
    return _CHECK_FAIL, (
        f"video_stream_count={profile.video_stream_count} "
        f"expected={contract.expected_video_stream_count}"
    )


def _check_video_codec(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    if contract.video_codec is None:
        return _CHECK_UNKNOWN, "no video-codec expectation declared"
    if profile.video_codec is None:
        return _CHECK_UNKNOWN, "video codec evidence absent"
    if profile.video_codec == contract.video_codec:
        return _CHECK_PASS, None
    return _CHECK_FAIL, f"video_codec={profile.video_codec!r} expected={contract.video_codec!r}"


def _check_pixel_format(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    if contract.pixel_format is None:
        return _CHECK_UNKNOWN, "no pixel-format expectation declared"
    observed = _norm(profile.pixel_format)
    if observed is None:
        return _CHECK_UNKNOWN, "pixel format evidence absent"
    expected = _norm(contract.pixel_format)
    if expected == "yuv420p" and observed in _YUV420P_SYNONYMS:
        return _CHECK_PASS, None
    if observed == expected:
        return _CHECK_PASS, None
    return _CHECK_FAIL, f"pixel_format={profile.pixel_format!r} expected={contract.pixel_format!r}"


def _check_bit_depth(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    if contract.bit_depth is None:
        return _CHECK_UNKNOWN, "no bit-depth expectation declared"
    if profile.bit_depth is None:
        return _CHECK_UNKNOWN, "bit depth evidence absent"
    if profile.bit_depth == contract.bit_depth:
        return _CHECK_PASS, None
    return _CHECK_FAIL, f"bit_depth={profile.bit_depth} expected={contract.bit_depth}"


def _check_width(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    if contract.expected_width is None:
        return _CHECK_UNKNOWN, "no width expectation declared"
    if profile.display_width is None:
        return _CHECK_UNKNOWN, "display width evidence absent"
    if profile.display_width == contract.expected_width:
        return _CHECK_PASS, None
    return _CHECK_FAIL, f"display_width={profile.display_width} expected={contract.expected_width}"


def _check_height(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    if contract.expected_height is None:
        return _CHECK_UNKNOWN, "no height expectation declared"
    if profile.display_height is None:
        return _CHECK_UNKNOWN, "display height evidence absent"
    if profile.display_height == contract.expected_height:
        return _CHECK_PASS, None
    return _CHECK_FAIL, f"display_height={profile.display_height} expected={contract.expected_height}"


def _check_fps(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    """Stage 15: NEVER a hardcoded fps requirement for a normalized
    source (`frame_rate_policy=PRESERVE_CFR`) -- purely informational
    there (always UNKNOWN, folded into optional_checks, never FAIL).
    Only `FRAME_RATE_POLICY_FIXED_FPS` (the real render contract) checks
    a specific number."""
    if contract.frame_rate_policy != FRAME_RATE_POLICY_FIXED_FPS or contract.fixed_fps is None:
        return _CHECK_UNKNOWN, "no fixed-fps expectation declared for this contract's frame-rate policy"
    if profile.effective_fps is None:
        return _CHECK_UNKNOWN, "effective fps evidence absent"
    if abs(profile.effective_fps - contract.fixed_fps) < 0.05:
        return _CHECK_PASS, None
    return _CHECK_FAIL, f"effective_fps={profile.effective_fps} expected={contract.fixed_fps}"


def _check_vfr_status(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    """Stage 15: both contracts ultimately require CFR post-processing
    (the canon's own "CFR where normalization was required; CFR source
    rates otherwise preserved" / the renderer's own fixed fps= filter) --
    never LIKELY_VFR/VFR, regardless of the specific numeric rate."""
    if profile.vfr_status == smp.VFR_STATUS_CFR:
        return _CHECK_PASS, None
    if profile.vfr_status == smp.VFR_STATUS_UNKNOWN:
        return _CHECK_UNKNOWN, "VFR status evidence absent"
    return _CHECK_FAIL, f"vfr_status={profile.vfr_status!r} expected=CFR"


def _check_rotation(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    if not contract.rotation_expected_zero_or_absent:
        return _CHECK_UNKNOWN, "no rotation expectation declared"
    if profile.rotation_degrees in (0, None):
        return _CHECK_PASS, None
    return _CHECK_FAIL, f"rotation_degrees={profile.rotation_degrees} expected=0/absent"


def _check_orientation(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    """Stage 12: rotation-tag absence alone is not proof the physical
    pixels are correctly oriented -- also require real DISPLAY-dimension
    evidence (never coded dimensions alone, per D-271's own `orientation_
    category`'s own documented convention)."""
    if not contract.orientation_evidence_required:
        return _CHECK_UNKNOWN, "no orientation expectation declared"
    if not profile.display_width or not profile.display_height:
        return _CHECK_UNKNOWN, "display dimension evidence absent"
    if profile.display_width > 0 and profile.display_height > 0:
        return _CHECK_PASS, None
    return _CHECK_FAIL, (
        f"display_width={profile.display_width} display_height={profile.display_height}"
    )


def _check_hdr_status(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    if contract.hdr_target is None:
        return _CHECK_UNKNOWN, "no HDR-target expectation declared"
    if profile.hdr_status == smp.HDR_STATUS_UNKNOWN:
        return _CHECK_UNKNOWN, "HDR status evidence absent"
    if profile.hdr_status == contract.hdr_target:
        return _CHECK_PASS, None
    return _CHECK_FAIL, f"hdr_status={profile.hdr_status!r} expected={contract.hdr_target!r}"


def _make_color_check(field_name: str, contract_field: str, check_name: str) -> Callable:
    def _check(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
        expected = getattr(contract, contract_field)
        if expected is None:
            return _CHECK_UNKNOWN, f"no {check_name.lower()} expectation declared"
        observed = getattr(profile, field_name)
        if observed is None:
            return _CHECK_UNKNOWN, f"{field_name} evidence absent"
        if _norm(observed) == _norm(expected):
            return _CHECK_PASS, None
        return _CHECK_FAIL, f"{field_name}={observed!r} expected={expected!r}"
    return _check


_check_color_primaries = _make_color_check("color_primaries", "color_primaries", CHECK_COLOR_PRIMARIES)
_check_color_transfer = _make_color_check("color_transfer", "color_transfer", CHECK_COLOR_TRANSFER)
_check_color_space = _make_color_check("color_space", "color_space", CHECK_COLOR_SPACE)
_check_color_range = _make_color_check("color_range", "color_range", CHECK_COLOR_RANGE)


def _check_timeline_start(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    """Stage 16: honest, stream-level-only evidence -- never claims
    packet-level monotonic PTS proof. Absent start-time evidence is
    UNKNOWN, never fabricated as PASS."""
    if not contract.timeline_start_required:
        return _CHECK_UNKNOWN, "no timeline-start expectation declared"
    candidates = [
        v for v in (profile.format_start_time, profile.video_stream_start_time, profile.audio_stream_start_time)
        if v is not None
    ]
    if not candidates:
        return _CHECK_UNKNOWN, "start-time evidence absent"
    if all(abs(v) < 0.01 for v in candidates):
        return _CHECK_PASS, None
    return _CHECK_FAIL, f"non-zero start time evidence: {candidates}"


def _check_audio_stream_count(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    if contract.audio_policy == AUDIO_POLICY_SINGLE_STREAM_REQUIRED:
        if profile.audio_stream_count == 1:
            return _CHECK_PASS, None
        return _CHECK_FAIL, f"audio_stream_count={profile.audio_stream_count} expected=1"
    # SINGLE_STREAM_OR_ABSENT (Stage 18: never MULTIPLE)
    if profile.audio_stream_count in (0, 1):
        return _CHECK_PASS, None
    return _CHECK_FAIL, f"audio_stream_count={profile.audio_stream_count} expected=0 or 1"


def _check_audio_codec(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    if contract.audio_codec is None:
        return _CHECK_UNKNOWN, "no audio-codec expectation declared"
    if profile.audio_presence == smp.AUDIO_MISSING:
        # Stage 3/17: absent audio is fine for a contract that does not
        # itself REQUIRE audio to exist (SINGLE_STREAM_OR_ABSENT).
        if contract.audio_policy == AUDIO_POLICY_SINGLE_STREAM_OR_ABSENT:
            return _CHECK_UNKNOWN, "audio absent (optional for this contract)"
        return _CHECK_FAIL, "audio_presence=AUDIO_MISSING but contract requires audio"
    if profile.audio_codec is None:
        return _CHECK_UNKNOWN, "audio codec evidence absent"
    if profile.audio_codec == contract.audio_codec:
        return _CHECK_PASS, None
    return _CHECK_FAIL, f"audio_codec={profile.audio_codec!r} expected={contract.audio_codec!r}"


def _check_audio_sample_rate(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    if contract.audio_sample_rate_hz is None:
        return _CHECK_UNKNOWN, "no audio-sample-rate expectation declared"
    if profile.audio_presence == smp.AUDIO_MISSING:
        if contract.audio_policy == AUDIO_POLICY_SINGLE_STREAM_OR_ABSENT:
            return _CHECK_UNKNOWN, "audio absent (optional for this contract)"
        return _CHECK_FAIL, "audio_presence=AUDIO_MISSING but contract requires audio"
    if profile.audio_sample_rate_hz is None:
        return _CHECK_UNKNOWN, "audio sample-rate evidence absent"
    if profile.audio_sample_rate_hz == contract.audio_sample_rate_hz:
        return _CHECK_PASS, None
    return _CHECK_FAIL, (
        f"audio_sample_rate_hz={profile.audio_sample_rate_hz} expected={contract.audio_sample_rate_hz}"
    )


def _check_audio_channels(profile: smp.SourceMediaProfile, contract: ExpectedMediaFormatContract):
    if contract.audio_channels is None:
        return _CHECK_UNKNOWN, "no audio-channels expectation declared"
    if profile.audio_presence == smp.AUDIO_MISSING:
        if contract.audio_policy == AUDIO_POLICY_SINGLE_STREAM_OR_ABSENT:
            return _CHECK_UNKNOWN, "audio absent (optional for this contract)"
        return _CHECK_FAIL, "audio_presence=AUDIO_MISSING but contract requires audio"
    if profile.audio_channels is None:
        return _CHECK_UNKNOWN, "audio channel evidence absent"
    if profile.audio_channels == contract.audio_channels:
        return _CHECK_PASS, None
    return _CHECK_FAIL, f"audio_channels={profile.audio_channels} expected={contract.audio_channels}"


_CHECK_FUNCS: dict[str, Callable] = {
    CHECK_CONTAINER: _check_container,
    CHECK_VIDEO_PRESENT: _check_video_present,
    CHECK_VIDEO_STREAM_COUNT: _check_video_stream_count,
    CHECK_VIDEO_CODEC: _check_video_codec,
    CHECK_PIXEL_FORMAT: _check_pixel_format,
    CHECK_BIT_DEPTH: _check_bit_depth,
    CHECK_WIDTH: _check_width,
    CHECK_HEIGHT: _check_height,
    CHECK_FPS: _check_fps,
    CHECK_VFR_STATUS: _check_vfr_status,
    CHECK_ROTATION: _check_rotation,
    CHECK_ORIENTATION: _check_orientation,
    CHECK_HDR_STATUS: _check_hdr_status,
    CHECK_COLOR_PRIMARIES: _check_color_primaries,
    CHECK_COLOR_TRANSFER: _check_color_transfer,
    CHECK_COLOR_SPACE: _check_color_space,
    CHECK_COLOR_RANGE: _check_color_range,
    CHECK_AUDIO_STREAM_COUNT: _check_audio_stream_count,
    CHECK_AUDIO_CODEC: _check_audio_codec,
    CHECK_AUDIO_SAMPLE_RATE: _check_audio_sample_rate,
    CHECK_AUDIO_CHANNELS: _check_audio_channels,
    CHECK_TIMELINE_START: _check_timeline_start,
}


def _profile_summary(profile: smp.SourceMediaProfile) -> dict:
    """Stage 30: bounded, secret-free diagnostics -- scalar facts only,
    never raw media bytes, never a filesystem path beyond what the
    profile itself already carries as its own identity."""
    return {
        "probe_status": profile.probe_status,
        "container_name": profile.container_name,
        "video_codec": profile.video_codec,
        "pixel_format": profile.pixel_format,
        "bit_depth": profile.bit_depth,
        "display_width": profile.display_width,
        "display_height": profile.display_height,
        "rotation_degrees": profile.rotation_degrees,
        "effective_fps": profile.effective_fps,
        "vfr_status": profile.vfr_status,
        "hdr_status": profile.hdr_status,
        "color_primaries": profile.color_primaries,
        "color_transfer": profile.color_transfer,
        "color_space": profile.color_space,
        "color_range": profile.color_range,
        "video_stream_count": profile.video_stream_count,
        "audio_stream_count": profile.audio_stream_count,
        "audio_codec": profile.audio_codec,
        "audio_sample_rate_hz": profile.audio_sample_rate_hz,
        "audio_channels": profile.audio_channels,
        "format_start_time": profile.format_start_time,
    }


def _contract_summary(contract: ExpectedMediaFormatContract) -> dict:
    return {
        "contract_id": contract.contract_id,
        "artifact_type": contract.artifact_type,
        "contract_version": contract.contract_version,
        "container": contract.container,
        "video_codec": contract.video_codec,
        "pixel_format": contract.pixel_format,
        "bit_depth": contract.bit_depth,
        "expected_width": contract.expected_width,
        "expected_height": contract.expected_height,
        "frame_rate_policy": contract.frame_rate_policy,
        "fixed_fps": contract.fixed_fps,
        "hdr_target": contract.hdr_target,
        "color_primaries": contract.color_primaries,
        "color_transfer": contract.color_transfer,
        "color_space": contract.color_space,
        "color_range": contract.color_range,
        "audio_policy": contract.audio_policy,
        "audio_codec": contract.audio_codec,
        "audio_sample_rate_hz": contract.audio_sample_rate_hz,
        "audio_channels": contract.audio_channels,
    }


def verify_output_format(
    profile: "smp.SourceMediaProfile",
    contract: ExpectedMediaFormatContract,
) -> OutputFormatQCResult:
    """Stage 5/6: the ONE QC entry point. Pure function of an already-
    probed profile and a declarative contract -- no I/O, no ffprobe/
    ffmpeg call, no media mutation (Stage 31/32).

    Overall `status`:
    - `UNKNOWN` when the underlying probe itself never completed
      (`profile.probe_status != COMPLETE`) -- verification could not be
      performed reliably at all (Stage 5's own "UNKNOWN" definition).
    - `FAIL` when any REQUIRED check reports a genuine mismatch.
    - `PARTIAL` when no REQUIRED check failed but at least one REQUIRED
      check's evidence was absent/unparseable (Stage 5's own "PARTIAL"
      definition -- "artifact probed, but required evidence missing").
    - `PASS` only when every REQUIRED check passed outright. Fails
      closed for delivery/normalization qualification (Stage 5's own
      "fail closed... where PASS is required")."""
    passed: list[str] = []
    failed: list[str] = []
    unknown: list[str] = []
    warnings: list[str] = []

    # Stage 5: only a genuinely FAILED probe makes verification wholly
    # unreliable. `PROBE_STATUS_PARTIAL` (D-271's own "most facts known,
    # some genuinely absent") is NOT the same thing -- it still carries
    # real, per-field evidence (e.g. a real `container_name`/`video_
    # presence` even when, say, `bit_depth` could not be determined), so
    # each individual check below already handles a missing field as its
    # own per-check UNKNOWN outcome; conflating PARTIAL with a hard probe
    # FAILURE would discard genuine evidence this module could otherwise
    # honestly report on.
    if profile.probe_status == smp.PROBE_STATUS_FAILED:
        return OutputFormatQCResult(
            status=STATUS_UNKNOWN,
            artifact_type=contract.artifact_type,
            contract_version=contract.contract_version,
            contract_id=contract.contract_id,
            required_checks=contract.required_checks,
            passed_checks=(), failed_checks=(), unknown_checks=tuple(contract.required_checks),
            warnings=(f"profile.probe_status={profile.probe_status!r} -- verification unreliable",),
            observed_profile_summary=_profile_summary(profile),
            expected_contract_summary=_contract_summary(contract),
        )

    all_checks = tuple(contract.required_checks) + tuple(contract.optional_checks)
    for check_name in all_checks:
        check_func = _CHECK_FUNCS.get(check_name)
        if check_func is None:
            unknown.append(check_name)
            warnings.append(f"{check_name}: no evaluator registered")
            continue
        outcome, detail = check_func(profile, contract)
        if outcome == _CHECK_PASS:
            passed.append(check_name)
        elif outcome == _CHECK_FAIL:
            failed.append(check_name)
            if detail:
                warnings.append(f"{check_name}: {detail}")
        else:
            unknown.append(check_name)
            if detail:
                warnings.append(f"{check_name}: {detail}")

    required_set = set(contract.required_checks)
    if any(name in required_set for name in failed):
        status = STATUS_FAIL
    elif any(name in required_set for name in unknown):
        status = STATUS_PARTIAL
    else:
        status = STATUS_PASS

    return OutputFormatQCResult(
        status=status,
        artifact_type=contract.artifact_type,
        contract_version=contract.contract_version,
        contract_id=contract.contract_id,
        required_checks=contract.required_checks,
        passed_checks=tuple(passed),
        failed_checks=tuple(failed),
        unknown_checks=tuple(unknown),
        warnings=tuple(warnings),
        observed_profile_summary=_profile_summary(profile),
        expected_contract_summary=_contract_summary(contract),
    )
