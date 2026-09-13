"""Source Media Probe / Format Classification Foundation -- D-271.

Post D-270 (renderer format/media-diversity hardening audit, Verdict A:
common mobile format foundation partially supported, clear P0 gaps
identified). This module answers ONE question, before any editorial or
render work touches a source file: **WHAT EXACTLY IS THIS FILE?**

`probe_source_media_profile` builds a typed, immutable `SourceMediaProfile`
from one bounded ffprobe call -- container, codec/profile, pixel format,
bit depth, coded vs. display dimensions, rotation (value + provenance),
avg/r frame rate, VFR status, color metadata, HDR status, audio codec/
sample-rate/channels, stream counts, start times/time bases. Every field
is either a real, ffprobe-derived fact or an honestly-reported `None`/
`UNKNOWN` -- nothing here is ever guessed or fabricated.

`classify_source_format` is a pure function of one `SourceMediaProfile`:
it never re-probes, never mutates the file, and never invokes ffmpeg. It
answers a SECOND, narrower question: given what we now know about this
file, is it SUPPORTED_NATIVE, NORMALIZATION_REQUIRED, UNSUPPORTED, or do
we have INSUFFICIENT_EVIDENCE to say -- with machine-readable reason
codes, never a bare verdict.

## What this module explicitly does NOT do (D-271's own binding scope)

No transcode, no color conversion, no HDR tone-map, no rotation
correction, no fps change, no codec change, no filtergraph change, no
upload rejection, no renderer mutation. This is a read-only classifier.
A file classified `UNSUPPORTED` or `NORMALIZATION_REQUIRED` today is
processed by the EXACT SAME unchanged renderer (`render.py`) as before
this module existed -- nothing here is wired into any live call site.
That wiring is explicitly D-272's own job (Stage 45), not this gate's.

## Relationship to existing probes (never replaced, never duplicated)

`cutsell_worker.media_probe.probe_media` (duration/width/height/fps/
has_audio, video-oriented, used by the live render/QC path) and the
narrower ad-hoc probes inside `visual_finishing_measurement.py`
(`_probe_rotation_degrees`) and `audio_finishing_measurement.py`
(`_probe_audio_stream_fields`) are all UNCHANGED by this gate -- D-270's
own finding was that these three each separately, narrowly re-solve a
piece of what a canonical source-format profile should express in one
place. This module is that one place, additive: nothing here replaces or
is consumed by any of those three today. A future gate MAY refactor them
to delegate here; this gate does not (Stage 32's own "prefer no behavior
change" instruction).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# =============================================================================
# Stage 27 -- probe status vocabulary
# =============================================================================

PROBE_STATUS_COMPLETE = "COMPLETE"
PROBE_STATUS_PARTIAL = "PARTIAL"
PROBE_STATUS_FAILED = "FAILED"

# =============================================================================
# Stage 3/4 -- presence vocabulary
# =============================================================================

VIDEO_PRESENT = "VIDEO_PRESENT"
VIDEO_MISSING = "VIDEO_MISSING"

AUDIO_PRESENT = "AUDIO_PRESENT"
AUDIO_MISSING = "AUDIO_MISSING"

# =============================================================================
# Stage 2 -- normalized container family vocabulary
# =============================================================================

CONTAINER_MP4 = "MP4"
CONTAINER_MOV = "MOV"
CONTAINER_M4V = "M4V"
CONTAINER_WEBM = "WEBM"
CONTAINER_MKV = "MKV"
CONTAINER_AVI = "AVI"
CONTAINER_UNKNOWN = "UNKNOWN"

# ffprobe's own `format_name` for the entire QuickTime/MP4 family is the
# SAME compound string ("mov,mp4,m4a,3gp,3g2,mj2") regardless of which
# member container it actually is -- ffprobe genuinely cannot distinguish
# MOV from MP4 from M4V by content alone for this family. This is the
# ONE place this module falls back to the file extension, and only as a
# tiebreaker WITHIN this already-format-confirmed family, never as the
# primary signal (Stage 2's own "do not rely solely on filename
# extension" instruction).
_QUICKTIME_FAMILY_MARKER = "mov,mp4"
_QUICKTIME_EXTENSION_MAP = {
    ".mov": CONTAINER_MOV,
    ".mp4": CONTAINER_MP4,
    ".m4v": CONTAINER_M4V,
}

# ffprobe's own matroska demuxer reports the SAME compound format_name
# ("matroska,webm") for both genuine MKV and genuine WEBM files -- WebM
# is a constrained Matroska profile, and ffprobe's format detection does
# not distinguish them by content either. This is the SECOND (and only
# other) place this module falls back to the file extension, for exactly
# the same reason as the QuickTime family above.
_MATROSKA_FAMILY_MARKERS = ("matroska", "webm")

# =============================================================================
# Stage 6 -- normalized video codec vocabulary
# =============================================================================

VIDEO_CODEC_H264 = "H264"
VIDEO_CODEC_HEVC = "HEVC"
VIDEO_CODEC_VP9 = "VP9"
VIDEO_CODEC_AV1 = "AV1"
VIDEO_CODEC_PRORES = "PRORES"
VIDEO_CODEC_MPEG4 = "MPEG4"
VIDEO_CODEC_UNKNOWN = "UNKNOWN"

_VIDEO_CODEC_MAP = {
    "h264": VIDEO_CODEC_H264,
    "hevc": VIDEO_CODEC_HEVC,
    "vp9": VIDEO_CODEC_VP9,
    "av1": VIDEO_CODEC_AV1,
    "prores": VIDEO_CODEC_PRORES,
    "mpeg4": VIDEO_CODEC_MPEG4,
}

# =============================================================================
# Stage 22 -- normalized audio codec vocabulary
# =============================================================================

AUDIO_CODEC_AAC = "AAC"
AUDIO_CODEC_PCM = "PCM"
AUDIO_CODEC_OPUS = "OPUS"
AUDIO_CODEC_MP3 = "MP3"
AUDIO_CODEC_UNKNOWN = "UNKNOWN"

# =============================================================================
# Stage 11/12 -- rotation vocabulary
# =============================================================================

ROTATION_SOURCE_DISPLAY_MATRIX = "DISPLAY_MATRIX"
ROTATION_SOURCE_ROTATE_TAG = "ROTATE_TAG"
ROTATION_SOURCE_NONE = "NONE"
ROTATION_SOURCE_UNKNOWN = "UNKNOWN"

_CANONICAL_ROTATIONS = (0, 90, 180, 270)

# =============================================================================
# Stage 15 -- VFR classification vocabulary
# =============================================================================

VFR_STATUS_CFR = "CFR"
VFR_STATUS_LIKELY_VFR = "LIKELY_VFR"
VFR_STATUS_VFR = "VFR"
VFR_STATUS_UNKNOWN = "UNKNOWN"

# =============================================================================
# Stage 21 -- HDR classification vocabulary
# =============================================================================

HDR_STATUS_SDR = "SDR"
HDR_STATUS_HDR_PQ = "HDR_PQ"
HDR_STATUS_HDR_HLG = "HDR_HLG"
HDR_STATUS_HDR_DOLBY_VISION = "HDR_DOLBY_VISION"
HDR_STATUS_HDR_OTHER = "HDR_OTHER"
HDR_STATUS_UNKNOWN = "UNKNOWN"

# Transfer characteristics this module treats as ordinary SDR when no
# PQ/HLG/Dolby-Vision evidence is present. Deliberately NOT including
# "unknown"/empty here -- an absent transfer tag means NO evidence either
# way, which is HDR_STATUS_UNKNOWN, never a fabricated SDR (Stage 21's
# own "do not classify all BT.2020 as HDR" extends to: do not classify
# absent evidence as SDR either).
_KNOWN_SDR_TRANSFERS = {"bt709", "smpte170m", "bt470bg", "gamma22", "gamma28", "smpte240m"}

# The real, documented ffprobe side-data-type string for a Dolby Vision
# configuration record -- the ONLY signal this module treats as genuine
# Dolby Vision evidence (Stage 21's own "only if real metadata evidence
# exists").
_DOLBY_VISION_SIDE_DATA_TYPE = "DOVI configuration record"

# =============================================================================
# Stage 28 -- source format classification vocabulary
# =============================================================================

SOURCE_FORMAT_CLASS_SUPPORTED_NATIVE = "SUPPORTED_NATIVE"
SOURCE_FORMAT_CLASS_NORMALIZATION_REQUIRED = "NORMALIZATION_REQUIRED"
SOURCE_FORMAT_CLASS_UNSUPPORTED = "UNSUPPORTED"
SOURCE_FORMAT_CLASS_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

# =============================================================================
# Stage 30 -- reason codes (non-blocking unless noted in classify_source_format)
# =============================================================================

REASON_ROTATION_METADATA_PRESENT = "ROTATION_METADATA_PRESENT"
REASON_HDR_INPUT = "HDR_INPUT"
REASON_LIKELY_VFR = "LIKELY_VFR"
REASON_MULTIPLE_VIDEO_STREAMS = "MULTIPLE_VIDEO_STREAMS"
REASON_MULTIPLE_AUDIO_STREAMS = "MULTIPLE_AUDIO_STREAMS"
REASON_MISSING_VIDEO = "MISSING_VIDEO"
REASON_UNKNOWN_CODEC = "UNKNOWN_CODEC"
REASON_TEN_BIT_VIDEO = "TEN_BIT_VIDEO"
REASON_UNSUPPORTED_CONTAINER = "UNSUPPORTED_CONTAINER"
REASON_AUDIO_MISSING = "AUDIO_MISSING"
REASON_LOW_RESOLUTION = "LOW_RESOLUTION"
REASON_HIGH_RESOLUTION = "HIGH_RESOLUTION"
REASON_RUNTIME_CAPABILITY_UNKNOWN = "RUNTIME_CAPABILITY_UNKNOWN"
REASON_PROBE_FAILED = "PROBE_FAILED"
REASON_ROTATION_MALFORMED = "ROTATION_MALFORMED"

# =============================================================================
# Stage 32 -- orientation vocabulary (a local, undependent copy -- this
# gate never imports from or modifies visual_finishing_measurement.py;
# see module docstring)
# =============================================================================

ORIENTATION_PORTRAIT = "PORTRAIT"
ORIENTATION_LANDSCAPE = "LANDSCAPE"
ORIENTATION_SQUARE = "SQUARE"
ORIENTATION_UNKNOWN = "UNKNOWN"

_FFPROBE_TIMEOUT_SEC = 30.0


# =============================================================================
# Stage 1 -- the profile type itself
# =============================================================================

@dataclass(frozen=True)
class SourceMediaProfile:
    """D-271 Stage 1: one immutable, typed snapshot of everything this
    module could honestly determine about a source file. A field is
    `None` (never a fabricated default) whenever the underlying evidence
    was absent or unparseable -- `probe_status`/`warnings`/`errors`
    record exactly why."""

    path: str
    probe_status: str

    container_name: str
    raw_format_name: str | None
    duration_sec: float | None
    file_size_bytes: int | None

    video_presence: str
    audio_presence: str

    video_stream_count: int
    audio_stream_count: int

    video_codec: str | None
    raw_video_codec: str | None
    video_profile: str | None
    pixel_format: str | None
    bit_depth: int | None

    coded_width: int | None
    coded_height: int | None
    display_width: int | None
    display_height: int | None

    rotation_degrees: int | None
    rotation_source: str

    avg_frame_rate: float | None
    r_frame_rate: float | None
    effective_fps: float | None
    vfr_status: str

    color_primaries: str | None
    color_transfer: str | None
    color_space: str | None
    color_range: str | None
    hdr_status: str

    audio_codec: str | None
    raw_audio_codec: str | None
    audio_sample_rate_hz: int | None
    audio_channels: int | None
    audio_channel_layout: str | None

    format_start_time: float | None
    video_stream_start_time: float | None
    audio_stream_start_time: float | None
    video_time_base: str | None
    audio_time_base: str | None

    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)


# =============================================================================
# Small, bounded, self-contained parsing helpers -- this module owns its
# own copies rather than reaching into media_probe.py/visual_finishing_
# measurement.py/audio_finishing_measurement.py's own private helpers,
# matching this codebase's established per-module convention (see D-269's
# own `_scope_component` precedent).
# =============================================================================

def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _parse_rational_rate(value: str | None) -> float | None:
    """Safely parses an ffprobe rational rate string ("30/1", "0/0",
    "N/A") into a float. Never raises, never divides by zero."""
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        num_str, _, den_str = value.partition("/")
        num = _safe_float(num_str)
        den = _safe_float(den_str)
        if num is None or den is None or den == 0:
            return None
        return num / den
    return _safe_float(value)


_PIX_FMT_BIT_DEPTH = {
    "yuv420p": 8, "yuvj420p": 8, "yuv422p": 8, "yuv444p": 8, "nv12": 8, "nv21": 8,
    "yuv420p10le": 10, "yuv420p10be": 10, "yuv422p10le": 10, "yuv444p10le": 10, "p010le": 10,
    "yuv420p12le": 12, "yuv422p12le": 12, "yuv444p12le": 12,
}


def _derive_bit_depth(bits_per_raw_sample, pix_fmt: str | None) -> int | None:
    """D-271 Stage 9: explicit ffprobe numeric field first, then a known
    `pix_fmt` mapping, else `None` (never guessed)."""
    explicit = _safe_int(bits_per_raw_sample)
    if explicit and explicit > 0:
        return explicit
    if pix_fmt:
        mapped = _PIX_FMT_BIT_DEPTH.get(pix_fmt.strip().lower())
        if mapped is not None:
            return mapped
    return None


def _normalize_container(format_name: str | None, path: str) -> tuple[str, str | None]:
    """D-271 Stage 2: returns (normalized_container, raw_format_name)."""
    raw = format_name or None
    text = (format_name or "").lower()
    if not text:
        return CONTAINER_UNKNOWN, raw
    if any(marker in text for marker in _MATROSKA_FAMILY_MARKERS):
        # ffprobe reports the SAME compound format_name for both genuine
        # MKV and genuine WEBM files -- extension is the only available
        # tiebreaker here too (mirrors the QuickTime-family exception
        # below), defaulting to MKV (the broader/superset format) when
        # the extension itself is ambiguous or absent.
        suffix = Path(path).suffix.lower()
        return (CONTAINER_WEBM if suffix == ".webm" else CONTAINER_MKV), raw
    if "avi" in text:
        return CONTAINER_AVI, raw
    if _QUICKTIME_FAMILY_MARKER in text:
        # ffprobe cannot distinguish MOV/MP4/M4V within this family from
        # content alone -- extension is the only available tiebreaker
        # HERE, and only here (Stage 2's own documented exception).
        suffix = Path(path).suffix.lower()
        return _QUICKTIME_EXTENSION_MAP.get(suffix, CONTAINER_MP4), raw
    return CONTAINER_UNKNOWN, raw


def _normalize_video_codec(codec_name: str | None) -> tuple[str | None, str | None]:
    if not codec_name:
        return None, None
    normalized = _VIDEO_CODEC_MAP.get(codec_name.strip().lower(), VIDEO_CODEC_UNKNOWN)
    return normalized, codec_name


def _normalize_audio_codec(codec_name: str | None) -> tuple[str | None, str | None]:
    if not codec_name:
        return None, None
    text = codec_name.strip().lower()
    if text == "aac":
        return AUDIO_CODEC_AAC, codec_name
    if text.startswith("pcm"):
        return AUDIO_CODEC_PCM, codec_name
    if text == "opus":
        return AUDIO_CODEC_OPUS, codec_name
    if text == "mp3":
        return AUDIO_CODEC_MP3, codec_name
    return AUDIO_CODEC_UNKNOWN, codec_name


def _normalize_rotation_degrees(raw: float) -> int | None:
    """D-271 Stage 11: normalizes an arbitrary rotation angle (including
    negative values, e.g. -90) to one of the four canonical values.
    Returns `None` (malformed/unsupported) for anything that does not
    cleanly match, e.g. 45 degrees -- never silently rounds to the
    nearest canonical value, which would fabricate precision that was
    never there."""
    normalized = round(raw) % 360
    if normalized in _CANONICAL_ROTATIONS:
        return normalized
    return None


def _extract_rotation(stream: dict) -> tuple[int | None, str, bool]:
    """D-271 Stage 11/12: side_data (`displaymatrix`) takes provenance
    priority over the legacy `rotate` stream tag when both are present
    (Stage 12's own ordering). Returns (rotation_degrees, rotation_source,
    malformed) -- `malformed=True` means rotation metadata was PRESENT
    but did not parse to a canonical value, distinct from genuinely
    absent metadata."""
    for entry in stream.get("side_data_list") or ():
        if "rotation" in entry:
            raw = _safe_float(entry.get("rotation"))
            if raw is None:
                return None, ROTATION_SOURCE_UNKNOWN, True
            normalized = _normalize_rotation_degrees(raw)
            if normalized is None:
                return None, ROTATION_SOURCE_DISPLAY_MATRIX, True
            return normalized, ROTATION_SOURCE_DISPLAY_MATRIX, False
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        raw = _safe_float(tags.get("rotate"))
        if raw is None:
            return None, ROTATION_SOURCE_UNKNOWN, True
        normalized = _normalize_rotation_degrees(raw)
        if normalized is None:
            return None, ROTATION_SOURCE_ROTATE_TAG, True
        return normalized, ROTATION_SOURCE_ROTATE_TAG, False
    return None, ROTATION_SOURCE_NONE, False


def _classify_vfr(avg_frame_rate: float | None, r_frame_rate: float | None) -> str:
    """D-271 Stage 15: this probe has only avg-vs-r-frame-rate evidence
    (no packet/frame-level timing) -- per this gate's own explicit
    instruction, that evidence can only ever honestly support `CFR` or
    `LIKELY_VFR`, never a confirmed hard `VFR` (which `VFR_STATUS_VFR`
    remains defined for -- a future gate with real frame-timing evidence
    may use it; this function never does)."""
    if avg_frame_rate is None or r_frame_rate is None:
        return VFR_STATUS_UNKNOWN
    if avg_frame_rate <= 0 or r_frame_rate <= 0:
        return VFR_STATUS_UNKNOWN
    if abs(avg_frame_rate - r_frame_rate) < 0.01:
        return VFR_STATUS_CFR
    return VFR_STATUS_LIKELY_VFR


def _has_dolby_vision_evidence(stream: dict) -> bool:
    for entry in stream.get("side_data_list") or ():
        if str(entry.get("side_data_type") or "") == _DOLBY_VISION_SIDE_DATA_TYPE:
            return True
    return False


def _classify_hdr(color_transfer: str | None, dolby_vision_evidence: bool) -> str:
    """D-271 Stage 21: PQ/HLG detected from `color_transfer` only;
    Dolby Vision only from real side-data evidence (never inferred from
    color primaries/transfer alone); absent transfer evidence is honestly
    `UNKNOWN`, never fabricated `SDR`; every other named transfer that is
    not a known SDR curve is `HDR_OTHER` rather than silently defaulted
    to `SDR` (Stage 21's own "do not classify all BT.2020 as HDR" cuts
    both ways -- this module never asserts SDR without evidence either)."""
    if dolby_vision_evidence:
        return HDR_STATUS_HDR_DOLBY_VISION
    text = (color_transfer or "").strip().lower()
    if text == "smpte2084":
        return HDR_STATUS_HDR_PQ
    if text == "arib-std-b67":
        return HDR_STATUS_HDR_HLG
    if not text:
        return HDR_STATUS_UNKNOWN
    if text in _KNOWN_SDR_TRANSFERS:
        return HDR_STATUS_SDR
    return HDR_STATUS_HDR_OTHER


def orientation_category(profile: SourceMediaProfile) -> str:
    """D-271 Stage 32: canonical orientation classification from DISPLAY
    dimensions (post-rotation-swap), never raw coded dimensions. This is
    a pure, standalone helper -- Visual Finishing is NOT modified or
    imported from by this gate (module docstring); a future gate may wire
    Visual Finishing's own orientation decision to call this instead of
    its current raw-coded-dimension logic, without this gate presuming to
    make that change itself."""
    width, height = profile.display_width, profile.display_height
    if not width or not height:
        return ORIENTATION_UNKNOWN
    if width == height:
        return ORIENTATION_SQUARE
    return ORIENTATION_PORTRAIT if height > width else ORIENTATION_LANDSCAPE


# =============================================================================
# Stage 1-27 -- the probe itself
# =============================================================================

def probe_source_media_profile(
    path: str,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> SourceMediaProfile:
    """D-271: the ONE bounded ffprobe call this module ever makes. Reads
    every field `SourceMediaProfile` can express from a single `-show_
    format -show_streams` JSON response -- container, both stream types,
    codec/profile/pixel-format/bit-depth, rotation, frame rates, color
    metadata, audio layout, start times/time bases. Never raises for a
    normal malformed-media case -- an ffprobe failure or an unreadable
    file produces a `PROBE_STATUS_FAILED` profile with `errors` populated,
    never an exception propagated to the caller."""
    warnings: list[str] = []
    errors: list[str] = []

    def _failed(reason: str) -> SourceMediaProfile:
        return SourceMediaProfile(
            path=path, probe_status=PROBE_STATUS_FAILED,
            container_name=CONTAINER_UNKNOWN, raw_format_name=None,
            duration_sec=None, file_size_bytes=None,
            video_presence=VIDEO_MISSING, audio_presence=AUDIO_MISSING,
            video_stream_count=0, audio_stream_count=0,
            video_codec=None, raw_video_codec=None, video_profile=None,
            pixel_format=None, bit_depth=None,
            coded_width=None, coded_height=None, display_width=None, display_height=None,
            rotation_degrees=None, rotation_source=ROTATION_SOURCE_UNKNOWN,
            avg_frame_rate=None, r_frame_rate=None, effective_fps=None, vfr_status=VFR_STATUS_UNKNOWN,
            color_primaries=None, color_transfer=None, color_space=None, color_range=None,
            hdr_status=HDR_STATUS_UNKNOWN,
            audio_codec=None, raw_audio_codec=None, audio_sample_rate_hz=None,
            audio_channels=None, audio_channel_layout=None,
            format_start_time=None, video_stream_start_time=None, audio_stream_start_time=None,
            video_time_base=None, audio_time_base=None,
            warnings=tuple(warnings), errors=tuple(errors + [reason]),
        )

    try:
        completed = runner(
            [
                "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path,
            ],
            capture_output=True, text=True, timeout=_FFPROBE_TIMEOUT_SEC,
        )
    except Exception as exc:
        return _failed(f"ffprobe_subprocess_error:{exc.__class__.__name__}")

    if completed.returncode != 0:
        return _failed(f"ffprobe_exit_nonzero:{completed.returncode}")

    try:
        payload = json.loads(completed.stdout or "{}")
    except Exception:
        return _failed("ffprobe_json_unparseable")

    fmt = payload.get("format") or {}
    streams = payload.get("streams") or []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    container_name, raw_format_name = _normalize_container(fmt.get("format_name"), path)
    if container_name == CONTAINER_UNKNOWN:
        warnings.append(f"unrecognized_container_format_name:{raw_format_name}")

    duration_sec = _safe_float(fmt.get("duration"))
    file_size_bytes = _safe_int(fmt.get("size"))
    if file_size_bytes is None:
        try:
            file_size_bytes = Path(path).stat().st_size
        except OSError:
            warnings.append("file_size_unavailable")

    video_presence = VIDEO_PRESENT if video_streams else VIDEO_MISSING
    audio_presence = AUDIO_PRESENT if audio_streams else AUDIO_MISSING
    if video_presence == VIDEO_MISSING:
        warnings.append("no_video_stream_found")

    video = video_streams[0] if video_streams else {}
    audio = audio_streams[0] if audio_streams else {}

    video_codec, raw_video_codec = _normalize_video_codec(video.get("codec_name"))
    if video_presence == VIDEO_PRESENT and video_codec == VIDEO_CODEC_UNKNOWN:
        warnings.append(f"unrecognized_video_codec:{raw_video_codec}")
    video_profile = video.get("profile") or None
    pixel_format = video.get("pix_fmt") or None
    bit_depth = _derive_bit_depth(video.get("bits_per_raw_sample"), pixel_format)
    if video_presence == VIDEO_PRESENT and bit_depth is None:
        warnings.append("bit_depth_unavailable")

    coded_width = _safe_int(video.get("coded_width") if video.get("coded_width") is not None else video.get("width"))
    coded_height = _safe_int(video.get("coded_height") if video.get("coded_height") is not None else video.get("height"))

    rotation_degrees, rotation_source, rotation_malformed = (
        _extract_rotation(video) if video_presence == VIDEO_PRESENT else (None, ROTATION_SOURCE_UNKNOWN, False)
    )
    if rotation_malformed:
        warnings.append("rotation_metadata_malformed")

    if coded_width and coded_height and rotation_degrees in (90, 270):
        display_width, display_height = coded_height, coded_width
    else:
        display_width, display_height = coded_width, coded_height

    avg_frame_rate = _parse_rational_rate(video.get("avg_frame_rate"))
    r_frame_rate = _parse_rational_rate(video.get("r_frame_rate"))
    effective_fps = avg_frame_rate if avg_frame_rate is not None else r_frame_rate
    vfr_status = _classify_vfr(avg_frame_rate, r_frame_rate) if video_presence == VIDEO_PRESENT else VFR_STATUS_UNKNOWN

    color_primaries = video.get("color_primaries") or None
    color_transfer = video.get("color_transfer") or None
    color_space = video.get("color_space") or None
    color_range = video.get("color_range") or None
    dolby_vision_evidence = _has_dolby_vision_evidence(video) if video_presence == VIDEO_PRESENT else False
    hdr_status = _classify_hdr(color_transfer, dolby_vision_evidence) if video_presence == VIDEO_PRESENT else HDR_STATUS_UNKNOWN

    audio_codec, raw_audio_codec = _normalize_audio_codec(audio.get("codec_name"))
    if audio_presence == AUDIO_PRESENT and audio_codec == AUDIO_CODEC_UNKNOWN:
        warnings.append(f"unrecognized_audio_codec:{raw_audio_codec}")
    audio_sample_rate_hz = _safe_int(audio.get("sample_rate"))
    audio_channels = _safe_int(audio.get("channels"))
    audio_channel_layout = audio.get("channel_layout") or None

    format_start_time = _safe_float(fmt.get("start_time"))
    video_stream_start_time = _safe_float(video.get("start_time")) if video_presence == VIDEO_PRESENT else None
    audio_stream_start_time = _safe_float(audio.get("start_time")) if audio_presence == AUDIO_PRESENT else None
    video_time_base = video.get("time_base") or None
    audio_time_base = audio.get("time_base") or None

    probe_status = PROBE_STATUS_COMPLETE if not warnings else PROBE_STATUS_PARTIAL

    return SourceMediaProfile(
        path=path, probe_status=probe_status,
        container_name=container_name, raw_format_name=raw_format_name,
        duration_sec=duration_sec, file_size_bytes=file_size_bytes,
        video_presence=video_presence, audio_presence=audio_presence,
        video_stream_count=len(video_streams), audio_stream_count=len(audio_streams),
        video_codec=video_codec, raw_video_codec=raw_video_codec, video_profile=video_profile,
        pixel_format=pixel_format, bit_depth=bit_depth,
        coded_width=coded_width, coded_height=coded_height,
        display_width=display_width, display_height=display_height,
        rotation_degrees=rotation_degrees, rotation_source=rotation_source,
        avg_frame_rate=avg_frame_rate, r_frame_rate=r_frame_rate,
        effective_fps=effective_fps, vfr_status=vfr_status,
        color_primaries=color_primaries, color_transfer=color_transfer,
        color_space=color_space, color_range=color_range, hdr_status=hdr_status,
        audio_codec=audio_codec, raw_audio_codec=raw_audio_codec,
        audio_sample_rate_hz=audio_sample_rate_hz, audio_channels=audio_channels,
        audio_channel_layout=audio_channel_layout,
        format_start_time=format_start_time,
        video_stream_start_time=video_stream_start_time, audio_stream_start_time=audio_stream_start_time,
        video_time_base=video_time_base, audio_time_base=audio_time_base,
        warnings=tuple(warnings), errors=tuple(errors),
    )


# =============================================================================
# Stage 28/29/30 -- classification (pure function of the profile only)
# =============================================================================

@dataclass(frozen=True)
class SourceFormatClassification:
    source_format_class: str
    reasons: tuple[str, ...] = field(default_factory=tuple)


def classify_source_format(profile: SourceMediaProfile) -> SourceFormatClassification:
    """D-271 Stage 28/29/30: a pure function of `profile` alone -- never
    re-probes, never invokes ffmpeg, never mutates anything. Conservative
    by design (Stage 29's own "do not silently broaden support beyond
    evidence"): a codec/container this function has not positively
    confirmed a normalization/support answer for is INSUFFICIENT_
    EVIDENCE, never optimistically SUPPORTED_NATIVE."""
    reasons: list[str] = []

    if profile.probe_status == PROBE_STATUS_FAILED:
        return SourceFormatClassification(SOURCE_FORMAT_CLASS_INSUFFICIENT_EVIDENCE, (REASON_PROBE_FAILED,))

    if profile.video_presence == VIDEO_MISSING:
        return SourceFormatClassification(SOURCE_FORMAT_CLASS_UNSUPPORTED, (REASON_MISSING_VIDEO,))

    if profile.container_name in (CONTAINER_MKV, CONTAINER_AVI, CONTAINER_UNKNOWN):
        return SourceFormatClassification(SOURCE_FORMAT_CLASS_UNSUPPORTED, (REASON_UNSUPPORTED_CONTAINER,))

    if profile.video_codec is None or profile.video_codec == VIDEO_CODEC_UNKNOWN:
        return SourceFormatClassification(SOURCE_FORMAT_CLASS_INSUFFICIENT_EVIDENCE, (REASON_UNKNOWN_CODEC,))

    if profile.video_stream_count > 1:
        reasons.append(REASON_MULTIPLE_VIDEO_STREAMS)
    if profile.audio_stream_count > 1:
        reasons.append(REASON_MULTIPLE_AUDIO_STREAMS)
    if profile.video_stream_count > 1 or profile.audio_stream_count > 1:
        # Stage 36: no stream-selection policy exists yet -- never
        # silently pick "the first one" and call it fully supported.
        return SourceFormatClassification(SOURCE_FORMAT_CLASS_INSUFFICIENT_EVIDENCE, tuple(reasons))

    blocking = False

    if profile.rotation_degrees not in (0, None):
        reasons.append(REASON_ROTATION_METADATA_PRESENT)
        blocking = True

    if profile.hdr_status in (HDR_STATUS_HDR_PQ, HDR_STATUS_HDR_HLG, HDR_STATUS_HDR_DOLBY_VISION, HDR_STATUS_HDR_OTHER):
        reasons.append(REASON_HDR_INPUT)
        blocking = True

    if profile.bit_depth is not None and profile.bit_depth > 8:
        reasons.append(REASON_TEN_BIT_VIDEO)
        blocking = True

    if profile.video_codec != VIDEO_CODEC_H264:
        # D-270's own finding: production decode capability for anything
        # other than H.264 (this pipeline's own native encode target) is
        # UNVERIFIED on the real worker image (Stage 41) -- never
        # silently promoted to SUPPORTED_NATIVE on local-sandbox evidence
        # alone.
        reasons.append(REASON_RUNTIME_CAPABILITY_UNKNOWN)
        blocking = True

    # Non-blocking, informational-only reasons (Stage 30: "reasons can be
    # non-blocking") -- D-270 already established the renderer normalizes
    # each of these structurally (fixed-fps output, anullsrc synthesis).
    if profile.vfr_status in (VFR_STATUS_LIKELY_VFR, VFR_STATUS_VFR):
        reasons.append(REASON_LIKELY_VFR)
    if profile.audio_presence == AUDIO_MISSING:
        reasons.append(REASON_AUDIO_MISSING)

    if blocking:
        return SourceFormatClassification(SOURCE_FORMAT_CLASS_NORMALIZATION_REQUIRED, tuple(reasons))

    return SourceFormatClassification(SOURCE_FORMAT_CLASS_SUPPORTED_NATIVE, tuple(reasons))


# =============================================================================
# Stage 37 -- resource-risk flags (no invented thresholds -- Stage 37's
# own "do not invent new thresholds unless an existing canonical
# threshold exists; reuse existing limits only"). Structural seams that
# never fire until a caller supplies a REAL existing canonical bound.
# =============================================================================

RESOURCE_FLAG_VERY_HIGH_RESOLUTION = "VERY_HIGH_RESOLUTION"
RESOURCE_FLAG_VERY_LONG_DURATION = "VERY_LONG_DURATION"
RESOURCE_FLAG_MANY_STREAMS = "MANY_STREAMS"
RESOURCE_FLAG_UNKNOWN_CODEC = "UNKNOWN_CODEC"


def resource_risk_flags(
    profile: SourceMediaProfile,
    *,
    max_pixel_count: int | None = None,
    max_duration_sec: float | None = None,
) -> tuple[str, ...]:
    """D-271 Stage 37: `max_pixel_count`/`max_duration_sec` default to
    `None` and NEVER fire on their own -- this codebase has no existing
    canonical whole-source resolution or duration ceiling (confirmed by
    inspection: `RENDER_FFMPEG_TIMEOUT_SEC`/`MAX_UPLOAD_BYTES` are a
    render-execution-time bound and an upload-size bound respectively,
    neither a resolution/duration bound), so this function does not
    invent one. A caller with a real, separately-authorized threshold may
    supply it; until then these two flags are a structural seam, not an
    active check -- exactly D-266's own timeout-seam precedent.
    `MANY_STREAMS`/`UNKNOWN_CODEC` reuse facts this module already has,
    no threshold needed."""
    flags: list[str] = []
    if (
        max_pixel_count is not None and profile.display_width and profile.display_height
        and profile.display_width * profile.display_height > max_pixel_count
    ):
        flags.append(RESOURCE_FLAG_VERY_HIGH_RESOLUTION)
    if max_duration_sec is not None and profile.duration_sec is not None and profile.duration_sec > max_duration_sec:
        flags.append(RESOURCE_FLAG_VERY_LONG_DURATION)
    if profile.video_stream_count > 1 or profile.audio_stream_count > 1:
        flags.append(RESOURCE_FLAG_MANY_STREAMS)
    if profile.video_codec == VIDEO_CODEC_UNKNOWN:
        flags.append(RESOURCE_FLAG_UNKNOWN_CODEC)
    return tuple(flags)


# =============================================================================
# Stage 40/41 -- local ffmpeg/ffprobe capability snapshot (LOCAL SANDBOX
# ONLY -- never asserted as production-worker truth; Stage 41's own
# explicit instruction)
# =============================================================================

@dataclass(frozen=True)
class LocalFfmpegCapabilitySnapshot:
    """D-271 Stage 40/41: what THIS runner's local ffmpeg/ffprobe can do
    -- explicitly, permanently scoped to the machine that ran the probe.
    Never conflate `hevc_decoder_present`/`hevc_encoder_present` here with
    a claim about the real production worker image; see `runtime_
    capability_status` below and D-270's own Stage 5/41 finding that the
    real image's ffmpeg build is independently unverified."""

    ffmpeg_version: str | None
    ffprobe_version: str | None
    hevc_decoder_present: bool
    hevc_encoder_present: bool
    av1_decoder_present: bool
    libx264_present: bool
    runtime_capability_status: str
    errors: tuple[str, ...] = field(default_factory=tuple)


RUNTIME_CAPABILITY_LOCAL_SANDBOX_ONLY = "LOCAL_SANDBOX_ONLY_NOT_PRODUCTION_VERIFIED"


def capture_local_ffmpeg_capability(
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> LocalFfmpegCapabilitySnapshot:
    """D-271 Stage 40: bounded, real subprocess calls against whatever
    ffmpeg/ffprobe this runner has -- no provider call, no network. The
    result is always labelled `RUNTIME_CAPABILITY_LOCAL_SANDBOX_ONLY_NOT_
    PRODUCTION_VERIFIED` (Stage 41's own binding instruction)."""
    errors: list[str] = []

    def _run(args: list[str]) -> str:
        try:
            completed = runner(args, capture_output=True, text=True, timeout=_FFPROBE_TIMEOUT_SEC)
            return completed.stdout or ""
        except Exception as exc:
            errors.append(f"{args[0]}_capability_probe_failed:{exc.__class__.__name__}")
            return ""

    ffmpeg_version_output = _run(["ffmpeg", "-version"])
    ffprobe_version_output = _run(["ffprobe", "-version"])
    decoders_output = _run(["ffmpeg", "-decoders"])
    encoders_output = _run(["ffmpeg", "-encoders"])

    def _first_line(text: str) -> str | None:
        first = text.splitlines()[0].strip() if text.splitlines() else ""
        return first or None

    return LocalFfmpegCapabilitySnapshot(
        ffmpeg_version=_first_line(ffmpeg_version_output),
        ffprobe_version=_first_line(ffprobe_version_output),
        hevc_decoder_present="hevc" in decoders_output.lower(),
        hevc_encoder_present="libx265" in encoders_output.lower() or "hevc_" in encoders_output.lower(),
        av1_decoder_present="av1" in decoders_output.lower(),
        libx264_present="libx264" in encoders_output.lower(),
        runtime_capability_status=RUNTIME_CAPABILITY_LOCAL_SANDBOX_ONLY,
        errors=tuple(errors),
    )


# =============================================================================
# Stage 42 -- technical QC interface design (a pure TYPE only -- no
# implementation, per this gate's own "Do NOT implement QC behavior in
# this gate unless a pure type is needed" instruction). A future gate
# implements `verify_output_format(final_path, expected_output_contract)`
# against `post_render_media_qc.py`'s own existing authority; this gate
# only names the shape of `expected_output_contract`.
# =============================================================================

@dataclass(frozen=True)
class OutputFormatContract:
    """D-271 Stage 42: the future output-format-compliance contract a
    technical QC seam would verify a rendered file against. Defined here
    as a pure type only -- `post_render_media_qc.py` is NOT modified by
    this gate (module docstring; D-271's own QC-authority-unchanged
    binding constraint)."""

    container: str
    video_codec: str
    pixel_format: str
    fps: float
    width: int
    height: int
    audio_codec: str
    sample_rate_hz: int
    channels: int
