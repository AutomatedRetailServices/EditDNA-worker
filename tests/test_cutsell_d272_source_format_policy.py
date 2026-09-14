"""D-272 -- SOURCE FORMAT POLICY / EARLY MEDIA GATE.

Post D-271 (source media probe + format classification foundation,
Verdict A). Proves `cutsell_worker.source_format_policy`'s
`SourceFormatPolicyDecision`, `evaluate_source_format_policy`,
`can_enter_editorial_pipeline`, and the integration seam
`evaluate_source_for_editorial_entry` -- against real ffmpeg-generated
synthetic fixtures and hand-built ffprobe-JSON payloads (matching
D-271's own established pattern for properties this ffmpeg build cannot
attach via a real fixture).

No transcode, no HDR tonemap, no rotation pixel transform, no fps
conversion, no codec conversion, no filtergraph change, no RAW, no
provider anywhere in this file. This gate deliberately does NOT wire the
seam into `worker_job.py` -- see `source_format_policy.py`'s own module
docstring; a regression firewall below confirms that file is untouched.
"""
from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from cutsell_worker import source_media_profile as smp
from cutsell_worker import source_format_policy as sfp

pytestmark_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")


def _source_without_docstrings(path: str) -> str:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body[0] = ast.Expr(value=ast.Constant(value=""))
    return ast.unparse(tree)


def _run_git_diff(rel_path: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", rel_path], capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


def _fake_runner(payload: dict, returncode: int = 0):
    def runner(cmd, **kwargs):
        return SimpleNamespace(returncode=returncode, stdout=json.dumps(payload), stderr="")
    return runner


def _profile_from_payload(payload: dict, path: str = "x.mp4") -> smp.SourceMediaProfile:
    return smp.probe_source_media_profile(path, runner=_fake_runner(payload))


def _h264_video_stream(**overrides) -> dict:
    stream = {
        "codec_type": "video", "codec_name": "h264", "coded_width": 320, "coded_height": 240,
        "pix_fmt": "yuv420p", "avg_frame_rate": "30/1", "r_frame_rate": "30/1", "bits_per_raw_sample": "8",
    }
    stream.update(overrides)
    return stream


def _payload(*streams, format_name="mov,mp4,m4a,3gp,3g2,mj2", duration="5.0") -> dict:
    return {"format": {"format_name": format_name, "duration": duration}, "streams": list(streams)}


# =============================================================================
# Synthetic fixtures (real ffmpeg, module-scoped -- reused across tests)
# =============================================================================

@pytest.fixture(scope="module")
def h264_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d272_h264")
    path = str(d / "h264.mp4")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", path])
    return path


@pytest.fixture(scope="module")
def h264_mov(tmp_path_factory):
    d = tmp_path_factory.mktemp("d272_mov")
    path = str(d / "h264.mov")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", path])
    return path


@pytest.fixture(scope="module")
def h264_24fps(tmp_path_factory):
    d = tmp_path_factory.mktemp("d272_24fps")
    path = str(d / "h264_24.mp4")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", path])
    return path


@pytest.fixture(scope="module")
def h264_60fps(tmp_path_factory):
    d = tmp_path_factory.mktemp("d272_60fps")
    path = str(d / "h264_60.mp4")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=60", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", path])
    return path


@pytest.fixture(scope="module")
def hevc_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d272_hevc")
    path = str(d / "hevc.mp4")
    try:
        _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
                 "-c:v", "libx265", "-pix_fmt", "yuv420p", "-tag:v", "hvc1", path])
    except subprocess.CalledProcessError:
        return None
    return path


@pytest.fixture(scope="module")
def hdr_pq_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d272_hdrpq")
    path = str(d / "hdr_pq.mp4")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p10le",
             "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", path])
    return path


@pytest.fixture(scope="module")
def hlg_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d272_hlg")
    path = str(d / "hlg.mp4")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p10le",
             "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", path])
    return path


@pytest.fixture(scope="module")
def no_audio_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d272_noaudio")
    path = str(d / "noaudio.mp4")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", path])
    return path


@pytest.fixture(scope="module")
def audio_only(tmp_path_factory):
    d = tmp_path_factory.mktemp("d272_audioonly")
    path = str(d / "audioonly.wav")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "1", path])
    return path


@pytest.fixture(scope="module")
def corrupt_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("d272_corrupt")
    path = str(d / "corrupt.mp4")
    Path(path).write_bytes(b"garbage bytes not a real media file" * 20)
    return path


@pytest.fixture(scope="module")
def mkv_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("d272_mkv")
    path = str(d / "x.mkv")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", path])
    return path


# =============================================================================
# Stage 1/3/36 -- policy version, single owner, immutability
# =============================================================================

def test_policy_version_single_owner():
    assert sfp.SOURCE_FORMAT_POLICY_VERSION == 1


@pytestmark_ffmpeg
def test_decision_immutable(h264_mp4):
    profile = smp.probe_source_media_profile(h264_mp4)
    decision = sfp.evaluate_source_format_policy(profile)
    with pytest.raises(Exception):
        decision.decision = "SOMETHING_ELSE"  # type: ignore[misc]


def test_decision_carries_policy_version():
    profile = _profile_from_payload(_payload(_h264_video_stream()))
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.policy_version == sfp.SOURCE_FORMAT_POLICY_VERSION


# =============================================================================
# Stage 8 -- H264 SDR ACCEPT (MP4 and MOV independently)
# =============================================================================

@pytestmark_ffmpeg
def test_h264_mp4_sdr_accept(h264_mp4):
    profile = smp.probe_source_media_profile(h264_mp4)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_ACCEPT


@pytestmark_ffmpeg
def test_h264_mov_sdr_accept(h264_mov):
    profile = smp.probe_source_media_profile(h264_mov)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_ACCEPT


# =============================================================================
# Stage 14 -- CFR policy: 24/30/60fps all ACCEPT, no output-fps policy change
# =============================================================================

@pytestmark_ffmpeg
def test_h264_cfr_24fps_accept(h264_24fps):
    profile = smp.probe_source_media_profile(h264_24fps)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_ACCEPT


@pytestmark_ffmpeg
def test_h264_cfr_60fps_accept(h264_60fps):
    profile = smp.probe_source_media_profile(h264_60fps)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_ACCEPT


# =============================================================================
# Stage 10 -- rotation NORMALIZE_REQUIRED (parser-level, per D-271's own
# established fixture-honesty precedent for rotation metadata)
# =============================================================================

def test_rotation_normalize_required():
    payload = _payload(_h264_video_stream(
        coded_width=1920, coded_height=1080,
        side_data_list=[{"side_data_type": "Display Matrix", "rotation": 90.0}],
    ))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_ROTATION_NORMALIZATION_REQUIRED in decision.normalization_reasons
    assert decision.can_enter_editorial_pipeline is False


def test_zero_rotation_never_flagged():
    payload = _payload(_h264_video_stream(tags={"rotate": "0"}))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert sfp.REASON_ROTATION_NORMALIZATION_REQUIRED not in decision.reason_codes


# =============================================================================
# Stage 11 -- HDR PQ/HLG NORMALIZE_REQUIRED
# =============================================================================

@pytestmark_ffmpeg
def test_hdr_pq_normalize_required(hdr_pq_mp4):
    profile = smp.probe_source_media_profile(hdr_pq_mp4)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_HDR_NORMALIZATION_REQUIRED in decision.normalization_reasons


@pytestmark_ffmpeg
def test_hlg_normalize_required(hlg_mp4):
    profile = smp.probe_source_media_profile(hlg_mp4)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_HDR_NORMALIZATION_REQUIRED in decision.normalization_reasons


def test_hdr_never_silently_accepted_into_sdr_renderer():
    payload = _payload(_h264_video_stream(color_transfer="smpte2084"))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.can_enter_editorial_pipeline is False


# =============================================================================
# Stage 12 -- BT.2020 alone is NOT automatically HDR
# =============================================================================

def test_bt2020_alone_not_automatically_hdr():
    payload = _payload(_h264_video_stream(color_primaries="bt2020"))
    profile = _profile_from_payload(payload)
    assert profile.hdr_status == smp.HDR_STATUS_UNKNOWN
    decision = sfp.evaluate_source_format_policy(profile)
    assert sfp.REASON_HDR_NORMALIZATION_REQUIRED not in decision.reason_codes
    assert sfp.REASON_COLOR_METADATA_UNCERTAIN in decision.warnings
    # a warning alone never blocks entry
    assert decision.decision == sfp.DECISION_ACCEPT


def test_dolby_vision_never_fabricated_in_policy():
    payload = _payload(_h264_video_stream(color_primaries="bt2020", color_transfer="smpte2084"))
    profile = _profile_from_payload(payload)
    assert profile.hdr_status != smp.HDR_STATUS_HDR_DOLBY_VISION
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED  # PQ, not fabricated Dolby Vision


# =============================================================================
# Stage 13 -- likely-VFR NORMALIZE_REQUIRED
# =============================================================================

def test_likely_vfr_normalize_required():
    payload = _payload(_h264_video_stream(avg_frame_rate="24/1", r_frame_rate="30000/1001"))
    profile = _profile_from_payload(payload)
    assert profile.vfr_status == smp.VFR_STATUS_LIKELY_VFR
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_VFR_NORMALIZATION_REQUIRED in decision.normalization_reasons


def test_output_fps_not_used_as_proof_editorial_timing_is_safe():
    """Stage 13's own explicit instruction: fixed output fps=30 must NOT
    be treated as proof that source VFR is safe for editorial timing --
    LIKELY_VFR still triggers NORMALIZE_REQUIRED regardless."""
    payload = _payload(_h264_video_stream(avg_frame_rate="24/1", r_frame_rate="30/1"))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED


# =============================================================================
# Stage 15 -- ten-bit conservative
# =============================================================================

def test_ten_bit_sdr_normalize_required():
    payload = _payload(_h264_video_stream(pix_fmt="yuv420p10le", bits_per_raw_sample="10"))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_TEN_BIT_NORMALIZATION_REQUIRED in decision.normalization_reasons


# =============================================================================
# Stage 16 -- pixel format (yuv422p/yuv444p) conservative
# =============================================================================

def test_yuv422p_normalization_required():
    payload = _payload(_h264_video_stream(pix_fmt="yuv422p", bits_per_raw_sample="8"))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_PIXEL_FORMAT_NORMALIZATION_REQUIRED in decision.normalization_reasons


def test_yuv444p_normalization_required():
    payload = _payload(_h264_video_stream(pix_fmt="yuv444p", bits_per_raw_sample="8"))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_PIXEL_FORMAT_NORMALIZATION_REQUIRED in decision.normalization_reasons


def test_yuv420p_never_flagged_pixel_format():
    payload = _payload(_h264_video_stream(pix_fmt="yuv420p"))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert sfp.REASON_PIXEL_FORMAT_NORMALIZATION_REQUIRED not in decision.reason_codes


def test_ten_bit_pix_fmt_not_double_counted_as_pixel_format_reason():
    payload = _payload(_h264_video_stream(pix_fmt="yuv420p10le", bits_per_raw_sample="10"))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert sfp.REASON_TEN_BIT_NORMALIZATION_REQUIRED in decision.normalization_reasons
    assert sfp.REASON_PIXEL_FORMAT_NORMALIZATION_REQUIRED not in decision.reason_codes


# =============================================================================
# Stage 17/18 -- missing audio (non-blocking) vs. missing video (REJECT)
# =============================================================================

@pytestmark_ffmpeg
def test_missing_audio_still_accepts(no_audio_mp4):
    profile = smp.probe_source_media_profile(no_audio_mp4)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_ACCEPT
    assert sfp.REASON_AUDIO_MISSING in decision.warnings


@pytestmark_ffmpeg
def test_missing_video_reject(audio_only):
    profile = smp.probe_source_media_profile(audio_only)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_REJECT
    assert decision.blocking_reasons == (sfp.REASON_MISSING_VIDEO,)
    assert decision.can_enter_editorial_pipeline is False


# =============================================================================
# Stage 23 -- failed / partial probe
# =============================================================================

@pytestmark_ffmpeg
def test_failed_probe_reject(corrupt_file):
    profile = smp.probe_source_media_profile(corrupt_file)
    assert profile.probe_status == smp.PROBE_STATUS_FAILED
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_REJECT
    assert decision.blocking_reasons == (sfp.REASON_PROBE_FAILED,)


def test_partial_probe_conservative_but_not_automatically_blocked():
    """A PARTIAL probe (e.g. no audio stream -- an expected, non-fatal
    gap) still proceeds to a real classification, never treated as
    automatically insufficient."""
    payload = _payload(_h264_video_stream())
    profile = _profile_from_payload(payload)
    assert profile.probe_status in (smp.PROBE_STATUS_COMPLETE, smp.PROBE_STATUS_PARTIAL)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.source_profile_status == profile.probe_status


# =============================================================================
# Stage 19/20 -- multi-stream INSUFFICIENT_EVIDENCE
# =============================================================================

def test_multiple_video_streams_insufficient_evidence():
    payload = _payload(_h264_video_stream(), _h264_video_stream(coded_width=100, coded_height=100))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.REASON_MULTIPLE_VIDEO_STREAMS in decision.blocking_reasons
    assert decision.can_enter_editorial_pipeline is False


def test_multiple_audio_streams_insufficient_evidence():
    payload = _payload(
        _h264_video_stream(),
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2},
    )
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.REASON_MULTIPLE_AUDIO_STREAMS in decision.blocking_reasons


# =============================================================================
# Stage 21/22 -- unknown/unsupported codec, container
# =============================================================================

def test_unknown_codec_insufficient_evidence():
    payload = _payload(_h264_video_stream(codec_name="exotic_xyz"))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.REASON_UNKNOWN_CODEC in decision.blocking_reasons


def test_known_unsupported_codec_structural_seam():
    """No codec is currently confirmed hard-unsupported (D-270's own
    finding) -- this proves the STRUCTURAL seam works if one ever is,
    without asserting any real codec is unsupported today."""
    payload = _payload(_h264_video_stream(codec_name="vp9"))
    profile = _profile_from_payload(payload)
    assert profile.video_codec == smp.VIDEO_CODEC_VP9
    original = sfp.KNOWN_UNSUPPORTED_VIDEO_CODECS
    try:
        object.__setattr__(sfp, "KNOWN_UNSUPPORTED_VIDEO_CODECS", frozenset({smp.VIDEO_CODEC_VP9}))
    except Exception:
        sfp.KNOWN_UNSUPPORTED_VIDEO_CODECS = frozenset({smp.VIDEO_CODEC_VP9})
    try:
        decision = sfp.evaluate_source_format_policy(profile)
        assert decision.decision == sfp.DECISION_REJECT
        assert sfp.REASON_UNSUPPORTED_CODEC in decision.blocking_reasons
    finally:
        sfp.KNOWN_UNSUPPORTED_VIDEO_CODECS = original


def test_no_codec_currently_hard_unsupported_by_default():
    assert sfp.KNOWN_UNSUPPORTED_VIDEO_CODECS == frozenset()


def test_unsupported_container_rejected():
    payload = {"format": {"format_name": "avi", "duration": "1.0"}, "streams": [_h264_video_stream()]}
    profile = _profile_from_payload(payload, path="x.avi")
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_REJECT
    assert sfp.REASON_UNSUPPORTED_CONTAINER in decision.blocking_reasons


@pytestmark_ffmpeg
def test_mkv_container_rejected(mkv_file):
    profile = smp.probe_source_media_profile(mkv_file)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_REJECT
    assert sfp.REASON_UNSUPPORTED_CONTAINER in decision.blocking_reasons


def test_unknown_container_insufficient_evidence():
    payload = {"format": {"format_name": "some_exotic_format", "duration": "1.0"}, "streams": [_h264_video_stream()]}
    profile = _profile_from_payload(payload, path="x.exotic")
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.REASON_UNKNOWN_CONTAINER in decision.blocking_reasons


# =============================================================================
# Stage 24 -- invalid dimensions
# =============================================================================

def test_invalid_dimensions_rejected():
    payload = _payload(_h264_video_stream(coded_width=0, coded_height=0))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_REJECT
    assert sfp.REASON_INVALID_DIMENSIONS in decision.blocking_reasons


def test_no_new_minimum_resolution_threshold_invented():
    """A small but valid (positive) resolution is never rejected merely
    for being small -- Stage 24's own 'no arbitrary new minimum quality
    threshold' instruction."""
    payload = _payload(_h264_video_stream(coded_width=64, coded_height=64))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision != sfp.DECISION_REJECT


# =============================================================================
# Stage 9/28 -- HEVC runtime capability gating
# =============================================================================

@pytestmark_ffmpeg
def test_hevc_runtime_unknown_not_native_accept(hevc_mp4):
    if hevc_mp4 is None:
        pytest.skip("HEVC encoder not available on this runner")
    profile = smp.probe_source_media_profile(hevc_mp4)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.REASON_HEVC_RUNTIME_UNVERIFIED in decision.blocking_reasons
    assert decision.can_enter_editorial_pipeline is False


@pytestmark_ffmpeg
def test_hevc_runtime_confirmed_normalizes_to_h264(hevc_mp4):
    """D-272B: capability-confirmed HEVC is never ACCEPT-native in the
    canonical V1 source contract -- it always requires normalization to
    canonical H.264, even when otherwise clean (reconciles D-272 with
    D-273/D-274A; supersedes this test's own pre-D-272B expectation)."""
    if hevc_mp4 is None:
        pytest.skip("HEVC encoder not available on this runner")
    profile = smp.probe_source_media_profile(hevc_mp4)
    decision = sfp.evaluate_source_format_policy(
        profile, runtime_capability=sfp.RuntimeCapabilityInput(hevc_decode_confirmed=True),
    )
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_HEVC_TO_H264_NORMALIZATION_REQUIRED in decision.normalization_reasons
    assert sfp.REASON_HEVC_RUNTIME_UNVERIFIED not in decision.reason_codes
    assert decision.user_facing_error_code == sfp.USER_FACING_VIDEO_REQUIRES_NORMALIZATION


def test_hevc_default_capability_input_never_confirms():
    """No caller in this codebase constructs a non-default
    RuntimeCapabilityInput today (Stage 28's own binding instruction) --
    proven here: the default is always unconfirmed."""
    default = sfp.RuntimeCapabilityInput()
    assert default.hevc_decode_confirmed is False
    assert default.av1_decode_confirmed is False


def test_av1_runtime_unknown_insufficient_evidence():
    payload = _payload(_h264_video_stream(codec_name="av1"))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.REASON_CODEC_RUNTIME_UNVERIFIED in decision.blocking_reasons


def test_prores_no_confirmation_mechanism_insufficient_evidence():
    payload = _payload(_h264_video_stream(codec_name="prores"))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.REASON_CODEC_RUNTIME_UNVERIFIED in decision.blocking_reasons


def test_mpeg4_no_confirmation_mechanism_insufficient_evidence():
    payload = _payload(_h264_video_stream(codec_name="mpeg4"))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE


# =============================================================================
# Stage 25/27 -- resource risk, no invented thresholds
# =============================================================================

def test_resource_flags_never_fire_without_explicit_threshold():
    payload = _payload(_h264_video_stream(coded_width=7680, coded_height=4320), duration="999999.0")
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert sfp.REASON_RESOURCE_RISK_HIGH_RESOLUTION not in decision.reason_codes
    assert sfp.REASON_RESOURCE_RISK_LONG_DURATION not in decision.reason_codes
    assert decision.decision == sfp.DECISION_ACCEPT


def test_resource_flags_fire_with_real_caller_supplied_threshold():
    payload = _payload(_h264_video_stream(coded_width=3840, coded_height=2160), duration="10.0")
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile, max_pixel_count=1920 * 1080, max_duration_sec=5.0)
    assert sfp.REASON_RESOURCE_RISK_HIGH_RESOLUTION in decision.warnings
    assert sfp.REASON_RESOURCE_RISK_LONG_DURATION in decision.warnings
    # warnings alone never block entry
    assert decision.decision == sfp.DECISION_ACCEPT


def test_no_new_numeric_threshold_invented_in_module_source():
    source = _source_without_docstrings("cutsell_worker/source_format_policy.py")
    for needle in ("3840", "2160", "4096", "7680", "MAX_RESOLUTION", "MAX_DURATION_SEC ="):
        assert needle not in source


# =============================================================================
# Stage 26/29/30 -- reason codes deterministic, blocking/normalization/warning
# =============================================================================

def test_reason_codes_deterministic_across_repeated_calls():
    payload = _payload(_h264_video_stream(tags={"rotate": "90"}))
    profile = _profile_from_payload(payload)
    a = sfp.evaluate_source_format_policy(profile)
    b = sfp.evaluate_source_format_policy(profile)
    assert a.decision == b.decision
    assert a.reason_codes == b.reason_codes


def test_blocking_normalization_warning_are_separate_buckets():
    payload = _payload(_h264_video_stream(tags={"rotate": "90"}))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.blocking_reasons == ()
    assert sfp.REASON_ROTATION_NORMALIZATION_REQUIRED in decision.normalization_reasons
    assert set(decision.blocking_reasons).isdisjoint(decision.normalization_reasons)
    assert set(decision.normalization_reasons).isdisjoint(decision.warnings)


def test_path_and_filename_independent():
    payload = _payload(_h264_video_stream())
    profile_a = _profile_from_payload(payload, path="a.mp4")
    profile_b = _profile_from_payload(payload, path="totally-different-name.mp4")
    decision_a = sfp.evaluate_source_format_policy(profile_a)
    decision_b = sfp.evaluate_source_format_policy(profile_b)
    assert decision_a.decision == decision_b.decision
    assert decision_a.reason_codes == decision_b.reason_codes


# =============================================================================
# Stage 31/38 -- editorial-entry helper
# =============================================================================

def test_accept_allows_editorial_entry():
    payload = _payload(_h264_video_stream())
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_ACCEPT
    assert sfp.can_enter_editorial_pipeline(decision) is True


def test_normalize_required_blocks_editorial_entry():
    payload = _payload(_h264_video_stream(tags={"rotate": "90"}))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.can_enter_editorial_pipeline(decision) is False


def test_reject_blocks_editorial_entry():
    payload = _payload()  # no streams -> missing video
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_REJECT
    assert sfp.can_enter_editorial_pipeline(decision) is False


def test_insufficient_evidence_blocks_editorial_entry():
    payload = _payload(_h264_video_stream(codec_name="unknown_codec_xyz"))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.can_enter_editorial_pipeline(decision) is False


# =============================================================================
# Stage 32/38 -- integration seam (no GPU/provider call for blocked states)
# =============================================================================

@pytestmark_ffmpeg
def test_integration_seam_accepts_clean_h264(h264_mp4):
    decision = sfp.evaluate_source_for_editorial_entry(h264_mp4)
    assert decision.decision == sfp.DECISION_ACCEPT
    assert decision.can_enter_editorial_pipeline is True


@pytestmark_ffmpeg
def test_integration_seam_blocks_corrupt_file_no_asr_or_gpu_call(corrupt_file):
    """Proves the seam costs exactly one bounded ffprobe call for a
    blocked state -- no ASR/GPU/provider import or call happens anywhere
    in the seam's own call graph (confirmed by source inspection below,
    and functionally here by the decision itself)."""
    decision = sfp.evaluate_source_for_editorial_entry(corrupt_file)
    assert decision.decision == sfp.DECISION_REJECT
    assert decision.can_enter_editorial_pipeline is False


def test_seam_never_imports_asr_or_gpu_modules():
    source = _source_without_docstrings("cutsell_worker/source_format_policy.py")
    for needle in ("FasterWhisperASR", "brain_runtime", "process_local_sources", "GPUExecutionProvider", "from .asr"):
        assert needle not in source


def test_worker_job_activation_deferred_to_d272a():
    """D-272 itself proved the seam works without wiring it into
    `worker_job.py` (rejecting a real upload is an editorial/product
    policy decision -- CLAUDE.md's own D-091 escalation condition A).
    D-272A is the separately-authorized Product Owner activation gate
    that legitimately extends `worker_job.py` -- this test only confirms
    D-272's own seam functions (`evaluate_source_for_editorial_entry`,
    `evaluate_source_format_policy`) still exist and are importable,
    rather than re-asserting the now-superseded "untouched" guard (see
    `test_cutsell_d272a_live_source_format_gate.py`'s own firewall)."""
    assert hasattr(sfp, "evaluate_source_for_editorial_entry")
    assert hasattr(sfp, "evaluate_source_format_policy")


# =============================================================================
# Stage 33 -- no silent fallback
# =============================================================================

def test_no_ffmpeg_try_fallback_language_in_source():
    """This module's own docstring quotes the FORBIDDEN phrase while
    describing that it never does that -- strip docstrings first (the
    same established false-positive-proofing technique) before checking."""
    source = _source_without_docstrings("cutsell_worker/source_format_policy.py")
    assert "let ffmpeg try" not in source.lower().replace("_", " ")


def test_unknown_codec_never_defaults_to_accept():
    payload = _payload(_h264_video_stream(codec_name="totally_unrecognized"))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision != sfp.DECISION_ACCEPT


# =============================================================================
# Stage 34 -- user-facing error code foundation
# =============================================================================

def test_user_facing_error_code_none_for_accept():
    payload = _payload(_h264_video_stream())
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.user_facing_error_code is None


def test_user_facing_error_code_video_corrupt():
    profile = smp.probe_source_media_profile("nope.mp4", runner=lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.user_facing_error_code == sfp.USER_FACING_VIDEO_CORRUPT


def test_user_facing_error_code_requires_normalization():
    payload = _payload(_h264_video_stream(tags={"rotate": "90"}))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.user_facing_error_code == sfp.USER_FACING_VIDEO_REQUIRES_NORMALIZATION


def test_user_facing_error_code_stream_ambiguous():
    payload = _payload(_h264_video_stream(), _h264_video_stream(coded_width=10, coded_height=10))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.user_facing_error_code == sfp.USER_FACING_VIDEO_STREAM_AMBIGUOUS


def test_user_facing_error_code_runtime_unverified():
    payload = _payload(_h264_video_stream(codec_name="hevc"))
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.user_facing_error_code == sfp.USER_FACING_RUNTIME_CODEC_SUPPORT_UNVERIFIED


# =============================================================================
# Stage 35 -- observability, no secrets
# =============================================================================

def test_diagnostics_fields_present_no_secrets():
    payload = _payload(_h264_video_stream())
    profile = _profile_from_payload(payload)
    decision = sfp.evaluate_source_format_policy(profile)
    for attr in (
        "policy_version", "decision", "reason_codes", "blocking_reasons",
        "normalization_reasons", "warnings", "source_profile_status", "source_format_class",
    ):
        assert hasattr(decision, attr)
    serialized = str(decision).lower()
    for forbidden in ("secret", "password", "authorization", "access_key"):
        assert forbidden not in serialized


# =============================================================================
# Stage 39/40 -- closed-track firewall + security posture
# =============================================================================

@pytest.mark.parametrize("rel_path", [
    "cutsell_worker/render.py",
    "cutsell_worker/render_delivery.py",
    "cutsell_worker/render_plan.py",
    "cutsell_worker/media_probe.py",
    "cutsell_worker/source_media_profile.py",
    "cutsell_worker/post_render_media_qc.py",
    "cutsell_worker/visual_finishing_executor.py",
    "cutsell_worker/visual_finishing_composition.py",
    "cutsell_worker/visual_finishing_measurement.py",
    "cutsell_worker/visual_finishing_policy.py",
    "cutsell_worker/audio_finishing_executor.py",
    "cutsell_worker/audio_finishing_composition.py",
    "cutsell_worker/audio_finishing_measurement.py",
    "cutsell_worker/boundary_engine_pass.py",
    "cutsell_worker/pacing_transition_decision.py",
    "cutsell_worker/post_render_watch_listen_qc.py",
    "cutsell_worker/live_render_qc.py",
    "cutsell_worker/finishing_contract.py",
    "cutsell_worker/export_job.py",
    "cutsell_worker/exports.py",
    "cutsell_worker/tenant_safe_delivery.py",
    # uploads.py removed: D-282 (a later, separately-authorized gate)
    # legitimately adds a voice-over upload allowlist to it --
    # self-resolving guard, same pattern as this file's own worker_job.py
    # precedent below.
    # worker_job.py deliberately removed from this list: D-272A (the
    # separately-authorized Product Owner activation gate) legitimately
    # extends it -- see test_cutsell_d272a_live_source_format_gate.py's
    # own firewall for the current authoritative unrelated-files list.
    "cutsell_worker/flow_b.py",
    "cutsell_worker/gpu_execution_provider.py",
])
def test_unrelated_authorities_unchanged(rel_path):
    assert _run_git_diff(rel_path) == "", f"D-272 must not touch {rel_path}"


def test_render_timeout_still_1200():
    from cutsell_worker import render
    assert render.RENDER_FFMPEG_TIMEOUT_SEC == 1200.0


def test_render_output_fps_still_30():
    from cutsell_worker import render
    assert render.RENDER_FPS_DEFAULT == 30


def test_codec_filtergraph_still_unchanged():
    source = Path("cutsell_worker/render.py").read_text(encoding="utf-8")
    assert '"libx264"' in source
    assert '"-crf", "20"' in source


def test_no_shell_true():
    source = Path("cutsell_worker/source_format_policy.py").read_text(encoding="utf-8")
    assert "shell=True" not in source


def test_no_network_or_credential_construction():
    source = _source_without_docstrings("cutsell_worker/source_format_policy.py")
    for needle in ("boto3", "requests.", "urllib.request", "AWS_SECRET", "Authorization"):
        assert needle not in source


def test_no_provider_no_raw_reference():
    source = _source_without_docstrings("cutsell_worker/source_format_policy.py")
    for needle in ("runpod", "RunPod", "modal.", "GPUExecutionProvider"):
        assert needle not in source


def test_no_encode_command_construction():
    source = _source_without_docstrings("cutsell_worker/source_format_policy.py")
    for needle in ("'-c:v'", '"-c:v"', "'-vf'", '"-vf"', "scale=", "pad=", "transpose="):
        assert needle not in source


def test_module_never_mutates_source_media_profile():
    """Checks for actual assignment to a profile attribute (single `=`,
    never `==`) -- a naive substring check would false-positive on this
    module's own legitimate `profile.hdr_status == ...` comparisons."""
    source = _source_without_docstrings("cutsell_worker/source_format_policy.py")
    assert "dataclasses.replace(profile" not in source
    for attr in ("rotation_degrees", "hdr_status", "video_codec", "container_name"):
        assert f"profile.{attr} = " not in source
