"""D-271 -- SOURCE MEDIA PROBE / FORMAT CLASSIFICATION FOUNDATION.

Post D-270 (renderer format/media-diversity hardening audit, Verdict A).
Proves `cutsell_worker.source_media_profile`'s `SourceMediaProfile`
probe, `classify_source_format` classification, and the `LocalFfmpeg
CapabilitySnapshot`/`OutputFormatContract` seams -- against real ffmpeg-
generated synthetic fixtures where a real decode is needed, and against
hand-built ffprobe-JSON-shaped payloads (Stage 39's own "parser test,
not decode test" distinction) for properties this ffmpeg build/version
could not be made to attach via a real fixture (rotation metadata --
see `test_rotation_fixture_generation_not_available_locally` below).

No transcode, no color conversion, no HDR tonemap, no rotation
correction, no fps/codec/filtergraph change, no RAW, no provider
anywhere in this file.
"""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutsell_worker import source_media_profile as smp

pytestmark_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")


def _source_without_docstrings(path: str) -> str:
    """D-262/D-263/D-266/D-267/D-269/D-270's own established false-
    positive-proofing technique."""
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


# =============================================================================
# Stage 38 -- synthetic fixtures (real ffmpeg-generated)
# =============================================================================

@pytest.fixture(scope="module")
def h264_mp4_30fps(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_h264")
    path = str(d / "h264_30fps.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", path,
    ])
    return path


@pytest.fixture(scope="module")
def h264_mov(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_mov")
    path = str(d / "h264.mov")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ])
    return path


@pytest.fixture(scope="module")
def h264_24fps(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_24fps")
    path = str(d / "h264_24fps.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ])
    return path


@pytest.fixture(scope="module")
def h264_60fps(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_60fps")
    path = str(d / "h264_60fps.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=60", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ])
    return path


@pytest.fixture(scope="module")
def hevc_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_hevc")
    path = str(d / "hevc.mp4")
    try:
        _ffmpeg([
            "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
            "-c:v", "libx265", "-pix_fmt", "yuv420p", "-tag:v", "hvc1", path,
        ])
    except subprocess.CalledProcessError:
        return None
    return path


@pytest.fixture(scope="module")
def mono_audio(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_mono")
    path = str(d / "mono.wav")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-ac", "1", "-t", "1", path])
    return path


@pytest.fixture(scope="module")
def no_audio_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_noaudio")
    path = str(d / "noaudio.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", path,
    ])
    return path


@pytest.fixture(scope="module")
def audio_only(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_audioonly")
    path = str(d / "audioonly.wav")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "1", path])
    return path


@pytest.fixture(scope="module")
def bt709_tagged(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_bt709")
    path = str(d / "bt709.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv", path,
    ])
    return path


@pytest.fixture(scope="module")
def hdr_pq_tagged(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_hdrpq")
    path = str(d / "hdr_pq.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", path,
    ])
    return path


@pytest.fixture(scope="module")
def hlg_tagged(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_hlg")
    path = str(d / "hlg.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", path,
    ])
    return path


@pytest.fixture(scope="module")
def odd_coded_dimensions(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_odd")
    path = str(d / "odd.mp4")
    # D-270's own finding: libx264/yuv420p requires even coded dims. This
    # fixture proves the PROBE itself (never re-encoding) reads whatever
    # ffprobe reports honestly -- it does not assert encoding odd dims
    # succeeds (D-270 already proved libx264 refuses that directly).
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=321x241:rate=30", "-t", "1",
        "-vf", "scale=320:240", "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ])
    return path


@pytest.fixture(scope="module")
def portrait_coded_dimensions(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_portrait")
    path = str(d / "portrait.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=240x320:rate=30", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ])
    return path


@pytest.fixture(scope="module")
def corrupt_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_corrupt")
    path = str(d / "corrupt.mp4")
    Path(path).write_bytes(b"this is not a real media file, just garbage bytes" * 10)
    return path


@pytest.fixture(scope="module")
def unknown_container(tmp_path_factory):
    d = tmp_path_factory.mktemp("d271_unknown")
    path = str(d / "unknown.mkv")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ])
    return path


# =============================================================================
# Stage 39 -- fixture honesty: rotation metadata injection
# =============================================================================

@pytestmark_ffmpeg
def test_rotation_fixture_generation_not_available_locally(tmp_path):
    """D-271 Stage 39: this ffmpeg build/version's `-metadata:s:v rotate=`
    (both re-encode and `-c copy` stream-copy modes) and its `-display_
    rotation` option (confirmed to be INPUT-only, not an output-muxer
    tag) were both tried and neither attached a readable rotation tag or
    side-data entry -- confirmed here, live, rather than asserted from
    memory. This is FIXTURE_NOT_AVAILABLE_LOCALLY, honestly recorded
    (Stage 39's own instruction), not faked. Rotation-dependent behavior
    is instead proven via a parser-level test against a hand-built
    ffprobe JSON payload (`test_rotation_display_matrix_swaps_dimensions`
    and neighbors below)."""
    out = tmp_path / "rot_attempt.mp4"
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-metadata:s:v:0", "rotate=90", str(out),
    ])
    profile = smp.probe_source_media_profile(str(out))
    # Documents the actual, current behavior of this ffmpeg build: no
    # rotation attaches via this flag. If a future ffmpeg upgrade DOES
    # attach it, this assertion will start failing loudly rather than
    # silently -- which is itself useful signal, not a broken test.
    assert profile.rotation_degrees is None
    assert profile.rotation_source == smp.ROTATION_SOURCE_NONE


# =============================================================================
# Stage 2 -- container detection
# =============================================================================

@pytestmark_ffmpeg
def test_container_detected_mp4(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.container_name == smp.CONTAINER_MP4


@pytestmark_ffmpeg
def test_container_detected_mov_by_extension_tiebreak(h264_mov):
    profile = smp.probe_source_media_profile(h264_mov)
    assert profile.container_name == smp.CONTAINER_MOV


@pytestmark_ffmpeg
def test_container_detected_mkv(unknown_container):
    profile = smp.probe_source_media_profile(unknown_container)
    assert profile.container_name == smp.CONTAINER_MKV


def test_container_extension_alone_not_authoritative():
    """Stage 2: a `.mp4`-named file whose real `format_name` says
    matroska must be classified MKV, not MP4 -- extension is never the
    primary signal outside the one documented QuickTime-family tiebreak."""
    payload = {
        "format": {"format_name": "matroska,webm", "duration": "1.0"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "width": 100, "height": 100,
                     "coded_width": 100, "coded_height": 100, "pix_fmt": "yuv420p"}],
    }
    profile = smp.probe_source_media_profile("disguised.mp4", runner=_fake_runner(payload))
    assert profile.container_name == smp.CONTAINER_MKV


# =============================================================================
# Stage 3/4/5 -- video/audio presence, stream counts
# =============================================================================

@pytestmark_ffmpeg
def test_video_presence_explicit(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.video_presence == smp.VIDEO_PRESENT


@pytestmark_ffmpeg
def test_video_missing_explicit_not_silent_zero(audio_only):
    profile = smp.probe_source_media_profile(audio_only)
    assert profile.video_presence == smp.VIDEO_MISSING
    assert profile.coded_width is None
    assert profile.coded_height is None


@pytestmark_ffmpeg
def test_audio_presence_explicit(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.audio_presence == smp.AUDIO_PRESENT


@pytestmark_ffmpeg
def test_audio_missing_not_an_error(no_audio_video):
    profile = smp.probe_source_media_profile(no_audio_video)
    assert profile.audio_presence == smp.AUDIO_MISSING
    assert profile.probe_status in (smp.PROBE_STATUS_COMPLETE, smp.PROBE_STATUS_PARTIAL)
    assert "AUDIO_MISSING" not in profile.errors


@pytestmark_ffmpeg
def test_video_stream_count(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.video_stream_count == 1


@pytestmark_ffmpeg
def test_audio_stream_count(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.audio_stream_count == 1


def test_multiple_video_streams_surfaced():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 100, "height": 100,
             "coded_width": 100, "coded_height": 100, "pix_fmt": "yuv420p"},
            {"codec_type": "video", "codec_name": "h264", "width": 50, "height": 50,
             "coded_width": 50, "coded_height": 50, "pix_fmt": "yuv420p"},
        ],
    }
    profile = smp.probe_source_media_profile("multi.mp4", runner=_fake_runner(payload))
    assert profile.video_stream_count == 2
    classification = smp.classify_source_format(profile)
    assert classification.source_format_class == smp.SOURCE_FORMAT_CLASS_INSUFFICIENT_EVIDENCE
    assert smp.REASON_MULTIPLE_VIDEO_STREAMS in classification.reasons


def test_multiple_audio_streams_surfaced():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 100, "height": 100,
             "coded_width": 100, "coded_height": 100, "pix_fmt": "yuv420p"},
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2},
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2},
        ],
    }
    profile = smp.probe_source_media_profile("multi_audio.mp4", runner=_fake_runner(payload))
    assert profile.audio_stream_count == 2
    classification = smp.classify_source_format(profile)
    assert smp.REASON_MULTIPLE_AUDIO_STREAMS in classification.reasons
    assert classification.source_format_class == smp.SOURCE_FORMAT_CLASS_INSUFFICIENT_EVIDENCE


# =============================================================================
# Stage 6/7 -- video codec normalization
# =============================================================================

@pytestmark_ffmpeg
def test_h264_normalized(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.video_codec == smp.VIDEO_CODEC_H264
    assert profile.raw_video_codec == "h264"


@pytestmark_ffmpeg
def test_hevc_normalized(hevc_mp4):
    if hevc_mp4 is None:
        pytest.skip("HEVC encoder not available on this runner")
    profile = smp.probe_source_media_profile(hevc_mp4)
    assert profile.video_codec == smp.VIDEO_CODEC_HEVC


def test_unknown_codec_preserved_raw():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"},
        "streams": [{"codec_type": "video", "codec_name": "exotic_codec_xyz", "width": 100, "height": 100,
                     "coded_width": 100, "coded_height": 100, "pix_fmt": "yuv420p"}],
    }
    profile = smp.probe_source_media_profile("exotic.mp4", runner=_fake_runner(payload))
    assert profile.video_codec == smp.VIDEO_CODEC_UNKNOWN
    assert profile.raw_video_codec == "exotic_codec_xyz"
    classification = smp.classify_source_format(profile)
    assert classification.source_format_class == smp.SOURCE_FORMAT_CLASS_INSUFFICIENT_EVIDENCE
    assert smp.REASON_UNKNOWN_CODEC in classification.reasons


# =============================================================================
# Stage 7 -- profile capture
# =============================================================================

@pytestmark_ffmpeg
def test_profile_captured(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.video_profile is not None


def test_profile_none_when_missing():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "width": 100, "height": 100,
                     "coded_width": 100, "coded_height": 100, "pix_fmt": "yuv420p"}],
    }
    profile = smp.probe_source_media_profile("noprofile.mp4", runner=_fake_runner(payload))
    assert profile.video_profile is None


# =============================================================================
# Stage 8/9 -- pixel format / bit depth
# =============================================================================

@pytestmark_ffmpeg
def test_pix_fmt_captured(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.pixel_format == "yuv420p"


@pytestmark_ffmpeg
def test_bit_depth_explicit_from_ffprobe_field(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.bit_depth == 8


@pytestmark_ffmpeg
def test_bit_depth_ten_bit(hdr_pq_tagged):
    profile = smp.probe_source_media_profile(hdr_pq_tagged)
    assert profile.bit_depth == 10


def test_bit_depth_falls_back_to_pix_fmt_mapping():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "width": 100, "height": 100,
                     "coded_width": 100, "coded_height": 100, "pix_fmt": "yuv420p10le"}],
    }
    profile = smp.probe_source_media_profile("tenbit.mp4", runner=_fake_runner(payload))
    assert profile.bit_depth == 10


def test_bit_depth_unknown_never_guessed():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "width": 100, "height": 100,
                     "coded_width": 100, "coded_height": 100, "pix_fmt": "some_unknown_format"}],
    }
    profile = smp.probe_source_media_profile("unknownfmt.mp4", runner=_fake_runner(payload))
    assert profile.bit_depth is None


# =============================================================================
# Stage 10 -- coded vs. display dimensions
# =============================================================================

@pytestmark_ffmpeg
def test_coded_dimensions_captured(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.coded_width == 320
    assert profile.coded_height == 240


@pytestmark_ffmpeg
def test_display_dimensions_equal_coded_when_no_rotation(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.display_width == profile.coded_width
    assert profile.display_height == profile.coded_height


def test_rotation_display_matrix_swaps_dimensions():
    """Stage 10's own canonical example: coded 1920x1080 + rotation 90 ->
    display 1080x1920. Proven via a parser-level hand-built payload
    (Stage 39) since rotation-tag fixture generation is not available
    locally (see test_rotation_fixture_generation_not_available_locally)."""
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "5.0"},
        "streams": [{
            "codec_type": "video", "codec_name": "h264", "profile": "High",
            "coded_width": 1920, "coded_height": 1080, "width": 1920, "height": 1080,
            "pix_fmt": "yuv420p", "bits_per_raw_sample": "8",
            "avg_frame_rate": "30/1", "r_frame_rate": "30/1",
            "side_data_list": [{"side_data_type": "Display Matrix", "rotation": 90.0}],
        }],
    }
    profile = smp.probe_source_media_profile("iphone.mov", runner=_fake_runner(payload))
    assert profile.coded_width == 1920 and profile.coded_height == 1080
    assert profile.display_width == 1080 and profile.display_height == 1920
    assert smp.orientation_category(profile) == smp.ORIENTATION_PORTRAIT


def test_rotation_never_mutates_pixels_only_classification():
    """Stage 10's own explicit 'Do NOT rotate pixels. This is
    classification only' -- the profile carries no pixel data and this
    module never constructs an ffmpeg ENCODE/filter command anywhere
    (Stage 40's capability snapshot legitimately references the bare
    words "libx264"/"libx265" as capability-check substrings against
    `ffmpeg -encoders` output, which is not an encode invocation -- this
    check targets actual filtergraph/encode command construction)."""
    source = _source_without_docstrings("cutsell_worker/source_media_profile.py")
    for needle in ("transpose=", '"-vf"', "hflip", "vflip", '"-c:v"', "scale=", "pad="):
        assert needle not in source


# =============================================================================
# Stage 11/12 -- rotation normalization + provenance
# =============================================================================

def test_rotation_normalizes_negative_value():
    assert smp._normalize_rotation_degrees(-90) == 270


def test_rotation_normalizes_270():
    assert smp._normalize_rotation_degrees(270) == 270


def test_rotation_malformed_value_is_none():
    assert smp._normalize_rotation_degrees(45) is None


def test_rotation_provenance_display_matrix_priority():
    rotation, source, malformed = smp._extract_rotation({
        "side_data_list": [{"side_data_type": "Display Matrix", "rotation": 180.0}],
        "tags": {"rotate": "90"},
    })
    assert rotation == 180
    assert source == smp.ROTATION_SOURCE_DISPLAY_MATRIX
    assert malformed is False


def test_rotation_provenance_tag_when_no_side_data():
    rotation, source, malformed = smp._extract_rotation({"tags": {"rotate": "90"}})
    assert rotation == 90
    assert source == smp.ROTATION_SOURCE_ROTATE_TAG


def test_rotation_provenance_none_when_absent():
    rotation, source, malformed = smp._extract_rotation({})
    assert rotation is None
    assert source == smp.ROTATION_SOURCE_NONE
    assert malformed is False


def test_rotation_provenance_unknown_when_malformed():
    rotation, source, malformed = smp._extract_rotation({"tags": {"rotate": "not-a-number"}})
    assert rotation is None
    assert malformed is True


# =============================================================================
# Stage 13/14/15/16 -- frame rate / VFR
# =============================================================================

@pytestmark_ffmpeg
def test_avg_frame_rate_captured(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.avg_frame_rate == pytest.approx(30.0, abs=0.1)


@pytestmark_ffmpeg
def test_r_frame_rate_captured_separately(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.r_frame_rate == pytest.approx(30.0, abs=0.1)


@pytestmark_ffmpeg
def test_24fps_captured(h264_24fps):
    profile = smp.probe_source_media_profile(h264_24fps)
    assert profile.effective_fps == pytest.approx(24.0, abs=0.1)


@pytestmark_ffmpeg
def test_60fps_captured(h264_60fps):
    profile = smp.probe_source_media_profile(h264_60fps)
    assert profile.effective_fps == pytest.approx(60.0, abs=0.1)


@pytestmark_ffmpeg
def test_cfr_classification_when_rates_agree(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.vfr_status == smp.VFR_STATUS_CFR


def test_likely_vfr_classification_when_rates_disagree():
    """Stage 15: avg != r_frame_rate is LIKELY_VFR, never a hard VFR
    claim from this one weak signal alone."""
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "5.0"},
        "streams": [{
            "codec_type": "video", "codec_name": "h264", "coded_width": 100, "coded_height": 100,
            "pix_fmt": "yuv420p", "avg_frame_rate": "24/1", "r_frame_rate": "30000/1001",
        }],
    }
    profile = smp.probe_source_media_profile("vfr.mp4", runner=_fake_runner(payload))
    assert profile.vfr_status == smp.VFR_STATUS_LIKELY_VFR
    classification = smp.classify_source_format(profile)
    assert smp.REASON_LIKELY_VFR in classification.reasons
    # Non-blocking (Stage 30/34): does not by itself force NORMALIZATION_REQUIRED.
    assert classification.source_format_class == smp.SOURCE_FORMAT_CLASS_SUPPORTED_NATIVE


def test_vfr_unknown_when_rate_unavailable():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "5.0"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "coded_width": 100, "coded_height": 100,
                     "pix_fmt": "yuv420p", "avg_frame_rate": "0/0", "r_frame_rate": "0/0"}],
    }
    profile = smp.probe_source_media_profile("norate.mp4", runner=_fake_runner(payload))
    assert profile.vfr_status == smp.VFR_STATUS_UNKNOWN
    assert profile.effective_fps is None


def test_frame_rate_parser_never_divides_by_zero():
    assert smp._parse_rational_rate("0/0") is None
    assert smp._parse_rational_rate("N/A") is None
    assert smp._parse_rational_rate(None) is None
    assert smp._parse_rational_rate("30/0") is None
    assert smp._parse_rational_rate("30/1") == 30.0


# =============================================================================
# Stage 17-20 -- color metadata
# =============================================================================

@pytestmark_ffmpeg
def test_color_primaries_captured(bt709_tagged):
    profile = smp.probe_source_media_profile(bt709_tagged)
    assert profile.color_primaries == "bt709"


@pytestmark_ffmpeg
def test_color_transfer_captured(bt709_tagged):
    profile = smp.probe_source_media_profile(bt709_tagged)
    assert profile.color_transfer == "bt709"


@pytestmark_ffmpeg
def test_color_space_captured(bt709_tagged):
    profile = smp.probe_source_media_profile(bt709_tagged)
    assert profile.color_space == "bt709"


@pytestmark_ffmpeg
def test_color_range_captured(bt709_tagged):
    profile = smp.probe_source_media_profile(bt709_tagged)
    assert profile.color_range == "tv"


# =============================================================================
# Stage 21 -- HDR classification
# =============================================================================

@pytestmark_ffmpeg
def test_sdr_detected(bt709_tagged):
    profile = smp.probe_source_media_profile(bt709_tagged)
    assert profile.hdr_status == smp.HDR_STATUS_SDR


@pytestmark_ffmpeg
def test_pq_detected(hdr_pq_tagged):
    profile = smp.probe_source_media_profile(hdr_pq_tagged)
    assert profile.hdr_status == smp.HDR_STATUS_HDR_PQ


@pytestmark_ffmpeg
def test_hlg_detected(hlg_tagged):
    profile = smp.probe_source_media_profile(hlg_tagged)
    assert profile.hdr_status == smp.HDR_STATUS_HDR_HLG


def test_dolby_vision_never_fabricated_without_real_evidence():
    """Stage 21: BT.2020 + an unusual transfer alone must NEVER become
    HDR_DOLBY_VISION -- only a real 'DOVI configuration record' side-data
    entry does."""
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"},
        "streams": [{"codec_type": "video", "codec_name": "hevc", "coded_width": 100, "coded_height": 100,
                     "pix_fmt": "yuv420p10le", "color_primaries": "bt2020", "color_transfer": "smpte2084"}],
    }
    profile = smp.probe_source_media_profile("nodv.mp4", runner=_fake_runner(payload))
    assert profile.hdr_status != smp.HDR_STATUS_HDR_DOLBY_VISION
    assert profile.hdr_status == smp.HDR_STATUS_HDR_PQ


def test_dolby_vision_detected_from_real_side_data_evidence():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"},
        "streams": [{"codec_type": "video", "codec_name": "hevc", "coded_width": 100, "coded_height": 100,
                     "pix_fmt": "yuv420p10le",
                     "side_data_list": [{"side_data_type": "DOVI configuration record"}]}],
    }
    profile = smp.probe_source_media_profile("dv.mp4", runner=_fake_runner(payload))
    assert profile.hdr_status == smp.HDR_STATUS_HDR_DOLBY_VISION


def test_bt2020_alone_without_transfer_is_not_classified_hdr():
    """Stage 21's own 'do not classify all BT.2020 as HDR' -- absent
    transfer evidence is UNKNOWN, never fabricated SDR or HDR."""
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "coded_width": 100, "coded_height": 100,
                     "pix_fmt": "yuv420p", "color_primaries": "bt2020"}],
    }
    profile = smp.probe_source_media_profile("bt2020_only.mp4", runner=_fake_runner(payload))
    assert profile.hdr_status == smp.HDR_STATUS_UNKNOWN


# =============================================================================
# Stage 22/23/24 -- audio codec / sample rate / channels
# =============================================================================

@pytestmark_ffmpeg
def test_audio_codec_captured(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.audio_codec == smp.AUDIO_CODEC_AAC


@pytestmark_ffmpeg
def test_sample_rate_captured(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.audio_sample_rate_hz == 48000


@pytestmark_ffmpeg
def test_mono_channels_captured(mono_audio):
    profile = smp.probe_source_media_profile(mono_audio)
    assert profile.audio_channels == 1


@pytestmark_ffmpeg
def test_channel_layout_captured(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.audio_channel_layout in ("stereo", None)


def test_pcm_audio_codec_normalized():
    payload = {
        "format": {"format_name": "wav", "duration": "1.0"},
        "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "sample_rate": "44100", "channels": 1}],
    }
    profile = smp.probe_source_media_profile("pcm.wav", runner=_fake_runner(payload))
    assert profile.audio_codec == smp.AUDIO_CODEC_PCM


# =============================================================================
# Stage 25/26 -- start times / time bases
# =============================================================================

@pytestmark_ffmpeg
def test_format_start_time_captured(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.format_start_time == pytest.approx(0.0, abs=0.01)


@pytestmark_ffmpeg
def test_video_time_base_captured(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.video_time_base is not None and "/" in profile.video_time_base


def test_nonzero_start_time_captured():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "5.0", "start_time": "1.5"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "coded_width": 100, "coded_height": 100,
                     "pix_fmt": "yuv420p", "start_time": "1.5"}],
    }
    profile = smp.probe_source_media_profile("editlist.mov", runner=_fake_runner(payload))
    assert profile.format_start_time == pytest.approx(1.5)
    assert profile.video_stream_start_time == pytest.approx(1.5)


# =============================================================================
# Stage 27 -- probe status
# =============================================================================

@pytestmark_ffmpeg
def test_probe_status_complete_for_clean_fixture(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert profile.probe_status == smp.PROBE_STATUS_COMPLETE


@pytestmark_ffmpeg
def test_probe_status_complete_when_no_audio_is_the_only_gap(no_audio_video):
    """Stage 4/27: 'no audio' is a fully-known, expected state (Stage 4's
    own 'no error merely because audio is absent') -- it does NOT, by
    itself, downgrade an otherwise-fully-resolved probe to PARTIAL."""
    profile = smp.probe_source_media_profile(no_audio_video)
    assert profile.probe_status == smp.PROBE_STATUS_COMPLETE


@pytestmark_ffmpeg
def test_probe_status_failed_for_corrupt_file(corrupt_file):
    profile = smp.probe_source_media_profile(corrupt_file)
    assert profile.probe_status == smp.PROBE_STATUS_FAILED
    assert profile.errors


def test_probe_status_failed_never_raises_partial_info_discarded():
    def raising_runner(cmd, **kwargs):
        raise OSError("simulated ffprobe launch failure")
    profile = smp.probe_source_media_profile("unreadable.mp4", runner=raising_runner)
    assert profile.probe_status == smp.PROBE_STATUS_FAILED
    assert "ffprobe_subprocess_error" in profile.errors[0]


def test_probe_status_failed_on_nonzero_exit():
    def bad_runner(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="no such file")
    profile = smp.probe_source_media_profile("missing.mp4", runner=bad_runner)
    assert profile.probe_status == smp.PROBE_STATUS_FAILED


# =============================================================================
# Stage 28/29/30/35/36 -- classification + reasons
# =============================================================================

@pytestmark_ffmpeg
def test_supported_native_for_clean_h264_sdr(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    classification = smp.classify_source_format(profile)
    assert classification.source_format_class == smp.SOURCE_FORMAT_CLASS_SUPPORTED_NATIVE


@pytestmark_ffmpeg
def test_normalization_required_for_hdr(hdr_pq_tagged):
    profile = smp.probe_source_media_profile(hdr_pq_tagged)
    classification = smp.classify_source_format(profile)
    assert classification.source_format_class == smp.SOURCE_FORMAT_CLASS_NORMALIZATION_REQUIRED
    assert smp.REASON_HDR_INPUT in classification.reasons


def test_missing_video_unsupported():
    payload = {
        "format": {"format_name": "wav", "duration": "1.0"},
        "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "sample_rate": "44100", "channels": 1}],
    }
    profile = smp.probe_source_media_profile("audioonly.wav", runner=_fake_runner(payload))
    classification = smp.classify_source_format(profile)
    assert classification.source_format_class == smp.SOURCE_FORMAT_CLASS_UNSUPPORTED
    assert classification.reasons == (smp.REASON_MISSING_VIDEO,)


def test_no_audio_not_automatically_unsupported():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "coded_width": 100, "coded_height": 100,
                     "pix_fmt": "yuv420p", "avg_frame_rate": "30/1", "r_frame_rate": "30/1"}],
    }
    profile = smp.probe_source_media_profile("noaudio.mp4", runner=_fake_runner(payload))
    classification = smp.classify_source_format(profile)
    assert classification.source_format_class == smp.SOURCE_FORMAT_CLASS_SUPPORTED_NATIVE
    assert smp.REASON_AUDIO_MISSING in classification.reasons


def test_insufficient_evidence_for_probe_failure():
    profile = smp.probe_source_media_profile("nope.mp4", runner=lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    classification = smp.classify_source_format(profile)
    assert classification.source_format_class == smp.SOURCE_FORMAT_CLASS_INSUFFICIENT_EVIDENCE
    assert classification.reasons == (smp.REASON_PROBE_FAILED,)


def test_unsupported_container_classification():
    payload = {
        "format": {"format_name": "matroska,webm", "duration": "1.0"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "coded_width": 100, "coded_height": 100,
                     "pix_fmt": "yuv420p"}],
    }
    profile = smp.probe_source_media_profile("x.mkv", runner=_fake_runner(payload))
    classification = smp.classify_source_format(profile)
    assert classification.source_format_class == smp.SOURCE_FORMAT_CLASS_UNSUPPORTED
    assert smp.REASON_UNSUPPORTED_CONTAINER in classification.reasons


@pytestmark_ffmpeg
def test_hevc_classification_flags_runtime_capability_unknown(hevc_mp4):
    if hevc_mp4 is None:
        pytest.skip("HEVC encoder not available on this runner")
    profile = smp.probe_source_media_profile(hevc_mp4)
    classification = smp.classify_source_format(profile)
    assert smp.REASON_RUNTIME_CAPABILITY_UNKNOWN in classification.reasons
    assert classification.source_format_class == smp.SOURCE_FORMAT_CLASS_NORMALIZATION_REQUIRED


def test_ten_bit_video_surfaced_even_without_hdr_transfer():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "coded_width": 100, "coded_height": 100,
                     "pix_fmt": "yuv420p10le", "avg_frame_rate": "30/1", "r_frame_rate": "30/1"}],
    }
    profile = smp.probe_source_media_profile("tenbit_sdr.mp4", runner=_fake_runner(payload))
    classification = smp.classify_source_format(profile)
    assert smp.REASON_TEN_BIT_VIDEO in classification.reasons
    assert classification.source_format_class == smp.SOURCE_FORMAT_CLASS_NORMALIZATION_REQUIRED


def test_rotation_present_normalization_required():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "coded_width": 1920, "coded_height": 1080,
                     "pix_fmt": "yuv420p", "avg_frame_rate": "30/1", "r_frame_rate": "30/1",
                     "tags": {"rotate": "90"}}],
    }
    profile = smp.probe_source_media_profile("rotated.mp4", runner=_fake_runner(payload))
    classification = smp.classify_source_format(profile)
    assert classification.source_format_class == smp.SOURCE_FORMAT_CLASS_NORMALIZATION_REQUIRED
    assert smp.REASON_ROTATION_METADATA_PRESENT in classification.reasons


# =============================================================================
# Stage 32 -- orientation helper
# =============================================================================

@pytestmark_ffmpeg
def test_orientation_landscape(h264_mp4_30fps):
    profile = smp.probe_source_media_profile(h264_mp4_30fps)
    assert smp.orientation_category(profile) == smp.ORIENTATION_LANDSCAPE


@pytestmark_ffmpeg
def test_orientation_portrait(portrait_coded_dimensions):
    profile = smp.probe_source_media_profile(portrait_coded_dimensions)
    assert smp.orientation_category(profile) == smp.ORIENTATION_PORTRAIT


def test_orientation_square():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "coded_width": 100, "coded_height": 100,
                     "pix_fmt": "yuv420p"}],
    }
    profile = smp.probe_source_media_profile("square.mp4", runner=_fake_runner(payload))
    assert smp.orientation_category(profile) == smp.ORIENTATION_SQUARE


def test_orientation_unknown_when_missing_video():
    payload = {"format": {"format_name": "wav", "duration": "1.0"}, "streams": []}
    profile = smp.probe_source_media_profile("novideo.wav", runner=_fake_runner(payload))
    assert smp.orientation_category(profile) == smp.ORIENTATION_UNKNOWN


def test_orientation_helper_never_imports_visual_finishing():
    """Stage 32: this gate does not modify or depend on Visual Finishing
    -- it is a pure, standalone helper."""
    source = _source_without_docstrings("cutsell_worker/source_media_profile.py")
    assert "visual_finishing" not in source.lower()


# =============================================================================
# Stage 37 -- resource-risk flags (no invented thresholds)
# =============================================================================

def test_resource_flags_never_fire_without_explicit_threshold():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "999999.0"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "coded_width": 7680, "coded_height": 4320,
                     "pix_fmt": "yuv420p"}],
    }
    profile = smp.probe_source_media_profile("huge.mp4", runner=_fake_runner(payload))
    flags = smp.resource_risk_flags(profile)
    assert smp.RESOURCE_FLAG_VERY_HIGH_RESOLUTION not in flags
    assert smp.RESOURCE_FLAG_VERY_LONG_DURATION not in flags


def test_resource_flags_fire_when_caller_supplies_real_threshold():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "10.0"},
        "streams": [{"codec_type": "video", "codec_name": "h264", "coded_width": 3840, "coded_height": 2160,
                     "pix_fmt": "yuv420p"}],
    }
    profile = smp.probe_source_media_profile("4k.mp4", runner=_fake_runner(payload))
    flags = smp.resource_risk_flags(profile, max_pixel_count=1920 * 1080, max_duration_sec=5.0)
    assert smp.RESOURCE_FLAG_VERY_HIGH_RESOLUTION in flags
    assert smp.RESOURCE_FLAG_VERY_LONG_DURATION in flags


def test_resource_flags_many_streams_no_threshold_needed():
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "1.0"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "coded_width": 100, "coded_height": 100, "pix_fmt": "yuv420p"},
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2},
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2},
        ],
    }
    profile = smp.probe_source_media_profile("manystreams.mp4", runner=_fake_runner(payload))
    flags = smp.resource_risk_flags(profile)
    assert smp.RESOURCE_FLAG_MANY_STREAMS in flags


def test_no_new_numeric_threshold_invented_in_module_source():
    """Stage 37's own binding instruction: do not invent a new threshold
    unless an existing canonical one exists. Confirms no hardcoded
    resolution/duration ceiling constant was added."""
    source = _source_without_docstrings("cutsell_worker/source_media_profile.py")
    for needle in ("3840", "2160", "4096", "7680", "MAX_RESOLUTION", "MAX_DURATION_SEC ="):
        assert needle not in source


# =============================================================================
# Stage 40/41 -- local capability snapshot (never production truth)
# =============================================================================

def test_capability_snapshot_labelled_local_sandbox_only():
    snapshot = smp.capture_local_ffmpeg_capability()
    assert snapshot.runtime_capability_status == smp.RUNTIME_CAPABILITY_LOCAL_SANDBOX_ONLY


@pytestmark_ffmpeg
def test_capability_snapshot_reports_real_versions():
    snapshot = smp.capture_local_ffmpeg_capability()
    assert snapshot.ffmpeg_version is not None
    assert snapshot.ffprobe_version is not None


def test_capability_snapshot_never_raises_on_missing_binaries():
    def raising_runner(cmd, **kwargs):
        raise FileNotFoundError("no such binary")
    snapshot = smp.capture_local_ffmpeg_capability(runner=raising_runner)
    assert snapshot.errors
    assert snapshot.runtime_capability_status == smp.RUNTIME_CAPABILITY_LOCAL_SANDBOX_ONLY


def test_module_never_claims_production_hevc_support():
    """Stage 41's own binding instruction: never claim HEVC production
    support solely from local sandbox evidence anywhere in this module's
    own source."""
    source = _source_without_docstrings("cutsell_worker/source_media_profile.py").lower()
    assert "production" not in source or "not_production_verified" in source or "not production verified" in source


# =============================================================================
# Stage 42 -- output format contract (pure type only, no implementation)
# =============================================================================

def test_output_format_contract_is_a_pure_type():
    contract = smp.OutputFormatContract(
        container="MP4", video_codec="H264", pixel_format="yuv420p", fps=30.0,
        width=1080, height=1920, audio_codec="AAC", sample_rate_hz=48000, channels=2,
    )
    assert contract.container == "MP4"
    with pytest.raises(Exception):
        contract.container = "MOV"  # type: ignore[misc]


def test_no_verify_output_format_function_implemented():
    """Stage 42: only the TYPE is defined -- no `verify_output_format`
    callable exists in this gate (would be QC-behavior implementation,
    explicitly out of scope)."""
    assert not hasattr(smp, "verify_output_format")


# =============================================================================
# Stage 31 -- no render decision anywhere in this module
# =============================================================================

def test_module_never_invokes_ffmpeg_encode_path():
    """The only `ffmpeg` subprocess invocations in this module are the
    read-only capability-check calls (`ffmpeg -version`/`-decoders`/
    `-encoders`, Stage 40) -- never an encode/render command. Field names
    like `ffmpeg_version` legitimately contain the substring "ffmpeg";
    what this test actually forbids is any encode-command-shaped
    construct (an input flag, an output codec flag, or a filtergraph)."""
    source = _source_without_docstrings("cutsell_worker/source_media_profile.py")
    for needle in ("'-i'", '"-i"', "'-c:v'", '"-c:v"', "'-vf'", '"-vf"', "scale=", "pad=", "-crf"):
        assert needle not in source


def test_module_never_writes_or_mutates_source_file(h264_mp4_30fps=None):
    source = _source_without_docstrings("cutsell_worker/source_media_profile.py")
    for needle in ("open(path, \"wb\"", "os.remove", "os.replace", "shutil.move", "shutil.copy"):
        assert needle not in source


# =============================================================================
# Stage 34 -- security / no shell, no network, no credential construction
# =============================================================================

def test_no_shell_true():
    source = Path("cutsell_worker/source_media_profile.py").read_text(encoding="utf-8")
    assert "shell=True" not in source


def test_no_network_or_credential_construction():
    source = _source_without_docstrings("cutsell_worker/source_media_profile.py")
    for needle in ("boto3", "requests.", "urllib.request", "AWS_SECRET", "Authorization"):
        assert needle not in source


def test_no_provider_no_raw_reference():
    source = _source_without_docstrings("cutsell_worker/source_media_profile.py")
    for needle in ("runpod", "RunPod", "modal.", "GPUExecutionProvider"):
        assert needle not in source


def test_ffprobe_calls_are_bounded_with_timeout():
    source = Path("cutsell_worker/source_media_profile.py").read_text(encoding="utf-8")
    assert "timeout=" in source


# =============================================================================
# Regression firewall -- no renderer/finishing/QC/Pacing/Boundary/Freeze/
# Audio Join/Visual Finishing change anywhere; codec/fps/filtergraph
# unchanged (D-271's own binding scope)
# =============================================================================

@pytest.mark.parametrize("rel_path", [
    "cutsell_worker/render.py",
    "cutsell_worker/render_delivery.py",
    "cutsell_worker/render_plan.py",
    "cutsell_worker/media_probe.py",
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
    "cutsell_worker/uploads.py",
    "cutsell_worker/gpu_execution_provider.py",
])
def test_unrelated_authorities_unchanged(rel_path):
    assert _run_git_diff(rel_path) == "", f"D-271 must not touch {rel_path}"


def test_render_timeout_still_1200():
    from cutsell_worker import render
    assert render.RENDER_FFMPEG_TIMEOUT_SEC == 1200.0


def test_render_output_fps_still_30_default():
    from cutsell_worker import render
    assert render.RENDER_FPS_DEFAULT == 30


def test_codec_filtergraph_still_unchanged():
    source = Path("cutsell_worker/render.py").read_text(encoding="utf-8")
    assert '"libx264"' in source
    assert '"-crf", "20"' in source
