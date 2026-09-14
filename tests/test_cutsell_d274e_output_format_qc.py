"""D-274E -- OUTPUT FORMAT TECHNICAL QC.

Post D-274D. Proves `cutsell_worker.output_format_qc` -- the ONE
canonical output-format verification authority -- against:

  - real, ffmpeg-generated normalized-source fixtures (positive)
  - real, ffmpeg-generated malformed/unnormalized fixtures (negative)
  - a real `render_preview` output against `FINAL_RENDER_OUTPUT_
    CONTRACT_V1` (proves the disclosed color-metadata gap honestly)
  - partial-evidence behavior (missing color/rotation/fps evidence never
    fabricates PASS)
  - the D-274D executor integration (D-272 ACCEPT alone is no longer
    sufficient; a genuine format-QC FAIL after ACCEPT still reports
    NORMALIZATION_VERIFICATION_FAILED; a QC PARTIAL/UNKNOWN after ACCEPT
    never regresses an already-legitimate normalization)
  - path/filename independence, no media mutation, security
  - an 8-file closed-track firewall

NO renderer/Pacing/Boundary/Freeze/Audio-Join/Audio-Finishing/Visual-
Finishing/Delivery change, no new normalization transform, no new codec
support, no HDR policy change, no live auto-normalization activation, no
RAW, no provider, no paid compute anywhere in this file.
"""
from __future__ import annotations

import dataclasses
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from cutsell_worker import output_format_qc as ofq
from cutsell_worker import render as render_module
from cutsell_worker.render_plan import RenderSegment
from cutsell_worker import source_format_policy as sfp
from cutsell_worker import source_media_profile as smp
from cutsell_worker import source_normalization_executor as exe
from cutsell_worker import source_normalization_plan as snp

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")

_TEST_TIMEOUT_SEC = 30.0


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True, shell=False)


def _exec(source_path, plan, *, output_directory, timeout_sec=_TEST_TIMEOUT_SEC, **kwargs):
    return exe.execute_source_normalization(
        source_path, plan, output_directory=output_directory, timeout_sec=timeout_sec, **kwargs,
    )


def _make_plan(**overrides) -> snp.SourceNormalizationPlan:
    defaults = dict(
        source_identity="test_source",
        source_profile_reference="PROFILE_OK",
        contract_version=snp.CANONICAL_SOURCE_MEDIA_CONTRACT_V1.contract_version,
        container_action=snp.ACTION_NO_ACTION,
        codec_action=snp.ACTION_NO_ACTION,
        rotation_action=snp.ACTION_NO_ACTION,
        frame_rate_action=snp.ACTION_NO_ACTION,
        hdr_action=snp.ACTION_NO_ACTION,
        bit_depth_action=snp.ACTION_NO_ACTION,
        pixel_format_action=snp.ACTION_NO_ACTION,
        timeline_action=snp.ACTION_NO_ACTION,
        audio_action=snp.ACTION_NO_ACTION,
        target_contract=snp.CANONICAL_SOURCE_MEDIA_CONTRACT_V1,
        reason_codes=(),
        executability=snp.EXECUTABILITY_EXECUTABLE,
        plan_identity="normplan_test_" + uuid.uuid4().hex[:16],
        target_fps=None,
    )
    defaults.update(overrides)
    return snp.SourceNormalizationPlan(**defaults)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def canonical_sdr_mp4(tmp_path_factory):
    """A real, fully canonical H264/yuv420p/8-bit/BT709/CFR/AAC fixture --
    the positive control for NORMALIZED_SOURCE_CONTRACT_V1."""
    d = tmp_path_factory.mktemp("d274e_canonical")
    path = str(d / "canonical.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
        "-c:a", "aac", path,
    ])
    return path


@pytest.fixture(scope="module")
def wrong_container_file(tmp_path_factory):
    """A genuine WEBM file -- never trust the `.mp4` extension alone
    (Stage 9/29)."""
    d = tmp_path_factory.mktemp("d274e_wrong_container")
    path = str(d / "not_really.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24", "-t", "1",
        "-c:v", "libvpx-vp9", path.replace(".mp4", ".webm"),
    ])
    real_webm = path.replace(".mp4", ".webm")
    Path(real_webm).rename(path)
    return path


@pytest.fixture(scope="module")
def wrong_codec_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274e_wrong_codec")
    path = str(d / "mpeg4.mp4")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24", "-t", "1", "-c:v", "mpeg4", path])
    return path


@pytest.fixture(scope="module")
def yuv444_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274e_yuv444")
    path = str(d / "yuv444.mp4")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv444p", path])
    return path


@pytest.fixture(scope="module")
def ten_bit_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274e_10bit")
    path = str(d / "10bit.mp4")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p10le", path])
    return path


@pytest.fixture(scope="module")
def hdr_remaining_mp4(tmp_path_factory):
    """A genuinely still-PQ-tagged file -- the "HDR remaining" negative
    fixture (Stage 27)."""
    d = tmp_path_factory.mktemp("d274e_hdr_remaining")
    path = str(d / "still_pq.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
        path,
    ])
    return path


@pytest.fixture(scope="module")
def wrong_color_primaries_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274e_wrong_primaries")
    path = str(d / "wrong_primaries.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-color_primaries", "bt2020", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
        path,
    ])
    return path


@pytest.fixture(scope="module")
def multi_video_stream_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274e_multivideo")
    path = str(d / "multivideo.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24",
        "-t", "1", "-map", "0:v", "-map", "1:v", "-c:v", "libx264", path,
    ])
    return path


@pytest.fixture(scope="module")
def missing_video_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274e_novideo")
    path = str(d / "audio_only.mp4")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "1", path])
    return path


@pytest.fixture(scope="module")
def genuine_vfr_mp4(tmp_path_factory):
    """A genuine (non-parser-only) VFR fixture via concat of two
    differently-clocked segments -- mirrors D-274B's own construction."""
    d = tmp_path_factory.mktemp("d274e_vfr")
    seg_a = d / "a.mp4"
    seg_b = d / "b.mp4"
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", str(seg_a)])
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30", "-t", "1", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", str(seg_b)])
    list_txt = d / "list.txt"
    list_txt.write_text(f"file '{seg_a.name}'\nfile '{seg_b.name}'\n", encoding="utf-8")
    out = d / "vfr.mp4"
    _ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", str(list_txt), "-c", "copy", str(out)])
    return str(out)


@pytest.fixture(scope="module")
def rendered_output_mp4(tmp_path_factory):
    """A real `render_preview` output -- the fixture proving the FINAL_
    RENDER_OUTPUT_CONTRACT_V1's own disclosed color-metadata gap
    honestly, not a synthetic stand-in for the real renderer."""
    d = tmp_path_factory.mktemp("d274e_rendered")
    source = d / "clip.mp4"
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
    ])
    seg = RenderSegment(clip_id="c1", source_asset_id="a1", source_path=str(source), start=0.0, end=1.5)
    out = str(d / "rendered.mp4")
    render_module.render_preview([seg], out)
    return out


# =============================================================================
# QC owner / version / immutability (Stage 1/2/24/25)
# =============================================================================

def test_qc_version_single_owner():
    assert ofq.OUTPUT_FORMAT_QC_VERSION == 1


def test_contracts_are_immutable_and_distinct():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ofq.NORMALIZED_SOURCE_CONTRACT_V1.container = "WEBM"  # type: ignore[misc]
    assert ofq.NORMALIZED_SOURCE_CONTRACT_V1.artifact_type == ofq.ARTIFACT_TYPE_NORMALIZED_SOURCE
    assert ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1.artifact_type == ofq.ARTIFACT_TYPE_FINAL_RENDER
    assert ofq.NORMALIZED_SOURCE_CONTRACT_V1 != ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1


def test_contract_deterministic_identity():
    """Stage 25: same field values -> equal contracts; no local paths
    anywhere in either canonical contract's own field values."""
    replica = dataclasses.replace(ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert replica == ofq.NORMALIZED_SOURCE_CONTRACT_V1
    for contract in (ofq.NORMALIZED_SOURCE_CONTRACT_V1, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1):
        for value in dataclasses.astuple(contract):
            assert not (isinstance(value, str) and value.startswith("/"))


def test_output_format_qc_result_is_frozen():
    result = ofq.verify_output_format(
        smp.SourceMediaProfile(
            path="x", probe_status=smp.PROBE_STATUS_FAILED, container_name=smp.CONTAINER_UNKNOWN,
            raw_format_name=None, duration_sec=None, file_size_bytes=None,
            video_presence=smp.VIDEO_MISSING, audio_presence=smp.AUDIO_MISSING,
            video_stream_count=0, audio_stream_count=0, video_codec=None, raw_video_codec=None,
            video_profile=None, pixel_format=None, bit_depth=None, coded_width=None, coded_height=None,
            display_width=None, display_height=None, rotation_degrees=None,
            rotation_source=smp.ROTATION_SOURCE_UNKNOWN, avg_frame_rate=None, r_frame_rate=None,
            effective_fps=None, vfr_status=smp.VFR_STATUS_UNKNOWN, color_primaries=None, color_transfer=None,
            color_space=None, color_range=None, hdr_status=smp.HDR_STATUS_UNKNOWN, audio_codec=None,
            raw_audio_codec=None, audio_sample_rate_hz=None, audio_channels=None, audio_channel_layout=None,
            format_start_time=None, video_stream_start_time=None, audio_stream_start_time=None,
            video_time_base=None, audio_time_base=None,
        ),
        ofq.NORMALIZED_SOURCE_CONTRACT_V1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = "PASS"  # type: ignore[misc]


# =============================================================================
# Positive: canonical normalized-source fixture -> PASS
# =============================================================================

def test_canonical_sdr_fixture_passes_normalized_source_contract(canonical_sdr_mp4):
    profile = smp.probe_source_media_profile(canonical_sdr_mp4)
    result = ofq.verify_output_format(profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert result.status == ofq.STATUS_PASS, result.warnings
    assert result.failed_checks == ()
    assert ofq.CHECK_CONTAINER in result.passed_checks
    assert ofq.CHECK_VIDEO_CODEC in result.passed_checks
    assert ofq.CHECK_HDR_STATUS in result.passed_checks
    assert ofq.CHECK_COLOR_PRIMARIES in result.passed_checks


def test_real_d274_hdr_normalized_outputs_pass(tmp_path):
    """Every real D-274D HDR-normalization output category -- proven via
    the real executor -- must pass the normalized-source contract."""
    from cutsell_worker import production_runtime_capability as prc

    src_dir = Path(tmp_path) / "hdr_src"
    src_dir.mkdir()
    pq_path = str(src_dir / "pq.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", pq_path,
    ])
    cap = prc.capture_local_sandbox_capability_for_testing()
    profile = smp.probe_source_media_profile(pq_path)
    decision = sfp.evaluate_source_format_policy(profile)
    plan = snp.build_source_normalization_plan("src", profile, decision, tonemap_available=cap.hdr_tonemap_usable).plan
    result = _exec(pq_path, plan, output_directory=str(tmp_path), tonemap_capability=cap)
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED, result.diagnostics
    qc = ofq.verify_output_format(result.normalized_profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert qc.status == ofq.STATUS_PASS, qc.warnings


def test_genuine_vfr_normalized_output_passes(genuine_vfr_mp4, tmp_path):
    profile = smp.probe_source_media_profile(genuine_vfr_mp4)
    assert profile.vfr_status == smp.VFR_STATUS_LIKELY_VFR
    plan = _make_plan(frame_rate_action=snp.ACTION_VFR_TO_CFR, target_fps=profile.effective_fps)
    result = _exec(genuine_vfr_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED, result.diagnostics
    qc = ofq.verify_output_format(result.normalized_profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert qc.status in (ofq.STATUS_PASS, ofq.STATUS_PARTIAL)
    assert qc.failed_checks == ()


# =============================================================================
# Negative fixtures -- each fails the correct named check
# =============================================================================

def test_wrong_container_fails_container_check(wrong_container_file):
    profile = smp.probe_source_media_profile(wrong_container_file)
    result = ofq.verify_output_format(profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert result.status == ofq.STATUS_FAIL
    assert ofq.CHECK_CONTAINER in result.failed_checks


def test_wrong_codec_fails_codec_check(wrong_codec_mp4):
    profile = smp.probe_source_media_profile(wrong_codec_mp4)
    result = ofq.verify_output_format(profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert result.status == ofq.STATUS_FAIL
    assert ofq.CHECK_VIDEO_CODEC in result.failed_checks


def test_yuv444_fails_pixel_format_check(yuv444_mp4):
    profile = smp.probe_source_media_profile(yuv444_mp4)
    result = ofq.verify_output_format(profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert result.status == ofq.STATUS_FAIL
    assert ofq.CHECK_PIXEL_FORMAT in result.failed_checks


def test_ten_bit_fails_bit_depth_check(ten_bit_mp4):
    profile = smp.probe_source_media_profile(ten_bit_mp4)
    result = ofq.verify_output_format(profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert result.status == ofq.STATUS_FAIL
    assert ofq.CHECK_BIT_DEPTH in result.failed_checks
    assert ofq.CHECK_PIXEL_FORMAT in result.failed_checks


def test_hdr_remaining_fails_hdr_and_color_checks(hdr_remaining_mp4):
    profile = smp.probe_source_media_profile(hdr_remaining_mp4)
    result = ofq.verify_output_format(profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert result.status == ofq.STATUS_FAIL
    assert ofq.CHECK_HDR_STATUS in result.failed_checks
    assert ofq.CHECK_COLOR_PRIMARIES in result.failed_checks
    assert ofq.CHECK_COLOR_TRANSFER in result.failed_checks


def test_wrong_color_primaries_fails_only_that_check(wrong_color_primaries_mp4):
    profile = smp.probe_source_media_profile(wrong_color_primaries_mp4)
    result = ofq.verify_output_format(profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert result.status == ofq.STATUS_FAIL
    assert ofq.CHECK_COLOR_PRIMARIES in result.failed_checks
    assert ofq.CHECK_COLOR_TRANSFER not in result.failed_checks
    assert ofq.CHECK_VIDEO_CODEC not in result.failed_checks


def test_multiple_video_streams_fails_stream_count_check(multi_video_stream_mp4):
    profile = smp.probe_source_media_profile(multi_video_stream_mp4)
    if profile.video_stream_count <= 1:
        pytest.skip("fixture did not genuinely retain two video streams on this ffmpeg build")
    result = ofq.verify_output_format(profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert result.status == ofq.STATUS_FAIL
    assert ofq.CHECK_VIDEO_STREAM_COUNT in result.failed_checks


def test_missing_video_fails_video_present_check(missing_video_mp4):
    profile = smp.probe_source_media_profile(missing_video_mp4)
    result = ofq.verify_output_format(profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert result.status == ofq.STATUS_FAIL
    assert ofq.CHECK_VIDEO_PRESENT in result.failed_checks


def test_vfr_status_fails_when_still_vfr(genuine_vfr_mp4):
    """An UNNORMALIZED genuinely-VFR source must fail VFR_STATUS -- proves
    the check actually discriminates, not just passes everything."""
    profile = smp.probe_source_media_profile(genuine_vfr_mp4)
    assert profile.vfr_status == smp.VFR_STATUS_LIKELY_VFR
    result = ofq.verify_output_format(profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert result.status == ofq.STATUS_FAIL
    assert ofq.CHECK_VFR_STATUS in result.failed_checks


# =============================================================================
# Partial evidence -- missing evidence never fabricates PASS
# =============================================================================

def test_missing_color_evidence_is_partial_not_pass(canonical_sdr_mp4):
    profile = smp.probe_source_media_profile(canonical_sdr_mp4)
    stripped = dataclasses.replace(
        profile, color_primaries=None, color_transfer=None, color_space=None,
        color_range=None, hdr_status=smp.HDR_STATUS_UNKNOWN,
    )
    result = ofq.verify_output_format(stripped, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert result.status == ofq.STATUS_PARTIAL
    assert result.failed_checks == ()
    for check in (ofq.CHECK_HDR_STATUS, ofq.CHECK_COLOR_PRIMARIES, ofq.CHECK_COLOR_TRANSFER,
                  ofq.CHECK_COLOR_SPACE, ofq.CHECK_COLOR_RANGE):
        assert check in result.unknown_checks


def test_missing_rotation_evidence_still_passes_when_none(canonical_sdr_mp4):
    """`rotation_degrees=None` is itself a valid PASS value (Stage 12's
    own "0 / absent"), never PARTIAL -- absence IS the expected state."""
    profile = smp.probe_source_media_profile(canonical_sdr_mp4)
    assert profile.rotation_degrees is None
    result = ofq.verify_output_format(profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert ofq.CHECK_ROTATION in result.passed_checks


def test_probe_failed_yields_overall_unknown():
    failed_profile = smp.SourceMediaProfile(
        path="x", probe_status=smp.PROBE_STATUS_FAILED, container_name=smp.CONTAINER_UNKNOWN,
        raw_format_name=None, duration_sec=None, file_size_bytes=None,
        video_presence=smp.VIDEO_MISSING, audio_presence=smp.AUDIO_MISSING,
        video_stream_count=0, audio_stream_count=0, video_codec=None, raw_video_codec=None,
        video_profile=None, pixel_format=None, bit_depth=None, coded_width=None, coded_height=None,
        display_width=None, display_height=None, rotation_degrees=None,
        rotation_source=smp.ROTATION_SOURCE_UNKNOWN, avg_frame_rate=None, r_frame_rate=None,
        effective_fps=None, vfr_status=smp.VFR_STATUS_UNKNOWN, color_primaries=None, color_transfer=None,
        color_space=None, color_range=None, hdr_status=smp.HDR_STATUS_UNKNOWN, audio_codec=None,
        raw_audio_codec=None, audio_sample_rate_hz=None, audio_channels=None, audio_channel_layout=None,
        format_start_time=None, video_stream_start_time=None, audio_stream_start_time=None,
        video_time_base=None, audio_time_base=None,
    )
    result = ofq.verify_output_format(failed_profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert result.status == ofq.STATUS_UNKNOWN


# =============================================================================
# Audio policy -- optional for normalized source, required for final render
# =============================================================================

def test_normalized_source_audio_absent_is_fine(canonical_sdr_mp4, tmp_path):
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
             "-an", str(tmp_path / "no_audio.mp4")])
    profile = smp.probe_source_media_profile(str(tmp_path / "no_audio.mp4"))
    assert profile.audio_presence == smp.AUDIO_MISSING
    result = ofq.verify_output_format(profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert result.status == ofq.STATUS_PASS, result.warnings


def test_final_render_requires_audio_stream(rendered_output_mp4):
    profile = smp.probe_source_media_profile(rendered_output_mp4)
    assert profile.audio_presence == smp.AUDIO_PRESENT
    assert profile.audio_stream_count == 1
    assert profile.audio_codec == smp.AUDIO_CODEC_AAC
    assert profile.audio_sample_rate_hz == 48000
    assert profile.audio_channels == 2
    result = ofq.verify_output_format(profile, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1)
    assert ofq.CHECK_AUDIO_STREAM_COUNT in result.passed_checks
    assert ofq.CHECK_AUDIO_CODEC in result.passed_checks
    assert ofq.CHECK_AUDIO_SAMPLE_RATE in result.passed_checks
    assert ofq.CHECK_AUDIO_CHANNELS in result.passed_checks


# =============================================================================
# Final render contract -- real render_preview output, honest gap
# =============================================================================

def test_real_render_output_matches_video_geometry_codec_fps(rendered_output_mp4):
    profile = smp.probe_source_media_profile(rendered_output_mp4)
    assert profile.video_codec == smp.VIDEO_CODEC_H264
    assert profile.pixel_format == "yuv420p"
    assert profile.bit_depth == 8
    assert profile.display_width == 1080
    assert profile.display_height == 1920
    assert abs(profile.effective_fps - 30.0) < 0.01
    assert profile.vfr_status == smp.VFR_STATUS_CFR
    result = ofq.verify_output_format(profile, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1)
    for check in (
        ofq.CHECK_CONTAINER, ofq.CHECK_VIDEO_CODEC, ofq.CHECK_PIXEL_FORMAT, ofq.CHECK_BIT_DEPTH,
        ofq.CHECK_WIDTH, ofq.CHECK_HEIGHT, ofq.CHECK_FPS, ofq.CHECK_VFR_STATUS, ofq.CHECK_ROTATION,
        ofq.CHECK_ORIENTATION, ofq.CHECK_TIMELINE_START,
    ):
        assert check in result.passed_checks, f"{check} unexpectedly not passed: {result.warnings}"


def test_real_render_output_honestly_reports_color_metadata_gap(rendered_output_mp4):
    """Stage 23: the ACTUAL, current, undisclosed-workaround result --
    render.py writes no explicit color tags, so this is PARTIAL (missing
    evidence), never a fabricated PASS, and never silently softened to
    match the gap."""
    profile = smp.probe_source_media_profile(rendered_output_mp4)
    assert profile.color_primaries is None
    assert profile.hdr_status == smp.HDR_STATUS_UNKNOWN
    result = ofq.verify_output_format(profile, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1)
    assert result.status == ofq.STATUS_PARTIAL
    for check in (ofq.CHECK_HDR_STATUS, ofq.CHECK_COLOR_PRIMARIES, ofq.CHECK_COLOR_TRANSFER,
                  ofq.CHECK_COLOR_SPACE, ofq.CHECK_COLOR_RANGE):
        assert check in result.unknown_checks
    assert result.failed_checks == ()


def test_final_render_contract_still_declares_the_true_requirement():
    """The contract itself must NOT be softened to quietly match the
    renderer's own current gap -- these checks stay REQUIRED, not
    removed or downgraded to OPTIONAL."""
    for check in (ofq.CHECK_HDR_STATUS, ofq.CHECK_COLOR_PRIMARIES, ofq.CHECK_COLOR_TRANSFER,
                  ofq.CHECK_COLOR_SPACE, ofq.CHECK_COLOR_RANGE):
        assert check in ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1.required_checks


# =============================================================================
# D-274D executor integration (Stage 19/20/21)
# =============================================================================

def test_executor_diagnostics_carry_format_qc_status(canonical_sdr_mp4, tmp_path):
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(canonical_sdr_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED, result.diagnostics
    assert "format_qc_status" in result.diagnostics
    assert result.diagnostics["format_qc_status"] in (ofq.STATUS_PASS, ofq.STATUS_PARTIAL)


def test_d272_accept_alone_no_longer_declared_sufficient_but_partial_never_regresses(canonical_sdr_mp4, tmp_path):
    """A rotation-only normalization (no HDR action, so never writes
    explicit color tags) still SUCCEEDS -- format-QC PARTIAL must never
    retroactively fail an already-legitimate D-272 ACCEPT (this gate's
    own explicit "never regress a today-successful case" design)."""
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_180)
    result = _exec(canonical_sdr_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED, result.diagnostics
    assert result.diagnostics["execution_status"] == "SUCCESS"


def test_format_qc_fail_after_d272_accept_flips_to_verification_failed(canonical_sdr_mp4, tmp_path, monkeypatch):
    """The genuine new safety-net path: force `verify_output_format` to
    report FAIL even though D-272 itself would ACCEPT, and confirm the
    executor correctly reclassifies the outcome as `NORMALIZATION_
    VERIFICATION_FAILED` -- reusing the EXISTING outcome category, never
    inventing a new one (Stage 20)."""
    forced_fail = ofq.OutputFormatQCResult(
        status=ofq.STATUS_FAIL, artifact_type=ofq.ARTIFACT_TYPE_NORMALIZED_SOURCE, contract_version=1,
        contract_id="FORCED_TEST", required_checks=(), passed_checks=(), failed_checks=("CONTAINER",),
        unknown_checks=(), warnings=("forced failure for test",), observed_profile_summary={},
        expected_contract_summary={},
    )
    monkeypatch.setattr(exe.ofq, "verify_output_format", lambda profile, contract: forced_fail)
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(canonical_sdr_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_VERIFICATION_FAILED
    assert result.diagnostics["execution_status"] == "FORMAT_QC_FAILED_AFTER_D272_ACCEPT"
    assert result.failure is not None
    assert result.failure.error_category == snp.NORMALIZATION_VERIFICATION_FAILED


def test_format_qc_never_triggers_a_second_normalization_pass(canonical_sdr_mp4, tmp_path, monkeypatch):
    """Stage 21: a format-QC-induced failure must never itself schedule a
    retry -- `MAX_NORMALIZATION_ATTEMPTS` stays 1, and the executor's own
    one-pass firewall is unaffected by format QC at all."""
    assert snp.MAX_NORMALIZATION_ATTEMPTS == 1
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(
        canonical_sdr_mp4, plan, output_directory=str(tmp_path), attempt_count=1,
    )
    assert result.outcome == snp.NORMALIZATION_FAILED
    assert result.failure.error_category == exe.FAILURE_SECOND_PASS_REJECTED


# =============================================================================
# Path/filename independence, no media mutation
# =============================================================================

def test_path_independent_same_bytes_same_result(canonical_sdr_mp4, tmp_path):
    """Copying the same bytes to a differently-named/pathed file must
    yield an identical QC result -- filename/path play no role."""
    import shutil as _shutil
    copy_path = tmp_path / "totally_different_name.mp4"
    _shutil.copy(canonical_sdr_mp4, copy_path)
    profile_a = smp.probe_source_media_profile(canonical_sdr_mp4)
    profile_b = smp.probe_source_media_profile(str(copy_path))
    result_a = ofq.verify_output_format(profile_a, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    result_b = ofq.verify_output_format(profile_b, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert result_a.status == result_b.status
    assert result_a.passed_checks == result_b.passed_checks
    assert result_a.failed_checks == result_b.failed_checks


def test_wrong_extension_does_not_fool_container_check(wrong_container_file):
    """`wrong_container_file` is a real WEBM saved with a `.mp4` name --
    the container check must key off real probed evidence, never the
    extension (Stage 9/29)."""
    assert wrong_container_file.endswith(".mp4")
    profile = smp.probe_source_media_profile(wrong_container_file)
    assert profile.container_name != smp.CONTAINER_MP4
    result = ofq.verify_output_format(profile, ofq.NORMALIZED_SOURCE_CONTRACT_V1)
    assert ofq.CHECK_CONTAINER in result.failed_checks


def test_verify_output_format_never_touches_the_filesystem():
    """Stage 31/32: `verify_output_format` is a pure function of an
    already-probed profile -- it must never itself call subprocess.
    Checked against the CODE body only (the docstring text legitimately
    mentions ffprobe/ffmpeg in prose describing what this function does
    NOT do)."""
    import ast
    import inspect
    import textwrap
    source = inspect.getsource(ofq.verify_output_format)
    tree = ast.parse(textwrap.dedent(source))
    func_def = tree.body[0]
    body_without_docstring = ast.Module(body=func_def.body[1:], type_ignores=[])
    code_only = ast.unparse(body_without_docstring)
    assert "subprocess" not in code_only
    assert "ffprobe" not in code_only
    assert "ffmpeg" not in code_only


# =============================================================================
# Security
# =============================================================================

def test_module_never_uses_shell_or_network():
    import inspect
    source = inspect.getsource(ofq)
    assert "shell=True" not in source
    assert "subprocess" not in source
    assert "socket" not in source
    assert "requests" not in source
    assert "urllib" not in source


def test_diagnostics_never_carry_secrets(rendered_output_mp4):
    profile = smp.probe_source_media_profile(rendered_output_mp4)
    result = ofq.verify_output_format(profile, ofq.FINAL_RENDER_OUTPUT_CONTRACT_V1)
    payload = str(result.observed_profile_summary) + str(result.expected_contract_summary) + str(result.warnings)
    for forbidden in ("REDIS_URL", "password", "secret", "AKIA"):
        assert forbidden not in payload


# =============================================================================
# Closed-track firewall -- everything else byte-for-byte unchanged
# =============================================================================

@pytest.mark.parametrize("relative_path", [
    "cutsell_worker/render.py",
    "cutsell_worker/source_format_policy.py",
    "cutsell_worker/source_media_profile.py",
    "cutsell_worker/source_normalization_plan.py",
    "cutsell_worker/render_delivery.py",
    "cutsell_worker/worker_job.py",
    "cutsell_worker/post_render_media_qc.py",
    "cutsell_worker/live_render_qc.py",
])
def test_closed_track_files_unmodified_by_this_gate(relative_path):
    """D-274E's own scope is additive-only: a NEW module (`output_format_
    qc.py`) plus a narrow, additive change inside `source_normalization_
    executor.py` (the format-QC gate described above). No other
    production module's own content changes as part of this gate."""
    result = subprocess.run(
        ["git", "diff", "--stat", "fcb57cf", "--", relative_path],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.stdout.strip() == "", f"{relative_path} was modified by D-274E: {result.stdout}"


def test_worker_job_never_references_output_format_qc():
    import inspect
    from cutsell_worker import worker_job
    source = inspect.getsource(worker_job)
    assert "output_format_qc" not in source
    assert "verify_output_format" not in source
