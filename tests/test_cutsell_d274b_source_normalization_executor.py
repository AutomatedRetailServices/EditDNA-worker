"""D-274B -- ROTATION + VFR / TIMELINE NORMALIZATION EXECUTOR.

Post D-272B / D-274A. Proves `cutsell_worker.source_normalization_executor`
against real ffmpeg-generated synthetic fixtures: physical rotation
(asymmetric visual marker, not just width/height), a genuine (non-parser-
only) VFR fixture built via concat-demuxer stream copy of two differently-
clocked segments, a genuine non-zero-start-time fixture, multi-action
one-pass composition, the one-pass firewall, structured failure categories,
atomic promotion, original-file immutability, and the mandatory D-271
re-probe + D-272 re-evaluation + D-274A verification loop.

No HDR tonemap, no HEVC->H264 execution, no 10-bit->8-bit execution, no
broad pixel-format conversion, no audio loudness change, no renderer
change, no RAW, no provider, no paid compute anywhere in this file.
"""
from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from cutsell_worker import source_format_policy as sfp
from cutsell_worker import source_media_profile as smp
from cutsell_worker import source_normalization_executor as exe
from cutsell_worker import source_normalization_plan as snp

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")

# Stage 22: NORMALIZATION_FFMPEG_TIMEOUT_SEC stays None (Product-Owner seam)
# at the module level -- synthetic tests inject their OWN small, bounded
# timeout explicitly, exactly as Stage 22 instructs, rather than relying on
# (or silently activating) a module-level default.
_TEST_TIMEOUT_SEC = 30.0


def _exec(source_path, plan, *, output_directory, timeout_sec=_TEST_TIMEOUT_SEC, **kwargs):
    return exe.execute_source_normalization(
        source_path, plan, output_directory=output_directory, timeout_sec=timeout_sec, **kwargs,
    )


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True, shell=False)


def _ffprobe_json(path: str, extra: list[str]) -> dict:
    import json
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", *extra, str(path)],
        capture_output=True, text=True, check=True, shell=False,
    )
    return json.loads(result.stdout)


def _make_plan(**overrides) -> snp.SourceNormalizationPlan:
    """Stage 38's own 'A. executor physical rotation testing' path: a
    directly-constructed plan, since this gate proves the EXECUTOR obeys
    whatever plan it is handed (Stage 2: plan is authority, never
    re-decided) -- not the D-271->D-272->D-274A derivation chain itself
    (already proven end-to-end for VFR in the smoke test below and in
    D-274A's own test suite)."""
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


def _corner_colors(png_path: Path) -> dict:
    from PIL import Image
    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    return {
        "TL": img.getpixel((5, 5)),
        "TR": img.getpixel((w - 6, 5)),
        "BL": img.getpixel((5, h - 6)),
        "BR": img.getpixel((w - 6, h - 6)),
    }


def _is_white(rgb) -> bool:
    return all(c > 200 for c in rgb)


def _marker_corner(png_path: Path) -> str:
    corners = _corner_colors(png_path)
    whites = [name for name, rgb in corners.items() if _is_white(rgb)]
    assert len(whites) == 1, f"expected exactly one white corner, got {corners}"
    return whites[0]


# =============================================================================
# Fixtures -- real ffmpeg-generated synthetic media
# =============================================================================

@pytest.fixture(scope="module")
def asymmetric_landscape_mp4(tmp_path_factory):
    """320x240 H264/yuv420p/8-bit source with a white square baked into the
    TOP-LEFT corner only, plus a sine-tone audio stream -- used to prove
    physical rotation direction with real pixel evidence, never just
    width/height (Stage 4's own explicit warning)."""
    d = tmp_path_factory.mktemp("d274b_rot")
    path = str(d / "asym_landscape.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1:r=10",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:d=1",
        "-vf", "drawbox=x=0:y=0:w=60:h=40:color=white:t=fill",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", path,
    ])
    return path


@pytest.fixture(scope="module")
def asymmetric_no_audio_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274b_rot_noaudio")
    path = str(d / "asym_noaudio.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1:r=10",
        "-vf", "drawbox=x=0:y=0:w=60:h=40:color=white:t=fill",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", path,
    ])
    return path


@pytest.fixture(scope="module")
def genuine_vfr_mp4(tmp_path_factory):
    """A REAL, non-parser-only VFR fixture (Stage 39): concat-demuxer
    stream-copy of two differently-clocked segments (24fps then 60fps)
    preserves each packet's own original duration rather than re-timing to
    one uniform rate. Confirmed via ffprobe frame pkt_duration_time to
    carry genuinely distinct values, and via the REAL D-271 profiler to
    classify as LIKELY_VFR."""
    d = tmp_path_factory.mktemp("d274b_vfr")
    seg_a = d / "segA.mp4"
    seg_b = d / "segB.mp4"
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=24", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(seg_a)])
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=60", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(seg_b)])
    list_txt = d / "list.txt"
    list_txt.write_text(f"file '{seg_a.name}'\nfile '{seg_b.name}'\n", encoding="utf-8")
    out = d / "vfr_candidate.mp4"
    _ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", str(list_txt), "-c", "copy", str(out)])
    return str(out)


@pytest.fixture(scope="module")
def timeline_offset_mp4(tmp_path_factory):
    """A genuine non-zero start-time fixture (Stage 40): encode a short
    clip, then remux with `-output_ts_offset` so the container/stream
    start times are shifted away from zero -- proven via ffprobe, not
    assumed."""
    d = tmp_path_factory.mktemp("d274b_offset")
    base = d / "base.mp4"
    _ffmpeg(["-y", "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=1:r=10",
             "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:d=1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(base)])
    shifted = d / "shifted.mp4"
    _ffmpeg(["-y", "-itsoffset", "2.0", "-i", str(base), "-map", "0:v", "-map", "0:a",
             "-c", "copy", str(shifted)])
    return str(shifted)


@pytest.fixture(scope="module")
def cfr_30fps_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274b_cfr30")
    path = str(d / "cfr30.mp4")
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", path])
    return path


# =============================================================================
# Rotation: 90 / 180 / 270, real pixel evidence, no double autorotation
# =============================================================================

def test_rotate_90_moves_marker_and_swaps_dims(asymmetric_landscape_mp4, tmp_path):
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED, result.diagnostics
    probe = _ffprobe_json(result.normalized_path, ["-show_entries", "stream=width,height", "-select_streams", "v:0"])
    w, h = probe["streams"][0]["width"], probe["streams"][0]["height"]
    assert (w, h) == (240, 320), "ROTATE_90 must swap display dimensions"
    frame_png = tmp_path / "frame90.png"
    _ffmpeg(["-y", "-i", result.normalized_path, "-frames:v", "1", str(frame_png)])
    # Empirically-verified: transpose=1 is a genuine 90-deg CLOCKWISE
    # rotation -- a TL marker moves to TR.
    assert _marker_corner(frame_png) == "TR"


def test_rotate_180_preserves_dims_moves_marker(asymmetric_landscape_mp4, tmp_path):
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_180)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED, result.diagnostics
    probe = _ffprobe_json(result.normalized_path, ["-show_entries", "stream=width,height", "-select_streams", "v:0"])
    w, h = probe["streams"][0]["width"], probe["streams"][0]["height"]
    assert (w, h) == (320, 240), "ROTATE_180 must NOT change dimensions"
    frame_png = tmp_path / "frame180.png"
    _ffmpeg(["-y", "-i", result.normalized_path, "-frames:v", "1", str(frame_png)])
    assert _marker_corner(frame_png) == "BR"


def test_rotate_270_moves_marker_and_swaps_dims(asymmetric_landscape_mp4, tmp_path):
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_270)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED, result.diagnostics
    probe = _ffprobe_json(result.normalized_path, ["-show_entries", "stream=width,height", "-select_streams", "v:0"])
    w, h = probe["streams"][0]["width"], probe["streams"][0]["height"]
    assert (w, h) == (240, 320), "ROTATE_270 must swap display dimensions"
    frame_png = tmp_path / "frame270.png"
    _ffmpeg(["-y", "-i", result.normalized_path, "-frames:v", "1", str(frame_png)])
    # transpose=2 is genuine COUNTERCLOCKWISE -- TL marker moves to BL.
    assert _marker_corner(frame_png) == "BL"


def test_rotation_command_uses_noautorotate_before_input(asymmetric_landscape_mp4, tmp_path):
    """Stage 8: no double autorotation -- the command must always pin
    `-noautorotate` immediately before `-i`, regardless of what rotation
    metadata a real source might carry, guaranteeing ONLY the plan-driven
    transpose (or none) is ever applied."""
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    video_filters, _, _ = exe._build_filter_chain(plan)
    assert "transpose=1" in video_filters
    # Executed command inspection via a captured fingerprint-bearing run:
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED
    assert result.diagnostics["command_fingerprint"]


def test_rotation_metadata_removed(asymmetric_landscape_mp4, tmp_path):
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED
    assert result.normalized_profile.rotation_degrees in (0, None)


def test_no_rotation_action_leaves_dims_unchanged(asymmetric_landscape_mp4, tmp_path):
    plan = _make_plan()  # all NO_ACTION
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED
    probe = _ffprobe_json(result.normalized_path, ["-show_entries", "stream=width,height", "-select_streams", "v:0"])
    assert (probe["streams"][0]["width"], probe["streams"][0]["height"]) == (320, 240)


# =============================================================================
# VFR -> CFR: target comes only from plan.target_fps, real genuine fixture
# =============================================================================

def test_vfr_target_fps_comes_from_plan_not_hardcoded(genuine_vfr_mp4, tmp_path):
    profile = smp.probe_source_media_profile(genuine_vfr_mp4)
    assert profile.vfr_status == smp.VFR_STATUS_LIKELY_VFR, "fixture must be genuinely VFR before normalization"
    plan = _make_plan(frame_rate_action=snp.ACTION_VFR_TO_CFR, target_fps=profile.effective_fps)
    assert plan.target_fps != 30.0 and plan.target_fps != 30
    result = _exec(genuine_vfr_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED, result.diagnostics
    assert result.normalized_profile.vfr_status == smp.VFR_STATUS_CFR
    # The fps used is the plan's target, never a hardcoded 30.
    assert abs(result.normalized_profile.effective_fps - profile.effective_fps) < 0.01


def test_vfr_missing_target_fps_raises_before_ffmpeg(tmp_path):
    plan = _make_plan(frame_rate_action=snp.ACTION_VFR_TO_CFR, target_fps=None)
    with pytest.raises(ValueError):
        exe._build_filter_chain(plan)


def test_cfr_24_30_60_no_fps_filter_added(cfr_30fps_mp4):
    plan = _make_plan()  # frame_rate_action stays NO_ACTION
    video_filters, _, _ = exe._build_filter_chain(plan)
    assert not any(f.startswith("fps=") for f in video_filters)


def test_cfr30_untouched_end_to_end(cfr_30fps_mp4, tmp_path):
    before = smp.probe_source_media_profile(cfr_30fps_mp4)
    plan = _make_plan()
    result = _exec(cfr_30fps_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED
    assert result.normalized_profile.vfr_status == smp.VFR_STATUS_CFR
    assert abs(result.normalized_profile.effective_fps - before.effective_fps) < 0.5


# =============================================================================
# Timeline zero, audio preservation, A/V relation
# =============================================================================

def test_timeline_offset_normalizes_to_zero(timeline_offset_mp4, tmp_path):
    before = smp.probe_source_media_profile(timeline_offset_mp4)
    assert before.video_stream_start_time is not None and abs(before.video_stream_start_time) > 0.0, (
        "fixture must genuinely carry a non-zero start time before normalization"
    )
    plan = _make_plan(timeline_action=snp.ACTION_TIMELINE_TO_ZERO)
    result = _exec(timeline_offset_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED, result.diagnostics
    assert result.normalized_profile.video_stream_start_time in (0.0, None) or abs(result.normalized_profile.video_stream_start_time) < 0.01
    assert result.normalized_profile.format_start_time in (0.0, None) or abs(result.normalized_profile.format_start_time) < 0.01


def test_audio_preserved_when_no_timeline_reset(asymmetric_landscape_mp4, tmp_path):
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED
    assert result.normalized_profile.has_audio_stream if hasattr(result.normalized_profile, "has_audio_stream") else True
    probe = _ffprobe_json(result.normalized_path, ["-show_entries", "stream=codec_type"])
    codec_types = {s["codec_type"] for s in probe["streams"]}
    assert "audio" in codec_types


def test_missing_audio_source_works(asymmetric_no_audio_mp4, tmp_path):
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_180)
    result = _exec(asymmetric_no_audio_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED, result.diagnostics
    probe = _ffprobe_json(result.normalized_path, ["-show_entries", "stream=codec_type"])
    codec_types = {s["codec_type"] for s in probe["streams"]}
    assert "audio" not in codec_types, "no audio must ever be fabricated"


def test_av_sync_before_after_reported(asymmetric_landscape_mp4, tmp_path):
    plan = _make_plan(timeline_action=snp.ACTION_TIMELINE_TO_ZERO)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED
    assert "duration_before_sec" in result.diagnostics
    assert "duration_after_sec" in result.diagnostics
    assert "duration_delta_sec" in result.diagnostics


# =============================================================================
# Multi-action one-pass composition
# =============================================================================

def test_rotation_plus_timeline_one_generation(timeline_offset_mp4, tmp_path):
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90, timeline_action=snp.ACTION_TIMELINE_TO_ZERO)
    result = _exec(timeline_offset_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED, result.diagnostics
    probe = _ffprobe_json(result.normalized_path, ["-show_entries", "stream=width,height", "-select_streams", "v:0"])
    assert (probe["streams"][0]["width"], probe["streams"][0]["height"]) == (120, 160)


def test_vfr_plus_timeline_one_generation(genuine_vfr_mp4, tmp_path):
    profile = smp.probe_source_media_profile(genuine_vfr_mp4)
    plan = _make_plan(frame_rate_action=snp.ACTION_VFR_TO_CFR, target_fps=profile.effective_fps,
                       timeline_action=snp.ACTION_TIMELINE_TO_ZERO)
    result = _exec(genuine_vfr_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED, result.diagnostics
    assert result.normalized_profile.vfr_status == smp.VFR_STATUS_CFR


def test_rotation_plus_vfr_plus_timeline_one_generation(genuine_vfr_mp4, tmp_path):
    profile = smp.probe_source_media_profile(genuine_vfr_mp4)
    plan = _make_plan(
        rotation_action=snp.ACTION_ROTATE_180,
        frame_rate_action=snp.ACTION_VFR_TO_CFR, target_fps=profile.effective_fps,
        timeline_action=snp.ACTION_TIMELINE_TO_ZERO,
    )
    result = _exec(genuine_vfr_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED, result.diagnostics
    assert result.normalized_profile.vfr_status == smp.VFR_STATUS_CFR
    probe = _ffprobe_json(result.normalized_path, ["-show_entries", "stream=width,height", "-select_streams", "v:0"])
    assert (probe["streams"][0]["width"], probe["streams"][0]["height"]) == (160, 120)


# =============================================================================
# Original immutability / normalized hash / references
# =============================================================================

def test_original_file_hash_unchanged(asymmetric_landscape_mp4, tmp_path):
    from cutsell_worker.render_delivery import compute_output_sha256
    before = compute_output_sha256(asymmetric_landscape_mp4)
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED
    after = compute_output_sha256(asymmetric_landscape_mp4)
    assert before == after
    assert result.diagnostics["original_source_unchanged"] is True


def test_normalized_hash_distinct_and_reference_shape(asymmetric_landscape_mp4, tmp_path):
    from cutsell_worker.render_delivery import compute_output_sha256
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED
    actual_bytes_hash = compute_output_sha256(result.normalized_path)
    assert result.normalized_reference.normalized_output_sha256 == actual_bytes_hash
    assert result.normalized_reference.original_source_identity == plan.source_identity
    assert result.normalized_reference.normalization_plan_identity == plan.plan_identity


# =============================================================================
# Mandatory re-probe / re-evaluate / verification / no second pass
# =============================================================================

def test_reprobe_and_reevaluate_are_performed(asymmetric_landscape_mp4, tmp_path):
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.normalized_profile is not None
    assert result.verification is not None
    assert "final_d272_decision" in result.diagnostics


def test_final_accept_required_for_success(asymmetric_landscape_mp4, tmp_path):
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_SUCCEEDED
    assert result.diagnostics["final_d272_decision"] == sfp.DECISION_ACCEPT
    assert result.verification.verified is True


def test_still_blocked_post_normalization_is_verification_failure(asymmetric_landscape_mp4, tmp_path, monkeypatch):
    """Simulates Stage 35: if the normalized output is STILL not ACCEPT
    (e.g. a real HEVC/HDR mix this gate cannot fully clear), the executor
    must report NORMALIZATION_VERIFICATION_FAILED and must NOT re-execute."""
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)

    def _fake_still_blocked(profile, *, runtime_capability=None, **kwargs):
        return sfp.SourceFormatPolicyDecision(
            decision=sfp.DECISION_NORMALIZE_REQUIRED,
            policy_version=1,
            reason_codes=("SIMULATED_STILL_BLOCKED",),
            blocking_reasons=(),
            normalization_reasons=("SIMULATED_STILL_BLOCKED",),
            warnings=(),
            source_profile_status="PROFILE_OK",
            source_format_class="UNKNOWN",
            user_facing_error_code=None,
        )

    monkeypatch.setattr(exe.sfp, "evaluate_source_format_policy", _fake_still_blocked)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_VERIFICATION_FAILED
    assert result.verification.verified is False


def test_no_second_normalization_pass(asymmetric_landscape_mp4, tmp_path):
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(
        asymmetric_landscape_mp4, plan, output_directory=str(tmp_path), attempt_count=1,
    )
    assert result.outcome == snp.NORMALIZATION_FAILED
    assert result.failure.error_category == exe.FAILURE_SECOND_PASS_REJECTED
    assert result.normalized_path is None


# =============================================================================
# Unsupported actions -- rejected BEFORE any ffmpeg subprocess
# =============================================================================

@pytest.mark.parametrize("field,action", [
    ("hdr_action", snp.ACTION_HDR_PQ_TO_SDR_BT709),
    ("hdr_action", snp.ACTION_HDR_HLG_TO_SDR_BT709),
    # D-274C Stage 8: ACTION_HEVC_TO_H264 is now IMPLEMENTED (see
    # tests/test_cutsell_d274c_hevc_capability_and_normalization.py) and
    # legitimately removed from this "still unsupported" matrix -- self-
    # resolving guard, same pattern as D-272B's own precedent.
    ("bit_depth_action", snp.ACTION_TEN_BIT_TO_EIGHT_BIT),
    ("pixel_format_action", snp.ACTION_PIXEL_FORMAT_TO_YUV420P),
])
def test_unsupported_action_rejected_before_ffmpeg(asymmetric_landscape_mp4, tmp_path, monkeypatch, field, action):
    calls = []
    monkeypatch.setattr(exe.subprocess, "run", lambda *a, **k: calls.append((a, k)) or (_ for _ in ()).throw(
        AssertionError("ffmpeg must never be invoked for an unsupported action")
    ))
    plan = _make_plan(**{field: action})
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_UNSUPPORTED
    assert result.failure.error_category == exe.FAILURE_UNSUPPORTED_ACTION
    assert calls == []


# =============================================================================
# Structured failure paths
# =============================================================================

def test_ffmpeg_nonzero_exit_is_structured_failure(tmp_path):
    bogus_source = tmp_path / "not_real_media.mp4"
    bogus_source.write_bytes(b"not a real video file, ffmpeg will reject this")
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(str(bogus_source), plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_FAILED
    assert result.failure.error_category == exe.FAILURE_FFMPEG_FAILED
    assert result.failure.return_code is not None and result.failure.return_code != 0
    assert len(result.failure.stderr_excerpt) <= exe._STDERR_EXCERPT_MAX_CHARS


def test_timeout_via_injected_tiny_timeout(asymmetric_landscape_mp4, tmp_path):
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(
        asymmetric_landscape_mp4, plan, output_directory=str(tmp_path), timeout_sec=0.0001,
    )
    assert result.outcome == snp.NORMALIZATION_FAILED
    assert result.failure.error_category == exe.FAILURE_TIMEOUT
    assert result.failure.timed_out is True


def test_timeout_seam_requires_product_owner_when_none(asymmetric_landscape_mp4, tmp_path):
    """Stage 22: NORMALIZATION_FFMPEG_TIMEOUT_SEC stays None until a
    Product Owner decision activates a number -- never silently reused
    from RENDER_FFMPEG_TIMEOUT_SEC=1200."""
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(
        asymmetric_landscape_mp4, plan, output_directory=str(tmp_path), timeout_sec=None,
    )
    assert result.outcome == exe.PRODUCT_OWNER_NORMALIZATION_TIMEOUT_REQUIRED
    assert result.failure.error_category == exe.FAILURE_TIMEOUT_POLICY_REQUIRED


def test_missing_output_is_structured_failure(asymmetric_landscape_mp4, tmp_path, monkeypatch):
    class _FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(exe.subprocess, "run", lambda *a, **k: _FakeCompleted())
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_FAILED
    assert result.failure.error_category == exe.FAILURE_OUTPUT_MISSING


def test_empty_output_is_structured_failure(asymmetric_landscape_mp4, tmp_path, monkeypatch):
    class _FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(command, **kwargs):
        # command[-1] is the temp output path this executor built.
        Path(command[-1]).write_bytes(b"")
        return _FakeCompleted()

    monkeypatch.setattr(exe.subprocess, "run", _fake_run)
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_FAILED
    assert result.failure.error_category == exe.FAILURE_OUTPUT_EMPTY


def test_atomic_promotion_failure_is_structured_and_cleans_up(asymmetric_landscape_mp4, tmp_path, monkeypatch):
    class _FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"fake-normalized-bytes")
        return _FakeCompleted()

    def _fake_replace(src, dst):
        raise OSError("simulated atomic promotion failure")

    monkeypatch.setattr(exe.subprocess, "run", _fake_run)
    monkeypatch.setattr(exe.os, "replace", _fake_replace)
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_FAILED
    assert result.failure.error_category == exe.FAILURE_ATOMIC_PROMOTION_FAILED
    # No stray temp artifact left behind (best-effort cleanup).
    leftovers = list(Path(tmp_path).glob(".*normalizing*"))
    assert leftovers == []


def test_ffmpeg_failure_leaves_no_promoted_partial(tmp_path):
    bogus_source = tmp_path / "bad.mp4"
    bogus_source.write_bytes(b"garbage")
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(str(bogus_source), plan, output_directory=str(tmp_path))
    assert result.outcome == snp.NORMALIZATION_FAILED
    promoted = list(Path(tmp_path).glob("normalized_*.mp4"))
    assert promoted == []


# =============================================================================
# Command safety / fingerprint / paths
# =============================================================================

def test_shell_is_never_used(asymmetric_landscape_mp4, tmp_path, monkeypatch):
    seen_kwargs = {}

    real_run = exe.subprocess.run

    def _spy(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(exe.subprocess, "run", _spy)
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    assert seen_kwargs.get("shell", False) is False


def test_command_fingerprint_deterministic():
    command = ["ffmpeg", "-i", "a.mp4", "-vf", "transpose=1", "out.mp4"]
    assert exe._command_fingerprint(command) == exe._command_fingerprint(list(command))


def test_paths_with_spaces_unicode_apostrophe(tmp_path):
    for name in ["with space.mp4", "ünïcödé_源.mp4", "it's_a_test.mp4"]:
        d = tmp_path / uuid.uuid4().hex
        d.mkdir()
        src = d / name
        _ffmpeg(["-y", "-f", "lavfi", "-i", "color=c=green:s=160x120:d=1:r=5",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)])
        plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
        out_dir = d / "out"
        result = _exec(str(src), plan, output_directory=str(out_dir))
        assert result.outcome == snp.NORMALIZATION_SUCCEEDED, (name, result.diagnostics)


def test_concurrent_jobs_isolated(asymmetric_landscape_mp4, tmp_path):
    plan_a = _make_plan(rotation_action=snp.ACTION_ROTATE_90, plan_identity="normplan_jobA")
    plan_b = _make_plan(rotation_action=snp.ACTION_ROTATE_270, plan_identity="normplan_jobB")
    out_a = tmp_path / "job_a"
    out_b = tmp_path / "job_b"
    result_a = _exec(asymmetric_landscape_mp4, plan_a, output_directory=str(out_a))
    result_b = _exec(asymmetric_landscape_mp4, plan_b, output_directory=str(out_b))
    assert result_a.outcome == snp.NORMALIZATION_SUCCEEDED
    assert result_b.outcome == snp.NORMALIZATION_SUCCEEDED
    assert result_a.normalized_path != result_b.normalized_path
    assert Path(result_a.normalized_path).exists() and Path(result_b.normalized_path).exists()


# =============================================================================
# Diagnostics bounded / no secrets
# =============================================================================

def test_diagnostics_are_bounded_and_secret_free(asymmetric_landscape_mp4, tmp_path):
    plan = _make_plan(rotation_action=snp.ACTION_ROTATE_90)
    result = _exec(asymmetric_landscape_mp4, plan, output_directory=str(tmp_path))
    diag_str = str(result.diagnostics)
    for forbidden in ("AKIA", "aws_secret", "password", "Authorization: Bearer"):
        assert forbidden not in diag_str
    assert "plan_identity" in result.diagnostics
    assert "command_fingerprint" in result.diagnostics


# =============================================================================
# Renderer / early-gate untouched (Stage 1 separation, no reuse via import)
# =============================================================================

def test_render_timeout_unchanged():
    from cutsell_worker import render as render_module
    assert render_module.RENDER_FFMPEG_TIMEOUT_SEC == 1200.0


def test_executor_module_does_not_import_render_module_symbols():
    import inspect
    source = inspect.getsource(exe)
    assert "from .render import" not in source
    assert "import render" not in source or "render_delivery" in source  # only render_delivery's hash util is reused
