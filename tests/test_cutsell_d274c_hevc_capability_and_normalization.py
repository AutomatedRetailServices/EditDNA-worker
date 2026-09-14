"""D-274C -- PRODUCTION HEVC CAPABILITY + HEVC->H264 NORMALIZATION.

Post D-274B. Proves (a) `cutsell_worker.production_runtime_capability`'s
typed production capability contract, its reuse of D-271's own bounded
ffmpeg decoder/encoder inspection mechanism, and its fail-closed bridge
into D-272's `RuntimeCapabilityInput`; and (b) that `cutsell_worker.
source_normalization_executor` now correctly EXECUTES `HEVC_TO_H264` for
confirmed-capability SDR 8-bit HEVC sources, while the pre-existing
unsupported-action firewall continues to reject any HEVC+HDR or
HEVC+10-bit composition before any ffmpeg call -- with real HEVC (libx265)
synthetic fixtures throughout, not merely metadata-shaped payloads.

No HDR tonemap, no PQ/HLG execution, no general 10-bit normalization, no
broad pixel-format normalization, no live auto-normalization activation,
no renderer/Pacing/Boundary/Freeze/Audio-Join/Audio-Finishing/Visual-
Finishing/delivery change, no RAW, no provider, no paid compute anywhere
in this file.
"""
from __future__ import annotations

import dataclasses
import shutil
import subprocess
from pathlib import Path

import pytest

from cutsell_worker import production_runtime_capability as prc
from cutsell_worker import source_format_policy as sfp
from cutsell_worker import source_media_profile as smp
from cutsell_worker import source_normalization_executor as exe
from cutsell_worker import source_normalization_plan as snp

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available on this runner",
)

_TEST_TIMEOUT_SEC = 30.0
_HEVC_CONFIRMED = sfp.RuntimeCapabilityInput(hevc_decode_confirmed=True)


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True, shell=False)


def _ffprobe_streams(path: str) -> list[dict]:
    import json
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True, check=True, shell=False,
    )
    return json.loads(result.stdout)["streams"]


def _build_plan(path: str, source_identity: str, *, capability=_HEVC_CONFIRMED):
    profile = smp.probe_source_media_profile(path)
    decision = sfp.evaluate_source_format_policy(profile, runtime_capability=capability)
    result = snp.build_source_normalization_plan(source_identity, profile, decision, runtime_capability=capability)
    return profile, decision, result


def _exec(source_path, plan, *, output_directory, timeout_sec=_TEST_TIMEOUT_SEC, **kwargs):
    return exe.execute_source_normalization(
        source_path, plan, output_directory=output_directory, timeout_sec=timeout_sec, **kwargs,
    )


# =============================================================================
# Fixtures -- real ffmpeg/libx265-generated synthetic HEVC media
# =============================================================================

@pytest.fixture(scope="module", autouse=True)
def _require_hevc_codec():
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    decoders = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"], capture_output=True, text=True).stdout
    if "libx265" not in encoders.lower() or "hevc" not in decoders.lower():
        pytest.skip("libx265 encoder / hevc decoder not available on this runner")


@pytest.fixture(scope="module")
def hevc_sdr_8bit_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274c_hevc")
    path = str(d / "hevc_sdr_8bit.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", "1", "-c:v", "libx265", "-x265-params", "log-level=none",
        "-pix_fmt", "yuv420p", "-c:a", "aac", path,
    ])
    return path


@pytest.fixture(scope="module")
def hevc_sdr_mov(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274c_hevc_mov")
    path = str(d / "hevc_sdr.mov")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
        "-c:v", "libx265", "-x265-params", "log-level=none", "-pix_fmt", "yuv420p", path,
    ])
    return path


@pytest.fixture(scope="module")
def hevc_10bit_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274c_hevc_10bit")
    path = str(d / "hevc_10bit.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
        "-c:v", "libx265", "-x265-params", "log-level=none", "-pix_fmt", "yuv420p10le", path,
    ])
    return path


@pytest.fixture(scope="module", params=["smpte2084", "arib-std-b67"])
def hevc_hdr_10bit_mp4(request, tmp_path_factory):
    """D-274D: a GENUINE HEVC + HDR (PQ or HLG) + 10-bit fixture -- real,
    ffprobe-readable `color_transfer` tags (`smpte2084`=PQ, `arib-std-
    b67`=HLG), not a plan/profile mismatch forced via `dataclasses.
    replace`. D-274D's own forensic proof (docs/CUTSELL_DECISIONS.md
    D-274D) found `zscale`'s own `transfer=linear` step requires the
    DECODED frames to genuinely carry a PQ/HLG transfer tag it can map
    from -- forcing a plan's `hdr_action` onto a genuinely-SDR source
    (as this file's own D-274C-era test originally did) fails inside
    ffmpeg itself ("no path between colorspaces"), never inside this
    executor's own code; a composition test must use a genuinely
    HDR-tagged source, exactly like this fixture."""
    d = tmp_path_factory.mktemp("d274d_hevc_hdr")
    path = str(d / f"hevc_hdr_{request.param}.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
        "-c:v", "libx265", "-x265-params", "log-level=none", "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020", "-color_trc", request.param, "-colorspace", "bt2020nc",
        path,
    ])
    return path


@pytest.fixture(scope="module")
def hevc_no_audio_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274c_hevc_noaudio")
    path = str(d / "hevc_noaudio.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
        "-c:v", "libx265", "-x265-params", "log-level=none", "-pix_fmt", "yuv420p", "-an", path,
    ])
    return path


@pytest.fixture(scope="module")
def hevc_asymmetric_mp4(tmp_path_factory):
    """A real asymmetric-marker HEVC fixture, for pixel-level rotation
    proof composed with HEVC_TO_H264 (Stage 15)."""
    d = tmp_path_factory.mktemp("d274c_hevc_asym")
    path = str(d / "hevc_asym.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1:r=10",
        "-vf", "drawbox=x=0:y=0:w=60:h=40:color=white:t=fill",
        "-c:v", "libx265", "-x265-params", "log-level=none", "-pix_fmt", "yuv420p", path,
    ])
    return path


@pytest.fixture(scope="module")
def hevc_offset_mp4(hevc_sdr_8bit_mp4, tmp_path_factory):
    """A genuine non-zero-start-time HEVC fixture (Stage 15's own
    "HEVC + timeline reset"), via -itsoffset remux."""
    d = tmp_path_factory.mktemp("d274c_hevc_offset")
    path = str(d / "hevc_offset.mp4")
    _ffmpeg(["-y", "-itsoffset", "2.0", "-i", hevc_sdr_8bit_mp4, "-map", "0:v", "-map", "0:a", "-c", "copy", path])
    return path


def _marker_corner(png_path: Path) -> str:
    from PIL import Image
    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    corners = {
        "TL": img.getpixel((5, 5)), "TR": img.getpixel((w - 6, 5)),
        "BL": img.getpixel((5, h - 6)), "BR": img.getpixel((w - 6, h - 6)),
    }
    whites = [name for name, rgb in corners.items() if all(c > 200 for c in rgb)]
    assert len(whites) == 1, corners
    return whites[0]


# =============================================================================
# production_runtime_capability -- typed contract + mechanism reuse
# =============================================================================

def test_capability_type_fields():
    cap = prc.ProductionRuntimeCapability(
        hevc_decoder_available=True, h264_encoder_available=True, ffmpeg_version="x",
        capability_source=prc.CAPABILITY_SOURCE_LOCAL_SANDBOX,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED,
    )
    assert cap.hevc_decoder_available is True
    assert cap.h264_encoder_available is True


def test_sandbox_capture_reuses_d271_mechanism():
    cap = prc.capture_local_sandbox_capability_for_testing()
    reference = smp.capture_local_ffmpeg_capability()
    assert cap.hevc_decoder_available == reference.hevc_decoder_present
    assert cap.h264_encoder_available == reference.libx264_present
    assert cap.capability_source == prc.CAPABILITY_SOURCE_LOCAL_SANDBOX


def test_production_capture_uses_distinct_label():
    cap = prc.capture_production_worker_capability()
    assert cap.capability_source == prc.CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK
    # Same mechanism -- same real ffmpeg answer -- as the sandbox path.
    sandbox_cap = prc.capture_local_sandbox_capability_for_testing()
    assert cap.hevc_decoder_available == sandbox_cap.hevc_decoder_available


def test_no_capture_function_ever_claims_established():
    """Stage 1's own 'do not use developer-sandbox capability as
    production truth': neither capture path can, on its own authority,
    claim PRODUCTION_CAPABILITY_ESTABLISHED -- only a future, separately-
    authorized activation gate (confirming via a real deployed-container
    log) may."""
    assert prc.capture_local_sandbox_capability_for_testing().production_verification_status == (
        prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED
    )
    assert prc.capture_production_worker_capability().production_verification_status == (
        prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED
    )


def test_missing_decoder_fails_closed():
    fake_snapshot = smp.LocalFfmpegCapabilitySnapshot(
        ffmpeg_version="fake", ffprobe_version="fake",
        hevc_decoder_present=False, hevc_encoder_present=False,
        av1_decoder_present=False, libx264_present=True,
        runtime_capability_status=smp.RUNTIME_CAPABILITY_LOCAL_SANDBOX_ONLY,
    )
    cap = prc._snapshot_to_capability(
        fake_snapshot, capability_source=prc.CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_ESTABLISHED,
    )
    assert cap.hevc_decoder_available is False
    bridged = prc.bridge_to_runtime_capability_input(cap)
    assert bridged.hevc_decode_confirmed is False


def test_missing_encoder_recorded_distinctly_from_decoder():
    fake_snapshot = smp.LocalFfmpegCapabilitySnapshot(
        ffmpeg_version="fake", ffprobe_version="fake",
        hevc_decoder_present=True, hevc_encoder_present=False,
        av1_decoder_present=False, libx264_present=False,
        runtime_capability_status=smp.RUNTIME_CAPABILITY_LOCAL_SANDBOX_ONLY,
    )
    cap = prc._snapshot_to_capability(
        fake_snapshot, capability_source=prc.CAPABILITY_SOURCE_LOCAL_SANDBOX,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED,
    )
    assert cap.hevc_decoder_available is True
    assert cap.h264_encoder_available is False


def test_bridge_fails_closed_when_not_established():
    """Stage 6: even a capability object carrying hevc_decoder_available
    =True must NOT confirm to D-272's policy unless production_
    verification_status is genuinely ESTABLISHED -- the honest 'no live
    activation performed this gate' invariant, enforced in code, not left
    to caller discipline."""
    cap = prc.capture_local_sandbox_capability_for_testing()
    bridged = prc.bridge_to_runtime_capability_input(cap)
    assert bridged.hevc_decode_confirmed is False


def test_bridge_confirms_when_simulated_established():
    """Structural proof only (mirrors D-272B's own Stage 7 precedent): IF
    a future gate genuinely marks a capability ESTABLISHED, the bridge
    correctly threads that into D-272's own RuntimeCapabilityInput -- this
    test simulates that marking at the test level, never as a code change
    that would itself constitute activation."""
    cap = prc.capture_local_sandbox_capability_for_testing()
    established = dataclasses.replace(cap, production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_ESTABLISHED)
    bridged = prc.bridge_to_runtime_capability_input(established)
    assert bridged.hevc_decode_confirmed is True
    assert bridged.av1_decode_confirmed is False  # Stage 6: never over-asserted


def test_bridge_never_asserts_av1():
    cap = prc.ProductionRuntimeCapability(
        hevc_decoder_available=True, h264_encoder_available=True, ffmpeg_version="x",
        capability_source=prc.CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_ESTABLISHED,
    )
    bridged = prc.bridge_to_runtime_capability_input(cap)
    assert bridged.av1_decode_confirmed is False


# =============================================================================
# D-272 policy matrix -- confirmed / unknown / unavailable
# =============================================================================

def test_hevc_confirmed_yields_normalize_required(hevc_sdr_8bit_mp4):
    _, decision, _ = _build_plan(hevc_sdr_8bit_mp4, "src")
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_HEVC_TO_H264_NORMALIZATION_REQUIRED in decision.normalization_reasons


def test_hevc_unknown_yields_insufficient_evidence(hevc_sdr_8bit_mp4):
    unconfirmed = sfp.RuntimeCapabilityInput()
    _, decision, _ = _build_plan(hevc_sdr_8bit_mp4, "src", capability=unconfirmed)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE


def test_h264_source_unaffected_still_accepts():
    """H264-unchanged regression: this gate touches nothing about the
    existing, non-HEVC ACCEPT path."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/h264.mp4"
        _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", path])
        profile = smp.probe_source_media_profile(path)
        decision = sfp.evaluate_source_format_policy(profile, runtime_capability=_HEVC_CONFIRMED)
        assert decision.decision == sfp.DECISION_ACCEPT


# =============================================================================
# Plan bridge -- D-272 -> D-274A, no re-derivation
# =============================================================================

def test_plan_maps_confirmed_hevc_to_action(hevc_sdr_8bit_mp4):
    _, _, result = _build_plan(hevc_sdr_8bit_mp4, "src")
    assert result.plan.codec_action == snp.ACTION_HEVC_TO_H264
    assert result.plan.is_executable is True


# =============================================================================
# Executor support -- SDR 8-bit HEVC normalizes successfully
# =============================================================================

def test_hevc_mp4_normalizes_to_canonical_output(hevc_sdr_8bit_mp4, tmp_path):
    _, _, result = _build_plan(hevc_sdr_8bit_mp4, "src")
    r = _exec(hevc_sdr_8bit_mp4, result.plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    p = r.normalized_profile
    assert p.video_codec == smp.VIDEO_CODEC_H264
    assert p.pixel_format == "yuv420p"
    assert p.bit_depth == 8
    assert p.container_name == "MP4"


def test_hevc_mov_normalizes_to_canonical_mp4(hevc_sdr_mov, tmp_path):
    _, _, result = _build_plan(hevc_sdr_mov, "src")
    r = _exec(hevc_sdr_mov, result.plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert r.normalized_profile.container_name == "MP4"
    assert r.normalized_profile.video_codec == smp.VIDEO_CODEC_H264


def test_hevc_no_audio_normalizes_without_fabricating_audio(hevc_no_audio_mp4, tmp_path):
    _, _, result = _build_plan(hevc_no_audio_mp4, "src")
    r = _exec(hevc_no_audio_mp4, result.plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    streams = _ffprobe_streams(r.normalized_path)
    assert not any(s["codec_type"] == "audio" for s in streams)


def test_hevc_audio_preserved(hevc_sdr_8bit_mp4, tmp_path):
    _, _, result = _build_plan(hevc_sdr_8bit_mp4, "src")
    r = _exec(hevc_sdr_8bit_mp4, result.plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED
    streams = _ffprobe_streams(r.normalized_path)
    assert any(s["codec_type"] == "audio" for s in streams)


def test_hevc_reprobe_and_reevaluation_performed(hevc_sdr_8bit_mp4, tmp_path):
    _, _, result = _build_plan(hevc_sdr_8bit_mp4, "src")
    r = _exec(hevc_sdr_8bit_mp4, result.plan, output_directory=str(tmp_path))
    assert r.normalized_profile is not None
    assert r.diagnostics["final_d272_decision"] == sfp.DECISION_ACCEPT
    assert r.verification.verified is True


def test_hevc_original_preserved_and_hash(hevc_sdr_8bit_mp4, tmp_path):
    from cutsell_worker.render_delivery import compute_output_sha256
    before = compute_output_sha256(hevc_sdr_8bit_mp4)
    _, _, result = _build_plan(hevc_sdr_8bit_mp4, "src")
    r = _exec(hevc_sdr_8bit_mp4, result.plan, output_directory=str(tmp_path))
    after = compute_output_sha256(hevc_sdr_8bit_mp4)
    assert before == after
    assert r.normalized_reference.normalized_output_sha256 == compute_output_sha256(r.normalized_path)


# =============================================================================
# Composition: HEVC + rotation / VFR / timeline (Stage 15)
# =============================================================================

def test_hevc_plus_rotation_one_generation(hevc_asymmetric_mp4, tmp_path):
    _, _, result = _build_plan(hevc_asymmetric_mp4, "src")
    plan = dataclasses.replace(result.plan, rotation_action=snp.ACTION_ROTATE_90)
    r = _exec(hevc_asymmetric_mp4, plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    streams = _ffprobe_streams(r.normalized_path)
    video = next(s for s in streams if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (240, 320)
    frame_png = tmp_path / "frame.png"
    _ffmpeg(["-y", "-i", r.normalized_path, "-frames:v", "1", str(frame_png)])
    assert _marker_corner(frame_png) == "TR"


def test_hevc_plus_timeline_reset_one_generation(hevc_offset_mp4, tmp_path):
    profile = smp.probe_source_media_profile(hevc_offset_mp4)
    assert abs(profile.video_stream_start_time or 0.0) > 0.0, "fixture must carry a real offset"
    _, _, result = _build_plan(hevc_offset_mp4, "src")
    assert result.plan.timeline_action == snp.ACTION_TIMELINE_TO_ZERO
    r = _exec(hevc_offset_mp4, result.plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert abs(r.normalized_profile.video_stream_start_time or 0.0) < 0.01


def test_hevc_plus_vfr_one_generation(hevc_sdr_8bit_mp4, tmp_path):
    """Reuses D-274B's own genuine-VFR-fixture technique, applied to HEVC
    segments: concat-demuxer stream copy of two differently-clocked HEVC
    segments preserves real per-frame timing variability."""
    seg_a = tmp_path / "segA.mp4"
    seg_b = tmp_path / "segB.mp4"
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=24", "-t", "1",
             "-c:v", "libx265", "-x265-params", "log-level=none", "-pix_fmt", "yuv420p", str(seg_a)])
    _ffmpeg(["-y", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=60", "-t", "1",
             "-c:v", "libx265", "-x265-params", "log-level=none", "-pix_fmt", "yuv420p", str(seg_b)])
    list_txt = tmp_path / "list.txt"
    list_txt.write_text(f"file '{seg_a.name}'\nfile '{seg_b.name}'\n", encoding="utf-8")
    vfr_hevc = tmp_path / "vfr_hevc.mp4"
    _ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", str(list_txt), "-c", "copy", str(vfr_hevc)])
    profile = smp.probe_source_media_profile(str(vfr_hevc))
    assert profile.vfr_status == smp.VFR_STATUS_LIKELY_VFR, "fixture must be genuinely VFR"
    _, _, result = _build_plan(str(vfr_hevc), "src")
    assert result.plan.codec_action == snp.ACTION_HEVC_TO_H264
    assert result.plan.frame_rate_action == snp.ACTION_VFR_TO_CFR
    out_dir = tmp_path / "out"
    r = _exec(str(vfr_hevc), result.plan, output_directory=str(out_dir))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert r.normalized_profile.vfr_status == smp.VFR_STATUS_CFR


def test_hevc_multi_action_one_generation(hevc_asymmetric_mp4, tmp_path):
    _, _, result = _build_plan(hevc_asymmetric_mp4, "src")
    plan = dataclasses.replace(
        result.plan, rotation_action=snp.ACTION_ROTATE_180, timeline_action=snp.ACTION_TIMELINE_TO_ZERO,
    )
    r = _exec(hevc_asymmetric_mp4, plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert r.normalized_profile.video_codec == smp.VIDEO_CODEC_H264


# =============================================================================
# HDR / 10-bit firewall -- rejected BEFORE ffmpeg, zero silent conversion
# =============================================================================

def test_hevc_plus_hdr_rejected_before_ffmpeg_when_HDR_TONEMAP_ACTIONS_frozen_empty(hevc_sdr_8bit_mp4, tmp_path, monkeypatch):
    """D-274C's own original scope note here ("HDR is out of scope for
    this gate, deferred to a later gate") is now superseded: D-274D
    IMPLEMENTS HDR tonemap, so `_UNSUPPORTED_ACTIONS` is genuinely empty
    and a plan combining HEVC_TO_H264 + an HDR tonemap action is now a
    real, executable, ffmpeg-invoking combination (see
    `test_hevc_plus_hdr_composes_in_one_generation` below for the real
    positive proof, and test_cutsell_d274d_hdr_pixel_format_
    normalization.py for the dedicated HDR gate's own coverage).

    This test is kept, RE-PURPOSED, to prove the general firewall
    mechanism itself still works for a genuinely unsupported action even
    when combined with HEVC -- self-resolving guard, same pattern as
    D-272B's own precedent -- by injecting a SYNTHETIC future action
    rather than asserting a real HDR action is still unsupported (which
    would now be false)."""
    calls = []
    monkeypatch.setattr(exe.subprocess, "run", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(
        AssertionError("must not call ffmpeg for HEVC + a genuinely unsupported action")
    ))
    fake_future_action = "FUTURE_UNIMPLEMENTED_ACTION_D274D_TEST_ONLY"
    monkeypatch.setattr(exe, "_UNSUPPORTED_ACTIONS", frozenset({fake_future_action}))
    _, _, result = _build_plan(hevc_sdr_8bit_mp4, "src")
    plan = dataclasses.replace(result.plan, hdr_action=fake_future_action)
    r = _exec(hevc_sdr_8bit_mp4, plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_UNSUPPORTED
    assert r.failure.error_category == exe.FAILURE_UNSUPPORTED_ACTION
    assert calls == []


def test_hevc_plus_hdr_composes_in_one_generation(hevc_hdr_10bit_mp4, tmp_path):
    """D-274D positive proof (this file's own self-resolving guard
    companion to the rename above): HEVC_TO_H264 + an HDR tonemap action
    (+ TEN_BIT_TO_EIGHT_BIT, since this genuine fixture is also 10-bit)
    on the SAME plan executes in ONE ffmpeg generation and reaches D-272
    ACCEPT -- mirrors `test_hevc_plus_vfr_one_generation`'s own pattern.
    Uses `hevc_hdr_10bit_mp4` (parametrized PQ/HLG, module-scoped) -- a
    GENUINELY HDR-tagged HEVC source, not a plan/profile mismatch (see
    that fixture's own docstring for why a forced mismatch fails inside
    ffmpeg itself, never inside this executor)."""
    profile = smp.probe_source_media_profile(hevc_hdr_10bit_mp4)
    assert profile.hdr_status in (smp.HDR_STATUS_HDR_PQ, smp.HDR_STATUS_HDR_HLG)
    assert profile.bit_depth == 10
    decision = sfp.evaluate_source_format_policy(profile, runtime_capability=_HEVC_CONFIRMED)
    plan_result = snp.build_source_normalization_plan(
        "src", profile, decision, runtime_capability=_HEVC_CONFIRMED, tonemap_available=True,
    )
    plan = plan_result.plan
    assert plan is not None and plan.is_executable
    assert plan.codec_action == snp.ACTION_HEVC_TO_H264
    assert plan.hdr_action in (snp.ACTION_HDR_PQ_TO_SDR_BT709, snp.ACTION_HDR_HLG_TO_SDR_BT709)
    assert plan.bit_depth_action == snp.ACTION_TEN_BIT_TO_EIGHT_BIT
    r = _exec(hevc_hdr_10bit_mp4, plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert r.normalized_profile.video_codec == smp.VIDEO_CODEC_H264
    assert r.normalized_profile.hdr_status == smp.HDR_STATUS_SDR
    assert r.normalized_profile.bit_depth == 8
    assert r.diagnostics["final_d272_decision"] == sfp.DECISION_ACCEPT


def test_hevc_dolby_vision_never_produces_executable_plan(hevc_sdr_8bit_mp4):
    """Dolby Vision is D-274A's own established `unsupported=True` branch
    (never a supported action at all, per D-274A Stage 7/14) -- confirmed
    still true post-D-274C: a plan carrying Dolby Vision reasoning is
    UNSUPPORTED at the plan-build level already, before the executor is
    even reached."""
    profile = smp.probe_source_media_profile(hevc_sdr_8bit_mp4)
    dv_profile = dataclasses.replace(profile, hdr_status=smp.HDR_STATUS_HDR_DOLBY_VISION)
    decision = sfp.evaluate_source_format_policy(dv_profile, runtime_capability=_HEVC_CONFIRMED)
    if decision.decision == sfp.DECISION_NORMALIZE_REQUIRED:
        result = snp.build_source_normalization_plan("src", dv_profile, decision, runtime_capability=_HEVC_CONFIRMED)
        assert result.outcome == snp.NORMALIZATION_UNSUPPORTED


def test_hevc_10bit_sdr_composes_in_one_generation(hevc_10bit_mp4, tmp_path):
    """D-274D self-resolving guard + positive proof: `ACTION_TEN_BIT_TO_
    EIGHT_BIT` is now implemented (D-274D's own empirical finding: the
    executor's pre-existing fixed `-pix_fmt yuv420p` output flag already
    downconverts 10-bit -> 8-bit with ZERO new filter code), so a genuine
    HEVC+10-bit SDR source now composes both actions in ONE ffmpeg
    generation and reaches D-272 ACCEPT -- this test used to assert the
    opposite (rejected before ffmpeg, back when 10-bit was still
    unsupported); renamed and re-purposed rather than deleted, per this
    codebase's own self-resolving-guard discipline."""
    profile = smp.probe_source_media_profile(hevc_10bit_mp4)
    assert profile.bit_depth == 10, "fixture must genuinely be 10-bit"
    _, _, result = _build_plan(hevc_10bit_mp4, "src")
    assert result.plan.codec_action == snp.ACTION_HEVC_TO_H264
    assert result.plan.bit_depth_action == snp.ACTION_TEN_BIT_TO_EIGHT_BIT
    r = _exec(hevc_10bit_mp4, result.plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert r.normalized_profile.video_codec == smp.VIDEO_CODEC_H264
    assert r.normalized_profile.bit_depth == 8
    assert r.diagnostics["final_d272_decision"] == sfp.DECISION_ACCEPT


# =============================================================================
# Codec-availability pre-check (Stage 4)
# =============================================================================

def test_h264_encoder_unavailable_rejected_before_ffmpeg(hevc_sdr_8bit_mp4, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(exe.subprocess, "run", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(
        AssertionError("must not call ffmpeg when h264 encoder unavailable")
    ))
    _, _, result = _build_plan(hevc_sdr_8bit_mp4, "src")
    unavailable = prc.ProductionRuntimeCapability(
        hevc_decoder_available=True, h264_encoder_available=False, ffmpeg_version="x",
        capability_source=prc.CAPABILITY_SOURCE_LOCAL_SANDBOX,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED,
    )
    r = _exec(hevc_sdr_8bit_mp4, result.plan, output_directory=str(tmp_path), codec_capability=unavailable)
    assert r.outcome == snp.NORMALIZATION_FAILED
    assert r.failure.error_category == exe.FAILURE_CODEC_UNAVAILABLE
    assert calls == []


def test_codec_capability_omitted_preserves_d274b_behavior(hevc_sdr_8bit_mp4, tmp_path):
    """No `codec_capability` argument at all -- every pre-D-274C call
    site/test -- must behave byte-identically to before this gate."""
    _, _, result = _build_plan(hevc_sdr_8bit_mp4, "src")
    r = _exec(hevc_sdr_8bit_mp4, result.plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED


# =============================================================================
# No live activation / timeout seam unchanged
# =============================================================================

def test_timeout_seam_still_unset_by_default():
    assert exe.NORMALIZATION_FFMPEG_TIMEOUT_SEC is None


def test_no_live_activation_worker_job_untouched():
    """D-274C's own scope: worker_job.py is inspected, never modified, by
    THIS gate -- confirmed via source-scan for any new normalization-
    executor reference (D-274C-A, a later separately-authorized gate,
    legitimately wires a capability-bridge reference into worker_job.py;
    that gate's own test file re-asserts what IS authorized there -- this
    assertion only ever concerned D-274C's own scope, never normalization
    execution)."""
    import inspect
    from cutsell_worker import worker_job
    source = inspect.getsource(worker_job)
    assert "source_normalization_executor" not in source
    assert "execute_source_normalization" not in source


def test_rq_worker_and_entrypoint_untouched():
    """Stage 2's own seam-identification, not seam-activation: rq_worker.py
    (the real worker-process startup entrypoint this gate identified) and
    entrypoint.sh must carry zero reference to this gate's new capability
    module -- activation is a separate, future gate."""
    root = Path(__file__).resolve().parents[1]
    rq_worker_source = (root / "rq_worker.py").read_text(encoding="utf-8")
    entrypoint_source = (root / "entrypoint.sh").read_text(encoding="utf-8")
    assert "production_runtime_capability" not in rq_worker_source
    assert "production_runtime_capability" not in entrypoint_source


# =============================================================================
# Closed-track firewall -- every other authority byte-for-byte unchanged
# =============================================================================

@pytest.mark.parametrize("relative_path", [
    "cutsell_worker/render.py",
    # D-274C-A (a later, separately-authorized gate) legitimately wires
    # worker_job.py's own evaluate_source_format_gate to a real capability
    # bridge -- self-resolving guard, same pattern as D-272B's own
    # precedent; worker_job.py entry removed here since this specific
    # assertion (byte-for-byte unchanged since D-274B) is no longer true
    # by design, not by regression.
    "cutsell_worker/source_format_policy.py",
    "cutsell_worker/source_media_profile.py",
    "cutsell_worker/render_delivery.py",
])
def test_closed_track_files_unmodified_by_this_gate(relative_path):
    """D-274C's own scope is additive-only: a NEW module
    (`production_runtime_capability.py`) plus a narrow, additive change
    inside `source_normalization_executor.py` (removing `ACTION_HEVC_TO_
    H264` from its own unsupported-action set, adding the optional
    `codec_capability` pre-check). No other production module's own
    content changes as part of this gate."""
    # Compares the CURRENT working tree (uncommitted changes included)
    # against D-274B's own HEAD -- never assumes this gate's own work is
    # already committed when the test runs.
    result = subprocess.run(
        ["git", "diff", "--stat", "e8f71c6", "--", relative_path],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.stdout.strip() == "", f"{relative_path} was modified by D-274C: {result.stdout}"
