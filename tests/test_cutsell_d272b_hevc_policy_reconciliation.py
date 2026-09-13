"""D-272B -- HEVC CANONICAL-NORMALIZATION POLICY RECONCILIATION.

Post D-274A. A narrow, Product-Owner-authorized reconciliation of D-272
with D-273/D-274A: `evaluate_source_format_policy` now emits
`REASON_HEVC_TO_H264_NORMALIZATION_REQUIRED` whenever a capability-
confirmed HEVC source needs normalizing, even when otherwise clean --
HEVC is never ACCEPT-native in the canonical V1 source contract. This
is NOT a general D-272 reopening: every other property (H.264,
rotation, HDR, VFR, 10-bit, pixel format, missing audio/video, multi-
stream, container) is unchanged and reproven by the existing D-272 test
suite (`test_cutsell_d272_source_format_policy.py`) plus the targeted
regressions below.

No normalization execution, no ffmpeg transcode, no rotation execution,
no VFR execution, no HDR tonemap, no renderer change, no RAW, no
provider anywhere in this file.
"""
from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutsell_worker import source_format_policy as sfp
from cutsell_worker import source_media_profile as smp
from cutsell_worker import source_normalization_plan as snp
from cutsell_worker import worker_job

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


def _base_profile(**overrides) -> smp.SourceMediaProfile:
    fields = dict(
        path="unused", probe_status=smp.PROBE_STATUS_COMPLETE,
        container_name=smp.CONTAINER_MP4, raw_format_name="mov,mp4", duration_sec=5.0, file_size_bytes=1000,
        video_presence=smp.VIDEO_PRESENT, audio_presence=smp.AUDIO_PRESENT,
        video_stream_count=1, audio_stream_count=1,
        video_codec=smp.VIDEO_CODEC_H264, raw_video_codec="h264", video_profile=None,
        pixel_format="yuv420p", bit_depth=8,
        coded_width=1920, coded_height=1080, display_width=1920, display_height=1080,
        rotation_degrees=0, rotation_source=smp.ROTATION_SOURCE_NONE,
        avg_frame_rate=30.0, r_frame_rate=30.0, effective_fps=30.0, vfr_status=smp.VFR_STATUS_CFR,
        color_primaries="bt709", color_transfer="bt709", color_space="bt709", color_range="tv",
        hdr_status=smp.HDR_STATUS_SDR,
        audio_codec="aac", raw_audio_codec="aac", audio_sample_rate_hz=48000, audio_channels=2,
        audio_channel_layout="stereo",
        format_start_time=0.0, video_stream_start_time=0.0, audio_stream_start_time=0.0,
        video_time_base="1/30000", audio_time_base="1/48000",
    )
    fields.update(overrides)
    return smp.SourceMediaProfile(**fields)


_HEVC_CAP = sfp.RuntimeCapabilityInput(hevc_decode_confirmed=True)


# ---------------------------------------------------------------------------
# Stage 10 test matrix, items 1-13
# ---------------------------------------------------------------------------

def test_1_h264_sdr_confirmed_accept():
    profile = _base_profile()
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_ACCEPT


def test_2_hevc_sdr_capability_confirmed_normalize_required():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc")
    decision = sfp.evaluate_source_format_policy(profile, runtime_capability=_HEVC_CAP)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_HEVC_TO_H264_NORMALIZATION_REQUIRED in decision.normalization_reasons
    assert decision.blocking_reasons == ()


def test_3_hevc_capability_unknown_insufficient_evidence():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc")
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.REASON_HEVC_RUNTIME_UNVERIFIED in decision.blocking_reasons


def test_4_hevc_missing_video_reject():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc",
                             video_presence=smp.VIDEO_MISSING)
    decision = sfp.evaluate_source_format_policy(profile, runtime_capability=_HEVC_CAP)
    assert decision.decision == sfp.DECISION_REJECT
    assert sfp.REASON_MISSING_VIDEO in decision.blocking_reasons


def test_4b_hevc_multi_stream_insufficient_evidence():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc",
                             video_stream_count=2)
    decision = sfp.evaluate_source_format_policy(profile, runtime_capability=_HEVC_CAP)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.REASON_MULTIPLE_VIDEO_STREAMS in decision.blocking_reasons


def test_5_hevc_plus_rotation_normalize_required():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc",
                             rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
                             display_width=1080, display_height=1920)
    decision = sfp.evaluate_source_format_policy(profile, runtime_capability=_HEVC_CAP)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_HEVC_TO_H264_NORMALIZATION_REQUIRED in decision.normalization_reasons
    assert sfp.REASON_ROTATION_NORMALIZATION_REQUIRED in decision.normalization_reasons


def test_6_hevc_plus_vfr_normalize_required():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc",
                             vfr_status=smp.VFR_STATUS_LIKELY_VFR, effective_fps=24.0)
    decision = sfp.evaluate_source_format_policy(profile, runtime_capability=_HEVC_CAP)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_HEVC_TO_H264_NORMALIZATION_REQUIRED in decision.normalization_reasons
    assert sfp.REASON_VFR_NORMALIZATION_REQUIRED in decision.normalization_reasons


def test_7_hevc_plus_ten_bit_normalize_required():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc",
                             bit_depth=10, pixel_format="yuv420p10le")
    decision = sfp.evaluate_source_format_policy(profile, runtime_capability=_HEVC_CAP)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_TEN_BIT_NORMALIZATION_REQUIRED in decision.normalization_reasons


def test_8_hevc_plus_yuv422_normalize_required():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc",
                             pixel_format="yuv422p")
    decision = sfp.evaluate_source_format_policy(profile, runtime_capability=_HEVC_CAP)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_PIXEL_FORMAT_NORMALIZATION_REQUIRED in decision.normalization_reasons


def test_9_hevc_plus_hdr_tonemap_confirmed_normalize_required_executable():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc",
                             hdr_status=smp.HDR_STATUS_HDR_PQ, color_transfer="smpte2084")
    decision = sfp.evaluate_source_format_policy(profile, runtime_capability=_HEVC_CAP)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    result = snp.build_source_normalization_plan("s1", profile, decision, runtime_capability=_HEVC_CAP,
                                                   tonemap_available=True)
    assert result.plan.codec_action == snp.ACTION_HEVC_TO_H264
    assert result.plan.hdr_action == snp.ACTION_HDR_PQ_TO_SDR_BT709
    assert result.plan.executability == snp.EXECUTABILITY_EXECUTABLE


def test_10_hevc_plus_hdr_capability_unknown_non_executable():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc",
                             hdr_status=smp.HDR_STATUS_HDR_PQ, color_transfer="smpte2084")
    decision = sfp.evaluate_source_format_policy(profile, runtime_capability=_HEVC_CAP)
    result = snp.build_source_normalization_plan("s1", profile, decision, runtime_capability=_HEVC_CAP)
    assert result.plan.executability == snp.EXECUTABILITY_CAPABILITY_UNVERIFIED
    assert "tonemap_available" in result.blocking_capability_gaps


def test_12_hevc_confirmed_user_facing_error_is_requires_normalization():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc")
    decision = sfp.evaluate_source_format_policy(profile, runtime_capability=_HEVC_CAP)
    assert decision.user_facing_error_code == sfp.USER_FACING_VIDEO_REQUIRES_NORMALIZATION


def test_13_hevc_unknown_capability_user_facing_error_unverified():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc")
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.user_facing_error_code == sfp.USER_FACING_RUNTIME_CODEC_SUPPORT_UNVERIFIED


def test_14_d274a_plan_contains_hevc_to_h264():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc")
    decision = sfp.evaluate_source_format_policy(profile, runtime_capability=_HEVC_CAP)
    result = snp.build_source_normalization_plan("s1", profile, decision, runtime_capability=_HEVC_CAP)
    assert result.plan.codec_action == snp.ACTION_HEVC_TO_H264


def test_15_multi_action_plan_identity_deterministic():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc",
                             rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
                             display_width=1080, display_height=1920)
    decision = sfp.evaluate_source_format_policy(profile, runtime_capability=_HEVC_CAP)
    r1 = snp.build_source_normalization_plan("s1", profile, decision, runtime_capability=_HEVC_CAP)
    r2 = snp.build_source_normalization_plan("s1", profile, decision, runtime_capability=_HEVC_CAP)
    assert r1.plan.plan_identity == r2.plan.plan_identity


def test_16_changed_hevc_action_changes_plan_identity():
    profile_hevc = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc")
    profile_h264 = _base_profile(video_codec=smp.VIDEO_CODEC_H264, raw_video_codec="h264",
                                  rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
                                  display_width=1080, display_height=1920)
    profile_hevc_rot = _base_profile(video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc",
                                      rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
                                      display_width=1080, display_height=1920)
    d_hevc = sfp.evaluate_source_format_policy(profile_hevc, runtime_capability=_HEVC_CAP)
    d_h264 = sfp.evaluate_source_format_policy(profile_h264)
    d_hevc_rot = sfp.evaluate_source_format_policy(profile_hevc_rot, runtime_capability=_HEVC_CAP)
    r_hevc = snp.build_source_normalization_plan("s1", profile_hevc, d_hevc, runtime_capability=_HEVC_CAP)
    r_h264 = snp.build_source_normalization_plan("s1", profile_h264, d_h264)
    r_hevc_rot = snp.build_source_normalization_plan("s1", profile_hevc_rot, d_hevc_rot, runtime_capability=_HEVC_CAP)
    assert r_hevc.plan.plan_identity != r_h264.plan.plan_identity
    assert r_hevc.plan.plan_identity != r_hevc_rot.plan.plan_identity


def test_17_no_normalization_execution_source_scan():
    for path in ("cutsell_worker/source_format_policy.py", "cutsell_worker/source_normalization_plan.py"):
        source = _source_without_docstrings(path)
        for banned in ("subprocess.run(", "subprocess.Popen(", "import subprocess"):
            assert banned not in source, f"{path} must not execute anything"


def test_18_no_ffmpeg_command_source_scan():
    for path in ("cutsell_worker/source_format_policy.py", "cutsell_worker/source_normalization_plan.py"):
        source = _source_without_docstrings(path)
        for banned in ("ffmpeg", "-vf", '"-c:v"', "scale=", "pad=", "hflip", "vflip"):
            assert banned not in source, f"{path} must not build an ffmpeg command"


def test_19_renderer_unchanged():
    assert _run_git_diff("cutsell_worker/render.py") == ""
    from cutsell_worker import render as render_module
    assert render_module.RENDER_FFMPEG_TIMEOUT_SEC == 1200.0
    assert render_module.RENDER_FPS_DEFAULT == 30


# ---------------------------------------------------------------------------
# Stage 7 -- live early gate: worker_job.py was NOT touched by D-272B, but
# its behavior for a source that DOES reach capability-confirmed HEVC now
# structurally blocks as NORMALIZE_REQUIRED (proven with a simulated
# capability signal -- worker_job.py itself never confirms HEVC capability
# today, an honest, separate, future activation, matching D-272A's own
# never-invent-a-production-capability discipline).
# ---------------------------------------------------------------------------

class _FakeBrain:
    backend = "local"
    external_calls_enabled = False
    hybrid_settings = SimpleNamespace(provider="none", primary_model="none")
    semantic_provider = None
    whole_video_provider = None
    visual_provider = None
    take_grouping_provider = None
    take_judge_provider = None
    clean_cut_provider = None
    composer_provider = None
    draft_review_provider = None
    editorial_judge = None


@pytest.fixture
def hevc_mp4(tmp_path_factory):
    path = tmp_path_factory.mktemp("d272b") / "hevc.mp4"
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "color=c=blue:s=320x240:d=1:r=30", "-c:v", "libx265", "-pix_fmt", "yuv420p",
         "-an", "-tag:v", "hvc1", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.skip("HEVC encoder not available on this runner")
    return str(path)


@pytest.fixture
def wired_worker_job(monkeypatch):
    calls = {"process_local_sources": 0}

    def fake_download_source(uri, destination, *, client=None):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(uri, destination)
        return destination

    def fake_process_local_sources(request, local_paths, **kwargs):
        calls["process_local_sources"] += 1
        return object()

    monkeypatch.setattr(worker_job, "validate_product_source_uri", lambda uri, **kw: None)
    monkeypatch.setattr(worker_job, "download_source", fake_download_source)
    monkeypatch.setattr(worker_job, "process_local_sources", fake_process_local_sources)
    monkeypatch.setattr(worker_job, "result_to_dict", lambda result: {"draft": {"selected": []}})
    monkeypatch.setattr(worker_job, "load_runtime_config", lambda: SimpleNamespace(asr_model="tiny"))
    monkeypatch.setattr(worker_job, "build_brain_runtime", lambda config: _FakeBrain())
    monkeypatch.setattr(worker_job, "create_initial_draft", lambda **kw: None)
    monkeypatch.setattr(worker_job, "safe_update_project", lambda **kw: {"status": "saved", "project_state": kw.get("state")})
    monkeypatch.setattr(worker_job, "publish_notification", lambda **kw: {"notification_id": "n1"})
    monkeypatch.setattr(worker_job, "generate_filmstrip", lambda *a, **kw: [])
    monkeypatch.setattr(worker_job, "waveform_peaks", lambda *a, **kw: [0.0])
    monkeypatch.setattr(worker_job, "store_timeline_assets", lambda **kw: {"status": "ok"})
    monkeypatch.setattr(worker_job, "record_processing_minutes", lambda **kw: None)
    monkeypatch.setattr(worker_job, "release_processing_slot", lambda **kw: None)
    return calls


def _payload(uri: str) -> dict:
    return {
        "project_id": "p1", "user_id": "u1",
        "sources": [{"source_asset_id": "s1", "uri": uri, "original_name": Path(uri).name}],
    }


@pytestmark_ffmpeg
def test_11a_hevc_live_gate_today_unconfirmed_insufficient_evidence(wired_worker_job, hevc_mp4):
    """worker_job.py's own evaluate_source_format_gate never passes a
    runtime_capability today (confirmed: zero occurrences in the module's
    source) -- so a real HEVC source live is INSUFFICIENT_EVIDENCE
    exactly as before D-272B. D-272B's new NORMALIZE_REQUIRED path only
    becomes reachable once a real production capability source is wired
    into the live gate (a separate, future activation)."""
    assert "runtime_capability" not in Path("cutsell_worker/worker_job.py").read_text(encoding="utf-8")
    with pytest.raises(worker_job.SourceFormatGateBlocked) as excinfo:
        worker_job.run_flow_b_job(_payload(hevc_mp4))
    assert wired_worker_job["process_local_sources"] == 0
    assert excinfo.value.blocked_sources[0]["decision"] == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert excinfo.value.primary_error_code == sfp.USER_FACING_RUNTIME_CODEC_SUPPORT_UNVERIFIED


@pytestmark_ffmpeg
def test_11b_hevc_live_gate_structural_proof_with_confirmed_capability(monkeypatch, wired_worker_job, hevc_mp4):
    """Stage 7's own structural claim: IF a real capability source were
    wired into the live gate (simulated here, not today's default),
    confirmed HEVC blocks as NORMALIZE_REQUIRED before process_local_
    sources, never bypassing to ACCEPT."""
    real_policy = worker_job.evaluate_source_format_policy

    def _confirmed(profile, **kwargs):
        kwargs["runtime_capability"] = sfp.RuntimeCapabilityInput(hevc_decode_confirmed=True)
        return real_policy(profile, **kwargs)

    monkeypatch.setattr(worker_job, "evaluate_source_format_policy", _confirmed)
    with pytest.raises(worker_job.SourceFormatGateBlocked) as excinfo:
        worker_job.run_flow_b_job(_payload(hevc_mp4))
    assert wired_worker_job["process_local_sources"] == 0
    assert excinfo.value.blocked_sources[0]["decision"] == sfp.DECISION_NORMALIZE_REQUIRED
    assert excinfo.value.primary_error_code == sfp.USER_FACING_VIDEO_REQUIRES_NORMALIZATION


# ---------------------------------------------------------------------------
# Stage 9 -- no broad policy reopen (H.264/rotation/HDR/VFR/10-bit/
# yuv422/yuv444/missing audio/missing video/multi-stream/container all
# unchanged except where HEVC composition legitimately combines them)
# ---------------------------------------------------------------------------

def test_h264_rotation_unchanged():
    profile = _base_profile(rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
                             display_width=1080, display_height=1920)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    assert decision.normalization_reasons == (sfp.REASON_ROTATION_NORMALIZATION_REQUIRED,)


def test_h264_hdr_unchanged():
    profile = _base_profile(hdr_status=smp.HDR_STATUS_HDR_PQ, color_transfer="smpte2084")
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.normalization_reasons == (sfp.REASON_HDR_NORMALIZATION_REQUIRED,)


def test_h264_missing_audio_unchanged():
    profile = _base_profile(audio_presence=smp.AUDIO_MISSING, audio_stream_count=0, audio_codec=None,
                             raw_audio_codec=None, audio_sample_rate_hz=None, audio_channels=None,
                             audio_channel_layout=None)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_ACCEPT
    assert decision.warnings == (sfp.REASON_AUDIO_MISSING,)


def test_h264_missing_video_unchanged():
    profile = _base_profile(video_presence=smp.VIDEO_MISSING)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_REJECT
    assert decision.blocking_reasons == (sfp.REASON_MISSING_VIDEO,)


def test_h264_multi_stream_unchanged():
    profile = _base_profile(audio_stream_count=2)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert decision.blocking_reasons == (sfp.REASON_MULTIPLE_AUDIO_STREAMS,)


def test_container_policy_unchanged():
    profile = _base_profile(container_name=smp.CONTAINER_MKV)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_REJECT
    assert decision.blocking_reasons == (sfp.REASON_UNSUPPORTED_CONTAINER,)


def test_av1_capability_gate_unchanged():
    profile = _base_profile(video_codec=smp.VIDEO_CODEC_AV1, raw_video_codec="av1")
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.REASON_CODEC_RUNTIME_UNVERIFIED in decision.blocking_reasons
    # AV1 confirmed does NOT trigger the new HEVC-specific reason -- this
    # gate's own scope is HEVC only.
    confirmed = sfp.evaluate_source_format_policy(
        profile, runtime_capability=sfp.RuntimeCapabilityInput(av1_decode_confirmed=True),
    )
    assert sfp.REASON_HEVC_TO_H264_NORMALIZATION_REQUIRED not in confirmed.normalization_reasons


# ---------------------------------------------------------------------------
# Closed-track firewall (Stage 11)
# ---------------------------------------------------------------------------

_FIREWALL_FILES = [
    "cutsell_worker/render.py",
    "cutsell_worker/render_delivery.py",
    "cutsell_worker/render_plan.py",
    "cutsell_worker/media_probe.py",
    "cutsell_worker/source_media_profile.py",
    "cutsell_worker/post_render_media_qc.py",
    "cutsell_worker/visual_finishing_measurement.py",
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
    "cutsell_worker/worker_job.py",
    "cutsell_worker/flow_b.py",
    "gpu_execution_provider.py",
]


@pytest.mark.parametrize("rel_path", _FIREWALL_FILES)
def test_unrelated_authorities_unchanged(rel_path):
    if not Path(rel_path).exists():
        pytest.skip(f"{rel_path} not present in this checkout")
    assert _run_git_diff(rel_path) == "", f"D-272B must not touch {rel_path}"


def test_no_secrets_in_source():
    for path in ("cutsell_worker/source_format_policy.py", "cutsell_worker/source_normalization_plan.py"):
        source = Path(path).read_text(encoding="utf-8")
        for banned in ("AKIA", "aws_secret", "BEGIN PRIVATE KEY", "api_key="):
            assert banned not in source


def test_compileall_clean():
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "py_compile",
         "cutsell_worker/source_format_policy.py", "cutsell_worker/source_normalization_plan.py"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
