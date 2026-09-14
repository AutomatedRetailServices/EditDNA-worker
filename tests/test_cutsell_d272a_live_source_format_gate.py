"""D-272A -- LIVE EARLY SOURCE FORMAT GATE ACTIVATION.

Post D-272 (source format policy foundation, Verdict A, seam proven but not
activated). Product Owner authorized live conservative gating: this gate
wires D-271's `probe_source_media_profile` + D-272's
`evaluate_source_format_policy` into the real
`cutsell_worker.worker_job.run_flow_b_job` call site, at the smallest safe
point -- after the existing per-source download/probe loop, before
`process_local_sources` (the ASR/GPU/semantic-reasoning entry point).

No transcode, no normalization implementation, no HDR tonemap, no rotation
pixel transform, no fps conversion, no codec conversion, no filtergraph
change, no RAW, no provider anywhere in this file. D-272 remains the sole
policy authority -- this file proves `worker_job.py` calls
`source_media_profile.probe_source_media_profile` and
`source_format_policy.evaluate_source_format_policy` directly rather than
reimplementing any codec/HDR/rotation/VFR/stream rule.
"""
from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutsell_worker import source_format_policy as sfp
from cutsell_worker import source_media_profile as smp
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


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def h264_mp4(tmp_path_factory):
    path = tmp_path_factory.mktemp("d272a") / "h264.mp4"
    _ffmpeg([
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=1:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    ])
    return str(path)


@pytest.fixture(scope="module")
def h264_mp4_no_audio(tmp_path_factory):
    path = tmp_path_factory.mktemp("d272a") / "h264_noaudio.mp4"
    _ffmpeg([
        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=1:r=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path),
    ])
    return str(path)


@pytest.fixture(scope="module")
def no_video_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("d272a") / "audio_only.wav"
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(path)])
    return str(path)


@pytest.fixture(scope="module")
def corrupt_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("d272a") / "corrupt.mp4"
    path.write_bytes(b"not a real video file")
    return str(path)


@pytest.fixture(scope="module")
def mkv_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("d272a") / "clip.mkv"
    _ffmpeg([
        "-f", "lavfi", "-i", "color=c=green:s=320x240:d=1:r=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path),
    ])
    return str(path)


def _rotated_profile(rotation_degrees: int = 90) -> smp.SourceMediaProfile:
    """Stage 15: a parser-controlled profile proving policy/live-gate
    integration -- not real-phone rotation decode proof (D-271 already
    established real rotation-tag injection is unavailable in this
    ffmpeg build)."""
    return smp.SourceMediaProfile(
        path="unused", probe_status=smp.PROBE_STATUS_COMPLETE,
        container_name=smp.CONTAINER_MP4, raw_format_name="mov,mp4", duration_sec=5.0, file_size_bytes=1000,
        video_presence=smp.VIDEO_PRESENT, audio_presence=smp.AUDIO_PRESENT,
        video_stream_count=1, audio_stream_count=1,
        video_codec=smp.VIDEO_CODEC_H264, raw_video_codec="h264", video_profile=None,
        pixel_format="yuv420p", bit_depth=8,
        coded_width=1920, coded_height=1080, display_width=1080, display_height=1920,
        rotation_degrees=rotation_degrees, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
        avg_frame_rate=30.0, r_frame_rate=30.0, effective_fps=30.0, vfr_status=smp.VFR_STATUS_CFR,
        color_primaries="bt709", color_transfer="bt709", color_space="bt709", color_range="tv",
        hdr_status=smp.HDR_STATUS_SDR,
        audio_codec="aac", raw_audio_codec="aac", audio_sample_rate_hz=48000, audio_channels=2,
        audio_channel_layout="stereo",
        format_start_time=0.0, video_stream_start_time=0.0, audio_stream_start_time=0.0,
        video_time_base="1/30000", audio_time_base="1/48000",
    )


def _hdr_profile(hdr_status: str) -> smp.SourceMediaProfile:
    base = _rotated_profile(rotation_degrees=0)
    from dataclasses import replace
    return replace(base, hdr_status=hdr_status, rotation_source=smp.ROTATION_SOURCE_NONE,
                    color_transfer="smpte2084" if hdr_status == smp.HDR_STATUS_HDR_PQ else "arib-std-b67")


def _vfr_profile() -> smp.SourceMediaProfile:
    from dataclasses import replace
    base = _rotated_profile(rotation_degrees=0)
    return replace(base, vfr_status=smp.VFR_STATUS_LIKELY_VFR, rotation_source=smp.ROTATION_SOURCE_NONE)


def _ten_bit_profile() -> smp.SourceMediaProfile:
    from dataclasses import replace
    base = _rotated_profile(rotation_degrees=0)
    return replace(base, bit_depth=10, pixel_format="yuv420p10le", rotation_source=smp.ROTATION_SOURCE_NONE)


# ---------------------------------------------------------------------------
# Stage 28/29 -- evaluate_source_format_gate() unit-level decision matrix
# ---------------------------------------------------------------------------

@pytestmark_ffmpeg
def test_h264_sdr_gate_accepts(h264_mp4):
    [diag] = worker_job.evaluate_source_format_gate({"s1": h264_mp4})
    assert diag["decision"] == sfp.DECISION_ACCEPT
    assert diag["source_asset_id"] == "s1"
    assert "path" not in diag
    assert h264_mp4 not in str(diag)


@pytestmark_ffmpeg
def test_h264_mp4_no_audio_gate_accepts(h264_mp4_no_audio):
    [diag] = worker_job.evaluate_source_format_gate({"s1": h264_mp4_no_audio})
    assert diag["decision"] == sfp.DECISION_ACCEPT
    assert sfp.REASON_AUDIO_MISSING in diag["warnings"]


@pytestmark_ffmpeg
def test_missing_video_gate_rejects(no_video_source):
    [diag] = worker_job.evaluate_source_format_gate({"s1": no_video_source})
    assert diag["decision"] == sfp.DECISION_REJECT
    assert sfp.REASON_MISSING_VIDEO in diag["blocking_reasons"]


@pytestmark_ffmpeg
def test_corrupt_source_gate_rejects(corrupt_source):
    [diag] = worker_job.evaluate_source_format_gate({"s1": corrupt_source})
    assert diag["decision"] == sfp.DECISION_REJECT
    assert diag["source_profile_status"] == smp.PROBE_STATUS_FAILED


@pytestmark_ffmpeg
def test_mkv_container_gate_rejects(mkv_source):
    [diag] = worker_job.evaluate_source_format_gate({"s1": mkv_source})
    assert diag["decision"] == sfp.DECISION_REJECT
    assert sfp.REASON_UNSUPPORTED_CONTAINER in diag["blocking_reasons"]


def test_rotation_gate_normalize_required(monkeypatch):
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: _rotated_profile(90))
    [diag] = worker_job.evaluate_source_format_gate({"s1": "x.mp4"})
    assert diag["decision"] == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_ROTATION_NORMALIZATION_REQUIRED in diag["normalization_reasons"]
    assert diag["user_facing_error_code"] == sfp.USER_FACING_VIDEO_REQUIRES_NORMALIZATION
    assert diag["rotation_degrees"] == 90


def test_pq_hdr_gate_normalize_required(monkeypatch):
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: _hdr_profile(smp.HDR_STATUS_HDR_PQ))
    [diag] = worker_job.evaluate_source_format_gate({"s1": "x.mp4"})
    assert diag["decision"] == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_HDR_NORMALIZATION_REQUIRED in diag["normalization_reasons"]
    assert diag["hdr_status"] == smp.HDR_STATUS_HDR_PQ


def test_hlg_gate_normalize_required(monkeypatch):
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: _hdr_profile(smp.HDR_STATUS_HDR_HLG))
    [diag] = worker_job.evaluate_source_format_gate({"s1": "x.mp4"})
    assert diag["decision"] == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_HDR_NORMALIZATION_REQUIRED in diag["normalization_reasons"]


def test_vfr_gate_normalize_required(monkeypatch):
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: _vfr_profile())
    [diag] = worker_job.evaluate_source_format_gate({"s1": "x.mp4"})
    assert diag["decision"] == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_VFR_NORMALIZATION_REQUIRED in diag["normalization_reasons"]


def test_ten_bit_gate_normalize_required(monkeypatch):
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: _ten_bit_profile())
    [diag] = worker_job.evaluate_source_format_gate({"s1": "x.mp4"})
    assert diag["decision"] == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_TEN_BIT_NORMALIZATION_REQUIRED in diag["normalization_reasons"]


def test_multi_video_stream_gate_insufficient_evidence(monkeypatch):
    from dataclasses import replace
    profile = replace(_rotated_profile(0), video_stream_count=2, rotation_source=smp.ROTATION_SOURCE_NONE)
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: profile)
    [diag] = worker_job.evaluate_source_format_gate({"s1": "x.mp4"})
    assert diag["decision"] == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.REASON_MULTIPLE_VIDEO_STREAMS in diag["blocking_reasons"]


def test_multi_audio_stream_gate_insufficient_evidence(monkeypatch):
    from dataclasses import replace
    profile = replace(_rotated_profile(0), audio_stream_count=2, rotation_source=smp.ROTATION_SOURCE_NONE)
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: profile)
    [diag] = worker_job.evaluate_source_format_gate({"s1": "x.mp4"})
    assert diag["decision"] == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.REASON_MULTIPLE_AUDIO_STREAMS in diag["blocking_reasons"]


def test_unknown_codec_gate_insufficient_evidence(monkeypatch):
    from dataclasses import replace
    profile = replace(_rotated_profile(0), video_codec=None, raw_video_codec="mystery",
                       rotation_source=smp.ROTATION_SOURCE_NONE)
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: profile)
    [diag] = worker_job.evaluate_source_format_gate({"s1": "x.mp4"})
    assert diag["decision"] == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.REASON_UNKNOWN_CODEC in diag["blocking_reasons"]


def test_hevc_runtime_unverified_gate_insufficient_evidence(monkeypatch):
    """D-274C-A now feeds a REAL, dynamically-established capability into
    this gate (see `test_gate_uses_established_worker_capability` and
    `test_gate_falls_back_to_insufficient_evidence_when_unestablished`
    below for the two live cases this supersedes) -- this specific test
    forces the pre-D-274C-A UNESTABLISHED case explicitly, since that is
    what its own name asserts ('runtime unverified'), rather than relying
    on an implicit always-off default that no longer reflects live
    behavior."""
    from dataclasses import replace
    from cutsell_worker import production_runtime_capability as prc
    from cutsell_worker import worker_runtime_capability as wrc
    profile = replace(_rotated_profile(0), video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc",
                       rotation_source=smp.ROTATION_SOURCE_NONE)
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: profile)
    unestablished = prc.ProductionRuntimeCapability(
        hevc_decoder_available=False, h264_encoder_available=False, ffmpeg_version=None,
        capability_source=prc.CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED,
    )
    monkeypatch.setattr(wrc, "get_worker_runtime_capability", lambda: unestablished)
    [diag] = worker_job.evaluate_source_format_gate({"s1": "x.mp4"})
    assert diag["decision"] == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert sfp.REASON_HEVC_RUNTIME_UNVERIFIED in diag["blocking_reasons"]


def test_invalid_dimensions_gate_rejects(monkeypatch):
    from dataclasses import replace
    profile = replace(_rotated_profile(0), coded_width=0, rotation_source=smp.ROTATION_SOURCE_NONE)
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: profile)
    [diag] = worker_job.evaluate_source_format_gate({"s1": "x.mp4"})
    assert diag["decision"] == sfp.DECISION_REJECT
    assert sfp.REASON_INVALID_DIMENSIONS in diag["blocking_reasons"]


def test_diagnostics_deterministic_and_multi_source_order_preserved(monkeypatch):
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: _rotated_profile(0))
    diags = worker_job.evaluate_source_format_gate({"a": "a.mp4", "b": "b.mp4", "c": "c.mp4"})
    assert [d["source_asset_id"] for d in diags] == ["a", "b", "c"]
    assert all(d["decision"] == sfp.DECISION_ACCEPT for d in diags)


def test_gate_passes_the_established_worker_capability_input(monkeypatch):
    """D-274C-A supersedes D-272A's own original Stage 7 ('no canonical
    production capability source exists, so never pass one') -- self-
    resolving guard, same pattern as D-272B's own precedent. The live
    gate now passes EXACTLY `worker_runtime_capability.get_worker_
    runtime_capability_input()`'s own result -- traced here with a
    distinctive sentinel to prove it is that call's result and not some
    other hardcoded value, never a duplicated derivation."""
    from cutsell_worker import source_format_policy as sfp_module
    from cutsell_worker import worker_runtime_capability as wrc
    captured = {}
    real = sfp.evaluate_source_format_policy

    def _spy(profile, **kwargs):
        captured.update(kwargs)
        return real(profile, **kwargs)

    sentinel = sfp_module.RuntimeCapabilityInput(hevc_decode_confirmed=True, av1_decode_confirmed=False)
    monkeypatch.setattr(worker_job, "evaluate_source_format_policy", _spy)
    monkeypatch.setattr(wrc, "get_worker_runtime_capability_input", lambda: sentinel)
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: _rotated_profile(0))
    worker_job.evaluate_source_format_gate({"s1": "x.mp4"})
    assert captured.get("runtime_capability") is sentinel


def test_gate_falls_back_to_insufficient_evidence_when_unestablished(monkeypatch):
    """The other half of the D-274C-A pairing: when the worker's own
    capability establishment did NOT genuinely succeed, the live gate's
    behavior is byte-identical to every pre-D-274C-A default (all-`False`
    RuntimeCapabilityInput) -- confirmed via the real bridge function,
    not a hand-rolled stand-in."""
    from dataclasses import replace
    from cutsell_worker import production_runtime_capability as prc
    from cutsell_worker import worker_runtime_capability as wrc
    profile = replace(_rotated_profile(0), video_codec=smp.VIDEO_CODEC_HEVC, raw_video_codec="hevc",
                       rotation_source=smp.ROTATION_SOURCE_NONE)
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: profile)
    unestablished = prc.ProductionRuntimeCapability(
        hevc_decoder_available=True, h264_encoder_available=True, ffmpeg_version="x",
        capability_source=prc.CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED,
    )
    monkeypatch.setattr(wrc, "get_worker_runtime_capability", lambda: unestablished)
    [diag] = worker_job.evaluate_source_format_gate({"s1": "x.mp4"})
    assert diag["decision"] == sfp.DECISION_INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# SourceFormatGateBlocked
# ---------------------------------------------------------------------------

def test_source_format_gate_blocked_carries_structured_diagnostics():
    blocked = [{
        "source_asset_id": "s1", "decision": sfp.DECISION_REJECT,
        "user_facing_error_code": sfp.USER_FACING_UNSUPPORTED_VIDEO_FORMAT,
    }]
    exc = worker_job.SourceFormatGateBlocked(blocked)
    assert exc.blocked_sources == blocked
    assert exc.primary_error_code == sfp.USER_FACING_UNSUPPORTED_VIDEO_FORMAT
    assert "s1" in str(exc)


# ---------------------------------------------------------------------------
# Stage 3-13/25 -- run_flow_b_job() live integration: ACCEPT proceeds,
# non-ACCEPT blocks before process_local_sources, job status/error mapping
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


def _payload(*, uri: str, source_asset_id: str = "s1", project_id: str = "p1", user_id: str = "u1") -> dict:
    return {
        "project_id": project_id,
        "user_id": user_id,
        "sources": [{"source_asset_id": source_asset_id, "uri": uri, "original_name": Path(uri).name}],
    }


def _multi_source_payload(uris: dict, *, project_id: str = "p1", user_id: str = "u1") -> dict:
    return {
        "project_id": project_id,
        "user_id": user_id,
        "sources": [
            {"source_asset_id": asset_id, "uri": uri, "original_name": Path(uri).name}
            for asset_id, uri in uris.items()
        ],
    }


@pytest.fixture
def wired_worker_job(monkeypatch):
    """Wires every non-format collaborator of run_flow_b_job to a
    lightweight fake/spy, leaving the real per-source loop, the real
    evaluate_source_format_gate (D-271/D-272), and the real exception
    handling path exercised end to end."""
    calls = {"process_local_sources": 0, "download_source_args": []}

    def fake_download_source(uri, destination, *, client=None):
        # uri is a real local fixture path in these tests (never S3) --
        # copy it into place exactly like a real download would produce.
        calls["download_source_args"].append((uri, destination))
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


@pytestmark_ffmpeg
def test_h264_sdr_live_accept_reaches_downstream(wired_worker_job, h264_mp4):
    result = worker_job.run_flow_b_job(_payload(uri=h264_mp4))
    assert wired_worker_job["process_local_sources"] == 1
    assert result["source_format_diagnostics"][0]["decision"] == sfp.DECISION_ACCEPT
    assert result["source_format_diagnostics"][0]["policy_version"] == sfp.SOURCE_FORMAT_POLICY_VERSION


@pytestmark_ffmpeg
def test_missing_video_live_reject_blocks_before_downstream(wired_worker_job, no_video_source):
    with pytest.raises(worker_job.SourceFormatGateBlocked) as excinfo:
        worker_job.run_flow_b_job(_payload(uri=no_video_source))
    assert wired_worker_job["process_local_sources"] == 0
    assert excinfo.value.primary_error_code == sfp.USER_FACING_UNSUPPORTED_VIDEO_FORMAT


@pytestmark_ffmpeg
def test_corrupt_source_live_reject_blocks_before_downstream(wired_worker_job, corrupt_source):
    """A file this corrupt already fails the pre-existing, narrower
    `media_probe.probe_media` call earlier in the SAME per-source loop
    (unchanged, out of this gate's scope) before this gate's own
    evaluate_source_format_gate step is ever reached -- so the exception
    raised here is that pre-existing one, not `SourceFormatGateBlocked`.
    Either way the real, load-bearing invariant holds: no downstream
    processing on corrupt media (Stage 19)."""
    with pytest.raises(Exception):
        worker_job.run_flow_b_job(_payload(uri=corrupt_source))
    assert wired_worker_job["process_local_sources"] == 0


@pytestmark_ffmpeg
def test_mkv_container_live_reject_blocks_before_downstream(wired_worker_job, mkv_source):
    with pytest.raises(worker_job.SourceFormatGateBlocked):
        worker_job.run_flow_b_job(_payload(uri=mkv_source))
    assert wired_worker_job["process_local_sources"] == 0


@pytestmark_ffmpeg
def test_rotation_live_normalize_required_now_attempts_normalization_then_blocks_on_timeout_policy(wired_worker_job, monkeypatch, h264_mp4):
    """D-272A's own original assertion here documented the PRE-D-274F
    terminal state: NORMALIZE_REQUIRED blocked immediately with
    `VIDEO_REQUIRES_NORMALIZATION`, no normalization attempt. D-274F (a
    later, separately-authorized, Product-Owner-authorized gate: "live
    auto-normalization activation") legitimately changes this: the source
    now enters `resolve_sources_for_editorial_entry`'s own NORMALIZE_
    REQUIRED branch, builds a plan, and attempts normalization -- but
    since `run_flow_b_job` (the real production call site) never
    overrides the still-absent canonical normalization timeout (Stage
    10), the executor's own timeout seam is what ultimately blocks this
    job, with the NEW `VIDEO_NORMALIZATION_TIMEOUT_POLICY_REQUIRED`
    code -- never bypassing to `process_local_sources` either way.
    Renamed + rewritten as a self-resolving guard rather than left
    failing or silently deleted (docs/CUTSELL_DECISIONS.md D-274F)."""
    # A real, valid fixture so the pre-existing, unrelated `probe_media`
    # call earlier in the loop succeeds -- only D-271's own probe is
    # replaced with a parser-controlled rotated profile (Stage 15: proves
    # policy/live-gate integration, not real-phone rotation decode).
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: _rotated_profile(90))
    with pytest.raises(worker_job.SourceFormatGateBlocked) as excinfo:
        worker_job.run_flow_b_job(_payload(uri=h264_mp4))
    assert wired_worker_job["process_local_sources"] == 0
    assert excinfo.value.blocked_sources[0]["decision"] == sfp.DECISION_NORMALIZE_REQUIRED
    assert excinfo.value.primary_error_code == worker_job.USER_FACING_VIDEO_NORMALIZATION_TIMEOUT_POLICY_REQUIRED


@pytestmark_ffmpeg
def test_hdr_pq_live_normalize_required_blocks_before_downstream(wired_worker_job, monkeypatch, h264_mp4):
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: _hdr_profile(smp.HDR_STATUS_HDR_PQ))
    with pytest.raises(worker_job.SourceFormatGateBlocked):
        worker_job.run_flow_b_job(_payload(uri=h264_mp4))
    assert wired_worker_job["process_local_sources"] == 0


@pytestmark_ffmpeg
def test_unknown_codec_live_insufficient_evidence_blocks_before_downstream(wired_worker_job, monkeypatch, h264_mp4):
    from dataclasses import replace
    profile = replace(_rotated_profile(0), video_codec=None, raw_video_codec="mystery",
                       rotation_source=smp.ROTATION_SOURCE_NONE)
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: profile)
    with pytest.raises(worker_job.SourceFormatGateBlocked) as excinfo:
        worker_job.run_flow_b_job(_payload(uri=h264_mp4))
    assert wired_worker_job["process_local_sources"] == 0
    assert excinfo.value.primary_error_code == sfp.USER_FACING_UNSUPPORTED_VIDEO_FORMAT


@pytestmark_ffmpeg
def test_multi_source_one_blocked_stops_whole_job(wired_worker_job, h264_mp4, no_video_source):
    with pytest.raises(worker_job.SourceFormatGateBlocked) as excinfo:
        worker_job.run_flow_b_job(_multi_source_payload({"good": h264_mp4, "bad": no_video_source}))
    assert wired_worker_job["process_local_sources"] == 0
    blocked_ids = {b["source_asset_id"] for b in excinfo.value.blocked_sources}
    assert blocked_ids == {"bad"}


@pytestmark_ffmpeg
def test_blocked_job_maps_to_existing_failed_state_and_error_code(wired_worker_job, no_video_source):
    states = []
    payloads = []
    monkeypatch = pytest.MonkeyPatch()
    try:
        def spy_update(**kw):
            states.append(kw.get("state"))
            return {"status": "saved", "project_state": kw.get("state")}

        def spy_notify(**kw):
            payloads.append(kw)
            return {"notification_id": "n1"}

        monkeypatch.setattr(worker_job, "safe_update_project", spy_update)
        monkeypatch.setattr(worker_job, "publish_notification", spy_notify)
        with pytest.raises(worker_job.SourceFormatGateBlocked):
            worker_job.run_flow_b_job(_payload(uri=no_video_source))
        assert "failed" in states
        # No new parallel job lifecycle (Stage 12): still the existing
        # failed-state vocabulary, with the D-272 error code preserved in
        # the notification payload rather than a generic exception name.
        failure_notifications = [p for p in payloads if p.get("kind") == "processing_failed"]
        assert failure_notifications
        assert failure_notifications[0]["payload"]["error"] == sfp.USER_FACING_UNSUPPORTED_VIDEO_FORMAT
        assert "source_format_gate" in failure_notifications[0]["payload"]
    finally:
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# Stage 26 -- no normalization/transcode/mutation performed by this gate
# ---------------------------------------------------------------------------

def test_worker_job_source_never_mutated_by_gate(monkeypatch, tmp_path):
    fixture = tmp_path / "f.mp4"
    fixture.write_bytes(b"original-bytes")
    monkeypatch.setattr(smp, "probe_source_media_profile", lambda path, **kw: _rotated_profile(90))
    worker_job.evaluate_source_format_gate({"s1": str(fixture)})
    assert fixture.read_bytes() == b"original-bytes"


def test_worker_job_source_no_transcode_normalization_language():
    """D-274F (a later, separately-authorized, Product-Owner-authorized
    gate: "live auto-normalization activation") legitimately calls the
    already-existing `worker_runtime_capability.get_worker_tonemap_
    available()` CAPABILITY CHECK from `worker_job.py` -- narrowed the
    banned `"tonemap"` substring to the actual ffmpeg FILTER-invocation
    form `"tonemap="` (e.g. `tonemap=hable`, as it genuinely appears in
    `source_normalization_executor.py`'s own filter chain), which still
    never appears in `worker_job.py` and still catches a real accidental
    filter-string duplication -- self-resolving guard, docs/CUTSELL_
    DECISIONS.md D-274F has the full disclosure."""
    source = _source_without_docstrings("cutsell_worker/worker_job.py")
    for banned in ("ffmpeg.input(", "-vf", '"-c:v"', "scale=", "pad=", "hflip", "vflip",
                   "tonemap=", "let ffmpeg try"):
        assert banned not in source


def test_worker_job_no_hardcoded_runtime_capability_confirmed():
    """No canonical production capability source exists yet -- confirm this
    gate never hardcodes hevc_decode_confirmed/av1_decode_confirmed True."""
    source = _source_without_docstrings("cutsell_worker/worker_job.py")
    assert "hevc_decode_confirmed=True" not in source
    assert "av1_decode_confirmed=True" not in source
    assert "RuntimeCapabilityInput(" not in source


# ---------------------------------------------------------------------------
# Stage 2/32 -- no duplicated policy; D-272 remains sole authority
# ---------------------------------------------------------------------------

def test_worker_job_reuses_d271_and_d272_functions_not_duplicated():
    source = _source_without_docstrings("cutsell_worker/worker_job.py")
    assert "smp.probe_source_media_profile(" in source
    assert "evaluate_source_format_policy(" in source
    # No independent codec/HDR/rotation/VFR literal vocabulary reimplemented
    # here -- those constants only ever come from the imported modules.
    for banned_literal in ('"HEVC"', '"h264"', "smpte2084", "arib-std-b67"):
        assert banned_literal not in source


def test_source_format_policy_module_activation_deferred_to_d272b():
    """D-272A itself never touched source_format_policy.py -- true at
    D-272A time. D-272B is the separately-authorized, narrow HEVC-policy
    reconciliation gate that legitimately extends it (see test_cutsell_
    d272_source_format_policy.py's own updated HEVC expectations); this
    test only confirms the module's own D-272A-era public surface
    (evaluate_source_format_policy, RuntimeCapabilityInput) still exists,
    rather than re-asserting the now-superseded "untouched" guard."""
    assert hasattr(sfp, "evaluate_source_format_policy")
    assert hasattr(sfp, "RuntimeCapabilityInput")


def test_source_media_profile_module_still_untouched_by_this_gate():
    assert _run_git_diff("cutsell_worker/source_media_profile.py") == ""


def test_flow_b_module_untouched_by_this_gate():
    assert _run_git_diff("cutsell_worker/flow_b.py") == ""


# ---------------------------------------------------------------------------
# Stage 27 -- firewall: unrelated authorities unchanged
# ---------------------------------------------------------------------------

_FIREWALL_FILES = [
    "cutsell_worker/render.py",
    "cutsell_worker/render_delivery.py",
    "cutsell_worker/render_plan.py",
    "cutsell_worker/media_probe.py",
    "cutsell_worker/post_render_media_qc.py",
    "cutsell_worker/visual_finishing_measurement.py",
    "cutsell_worker/boundary_engine_pass.py",
    "cutsell_worker/pacing_transition_decision.py",
    "cutsell_worker/post_render_watch_listen_qc.py",
    "cutsell_worker/live_render_qc.py",
    "cutsell_worker/finishing_contract.py",
    "cutsell_worker/export_job.py",
    "cutsell_worker/exports.py",
    "cutsell_worker/tenant_safe_delivery.py",
    "cutsell_worker/uploads.py",
    "gpu_execution_provider.py",
]


@pytest.mark.parametrize("rel_path", _FIREWALL_FILES)
def test_unrelated_authorities_unchanged(rel_path):
    if not Path(rel_path).exists():
        pytest.skip(f"{rel_path} not present in this checkout")
    assert _run_git_diff(rel_path) == ""


# ---------------------------------------------------------------------------
# Security / no-secrets
# ---------------------------------------------------------------------------

def test_worker_job_source_has_no_secrets():
    source = Path("cutsell_worker/worker_job.py").read_text(encoding="utf-8")
    for banned in ("AKIA", "aws_secret", "BEGIN PRIVATE KEY", "api_key="):
        assert banned not in source


def test_compileall_clean():
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", "cutsell_worker/worker_job.py"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
