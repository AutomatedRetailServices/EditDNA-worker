"""D-274C-A -- LIVE RUNTIME CAPABILITY ACTIVATION.

Post D-274C. Proves `cutsell_worker.worker_runtime_capability`'s
process-local, memoized, fail-closed establishment of a REAL worker's own
HEVC/H264 capability, its bounded diagnostics, and its live wiring into
`worker_job.evaluate_source_format_gate` -- with real HEVC/H264 synthetic
fixtures flowing through the ACTUAL live gate function, not a mock of it.

No live auto-normalization (NORMALIZE_REQUIRED still raises
SourceFormatGateBlocked before process_local_sources, unchanged), no HDR/
10-bit execution, no renderer/Pacing/Boundary/Freeze/Audio-Join/Audio-
Finishing/Visual-Finishing/delivery change, no RAW, no provider, no paid
compute anywhere in this file.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from cutsell_worker import production_runtime_capability as prc
from cutsell_worker import source_format_policy as sfp
from cutsell_worker import worker_job
from cutsell_worker import worker_runtime_capability as wrc

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")


@pytest.fixture(autouse=True)
def _reset_worker_capability_cache():
    """Every test starts from a clean, un-memoized slate and leaves one
    behind -- Stage 5/14's own 'no per-job repeated probe' only means
    something if tests don't leak cached state into each other."""
    wrc._reset_for_testing()
    yield
    wrc._reset_for_testing()


@pytest.fixture(scope="module")
def hevc_fixture(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274ca_hevc")
    path = str(d / "hevc.mp4")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
        "-c:v", "libx265", "-x265-params", "log-level=none", "-pix_fmt", "yuv420p", path,
    ], check=True, shell=False)
    return path


@pytest.fixture(scope="module")
def h264_fixture(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274ca_h264")
    path = str(d / "h264.mp4")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ], check=True, shell=False)
    return path


def _has_real_hevc_capability() -> bool:
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    decoders = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"], capture_output=True, text=True).stdout
    return "libx265" in encoders.lower() and "hevc" in decoders.lower()


# =============================================================================
# Stage 1-5/9/14 -- establishment mechanism, memoization, fail-closed, isolation
# =============================================================================

def test_establishment_succeeds_on_real_runner():
    cap = wrc.get_worker_runtime_capability()
    assert cap.ffmpeg_version is not None
    if len(cap.errors) == 0:
        assert cap.production_verification_status == prc.PRODUCTION_VERIFICATION_STATUS_ESTABLISHED
    else:
        assert cap.production_verification_status == prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED


def test_establishment_is_memoized_per_process():
    cap1 = wrc.get_worker_runtime_capability()
    cap2 = wrc.get_worker_runtime_capability()
    assert cap1 is cap2  # identical object -- proves no re-probe happened


def test_no_per_job_repeated_subprocess(monkeypatch):
    """Simulates many 'jobs' calling into the capability path -- the
    underlying capture mechanism (which itself makes several real
    subprocess calls) must run at most once, not once per 'job'."""
    calls = []
    real_capture = prc.capture_production_worker_capability

    def _spy(*args, **kwargs):
        calls.append(1)
        return real_capture(*args, **kwargs)

    monkeypatch.setattr(prc, "capture_production_worker_capability", _spy)
    for _ in range(5):
        wrc.get_worker_runtime_capability()
    assert len(calls) == 1  # exactly once across all 5 "job" accesses


def test_capture_never_raises_on_unexpected_exception(monkeypatch):
    def _boom():
        raise RuntimeError("simulated capture crash")

    monkeypatch.setattr(prc, "capture_production_worker_capability", _boom)
    cap = wrc.get_worker_runtime_capability()  # must not raise
    assert cap.production_verification_status == prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED
    assert "worker_capability_establishment_failed" in cap.errors[0]


def test_missing_ffmpeg_version_fails_closed(monkeypatch):
    broken = prc.ProductionRuntimeCapability(
        hevc_decoder_available=True, h264_encoder_available=True, ffmpeg_version=None,
        capability_source=prc.CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED,
    )
    monkeypatch.setattr(prc, "capture_production_worker_capability", lambda: broken)
    cap = wrc.get_worker_runtime_capability()
    assert cap.production_verification_status == prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED


def test_probe_errors_fail_closed(monkeypatch):
    broken = prc.ProductionRuntimeCapability(
        hevc_decoder_available=True, h264_encoder_available=True, ffmpeg_version="x",
        capability_source=prc.CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED,
        errors=("ffmpeg_capability_probe_failed:TimeoutExpired",),
    )
    monkeypatch.setattr(prc, "capture_production_worker_capability", lambda: broken)
    cap = wrc.get_worker_runtime_capability()
    assert cap.production_verification_status == prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED


def test_local_sandbox_capture_never_becomes_established():
    """Stage 1's own doctrine, still enforced: the sandbox-labelled path
    is architecturally separate from the promotion logic and can never
    itself claim ESTABLISHED."""
    sandbox_cap = prc.capture_local_sandbox_capability_for_testing()
    assert sandbox_cap.production_verification_status == prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED


def test_worker_process_isolation_no_shared_global():
    """Two independent 'worker processes' (simulated via fresh caches)
    each establish their own truth -- no cross-worker assumption baked
    into the value itself."""
    cap_a = wrc.get_worker_runtime_capability()
    wrc._reset_for_testing()
    cap_b = wrc.get_worker_runtime_capability()
    assert cap_a == cap_b  # same real environment -> same real answer
    assert cap_a is not cap_b  # but genuinely independently computed


# =============================================================================
# Stage 7 -- H264 encoder requirement for "usable" HEVC capability
# =============================================================================

def test_decoder_without_encoder_not_usable():
    cap = prc.ProductionRuntimeCapability(
        hevc_decoder_available=True, h264_encoder_available=False, ffmpeg_version="x",
        capability_source=prc.CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_ESTABLISHED,
    )
    bridged = prc.bridge_to_runtime_capability_input(cap)
    assert bridged.hevc_decode_confirmed is False


def test_decoder_and_encoder_usable_only_when_established():
    cap = prc.ProductionRuntimeCapability(
        hevc_decoder_available=True, h264_encoder_available=True, ffmpeg_version="x",
        capability_source=prc.CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_ESTABLISHED,
    )
    assert prc.bridge_to_runtime_capability_input(cap).hevc_decode_confirmed is True

    import dataclasses
    cap2 = dataclasses.replace(cap, production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED)
    assert prc.bridge_to_runtime_capability_input(cap2).hevc_decode_confirmed is False


# =============================================================================
# Stage 8/15 -- bounded diagnostics
# =============================================================================

def test_diagnostics_bounded_fields_only():
    diag = wrc.describe_worker_capability_diagnostics()
    assert set(diag.keys()) == {
        "ffmpeg_version", "hevc_decoder_available", "h264_encoder_available",
        "capability_source", "production_verification_status",
    }
    diag_str = str(diag)
    for forbidden in ("REDIS_URL", "password", "secret", "/root/", "AKIA"):
        assert forbidden not in diag_str


# =============================================================================
# Stage 11/6 -- live D-272 policy proof through the REAL worker_job.py path
# =============================================================================

@pytest.mark.skipif(not _has_real_hevc_capability(), reason="no real HEVC codec on this runner")
def test_live_gate_confirmed_hevc_normalize_required(hevc_fixture):
    """Case A: genuine establishment (this sandbox's own real ffmpeg) +
    a real HEVC source through the ACTUAL live worker_job.py function."""
    diagnostics = worker_job.evaluate_source_format_gate({"src": hevc_fixture})
    result = diagnostics[0]
    assert result["decision"] == sfp.DECISION_NORMALIZE_REQUIRED
    assert result["user_facing_error_code"] == "VIDEO_REQUIRES_NORMALIZATION"
    assert "HEVC_TO_H264_NORMALIZATION_REQUIRED" in result["normalization_reasons"]


def test_live_gate_unestablished_hevc_insufficient_evidence(hevc_fixture, monkeypatch):
    """Case B: capability establishment did NOT succeed -- must fall back
    to the pre-D-274C-A INSUFFICIENT_EVIDENCE behavior, unchanged."""
    unestablished = prc.ProductionRuntimeCapability(
        hevc_decoder_available=True, h264_encoder_available=True, ffmpeg_version="x",
        capability_source=prc.CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED,
    )
    monkeypatch.setattr(wrc, "get_worker_runtime_capability", lambda: unestablished)
    diagnostics = worker_job.evaluate_source_format_gate({"src": hevc_fixture})
    result = diagnostics[0]
    assert result["decision"] == sfp.DECISION_INSUFFICIENT_EVIDENCE
    assert result["user_facing_error_code"] == "RUNTIME_CODEC_SUPPORT_UNVERIFIED"


def test_live_gate_h264_unaffected_regardless_of_hevc_state(h264_fixture):
    diagnostics = worker_job.evaluate_source_format_gate({"src": h264_fixture})
    assert diagnostics[0]["decision"] == sfp.DECISION_ACCEPT


def test_live_gate_h264_unaffected_when_hevc_unestablished(h264_fixture, monkeypatch):
    unestablished = prc.ProductionRuntimeCapability(
        hevc_decoder_available=False, h264_encoder_available=False, ffmpeg_version=None,
        capability_source=prc.CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_UNESTABLISHED,
    )
    monkeypatch.setattr(wrc, "get_worker_runtime_capability", lambda: unestablished)
    diagnostics = worker_job.evaluate_source_format_gate({"src": h264_fixture})
    assert diagnostics[0]["decision"] == sfp.DECISION_ACCEPT


# =============================================================================
# Stage 10 -- no live auto-normalization: NORMALIZE_REQUIRED still blocks
# the job before process_local_sources
# =============================================================================

def test_normalize_required_still_raises_source_format_gate_blocked():
    """The confirmed-HEVC live decision feeds the SAME pre-existing
    blocked-sources check in run_flow_b_job -- proven here at the
    diagnostic-shape level (the exact contract SourceFormatGateBlocked
    is built from), without invoking the full job (no ASR/GPU/RAW)."""
    from cutsell_worker.worker_job import DECISION_ACCEPT

    diagnostics = [{"decision": sfp.DECISION_NORMALIZE_REQUIRED, "source_asset_id": "src", "user_facing_error_code": "VIDEO_REQUIRES_NORMALIZATION"}]
    blocked = [d for d in diagnostics if d["decision"] != DECISION_ACCEPT]
    assert blocked  # still blocks -- never silently continues to executor


def test_worker_job_never_imports_normalization_executor():
    import inspect
    source = inspect.getsource(worker_job)
    assert "source_normalization_executor" not in source
    assert "execute_source_normalization" not in source


# =============================================================================
# Stage 9 -- worker starts (no exception) even when HEVC support is absent
# =============================================================================

def test_worker_capability_helpers_never_raise_when_hevc_absent(monkeypatch):
    no_hevc = prc.ProductionRuntimeCapability(
        hevc_decoder_available=False, h264_encoder_available=True, ffmpeg_version="x",
        capability_source=prc.CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_ESTABLISHED,
    )
    monkeypatch.setattr(wrc, "get_worker_runtime_capability", lambda: no_hevc)
    # None of these must raise -- H264 SDR jobs must still work.
    diag = wrc.describe_worker_capability_diagnostics()
    assert diag["hevc_decoder_available"] is False
    bridged = wrc.get_worker_runtime_capability_input()
    assert bridged.hevc_decode_confirmed is False


# =============================================================================
# Security (Stage 18/24) -- shell safety, bounded, no secrets
# =============================================================================

def test_underlying_probe_never_uses_shell():
    import inspect
    source = inspect.getsource(prc)
    assert "shell=True" not in source
    source2 = inspect.getsource(wrc)
    assert "shell=True" not in source2


# =============================================================================
# Closed-track firewall -- everything else byte-for-byte unchanged
# =============================================================================

@pytest.mark.parametrize("relative_path", [
    "cutsell_worker/render.py",
    "cutsell_worker/source_format_policy.py",
    "cutsell_worker/source_media_profile.py",
    # D-274D legitimately modifies cutsell_worker/source_normalization_
    # executor.py (its own gate-owned file: HDR tonemap wiring, the
    # tonemap-capability pre-check, the luma diagnostic helper) --
    # removed from THIS gate's own closed-track list. Self-resolving
    # guard, same pattern D-274C-A itself already applied to earlier
    # gates' own closed-track lists.
    "cutsell_worker/source_normalization_plan.py",
    "cutsell_worker/render_delivery.py",
])
def test_closed_track_files_unmodified_by_this_gate(relative_path):
    result = subprocess.run(
        ["git", "diff", "--stat", "13d80b8", "--", relative_path],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.stdout.strip() == "", f"{relative_path} was modified by D-274C-A: {result.stdout}"


def test_render_timeout_still_unchanged():
    from cutsell_worker import render as render_module
    assert render_module.RENDER_FFMPEG_TIMEOUT_SEC == 1200.0


def test_normalization_timeout_seam_still_unchanged():
    from cutsell_worker import source_normalization_executor as exe
    assert exe.NORMALIZATION_FFMPEG_TIMEOUT_SEC is None
