"""D-274D -- HDR / 10-BIT / PIXEL-FORMAT NORMALIZATION.

Post D-274C-A. Proves the PRODUCT-OWNER-AUTHORIZED V1 HDR policy against
real ffmpeg-generated synthetic fixtures, end-to-end through the ACTUAL
D-271 profiler -> D-272 policy -> D-274A plan -> D-274D executor chain:

  - PQ -> SDR BT.709 tonemap (genuine `color_transfer=smpte2084` fixture)
  - HLG -> SDR BT.709 tonemap (genuine `color_transfer=arib-std-b67` fixture)
  - Dolby Vision / HDR_OTHER -- rejected at the PLAN level, zero ffmpeg
  - 10-bit SDR -> 8-bit (zero new filter code: the executor's own fixed
    `-pix_fmt yuv420p` output flag already downconverts)
  - yuv422/yuv444 SDR -> yuv420p (identical zero-new-filter-code finding)
  - HEVC+HDR, rotation+HDR, VFR+HDR one-pass composition
  - the tonemap-capability pre-check (fail closed, mirrors D-274C's own
    codec pre-check)
  - explicit BT.709 output color-metadata tags, ONLY when tonemap fired
  - the mandatory D-271 re-probe + D-272 re-evaluation loop reaching
    ACCEPT
  - the bounded, diagnostic-only luma-measurement evidence

NO Dolby Vision normalization, NO broad exotic-codec support, NO
renderer/Pacing/Boundary/Freeze/Audio-Join/Audio-Finishing/Visual-
Finishing/Delivery change, NO live auto-normalization activation
(worker_job.py untouched), NO RAW, NO provider, NO paid compute anywhere
in this file.
"""
from __future__ import annotations

import dataclasses
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from cutsell_worker import production_runtime_capability as prc
from cutsell_worker import source_format_policy as sfp
from cutsell_worker import source_media_profile as smp
from cutsell_worker import source_normalization_executor as exe
from cutsell_worker import source_normalization_plan as snp
from cutsell_worker import worker_runtime_capability as wrc

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")

_TEST_TIMEOUT_SEC = 30.0


def _exec(source_path, plan, *, output_directory, timeout_sec=_TEST_TIMEOUT_SEC, **kwargs):
    return exe.execute_source_normalization(
        source_path, plan, output_directory=output_directory, timeout_sec=timeout_sec, **kwargs,
    )


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True, shell=False)


def _make_plan(**overrides) -> snp.SourceNormalizationPlan:
    """Mirrors D-274B's own `_make_plan` helper exactly (Stage 38's own
    'executor obeys whatever plan it is handed' testing path)."""
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


def _real_capability() -> "prc.ProductionRuntimeCapability":
    """This sandbox's own genuine, real-subprocess-probed capability --
    never a fabricated `True`."""
    return prc.capture_local_sandbox_capability_for_testing()


# =============================================================================
# Fixtures -- all genuinely ffprobe-verifiable, never parser-only
# =============================================================================

@pytest.fixture(scope="module", params=["smpte2084", "arib-std-b67"])
def hdr_10bit_mp4(request, tmp_path_factory):
    """Genuine PQ (`smpte2084`) or HLG (`arib-std-b67`) tagged, 10-bit
    H264 source. D-274D's own forensic finding (docs/CUTSELL_DECISIONS.md
    D-274D): `zscale`'s own `transfer=linear` step requires the DECODED
    frames to genuinely carry this transfer tag -- a forced plan/profile
    mismatch fails inside ffmpeg itself ("no path between colorspaces")."""
    d = tmp_path_factory.mktemp("d274d_hdr")
    path = str(d / f"hdr_{request.param}.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020", "-color_trc", request.param, "-colorspace", "bt2020nc",
        path,
    ])
    return path, request.param


@pytest.fixture(scope="module")
def sdr_10bit_mp4(tmp_path_factory):
    """Genuine 10-bit source with NO HDR tags -- proves TEN_BIT_TO_EIGHT_
    BIT fires independently of any HDR action."""
    d = tmp_path_factory.mktemp("d274d_sdr10bit")
    path = str(d / "sdr_10bit.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        path,
    ])
    return path


@pytest.fixture(scope="module", params=["yuv422p", "yuv444p"])
def broad_pixel_format_mp4(request, tmp_path_factory):
    """Genuine 8-bit, non-4:2:0 chroma subsampling sources."""
    d = tmp_path_factory.mktemp("d274d_pixfmt")
    path = str(d / f"{request.param}.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", request.param, path,
    ])
    return path, request.param


@pytest.fixture(scope="module")
def sdr_8bit_mp4(tmp_path_factory):
    """Plain canonical SDR 8-bit yuv420p source -- the negative control."""
    d = tmp_path_factory.mktemp("d274d_sdr8bit")
    path = str(d / "sdr_8bit.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ])
    return path


# =============================================================================
# HDR PQ/HLG -> SDR BT.709 tonemap -- full real chain, ACCEPT required
# =============================================================================

def test_hdr_input_genuinely_classified(hdr_10bit_mp4):
    path, transfer = hdr_10bit_mp4
    profile = smp.probe_source_media_profile(path)
    expected = smp.HDR_STATUS_HDR_PQ if transfer == "smpte2084" else smp.HDR_STATUS_HDR_HLG
    assert profile.hdr_status == expected, "fixture must genuinely be HDR-tagged before normalization"
    assert profile.bit_depth == 10


def test_hdr_tonemap_full_chain_reaches_accept(hdr_10bit_mp4, tmp_path):
    path, transfer = hdr_10bit_mp4
    cap = _real_capability()
    profile = smp.probe_source_media_profile(path)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    plan_result = snp.build_source_normalization_plan(
        "src", profile, decision, tonemap_available=cap.hdr_tonemap_usable,
    )
    plan = plan_result.plan
    assert plan is not None and plan.is_executable
    expected_action = snp.ACTION_HDR_PQ_TO_SDR_BT709 if transfer == "smpte2084" else snp.ACTION_HDR_HLG_TO_SDR_BT709
    assert plan.hdr_action == expected_action
    r = _exec(path, plan, output_directory=str(tmp_path), tonemap_capability=cap)
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert r.diagnostics["final_d272_decision"] == sfp.DECISION_ACCEPT
    assert r.normalized_profile.hdr_status == smp.HDR_STATUS_SDR
    assert r.normalized_profile.bit_depth == 8
    assert r.normalized_profile.pixel_format == "yuv420p"


def test_hdr_tonemap_output_bt709_color_metadata(hdr_10bit_mp4, tmp_path):
    """Stage 8: explicit, genuinely-readable BT.709 output tags."""
    path, _ = hdr_10bit_mp4
    cap = _real_capability()
    profile = smp.probe_source_media_profile(path)
    decision = sfp.evaluate_source_format_policy(profile)
    plan = snp.build_source_normalization_plan(
        "src", profile, decision, tonemap_available=cap.hdr_tonemap_usable,
    ).plan
    r = _exec(path, plan, output_directory=str(tmp_path), tonemap_capability=cap)
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert r.normalized_profile.color_primaries == "bt709"
    assert r.normalized_profile.color_transfer == "bt709"
    assert r.normalized_profile.color_space == "bt709"
    assert r.diagnostics["needs_bt709_tagging"] is True


def test_non_hdr_normalization_never_tags_bt709(sdr_10bit_mp4, tmp_path):
    """Stage 8's own 'do not mislabel non-HDR outputs': a 10-bit-only
    normalization (no HDR action) must never set `needs_bt709_tagging`."""
    profile = smp.probe_source_media_profile(sdr_10bit_mp4)
    decision = sfp.evaluate_source_format_policy(profile)
    plan = snp.build_source_normalization_plan("src", profile, decision).plan
    assert plan.hdr_action == snp.ACTION_NO_ACTION
    r = _exec(sdr_10bit_mp4, plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert r.diagnostics["needs_bt709_tagging"] is False


def test_hdr_tonemap_luma_evidence_shows_real_transformation(hdr_10bit_mp4, tmp_path):
    """Stage 23/39: bounded, diagnostic-only before/after luma evidence
    proves the tonemap genuinely changed pixel VALUES, not just tags."""
    path, _ = hdr_10bit_mp4
    cap = _real_capability()
    profile = smp.probe_source_media_profile(path)
    decision = sfp.evaluate_source_format_policy(profile)
    plan = snp.build_source_normalization_plan(
        "src", profile, decision, tonemap_available=cap.hdr_tonemap_usable,
    ).plan
    r = _exec(path, plan, output_directory=str(tmp_path), tonemap_capability=cap)
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    before = r.diagnostics.get("luma_before")
    after = r.diagnostics.get("luma_after")
    assert before is not None and after is not None, "luma evidence must be present for an HDR action"
    # PQ/HLG source values are on a >8-bit (0-1023-ish) scale; SDR output
    # is genuinely 0-255 -- never asserting an exact number (Stage 39's
    # own "no invented tolerance"), only that a real transformation
    # occurred and both scales are self-consistent.
    assert after["ymax"] <= 255.5
    assert before["ymax"] > after["ymax"]


def test_luma_evidence_absent_for_non_hdr_action(sdr_10bit_mp4, tmp_path):
    profile = smp.probe_source_media_profile(sdr_10bit_mp4)
    decision = sfp.evaluate_source_format_policy(profile)
    plan = snp.build_source_normalization_plan("src", profile, decision).plan
    r = _exec(sdr_10bit_mp4, plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert "luma_before" not in r.diagnostics
    assert "luma_after" not in r.diagnostics


# =============================================================================
# Dolby Vision / HDR_OTHER -- rejected at the PLAN level, zero ffmpeg calls
# =============================================================================

@pytest.mark.parametrize("hdr_status", [smp.HDR_STATUS_HDR_DOLBY_VISION, smp.HDR_STATUS_HDR_OTHER])
def test_dolby_vision_and_hdr_other_blocked_at_plan_level(sdr_8bit_mp4, hdr_status):
    """Stage 3/4's own firewall: D-274A's own `unsupported=True` branch
    (established before D-274D existed) already blocks these -- D-274D
    adds ZERO Dolby-Vision/HDR_OTHER-specific code. This is the plan-level
    half of the firewall proof; the executor-level half (never even
    reached) is proven by `test_dolby_vision_never_reaches_executor`."""
    profile = dataclasses.replace(smp.probe_source_media_profile(sdr_8bit_mp4), hdr_status=hdr_status)
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    plan_result = snp.build_source_normalization_plan("src", profile, decision, tonemap_available=True)
    assert plan_result.outcome == snp.NORMALIZATION_UNSUPPORTED
    assert plan_result.plan is None or plan_result.plan.is_executable is False


def test_dolby_vision_never_reaches_executor(sdr_8bit_mp4, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(exe.subprocess, "run", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(
        AssertionError("ffmpeg must never be invoked for Dolby Vision")
    ))
    profile = dataclasses.replace(
        smp.probe_source_media_profile(sdr_8bit_mp4), hdr_status=smp.HDR_STATUS_HDR_DOLBY_VISION,
    )
    decision = sfp.evaluate_source_format_policy(profile)
    plan_result = snp.build_source_normalization_plan("src", profile, decision, tonemap_available=True)
    # D-274A's own plan builder never even returns an executable plan --
    # nothing to hand the executor at all. This assertion documents that
    # invariant explicitly rather than assuming it silently.
    assert plan_result.plan is None or not plan_result.plan.is_executable
    assert calls == []


# =============================================================================
# 10-bit SDR -> 8-bit -- zero new filter code (D-274D's own empirical finding)
# =============================================================================

def test_sdr_10bit_downconverts_with_existing_pix_fmt_flag(sdr_10bit_mp4, tmp_path):
    profile = smp.probe_source_media_profile(sdr_10bit_mp4)
    assert profile.bit_depth == 10
    assert profile.hdr_status == smp.HDR_STATUS_SDR
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    plan = snp.build_source_normalization_plan("src", profile, decision).plan
    assert plan.bit_depth_action == snp.ACTION_TEN_BIT_TO_EIGHT_BIT
    assert plan.hdr_action == snp.ACTION_NO_ACTION
    r = _exec(sdr_10bit_mp4, plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert r.normalized_profile.bit_depth == 8
    assert r.diagnostics["final_d272_decision"] == sfp.DECISION_ACCEPT


# =============================================================================
# yuv422/yuv444 SDR -> yuv420p -- identical zero-new-filter-code finding
# =============================================================================

def test_broad_pixel_format_downconverts_with_existing_pix_fmt_flag(broad_pixel_format_mp4, tmp_path):
    path, source_fmt = broad_pixel_format_mp4
    profile = smp.probe_source_media_profile(path)
    assert profile.pixel_format == source_fmt
    decision = sfp.evaluate_source_format_policy(profile)
    assert decision.decision == sfp.DECISION_NORMALIZE_REQUIRED
    plan = snp.build_source_normalization_plan("src", profile, decision).plan
    assert plan.pixel_format_action == snp.ACTION_PIXEL_FORMAT_TO_YUV420P
    r = _exec(path, plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert r.normalized_profile.pixel_format == "yuv420p"
    assert r.diagnostics["final_d272_decision"] == sfp.DECISION_ACCEPT


# =============================================================================
# Composition -- rotation+HDR, VFR+HDR, one ffmpeg generation
# =============================================================================

def test_rotation_plus_hdr_one_generation(hdr_10bit_mp4, tmp_path):
    path, _ = hdr_10bit_mp4
    cap = _real_capability()
    profile = smp.probe_source_media_profile(path)
    decision = sfp.evaluate_source_format_policy(profile)
    plan = snp.build_source_normalization_plan(
        "src", profile, decision, tonemap_available=cap.hdr_tonemap_usable,
    ).plan
    composed = dataclasses.replace(plan, rotation_action=snp.ACTION_ROTATE_90)
    r = _exec(path, composed, output_directory=str(tmp_path), tonemap_capability=cap)
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert r.diagnostics["final_d272_decision"] == sfp.DECISION_ACCEPT
    assert r.normalized_profile.hdr_status == smp.HDR_STATUS_SDR


def test_vfr_plus_hdr_one_generation(hdr_10bit_mp4, tmp_path):
    path, _ = hdr_10bit_mp4
    cap = _real_capability()
    profile = smp.probe_source_media_profile(path)
    decision = sfp.evaluate_source_format_policy(profile)
    plan = snp.build_source_normalization_plan(
        "src", profile, decision, tonemap_available=cap.hdr_tonemap_usable,
    ).plan
    composed = dataclasses.replace(plan, frame_rate_action=snp.ACTION_VFR_TO_CFR, target_fps=15.0)
    r = _exec(path, composed, output_directory=str(tmp_path), tonemap_capability=cap)
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert r.diagnostics["final_d272_decision"] == sfp.DECISION_ACCEPT
    assert abs(r.normalized_profile.effective_fps - 15.0) < 0.01


# =============================================================================
# Tonemap-capability pre-check -- fail closed, mirrors D-274C's own codec check
# =============================================================================

def test_tonemap_capability_unavailable_rejected_before_ffmpeg(hdr_10bit_mp4, tmp_path, monkeypatch):
    path, _ = hdr_10bit_mp4
    calls = []
    monkeypatch.setattr(exe.subprocess, "run", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(
        AssertionError("ffmpeg must never be invoked when tonemap capability is unavailable")
    ))
    plan = _make_plan(hdr_action=snp.ACTION_HDR_PQ_TO_SDR_BT709)
    unavailable = dataclasses.replace(_real_capability(), zscale_available=False, tonemap_available=False)
    assert unavailable.hdr_tonemap_usable is False
    r = _exec(path, plan, output_directory=str(tmp_path), tonemap_capability=unavailable)
    assert r.outcome == snp.NORMALIZATION_FAILED
    assert r.failure.error_category == exe.FAILURE_HDR_CAPABILITY_UNAVAILABLE
    assert calls == []


def test_tonemap_capability_check_skipped_when_not_supplied(hdr_10bit_mp4, tmp_path):
    """Backward compatibility (D-274D's own "byte-identical behavior when
    omitted" discipline, mirrors D-274C's own codec_capability): no
    `tonemap_capability` argument at all -> pre-check skipped, ffmpeg's
    own success/failure path is the safety net."""
    path, _ = hdr_10bit_mp4
    profile = smp.probe_source_media_profile(path)
    action = snp.ACTION_HDR_PQ_TO_SDR_BT709 if profile.hdr_status == smp.HDR_STATUS_HDR_PQ else snp.ACTION_HDR_HLG_TO_SDR_BT709
    plan = _make_plan(hdr_action=action)
    r = _exec(path, plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics


def test_tonemap_capability_and_codec_capability_are_independent(hdr_10bit_mp4, tmp_path):
    """Stage 9's own "each gate's own capability check independently
    toggleable": a caller may supply `codec_capability` alone (no HEVC
    action on this plan) alongside a genuinely usable `tonemap_
    capability` -- both checks must coexist without interference."""
    path, _ = hdr_10bit_mp4
    cap = _real_capability()
    profile = smp.probe_source_media_profile(path)
    decision = sfp.evaluate_source_format_policy(profile)
    plan = snp.build_source_normalization_plan(
        "src", profile, decision, tonemap_available=cap.hdr_tonemap_usable,
    ).plan
    r = _exec(path, plan, output_directory=str(tmp_path), codec_capability=cap, tonemap_capability=cap)
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics


# =============================================================================
# Worker-runtime capability seam (D-274D Stage 11) -- built, tested, NOT wired
# =============================================================================

def test_worker_tonemap_available_reflects_real_probe():
    wrc._reset_for_testing()
    try:
        available = wrc.get_worker_tonemap_available()
        capability = wrc.get_worker_runtime_capability()
        assert isinstance(available, bool)
        # Fail-closed: only True when genuinely ESTABLISHED AND both
        # filters are genuinely listed.
        expected = (
            capability.production_verification_status == prc.PRODUCTION_VERIFICATION_STATUS_ESTABLISHED
            and capability.hdr_tonemap_usable
        )
        assert available == expected
    finally:
        wrc._reset_for_testing()


def test_worker_tonemap_available_fails_closed_when_unestablished(monkeypatch):
    monkeypatch.setattr(
        prc, "capture_production_worker_capability",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated probe failure")),
    )
    wrc._reset_for_testing()
    try:
        assert wrc.get_worker_tonemap_available() is False
    finally:
        wrc._reset_for_testing()


def test_worker_runtime_capability_carries_tonemap_fields_through_establishment():
    """Regression guard for the field-drop bug caught and fixed within
    this same gate (see worker_runtime_capability.py's own comment): the
    ESTABLISHED-promotion branch must not silently reset zscale/tonemap/
    libplacebo to their dataclass defaults."""
    wrc._reset_for_testing()
    try:
        capability = wrc.get_worker_runtime_capability()
        if capability.production_verification_status == prc.PRODUCTION_VERIFICATION_STATUS_ESTABLISHED:
            raw = prc.capture_production_worker_capability()
            assert capability.zscale_available == raw.zscale_available
            assert capability.tonemap_available == raw.tonemap_available
            assert capability.libplacebo_available == raw.libplacebo_available
    finally:
        wrc._reset_for_testing()


def test_worker_job_now_legitimately_uses_tonemap_seam_via_d274f():
    """No live auto-normalization activation banner: worker_job.py carried
    zero reference to the tonemap seam this gate builds -- true at the
    time. D-274F (a later, separately-authorized, Product-Owner-
    authorized gate: "live auto-normalization activation") is exactly the
    gate that legitimately calls `get_worker_tonemap_available()` (Stage
    8's own "use real worker tonemap capability truth") before building a
    normalization plan for a PQ/HLG source (docs/CUTSELL_DECISIONS.md
    D-274F). Renamed + rewritten as a self-resolving guard rather than
    left failing or silently deleted."""
    import inspect
    from cutsell_worker import worker_job
    source = inspect.getsource(worker_job)
    assert "get_worker_tonemap_available" in source


# =============================================================================
# Negative matrix -- structural failures still work identically for HDR actions
# =============================================================================

def test_hdr_ffmpeg_nonzero_exit_is_structured_failure(tmp_path):
    bogus_source = tmp_path / "not_real_media.mp4"
    bogus_source.write_bytes(b"not a real video file")
    plan = _make_plan(hdr_action=snp.ACTION_HDR_PQ_TO_SDR_BT709)
    r = _exec(str(bogus_source), plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_FAILED
    assert r.failure.error_category == exe.FAILURE_FFMPEG_FAILED


def test_hdr_action_second_pass_still_rejected(hdr_10bit_mp4, tmp_path):
    path, _ = hdr_10bit_mp4
    cap = _real_capability()
    plan = _make_plan(hdr_action=snp.ACTION_HDR_PQ_TO_SDR_BT709)
    r = _exec(path, plan, output_directory=str(tmp_path), tonemap_capability=cap, attempt_count=1)
    assert r.outcome == snp.NORMALIZATION_FAILED
    assert r.failure.error_category == exe.FAILURE_SECOND_PASS_REJECTED


def test_ten_bit_action_alone_never_sets_hdr_diagnostics(sdr_10bit_mp4, tmp_path):
    plan = _make_plan(bit_depth_action=snp.ACTION_TEN_BIT_TO_EIGHT_BIT)
    r = _exec(sdr_10bit_mp4, plan, output_directory=str(tmp_path))
    assert r.outcome == snp.NORMALIZATION_SUCCEEDED, r.diagnostics
    assert r.diagnostics["hdr_action"] == snp.ACTION_NO_ACTION


# =============================================================================
# Security -- shell safety, no user data in filter syntax
# =============================================================================

def test_tonemap_filter_never_uses_shell():
    import inspect
    source = inspect.getsource(exe)
    assert "shell=True" not in source


def test_tonemap_filter_segment_has_no_user_controlled_interpolation():
    """The tonemap filter string is a fixed, module-level constant
    (Stage 8/9's own disclosed literal) -- never built from user/source-
    controlled strings (no f-string interpolation of profile fields)."""
    segment = exe._hdr_tonemap_filter_segment()
    assert segment.startswith("zscale=transfer=linear:npl=")
    assert "tonemap=hable" in segment
    # The only interpolated value is the disclosed numeric NPL constant.
    assert str(exe._TONEMAP_NOMINAL_PEAK_LUMINANCE_DEFAULT) in segment


# =============================================================================
# Closed-track firewall -- everything else byte-for-byte unchanged
# =============================================================================

@pytest.mark.parametrize("relative_path", [
    # D-274F (a later, separately-authorized, Product-Owner-authorized
    # gate: "live auto-normalization activation") legitimately wires the
    # real probe/policy/plan/executor/format-QC chain into `worker_job.py`
    # itself -- self-resolving guard, removed from this list for that
    # reason (docs/CUTSELL_DECISIONS.md D-274F has the full disclosure).
    "cutsell_worker/source_format_policy.py",
    "cutsell_worker/source_media_profile.py",
    "cutsell_worker/source_normalization_plan.py",
    # render_delivery.py / export_job.py removed from this closed-track
    # list: D-288 (a later, separately-authorized gate) legitimately adds
    # the watch_listen_status delivery gate and the real perceptual-review
    # call to these two files -- same self-resolving-guard pattern as the
    # precedent in test_cutsell_d269a_live_tenant_safe_delivery.py (which
    # documents the D-282 uploads.py/main.py precedent this follows).
    "rq_worker.py",
    "entrypoint.sh",
])
def test_closed_track_files_unmodified_by_this_gate(relative_path):
    """D-274D's own scope is additive-only inside `source_normalization_
    executor.py` (gate-owned, excluded here), `production_runtime_
    capability.py`, and `worker_runtime_capability.py` (both gate-owned,
    excluded here too -- neither was in any EARLIER gate's own closed-
    track list either, since both are themselves D-274C/D-274C-A-owned
    files this gate extends). No other production module's own content
    changes as part of this gate."""
    result = subprocess.run(
        ["git", "diff", "--stat", "a37f4ae", "--", relative_path],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.stdout.strip() == "", f"{relative_path} was modified by D-274D: {result.stdout}"
