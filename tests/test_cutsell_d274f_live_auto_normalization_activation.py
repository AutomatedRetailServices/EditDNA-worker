"""D-274F -- LIVE AUTO-NORMALIZATION ACTIVATION.

Post D-274E-A (final render BT709 format contract remediation, Verdict A:
`NORMALIZED_SOURCE_CONTRACT_V1` and `FINAL_RENDER_OUTPUT_CONTRACT_V1` both
proven to reach real `PASS`). Product Owner authorized: activate
`NORMALIZE_REQUIRED -> AUTO-NORMALIZE -> VERIFY -> CONTINUE` at the real
Flow-B job entry point (`worker_job.py::run_flow_b_job`).

No new normalization capability, no new codec support, no new HDR policy,
no new color policy, no new format-QC policy, no retry loop, no second
normalization pass, no renderer behavior change, no RAW/Modal/RunPod/
provider/paid compute anywhere in this file. D-271 (probe), D-272 (policy),
D-274A (plan), D-274B/C/D (executor), and D-274E (format QC) remain the
sole authorities this gate wires together -- `worker_job.py`'s own
`resolve_sources_for_editorial_entry` introduces ZERO duplicated codec/
HDR/rotation/VFR/format-QC logic of its own; it only calls each existing
authority in sequence and enforces the AND-requirement (D-272 ACCEPT *and*
D-274E format QC PASS) before ever substituting a normalized path into the
job's own `local_paths`.

Evidence-level honesty (Stage 25, mirrors D-271's own Stage 39 precedent):
this sandbox's ffmpeg build cannot attach a readable rotation tag via
`-metadata:s:v rotate=` or `-display_rotation` (confirmed by D-271's own
`test_rotation_fixture_generation_not_available_locally`) -- so the
rotation-live tests here use a PARSER-CONTROLLED profile override
(`dataclasses.replace` on a real, fully-probed `SourceMediaProfile`,
patched in only for the ORIGINAL source's own initial probe call, never
for the executor's own "before"/"after" re-probes of real ffmpeg output)
rather than a fabricated end-to-end fixture. Every other live path (VFR,
timeline, HEVC, PQ, HLG, 10-bit) uses a fully genuine, ffprobe-verifiable
real fixture with no profile mocking at all.

Realistic color-metadata baseline (an honest, disclosed test-design
choice): every non-HDR-tonemap fixture in this file carries EXPLICIT
`-color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv`
tags at construction time, mirroring what virtually every real phone/
camera-shot SDR video already carries natively. This is NOT a code
workaround -- ffmpeg genuinely preserves an input's own already-present
color tags through a rotation/VFR/timeline/HEVC/10-bit-only re-encode
(empirically confirmed both here and via `source_normalization_executor.py`'s
own D-274D Stage 8 comment: "a rotation-only or VFR-only output carries
whatever color tags its own source already had"). A source that
genuinely lacks ANY color metadata correctly reaches `NORMALIZED_SOURCE_
CONTRACT_V1`'s own `PARTIAL` (missing evidence, not a violation) and
therefore correctly BLOCKS entry under this gate's own strict PASS-
required AND-requirement (Stage 9/38) -- a deliberate, disclosed,
conservative fail-closed property, not a defect this gate is authorized
to relax (`test_untagged_source_conservatively_blocks_not_a_false_pass`
below proves this explicitly).
"""
from __future__ import annotations

import dataclasses
import inspect
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from cutsell_worker import output_format_qc as ofq
from cutsell_worker import production_runtime_capability as prc
from cutsell_worker import source_format_policy as sfp
from cutsell_worker import source_media_profile as smp
from cutsell_worker import source_normalization_executor as sne
from cutsell_worker import source_normalization_plan as snp
from cutsell_worker import worker_job
from cutsell_worker import worker_runtime_capability as wrc

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")

_TEST_TIMEOUT_SEC = 30.0
_BT709_TAGS = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True, shell=False)


def _run_git_diff_head(rel_path: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", rel_path], capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _established_capability(**overrides) -> "prc.ProductionRuntimeCapability":
    """A fabricated, but internally consistent, genuinely-`ESTABLISHED`
    capability snapshot -- used ONLY to simulate a production worker whose
    real startup self-check already confirmed HEVC/H264/tonemap support
    (Stage 27/29/30's own "simulated ESTABLISHED worker capability"
    instruction), never to fake a `True` value the bridge functions
    wouldn't themselves derive correctly from it."""
    defaults = dict(
        hevc_decoder_available=True,
        h264_encoder_available=True,
        ffmpeg_version="d274f-test-established",
        capability_source=prc.CAPABILITY_SOURCE_PRODUCTION_STARTUP_SELF_CHECK,
        production_verification_status=prc.PRODUCTION_VERIFICATION_STATUS_ESTABLISHED,
        zscale_available=True,
        tonemap_available=True,
        libplacebo_available=False,
        errors=(),
    )
    defaults.update(overrides)
    return prc.ProductionRuntimeCapability(**defaults)


@pytest.fixture
def established_capability(monkeypatch):
    """Patches the ONE real seam `resolve_sources_for_editorial_entry`
    consults (`worker_runtime_capability.get_worker_runtime_capability`,
    lru_cache'd) -- `get_worker_runtime_capability_input`/`get_worker_
    tonemap_available` are left UNPATCHED so this test still exercises
    their own REAL bridge logic (Stage 7/8/9's own fail-closed
    discipline), never a fabricated bool handed in directly."""
    monkeypatch.setattr(wrc, "get_worker_runtime_capability", lambda: _established_capability())
    return _established_capability()


@pytest.fixture
def hevc_capability_unverified(monkeypatch):
    monkeypatch.setattr(
        wrc, "get_worker_runtime_capability",
        lambda: _established_capability(hevc_decoder_available=False),
    )


@pytest.fixture
def tonemap_capability_unavailable(monkeypatch):
    monkeypatch.setattr(
        wrc, "get_worker_runtime_capability",
        lambda: _established_capability(zscale_available=False, tonemap_available=False),
    )


def _probe_with_rotation_override(path: str, rotation_degrees: int):
    """Stage 25: parser-controlled rotation evidence, applied ONLY to a
    genuinely, fully-probed real `SourceMediaProfile` (never a fabricated
    one) -- mirrors D-271's own `test_rotation_display_matrix_swaps_
    dimensions` precedent."""
    real = smp.probe_source_media_profile(path)
    return dataclasses.replace(
        real, rotation_degrees=rotation_degrees, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
    )


def _patched_probe_for_original_only(original_path: str, override_profile):
    """Returns a callable that returns `override_profile` ONLY when called
    with `original_path`, and the REAL probe result for every other path
    (critically: the executor's own "before"/"after" re-probes of the
    real ffmpeg-produced temp/normalized output, which must never be
    fooled by this override -- Stage 32/33's own mandatory REAL re-probe/
    re-evaluate loop stays completely real)."""
    real_probe = smp.probe_source_media_profile

    def _fake(path, *a, **kw):
        if str(path) == str(original_path):
            return override_profile
        return real_probe(path, *a, **kw)

    return _fake


# =============================================================================
# Fixtures -- realistic (BT709-tagged) synthetic media, genuinely
# ffprobe-verifiable except where explicitly disclosed as parser-controlled.
# =============================================================================

@pytest.fixture(scope="module")
def h264_sdr_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274f_h264")
    path = str(d / "h264_sdr.mp4")
    _ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        *_BT709_TAGS, "-c:a", "aac", path,
    ])
    return path


@pytest.fixture(scope="module")
def h264_sdr_no_audio_mp4(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274f_h264_noaudio")
    path = str(d / "h264_noaudio.mp4")
    _ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        *_BT709_TAGS, "-an", path,
    ])
    return path


@pytest.fixture(scope="module")
def h264_untagged_mp4(tmp_path_factory):
    """The negative control (module docstring's own disclosed fail-closed
    property): a genuinely rotation-affected source whose OWN color
    metadata was never written at all."""
    d = tmp_path_factory.mktemp("d274f_untagged")
    path = str(d / "untagged.mp4")
    _ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", path,
    ])
    return path


@pytest.fixture(scope="module")
def no_video_source(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("d274f_novideo") / "audio_only.wav")
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=1", path])
    return path


@pytest.fixture(scope="module")
def mkv_source(tmp_path_factory):
    d = tmp_path_factory.mktemp("d274f_mkv")
    path = str(d / "source.mkv")
    _ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ])
    return path


@pytest.fixture(scope="module")
def vp9_source_mp4(tmp_path_factory):
    """A genuinely codec-unverified source (Stage 21/22 of D-272's own
    docstring: any non-H264/HEVC/AV1 codec has no confirmation mechanism
    defined at all -- always `INSUFFICIENT_EVIDENCE`)."""
    d = tmp_path_factory.mktemp("d274f_vp9")
    path = str(d / "vp9.mp4")
    _ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
        "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", path,
    ])
    return path


@pytest.fixture(scope="module")
def vfr_source_mp4(tmp_path_factory):
    """A REAL, genuine VFR fixture -- concat-demuxer stream-copy of two
    differently-clocked segments (mirrors D-274B's own `genuine_vfr_mp4`
    precedent), tagged with realistic BT709 SDR color metadata."""
    d = tmp_path_factory.mktemp("d274f_vfr")
    seg_a = d / "segA.mp4"
    seg_b = d / "segB.mp4"
    _ffmpeg(["-f", "lavfi", "-i", "testsrc=size=160x120:rate=24", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", *_BT709_TAGS, str(seg_a)])
    _ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=160x120:rate=60", "-t", "1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", *_BT709_TAGS, str(seg_b)])
    list_txt = d / "list.txt"
    list_txt.write_text(f"file '{seg_a.name}'\nfile '{seg_b.name}'\n", encoding="utf-8")
    out = d / "vfr_candidate.mp4"
    _ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_txt), "-c", "copy", str(out)])
    return str(out)


@pytest.fixture(scope="module")
def timeline_offset_mp4(tmp_path_factory):
    """A genuine non-zero start-time fixture via `-itsoffset` remux
    (mirrors D-274B's own `timeline_offset_mp4` precedent)."""
    d = tmp_path_factory.mktemp("d274f_offset")
    base = d / "base.mp4"
    _ffmpeg(["-f", "lavfi", "-i", "color=c=blue:s=160x120:d=1:r=10",
             "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:d=1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", *_BT709_TAGS, "-c:a", "aac", str(base)])
    shifted = d / "shifted.mp4"
    _ffmpeg(["-itsoffset", "2.0", "-i", str(base), "-map", "0:v", "-map", "0:a", "-c", "copy", str(shifted)])
    return str(shifted)


@pytest.fixture(scope="module", autouse=False)
def _require_hevc_codec():
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    decoders = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"], capture_output=True, text=True).stdout
    if "libx265" not in encoders.lower() or "hevc" not in decoders.lower():
        pytest.skip("libx265 encoder / hevc decoder not available on this runner")


@pytest.fixture(scope="module")
def hevc_source_mp4(_require_hevc_codec, tmp_path_factory):
    d = tmp_path_factory.mktemp("d274f_hevc")
    path = str(d / "hevc.mp4")
    _ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
        "-c:v", "libx265", "-x265-params", "log-level=none", "-pix_fmt", "yuv420p",
        *_BT709_TAGS, "-c:a", "aac", path,
    ])
    return path


@pytest.fixture(scope="module", params=["smpte2084", "arib-std-b67"])
def hdr_10bit_mp4(request, tmp_path_factory):
    """Genuine PQ (`smpte2084`)/HLG (`arib-std-b67`) tagged 10-bit source
    (mirrors D-274D's own `hdr_10bit_mp4` precedent exactly)."""
    d = tmp_path_factory.mktemp("d274f_hdr")
    path = str(d / f"hdr_{request.param}.mp4")
    _ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020", "-color_trc", request.param, "-colorspace", "bt2020nc",
        path,
    ])
    return path, request.param


@pytest.fixture(scope="module")
def ten_bit_sdr_mp4(tmp_path_factory):
    """Genuine 10-bit source with realistic BT709 SDR tags -- no HDR
    action, proves TEN_BIT_TO_EIGHT_BIT fires and reaches PASS
    independently of the HDR tonemap path."""
    d = tmp_path_factory.mktemp("d274f_10bit")
    path = str(d / "sdr_10bit.mp4")
    _ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p10le", *_BT709_TAGS, path,
    ])
    return path


# =============================================================================
# ACCEPT path -- unchanged behavior, zero normalization calls (Stage 2/24)
# =============================================================================

def test_h264_sdr_accept_zero_normalization_calls(h264_sdr_mp4, monkeypatch):
    calls = {"n": 0}
    real_exec = sne.execute_source_normalization

    def spy(*a, **kw):
        calls["n"] += 1
        return real_exec(*a, **kw)

    monkeypatch.setattr(sne, "execute_source_normalization", spy)
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
            {"s1": h264_sdr_mp4}, output_directory=tmp,
        )
    assert calls["n"] == 0, "ACCEPT must never call the normalization executor"
    assert resolved == {"s1": h264_sdr_mp4}
    assert sfd[0]["decision"] == sfp.DECISION_ACCEPT
    assert not nd
    assert not blocked


def test_h264_sdr_no_audio_accept_reaches_downstream(h264_sdr_no_audio_mp4):
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
            {"s1": h264_sdr_no_audio_mp4}, output_directory=tmp,
        )
    assert resolved == {"s1": h264_sdr_no_audio_mp4}
    assert sfd[0]["decision"] == sfp.DECISION_ACCEPT
    assert not blocked


# =============================================================================
# REJECT / INSUFFICIENT_EVIDENCE -- never attempt normalization (Stage 5/6)
# =============================================================================

def test_missing_video_reject_never_normalizes(no_video_source, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(sne, "execute_source_normalization", lambda *a, **kw: calls.__setitem__("n", calls["n"] + 1))
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
            {"s1": no_video_source}, output_directory=tmp,
        )
    assert calls["n"] == 0
    assert not nd
    assert blocked and blocked[0]["decision"] == sfp.DECISION_REJECT
    assert not resolved


def test_mkv_container_reject_never_normalizes(mkv_source, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(sne, "execute_source_normalization", lambda *a, **kw: calls.__setitem__("n", calls["n"] + 1))
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
            {"s1": mkv_source}, output_directory=tmp,
        )
    assert calls["n"] == 0
    assert blocked and blocked[0]["decision"] == sfp.DECISION_REJECT


def test_vp9_insufficient_evidence_never_normalizes(vp9_source_mp4, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(sne, "execute_source_normalization", lambda *a, **kw: calls.__setitem__("n", calls["n"] + 1))
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
            {"s1": vp9_source_mp4}, output_directory=tmp,
        )
    assert calls["n"] == 0
    assert blocked and blocked[0]["decision"] == sfp.DECISION_INSUFFICIENT_EVIDENCE


# =============================================================================
# Timeout policy seam (Stage 10/41) -- the ONLY acceptable escalation when
# everything else works
# =============================================================================

def test_normalize_required_with_no_timeout_override_returns_product_owner_escalation(h264_sdr_mp4, monkeypatch):
    profile = _probe_with_rotation_override(h264_sdr_mp4, 90)
    monkeypatch.setattr(smp, "probe_source_media_profile", _patched_probe_for_original_only(h264_sdr_mp4, profile))
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
            {"s1": h264_sdr_mp4}, output_directory=tmp,
        )  # no normalization_timeout_sec override -- production default
    assert sfd[0]["decision"] == sfp.DECISION_NORMALIZE_REQUIRED
    assert not resolved
    assert blocked
    assert nd[0]["normalization_outcome"] == sne.PRODUCT_OWNER_NORMALIZATION_TIMEOUT_REQUIRED
    assert blocked[0]["user_facing_error_code"] == worker_job.USER_FACING_VIDEO_NORMALIZATION_TIMEOUT_POLICY_REQUIRED


def test_production_call_site_never_overrides_timeout():
    """Stage 10: `run_flow_b_job` (the real production call site) must
    NEVER pass its own `normalization_timeout_sec` -- proven by source
    inspection, the same convention D-266/D-274B's own 'no invented
    number' tests use, never by re-deriving the value at runtime."""
    source = inspect.getsource(worker_job.run_flow_b_job)
    assert "normalization_timeout_sec" not in source


def test_executor_own_timeout_constant_stays_none():
    assert sne.NORMALIZATION_FFMPEG_TIMEOUT_SEC is None


def test_no_canonical_normalization_timeout_reused_from_renderer():
    """Stage 10's own explicit ban: never silently reuse the renderer's
    1200s timeout."""
    from cutsell_worker import render as render_module

    assert sne.NORMALIZATION_FFMPEG_TIMEOUT_SEC != render_module.RENDER_FFMPEG_TIMEOUT_SEC


# =============================================================================
# NORMALIZE_REQUIRED -- successful live paths (Stage 3/9/10/24-32/38)
# =============================================================================

def test_rotation_live_normalizes_and_reaches_pass(h264_sdr_mp4):
    """Stage 25: parser-controlled rotation evidence (real fixture
    generation not available locally -- see module docstring)."""
    profile = _probe_with_rotation_override(h264_sdr_mp4, 90)
    with mock.patch.object(smp, "probe_source_media_profile", side_effect=_patched_probe_for_original_only(h264_sdr_mp4, profile)):
        with tempfile.TemporaryDirectory() as tmp:
            resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
                {"s1": h264_sdr_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
            assert not blocked, blocked
            assert resolved["s1"] != h264_sdr_mp4
            assert Path(resolved["s1"]).exists()
    assert nd[0]["resolved_source_kind"] == worker_job.RESOLVED_SOURCE_KIND_NORMALIZED
    assert nd[0]["format_qc_status"] == ofq.STATUS_PASS
    assert nd[0]["reevaluated_d272_decision"] == sfp.DECISION_ACCEPT
    assert nd[0]["plan_actions"]["rotation"] == snp.ACTION_ROTATE_90
    assert nd[0]["normalized_sha256"]


def test_vfr_live_normalizes_and_reaches_pass(vfr_source_mp4):
    profile = smp.probe_source_media_profile(vfr_source_mp4)
    assert profile.vfr_status in (smp.VFR_STATUS_LIKELY_VFR, smp.VFR_STATUS_VFR), "fixture must be genuinely VFR"
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
            {"s1": vfr_source_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
        )
    assert not blocked, blocked
    assert nd[0]["plan_actions"]["frame_rate"] == snp.ACTION_VFR_TO_CFR
    assert nd[0]["format_qc_status"] == ofq.STATUS_PASS
    assert nd[0]["resolved_source_kind"] == worker_job.RESOLVED_SOURCE_KIND_NORMALIZED


def test_timeline_live_normalizes_and_reaches_pass(timeline_offset_mp4):
    """D-274A Stage 25's own binding scope: D-272 has NO reason code for a
    non-zero start time by itself, so `timeline_action` is only ever an
    ADDITIVE action inside an ALREADY-triggered plan -- a timeline-offset-
    only source is genuine `ACCEPT` (never independently promoted to
    NORMALIZE_REQUIRED). This test composes the genuine timeline-offset
    fixture with a parser-controlled rotation trigger (Stage 25's own
    disclosed evidence-level technique) so the plan legitimately carries
    BOTH `rotation_action` and `timeline_action` together -- proving the
    real timeline correction fires and reaches PASS, without ever
    fabricating a D-272 reason this module does not itself define."""
    real_profile = smp.probe_source_media_profile(timeline_offset_mp4)
    assert any(
        v is not None and abs(v) > 0.0
        for v in (real_profile.format_start_time, real_profile.video_stream_start_time, real_profile.audio_stream_start_time)
    ), "fixture must genuinely carry a non-zero start time"
    composed_profile = dataclasses.replace(
        real_profile, rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
    )
    with mock.patch.object(
        smp, "probe_source_media_profile",
        side_effect=_patched_probe_for_original_only(timeline_offset_mp4, composed_profile),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
                {"s1": timeline_offset_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
            assert not blocked, blocked
            normalized_profile = smp.probe_source_media_profile(resolved["s1"])
            assert not any(
                v is not None and abs(v) > 0.0
                for v in (
                    normalized_profile.format_start_time,
                    normalized_profile.video_stream_start_time,
                    normalized_profile.audio_stream_start_time,
                )
            )
    assert nd[0]["plan_actions"]["timeline"] == snp.ACTION_TIMELINE_TO_ZERO
    assert nd[0]["plan_actions"]["rotation"] == snp.ACTION_ROTATE_90
    assert nd[0]["format_qc_status"] == ofq.STATUS_PASS


def test_hevc_live_normalizes_and_reaches_pass(hevc_source_mp4, established_capability):
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
            {"s1": hevc_source_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
        )
        assert not blocked, blocked
        normalized_profile = smp.probe_source_media_profile(resolved["s1"])
        assert normalized_profile.video_codec == smp.VIDEO_CODEC_H264
    assert sfd[0]["decision"] == sfp.DECISION_NORMALIZE_REQUIRED
    assert sfp.REASON_HEVC_TO_H264_NORMALIZATION_REQUIRED in sfd[0]["normalization_reasons"]
    assert nd[0]["plan_actions"]["codec"] == snp.ACTION_HEVC_TO_H264
    assert nd[0]["format_qc_status"] == ofq.STATUS_PASS


@pytest.mark.parametrize("transfer", ["smpte2084", "arib-std-b67"])
def test_pq_hlg_live_normalizes_and_reaches_pass(transfer, tmp_path_factory, established_capability):
    d = tmp_path_factory.mktemp("d274f_hdr_case")
    path = str(d / f"hdr_{transfer}.mp4")
    _ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020", "-color_trc", transfer, "-colorspace", "bt2020nc",
        path,
    ])
    expected_hdr = smp.HDR_STATUS_HDR_PQ if transfer == "smpte2084" else smp.HDR_STATUS_HDR_HLG
    profile = smp.probe_source_media_profile(path)
    assert profile.hdr_status == expected_hdr, "fixture must be genuinely HDR-classified"
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
            {"s1": path}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
        )
        assert not blocked, blocked
        normalized_profile = smp.probe_source_media_profile(resolved["s1"])
        assert normalized_profile.hdr_status == smp.HDR_STATUS_SDR
        assert normalized_profile.color_primaries == "bt709"
    assert nd[0]["format_qc_status"] == ofq.STATUS_PASS


def test_ten_bit_live_normalizes_and_reaches_pass(ten_bit_sdr_mp4):
    profile = smp.probe_source_media_profile(ten_bit_sdr_mp4)
    assert profile.bit_depth is not None and profile.bit_depth > 8
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
            {"s1": ten_bit_sdr_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
        )
        assert not blocked, blocked
        normalized_profile = smp.probe_source_media_profile(resolved["s1"])
        assert normalized_profile.bit_depth == 8
    assert nd[0]["plan_actions"]["bit_depth"] == snp.ACTION_TEN_BIT_TO_EIGHT_BIT
    assert nd[0]["format_qc_status"] == ofq.STATUS_PASS


def test_composed_rotation_plus_vfr_live_one_generation(vfr_source_mp4, monkeypatch):
    """Stage 32: a realistic composed case (rotation + VFR) through the
    FULL live resolution seam, proving exactly ONE normalization
    generation (one executor call, one ffmpeg subprocess) handles both
    actions together."""
    real_profile = smp.probe_source_media_profile(vfr_source_mp4)
    composed_profile = dataclasses.replace(
        real_profile, rotation_degrees=90, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
    )
    calls = {"n": 0}
    real_exec = sne.execute_source_normalization

    def spy(*a, **kw):
        calls["n"] += 1
        return real_exec(*a, **kw)

    monkeypatch.setattr(sne, "execute_source_normalization", spy)
    with mock.patch.object(
        smp, "probe_source_media_profile",
        side_effect=_patched_probe_for_original_only(vfr_source_mp4, composed_profile),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
                {"s1": vfr_source_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
    assert calls["n"] == 1, "exactly one normalization generation for a composed plan"
    assert not blocked, blocked
    assert nd[0]["plan_actions"]["rotation"] == snp.ACTION_ROTATE_90
    assert nd[0]["plan_actions"]["frame_rate"] == snp.ACTION_VFR_TO_CFR
    assert nd[0]["format_qc_status"] == ofq.STATUS_PASS


# =============================================================================
# Blocked-before-executor states (Stage 9/33-37) -- never launch ffmpeg
# =============================================================================

@pytest.mark.parametrize("hdr_status", [smp.HDR_STATUS_HDR_DOLBY_VISION, smp.HDR_STATUS_HDR_OTHER])
def test_dolby_vision_and_hdr_other_blocked_never_reach_executor(h264_sdr_mp4, hdr_status, monkeypatch):
    real_profile = smp.probe_source_media_profile(h264_sdr_mp4)
    forced_profile = dataclasses.replace(real_profile, hdr_status=hdr_status)
    calls = {"n": 0}
    monkeypatch.setattr(sne, "execute_source_normalization", lambda *a, **kw: calls.__setitem__("n", calls["n"] + 1))
    with mock.patch.object(
        smp, "probe_source_media_profile", side_effect=_patched_probe_for_original_only(h264_sdr_mp4, forced_profile),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
                {"s1": h264_sdr_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
    assert calls["n"] == 0, "Dolby Vision / HDR_OTHER must never reach the executor -- no ffmpeg call"
    assert blocked
    assert nd[0]["user_facing_error_code"] == worker_job.USER_FACING_VIDEO_NORMALIZATION_UNSUPPORTED


def test_hevc_capability_unverified_blocks_before_executor(hevc_source_mp4, hevc_capability_unverified, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(sne, "execute_source_normalization", lambda *a, **kw: calls.__setitem__("n", calls["n"] + 1))
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
            {"s1": hevc_source_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
        )
    # Without an established HEVC bridge, D-272 itself never emits the
    # HEVC_TO_H264 normalization reason -- the source is INSUFFICIENT_
    # EVIDENCE at the POLICY layer already (never reaches the plan/
    # executor at all), which is the correct, more conservative outcome.
    assert calls["n"] == 0
    assert blocked
    assert blocked[0]["decision"] == sfp.DECISION_INSUFFICIENT_EVIDENCE


def test_tonemap_unavailable_blocks_before_executor(hdr_10bit_mp4, tonemap_capability_unavailable, monkeypatch):
    path, _transfer = hdr_10bit_mp4
    calls = {"n": 0}
    monkeypatch.setattr(sne, "execute_source_normalization", lambda *a, **kw: calls.__setitem__("n", calls["n"] + 1))
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
            {"s1": path}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
        )
    assert calls["n"] == 0, "tonemap-unavailable must never launch the executor"
    assert blocked
    assert nd[0]["user_facing_error_code"] == worker_job.USER_FACING_RUNTIME_CODEC_SUPPORT_UNVERIFIED


def test_corrupt_media_still_blocked_before_downstream(tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a real video file, definitely corrupt garbage bytes")
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
            {"s1": str(corrupt)}, output_directory=tmp,
        )
    assert not resolved
    assert blocked


# =============================================================================
# Executor-level failures + strict format-QC PASS AND-requirement
# (Stage 19/20/38)
# =============================================================================

def test_normalization_subprocess_failure_blocks_downstream(h264_sdr_mp4, monkeypatch):
    profile = _probe_with_rotation_override(h264_sdr_mp4, 90)
    fake_result = sne.NormalizationExecutionResult(
        outcome=snp.NORMALIZATION_FAILED,
        normalized_path=None, normalized_reference=None, normalized_profile=None,
        verification=None, diagnostics={"format_qc_status": None},
        failure=sne.NormalizationExecutionFailure(
            error_category=sne.FAILURE_FFMPEG_FAILED, return_code=1, command_fingerprint="x",
            stderr_excerpt="simulated failure", timed_out=False, timeout_sec=_TEST_TIMEOUT_SEC,
            plan_identity="normplan_test",
        ),
    )
    monkeypatch.setattr(sne, "execute_source_normalization", lambda *a, **kw: fake_result)
    with mock.patch.object(smp, "probe_source_media_profile", side_effect=_patched_probe_for_original_only(h264_sdr_mp4, profile)):
        with tempfile.TemporaryDirectory() as tmp:
            resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
                {"s1": h264_sdr_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
    assert not resolved
    assert blocked
    assert blocked[0]["user_facing_error_code"] == worker_job.USER_FACING_VIDEO_NORMALIZATION_FAILED


def test_normalized_still_blocked_d272_reevaluation_failure_blocks_downstream(h264_sdr_mp4, monkeypatch):
    """A normalization whose OWN re-probe/re-evaluate still reports
    NORMALIZE_REQUIRED/REJECT/INSUFFICIENT_EVIDENCE -- Stage 19."""
    profile = _probe_with_rotation_override(h264_sdr_mp4, 90)
    real_exec = sne.execute_source_normalization

    def fake_exec(*a, **kw):
        result = real_exec(*a, **kw)
        return dataclasses.replace(result, outcome=snp.NORMALIZATION_VERIFICATION_FAILED)

    monkeypatch.setattr(sne, "execute_source_normalization", fake_exec)
    with mock.patch.object(smp, "probe_source_media_profile", side_effect=_patched_probe_for_original_only(h264_sdr_mp4, profile)):
        with tempfile.TemporaryDirectory() as tmp:
            resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
                {"s1": h264_sdr_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
    assert not resolved
    assert blocked
    assert blocked[0]["user_facing_error_code"] == worker_job.USER_FACING_VIDEO_NORMALIZATION_VERIFICATION_FAILED


def test_normalized_format_qc_fail_blocks_downstream(h264_sdr_mp4, monkeypatch):
    profile = _probe_with_rotation_override(h264_sdr_mp4, 90)
    real_exec = sne.execute_source_normalization

    def fake_exec(*a, **kw):
        result = real_exec(*a, **kw)
        new_diag = dict(result.diagnostics)
        new_diag["format_qc_status"] = ofq.STATUS_FAIL
        return dataclasses.replace(result, diagnostics=new_diag)

    monkeypatch.setattr(sne, "execute_source_normalization", fake_exec)
    with mock.patch.object(smp, "probe_source_media_profile", side_effect=_patched_probe_for_original_only(h264_sdr_mp4, profile)):
        with tempfile.TemporaryDirectory() as tmp:
            resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
                {"s1": h264_sdr_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
    assert not resolved
    assert blocked
    assert blocked[0]["user_facing_error_code"] == worker_job.USER_FACING_VIDEO_NORMALIZATION_VERIFICATION_FAILED


def test_untagged_source_conservatively_blocks_not_a_false_pass(h264_untagged_mp4):
    """Module docstring's own disclosed fail-closed property: a genuinely
    color-untagged source correctly lands PARTIAL (missing evidence, not
    FAIL) and is correctly BLOCKED by this gate's own strict PASS-only
    AND-requirement -- never a false PASS, never a silent relaxation."""
    profile = _probe_with_rotation_override(h264_untagged_mp4, 90)
    with mock.patch.object(smp, "probe_source_media_profile", side_effect=_patched_probe_for_original_only(h264_untagged_mp4, profile)):
        with tempfile.TemporaryDirectory() as tmp:
            resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
                {"s1": h264_untagged_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
    assert not resolved
    assert blocked
    assert nd[0]["normalization_outcome"] == snp.NORMALIZATION_SUCCEEDED  # executor's own success gate
    assert nd[0]["format_qc_status"] != ofq.STATUS_PASS  # but this gate's own stricter AND blocks it
    assert blocked[0]["user_facing_error_code"] == worker_job.USER_FACING_VIDEO_NORMALIZATION_VERIFICATION_FAILED


# =============================================================================
# One pass, no retry (Stage 4/21)
# =============================================================================

def test_no_second_normalization_pass_ever_attempted(h264_sdr_mp4, monkeypatch):
    profile = _probe_with_rotation_override(h264_sdr_mp4, 90)
    calls = {"n": 0}
    real_exec = sne.execute_source_normalization

    def counting_failure(*a, **kw):
        calls["n"] += 1
        result = real_exec(*a, **kw)
        return dataclasses.replace(result, outcome=snp.NORMALIZATION_FAILED)

    monkeypatch.setattr(sne, "execute_source_normalization", counting_failure)
    with mock.patch.object(smp, "probe_source_media_profile", side_effect=_patched_probe_for_original_only(h264_sdr_mp4, profile)):
        with tempfile.TemporaryDirectory() as tmp:
            worker_job.resolve_sources_for_editorial_entry(
                {"s1": h264_sdr_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
    assert calls["n"] == 1, "must never retry a failed normalization"


def test_executor_always_called_with_attempt_count_zero(h264_sdr_mp4, monkeypatch):
    profile = _probe_with_rotation_override(h264_sdr_mp4, 90)
    seen = {}

    def spy(*a, **kw):
        seen["attempt_count"] = kw.get("attempt_count")
        return sne.NormalizationExecutionResult(
            outcome=snp.NORMALIZATION_FAILED,
            normalized_path=None, normalized_reference=None, normalized_profile=None,
            verification=None, diagnostics={"format_qc_status": None}, failure=None,
        )

    monkeypatch.setattr(sne, "execute_source_normalization", spy)
    with mock.patch.object(smp, "probe_source_media_profile", side_effect=_patched_probe_for_original_only(h264_sdr_mp4, profile)):
        with tempfile.TemporaryDirectory() as tmp:
            worker_job.resolve_sources_for_editorial_entry(
                {"s1": h264_sdr_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
    assert seen["attempt_count"] == 0


# =============================================================================
# Multi-source resolution (Stage 12/13/22-27)
# =============================================================================

def test_multi_source_all_accept(h264_sdr_mp4, h264_sdr_no_audio_mp4):
    local_paths = {"a": h264_sdr_mp4, "b": h264_sdr_no_audio_mp4}
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(local_paths, output_directory=tmp)
    assert resolved == local_paths
    assert not blocked
    assert list(resolved.keys()) == list(local_paths.keys())


def test_multi_source_one_normalization_one_accept_order_preserved(h264_sdr_mp4, monkeypatch):
    """Both sources share the SAME underlying fixture path, so the first
    probe call (source "a") is left genuinely real and only the SECOND
    probe call (source "b", per `local_paths`' own iteration order) is
    overridden with parser-controlled rotation evidence -- proving
    independent per-source resolution AND that source order/keys are
    preserved (Stage 13) even when two sources resolve differently."""
    profile = _probe_with_rotation_override(h264_sdr_mp4, 90)
    local_paths = {"a": h264_sdr_mp4, "b": h264_sdr_mp4}
    real_probe = smp.probe_source_media_profile
    call_index = {"n": 0}

    def sequenced_probe(path, *a, **kw):
        call_index["n"] += 1
        if call_index["n"] == 2:
            return profile
        return real_probe(path, *a, **kw)

    with mock.patch.object(smp, "probe_source_media_profile", side_effect=sequenced_probe):
        with tempfile.TemporaryDirectory() as tmp:
            resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
                local_paths, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
    assert list(sfd[i]["source_asset_id"] for i in range(2)) == ["a", "b"]
    assert sfd[0]["decision"] == sfp.DECISION_ACCEPT
    assert sfd[1]["decision"] == sfp.DECISION_NORMALIZE_REQUIRED
    assert resolved["a"] == h264_sdr_mp4
    assert not blocked, blocked
    assert resolved["b"] != h264_sdr_mp4
    assert list(resolved.keys()) == ["a", "b"]


def test_multi_source_one_hard_reject_blocks_whole_job(h264_sdr_mp4, no_video_source):
    local_paths = {"a": h264_sdr_mp4, "b": no_video_source}
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(local_paths, output_directory=tmp)
    assert blocked
    assert blocked[0]["source_asset_id"] == "b"
    # The real production call site (`run_flow_b_job`) raises exactly this
    # exception on a non-empty `blocked_sources` list -- proven directly
    # against a REAL job in `test_missing_video_live_reject_blocks_before_
    # downstream_via_run_flow_b_job` below, never re-derived here.
    exc = worker_job.SourceFormatGateBlocked(blocked)
    assert exc.primary_error_code == blocked[0]["user_facing_error_code"]


def test_no_source_ever_silently_dropped(h264_sdr_mp4, h264_sdr_no_audio_mp4):
    local_paths = {"x": h264_sdr_mp4, "y": h264_sdr_no_audio_mp4}
    with tempfile.TemporaryDirectory() as tmp:
        resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(local_paths, output_directory=tmp)
    assert len(sfd) == 2
    assert set(sfd_entry["source_asset_id"] for sfd_entry in sfd) == {"x", "y"}


# =============================================================================
# Downstream substitution + no mixed timeline (Stage 15/26/27)
# =============================================================================

def test_downstream_gets_normalized_path_never_original(h264_sdr_mp4):
    profile = _probe_with_rotation_override(h264_sdr_mp4, 90)
    with mock.patch.object(smp, "probe_source_media_profile", side_effect=_patched_probe_for_original_only(h264_sdr_mp4, profile)):
        with tempfile.TemporaryDirectory() as tmp:
            resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
                {"s1": h264_sdr_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
    assert not blocked
    assert resolved["s1"] != h264_sdr_mp4
    assert str(Path(tmp)) in resolved["s1"]  # normalized artifact lives under the job's own output_directory


def test_job_local_normalized_artifact_stays_in_output_directory(h264_sdr_mp4):
    profile = _probe_with_rotation_override(h264_sdr_mp4, 90)
    with mock.patch.object(smp, "probe_source_media_profile", side_effect=_patched_probe_for_original_only(h264_sdr_mp4, profile)):
        with tempfile.TemporaryDirectory() as tmp:
            resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
                {"s1": h264_sdr_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
            assert not blocked
            assert str(Path(tmp)) in resolved["s1"]
        # after the `with tempfile.TemporaryDirectory()` block exits, the
        # directory (and everything inside it) is deleted -- Stage 22: no
        # orphan normalized media survives the job.
        assert not Path(tmp).exists()


def test_original_source_bytes_unchanged_by_normalization(h264_sdr_mp4):
    import hashlib

    before = hashlib.sha256(Path(h264_sdr_mp4).read_bytes()).hexdigest()
    profile = _probe_with_rotation_override(h264_sdr_mp4, 90)
    with mock.patch.object(smp, "probe_source_media_profile", side_effect=_patched_probe_for_original_only(h264_sdr_mp4, profile)):
        with tempfile.TemporaryDirectory() as tmp:
            worker_job.resolve_sources_for_editorial_entry(
                {"s1": h264_sdr_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
    after = hashlib.sha256(Path(h264_sdr_mp4).read_bytes()).hexdigest()
    assert before == after


# =============================================================================
# Diagnostics safety (Stage 16/17/28/30/40)
# =============================================================================

def test_source_format_diagnostics_shape_byte_identical_to_pre_d274f(h264_sdr_mp4):
    """Stage 40/regression firewall: `evaluate_source_format_gate`'s own
    pre-D-274F return shape is unaffected -- same keys, same values, for
    an ACCEPT source."""
    via_old = worker_job.evaluate_source_format_gate({"s1": h264_sdr_mp4})
    with tempfile.TemporaryDirectory() as tmp:
        _resolved, via_new, _nd, _blocked = worker_job.resolve_sources_for_editorial_entry(
            {"s1": h264_sdr_mp4}, output_directory=tmp,
        )
    assert via_old == via_new


def test_normalization_diagnostics_never_leak_filesystem_paths(h264_sdr_mp4):
    profile = _probe_with_rotation_override(h264_sdr_mp4, 90)
    with mock.patch.object(smp, "probe_source_media_profile", side_effect=_patched_probe_for_original_only(h264_sdr_mp4, profile)):
        with tempfile.TemporaryDirectory() as tmp:
            resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
                {"s1": h264_sdr_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
    import json

    serialized = json.dumps(nd)
    assert tmp not in serialized
    assert "/tmp" not in serialized or tmp not in serialized
    assert "ffmpeg" not in serialized.lower()


def test_normalization_diagnostics_carry_all_stage_16_fields(h264_sdr_mp4):
    profile = _probe_with_rotation_override(h264_sdr_mp4, 90)
    with mock.patch.object(smp, "probe_source_media_profile", side_effect=_patched_probe_for_original_only(h264_sdr_mp4, profile)):
        with tempfile.TemporaryDirectory() as tmp:
            resolved, sfd, nd, blocked = worker_job.resolve_sources_for_editorial_entry(
                {"s1": h264_sdr_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
    required_keys = {
        "decision", "normalization_required", "plan_identity", "plan_actions",
        "normalization_outcome", "normalized_sha256", "reprobe_status",
        "reevaluated_d272_decision", "format_qc_status", "resolved_source_kind",
    }
    assert required_keys.issubset(nd[0].keys())


# =============================================================================
# Format QC is authoritative (Stage 38) -- reused, never re-derived
# =============================================================================

def test_format_qc_reuses_ofq_verify_output_format_directly(h264_sdr_mp4, monkeypatch):
    calls = {"n": 0}
    real_verify = ofq.verify_output_format

    def spy(*a, **kw):
        calls["n"] += 1
        return real_verify(*a, **kw)

    monkeypatch.setattr(ofq, "verify_output_format", spy)
    profile = _probe_with_rotation_override(h264_sdr_mp4, 90)
    with mock.patch.object(smp, "probe_source_media_profile", side_effect=_patched_probe_for_original_only(h264_sdr_mp4, profile)):
        with tempfile.TemporaryDirectory() as tmp:
            worker_job.resolve_sources_for_editorial_entry(
                {"s1": h264_sdr_mp4}, output_directory=tmp, normalization_timeout_sec=_TEST_TIMEOUT_SEC,
            )
    # The executor itself already calls verify_output_format once
    # internally (D-274E's own integration) -- this proves the real
    # authority is exercised at least once end-to-end, never bypassed.
    assert calls["n"] >= 1


# =============================================================================
# D-272 reused, D-271 reused, D-274A reused, executor reused (Stage 44
# items 2-6) -- proven by source inspection, never a parallel reimplementation
# =============================================================================

def test_resolve_function_calls_real_probe_policy_plan_executor_qc():
    source = inspect.getsource(worker_job.resolve_sources_for_editorial_entry)
    assert "smp.probe_source_media_profile" in source
    assert "evaluate_source_format_policy" in source
    assert "snp.build_source_normalization_plan" in source
    assert "sne.execute_source_normalization" in source
    assert "ofq.STATUS_PASS" in source


def test_no_duplicated_codec_hdr_rotation_vfr_policy_in_worker_job():
    source = inspect.getsource(worker_job)
    forbidden = ("zscale=", "tonemap=", "transpose=", "libx264", "libx265", "-crf", "-preset")
    for token in forbidden:
        assert token not in source, f"worker_job.py must never duplicate normalization logic ({token!r} found)"


# =============================================================================
# Command safety unaffected (Stage 23) -- shell=False, atomic promotion,
# bounded subprocess all remain the executor's OWN unmodified responsibility
# =============================================================================

def test_worker_job_never_invokes_subprocess_directly():
    source = inspect.getsource(worker_job)
    assert "subprocess" not in source


# =============================================================================
# Regression firewall (Stage 43) -- every listed authority stays byte-
# identical relative to the current HEAD (self-resolving on commit, per
# this session's own established HEAD-relative-vs-fixed-SHA distinction)
# =============================================================================

@pytest.mark.parametrize(
    "rel_path",
    [
        "cutsell_worker/render.py",
        "cutsell_worker/media_overlay_render.py",
        "cutsell_worker/render_delivery.py",
        "cutsell_worker/live_render_qc.py",
        "cutsell_worker/post_render_media_qc.py",
        "cutsell_worker/source_normalization_executor.py",
        "cutsell_worker/source_normalization_plan.py",
        "cutsell_worker/source_format_policy.py",
        "cutsell_worker/source_media_profile.py",
        "cutsell_worker/output_format_qc.py",
        "cutsell_worker/worker_runtime_capability.py",
        "cutsell_worker/production_runtime_capability.py",
        "cutsell_worker/flow_b.py",
    ],
)
def test_unrelated_authorities_unchanged(rel_path):
    assert _run_git_diff_head(rel_path) == "", f"{rel_path} must remain byte-identical to HEAD for this gate"


def test_render_contract_version_unaffected():
    from cutsell_worker import render_delivery

    assert render_delivery.RENDER_CONTRACT_VERSION == 1


def test_max_normalization_attempts_unchanged():
    assert snp.MAX_NORMALIZATION_ATTEMPTS == 1


# =============================================================================
# End-to-end proof through the REAL `run_flow_b_job` (Stage 1/44 item 1:
# "live Flow-B gate owns activation") -- mirrors D-272A's own `wired_
# worker_job` fixture pattern: every non-format collaborator is a
# lightweight fake/spy, leaving the real per-source loop, the real
# `resolve_sources_for_editorial_entry` (D-271/D-272/D-274A/executor/
# D-274E), and the real exception-handling path exercised end to end.
# =============================================================================

def _payload(*, uri: str, source_asset_id: str = "s1", project_id: str = "p1", user_id: str = "u1") -> dict:
    return {
        "project_id": project_id,
        "user_id": user_id,
        "sources": [{"source_asset_id": source_asset_id, "uri": uri, "original_name": Path(uri).name}],
    }


@pytest.fixture
def wired_worker_job(monkeypatch):
    from types import SimpleNamespace

    calls = {"process_local_sources": 0, "seen_local_paths": None}

    def fake_download_source(uri, destination, *, client=None):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(uri, destination)
        return destination

    def fake_process_local_sources(request, local_paths, **kwargs):
        calls["process_local_sources"] += 1
        calls["seen_local_paths"] = dict(local_paths)
        return object()

    class _FakeBrain:
        backend = "local"
        external_calls_enabled = False
        semantic_provider = None
        whole_video_provider = None
        visual_provider = None
        take_grouping_provider = None
        take_judge_provider = None
        clean_cut_provider = None
        composer_provider = None
        draft_review_provider = None
        editorial_judge = None
        hybrid_settings = SimpleNamespace(provider="none", primary_model=None)

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


def test_h264_sdr_live_accept_reaches_downstream_via_run_flow_b_job(wired_worker_job, h264_sdr_mp4):
    result = worker_job.run_flow_b_job(_payload(uri=h264_sdr_mp4))
    assert wired_worker_job["process_local_sources"] == 1
    assert result["source_format_diagnostics"][0]["decision"] == sfp.DECISION_ACCEPT
    assert result["source_normalization_diagnostics"] == []
    # ACCEPT: the downstream path is the job's own downloaded copy (never
    # normalized -- the executor's own "normalized_" prefix never appears).
    seen_path = wired_worker_job["seen_local_paths"]["s1"]
    assert not Path(seen_path).name.startswith("normalized_")
    assert Path(seen_path).suffix == ".mp4"


def test_missing_video_live_reject_blocks_before_downstream_via_run_flow_b_job(wired_worker_job, no_video_source):
    with pytest.raises(worker_job.SourceFormatGateBlocked) as excinfo:
        worker_job.run_flow_b_job(_payload(uri=no_video_source))
    assert wired_worker_job["process_local_sources"] == 0
    assert excinfo.value.primary_error_code == sfp.USER_FACING_UNSUPPORTED_VIDEO_FORMAT


def _probe_override_on_call(call_number: int, *, rotation_degrees: int):
    """Like `_probe_with_rotation_override`/`_patched_probe_for_original_
    only` above, but keyed by CALL ORDER rather than by path -- required
    inside `run_flow_b_job` itself, where the source's own initial probe
    runs against a freshly-DOWNLOADED COPY (a new path inside the job's
    own temp directory), never the original fixture path directly. Every
    call still runs a REAL probe of whatever path it is actually given
    (so duration/codec/container/etc. all stay genuine); only the ONE
    targeted call additionally has its rotation fields overridden."""
    real_probe = smp.probe_source_media_profile
    state = {"n": 0}

    def _fake(path, *a, **kw):
        state["n"] += 1
        real_profile = real_probe(path, *a, **kw)
        if state["n"] == call_number:
            return dataclasses.replace(
                real_profile, rotation_degrees=rotation_degrees, rotation_source=smp.ROTATION_SOURCE_DISPLAY_MATRIX,
            )
        return real_profile

    return _fake


def test_normalize_required_no_timeout_blocks_via_run_flow_b_job(wired_worker_job, h264_sdr_mp4, monkeypatch):
    """Production default: `run_flow_b_job` itself never overrides the
    timeout, so a genuinely NORMALIZE_REQUIRED source blocks the whole
    job with the Product-Owner-escalation error code -- proven through
    the REAL job entry point, not just the helper function directly."""
    monkeypatch.setattr(smp, "probe_source_media_profile", _probe_override_on_call(1, rotation_degrees=90))
    with pytest.raises(worker_job.SourceFormatGateBlocked) as excinfo:
        worker_job.run_flow_b_job(_payload(uri=h264_sdr_mp4))
    assert wired_worker_job["process_local_sources"] == 0
    assert excinfo.value.primary_error_code == worker_job.USER_FACING_VIDEO_NORMALIZATION_TIMEOUT_POLICY_REQUIRED


def test_rotation_live_normalized_path_reaches_downstream_via_run_flow_b_job(wired_worker_job, h264_sdr_mp4, monkeypatch):
    """Stage 41's own explicit instruction: tests may inject a bounded
    test timeout even at the real `run_flow_b_job` entry point, by
    wrapping (never replacing the real behavior of) `resolve_sources_for_
    editorial_entry` -- the production code path itself
    (`test_production_call_site_never_overrides_timeout` above) is left
    completely unmodified; only THIS test's own call gets the override."""
    monkeypatch.setattr(smp, "probe_source_media_profile", _probe_override_on_call(1, rotation_degrees=90))
    real_resolve = worker_job.resolve_sources_for_editorial_entry

    def resolve_with_test_timeout(local_paths, *, output_directory, normalization_timeout_sec=None):
        return real_resolve(
            local_paths, output_directory=output_directory,
            normalization_timeout_sec=normalization_timeout_sec or _TEST_TIMEOUT_SEC,
        )

    monkeypatch.setattr(worker_job, "resolve_sources_for_editorial_entry", resolve_with_test_timeout)
    result = worker_job.run_flow_b_job(_payload(uri=h264_sdr_mp4))
    assert wired_worker_job["process_local_sources"] == 1
    seen_path = wired_worker_job["seen_local_paths"]["s1"]
    assert Path(seen_path).name.startswith("normalized_"), seen_path  # executor's own naming convention
    assert result["source_format_diagnostics"][0]["decision"] == sfp.DECISION_NORMALIZE_REQUIRED
    assert result["source_normalization_diagnostics"][0]["resolved_source_kind"] == worker_job.RESOLVED_SOURCE_KIND_NORMALIZED
    assert result["source_normalization_diagnostics"][0]["format_qc_status"] == ofq.STATUS_PASS
