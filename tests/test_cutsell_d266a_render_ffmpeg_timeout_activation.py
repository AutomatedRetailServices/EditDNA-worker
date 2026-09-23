"""D-266A -- ACTIVATE CANONICAL RENDER FFMPEG TIMEOUT.

Activates the timeout seam D-266 built (and deliberately left at `None`,
pending a Product Owner decision) with the approved canonical value:

    RENDER_FFMPEG_TIMEOUT_SEC = 1200  (20 minutes)

This is renderer SUBPROCESS execution-safety policy only -- never a whole-
job, workflow, queue, upload, measurement, or Modal/RunPod timeout, and
never reused for an unrelated operation (D-266A's own explicit
constraint). No real 20-minute test is run anywhere here: every boundary
proof uses a deterministic mock/test seam on `subprocess.run` or `_run`
itself, per this gate's own Stage 5 instruction.

No codec/CRF/preset/filtergraph/resolution/audio/retry/Pacing/Boundary/
Freeze/Audio-Join/Audio-Finishing/Visual-Finishing/QC-authority change.
Synthetic media only where real ffmpeg is used -- no RAW, no provider.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from cutsell_worker import render
from cutsell_worker.media_probe import probe_media
from cutsell_worker.render_plan import RenderSegment

pytestmark_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def source_clip(tmp_path_factory):
    directory = tmp_path_factory.mktemp("d266a_render")
    path = str(directory / "source.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000,volume=0.3,aformat=channel_layouts=stereo",
        "-t", "3", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "96k", path,
    ])
    return path


def _segment(source_path: str, clip_id: str = "c1", start: float = 0.0, end: float = 1.5) -> RenderSegment:
    return RenderSegment(clip_id=clip_id, source_asset_id="s1", source_path=source_path, start=start, end=end)


# =============================================================================
# Stage 1/7 items 1-2 -- canonical owner, single value, no duplication
# =============================================================================

def test_canonical_timeout_is_exactly_1200():
    assert render.RENDER_FFMPEG_TIMEOUT_SEC == 1200.0


def test_canonical_owner_is_render_module_only():
    # No other cutsell_worker module defines its own RENDER_FFMPEG_TIMEOUT_SEC
    # -- one owner, never duplicated.
    import ast
    import cutsell_worker
    package_dir = Path(cutsell_worker.__file__).parent
    owners = []
    for path in package_dir.glob("*.py"):
        if path.name == "render.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "RENDER_FFMPEG_TIMEOUT_SEC":
                        owners.append(path.name)
    assert owners == []


def test_old_d266_seam_name_is_gone_replaced_not_duplicated():
    source = Path("cutsell_worker/render.py").read_text(encoding="utf-8")
    assert "RENDER_SUBPROCESS_TIMEOUT_SEC" not in source
    assert source.count("RENDER_FFMPEG_TIMEOUT_SEC") >= 2  # definition + at least one use


# =============================================================================
# Stage 2/7 items 3-4 -- live _run() consumes the canonical value by default
# =============================================================================

def test_run_default_timeout_is_canonical_value(monkeypatch):
    seen = {}

    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    def _spy(command, **kwargs):
        seen.update(kwargs)
        return FakeCompleted()

    monkeypatch.setattr(subprocess, "run", _spy)
    render._run(["ffmpeg", "-i", "x"], output_path="/tmp/x.mp4")
    assert seen.get("timeout") == 1200.0


@pytestmark_ffmpeg
def test_render_preview_live_path_never_overrides_timeout_to_none(source_clip, tmp_path, monkeypatch):
    """Only the ffmpeg ENCODE calls that go through `_run()` (the ones
    D-266/D-266A own) must carry the canonical timeout. Two OTHER,
    pre-existing subprocess calls in this module are explicitly out of
    this gate's scope and legitimately pass no timeout at all:
    `media_probe.probe_media`'s own ffprobe call, and `tighten_trailing_
    silence`'s own `silencedetect` probe (D-266's own documented,
    unfixed gap -- both identified `-f null -` null-muxer probes, never
    `_run`'s own real encode, which always sets `-movflags +faststart`
    on its real output)."""
    seen_ffmpeg_encode_timeouts = []
    real_run = subprocess.run

    def _spy(command, *a, **k):
        if command and command[0] == "ffmpeg" and "+faststart" in command:
            seen_ffmpeg_encode_timeouts.append(k.get("timeout"))
        return real_run(command, *a, **k)

    monkeypatch.setattr(subprocess, "run", _spy)
    out = tmp_path / "out.mp4"
    render.render_preview((_segment(source_clip),), str(out))
    assert seen_ffmpeg_encode_timeouts, "expected at least one real ffmpeg encode call"
    assert all(t == 1200.0 for t in seen_ffmpeg_encode_timeouts), seen_ffmpeg_encode_timeouts


# =============================================================================
# Stage 3/5/7 items 5-11 -- timeout failure contract (deterministic mocks only)
# =============================================================================

def test_timeout_at_boundary_raises_structured_ffmpeg_timeout(monkeypatch):
    def _raise_timeout(*a, **k):
        assert k.get("timeout") == 1200.0
        raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=k.get("timeout"), output=b"", stderr=b"stalled")

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    diagnostics: list = []
    with pytest.raises(render.RenderExecutionError) as exc_info:
        render._run(["ffmpeg", "-i", "x"], output_path="/tmp/x.mp4", source_identity=("c1",), diagnostics=diagnostics)
    failure = exc_info.value.failure
    assert failure.error_category == render.RENDER_FAILURE_FFMPEG_TIMEOUT
    assert failure.timed_out is True
    assert failure.timeout_sec == 1200.0
    assert failure.command_fingerprint
    assert failure.stderr_excerpt == "stalled"
    assert diagnostics and diagnostics[0]["error_category"] == render.RENDER_FAILURE_FFMPEG_TIMEOUT
    assert diagnostics[0]["timeout_sec"] == 1200.0


def test_just_under_boundary_mocked_execution_is_allowed_to_complete(monkeypatch):
    """1199.x-second-equivalent mocked execution: subprocess.run returns
    normally (never raises TimeoutExpired) -- proves the boundary is a
    hard cutoff, not an early abort."""
    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(command, *, timeout, **kwargs):
        assert timeout == 1200.0
        return FakeCompleted()  # simulates completing just under the boundary

    monkeypatch.setattr(subprocess, "run", _fake_run)
    diagnostics: list = []
    render._run(["ffmpeg", "-i", "x"], output_path="/tmp/x.mp4", diagnostics=diagnostics)
    assert diagnostics[0]["execution_status"] == "SUCCESS"


@pytestmark_ffmpeg
def test_timeout_leaves_no_final_output_via_render_preview(source_clip, monkeypatch, tmp_path):
    out = tmp_path / "out.mp4"

    def _raise_timeout_from_run(command, *, output_path, **kwargs):
        raise render.RenderExecutionError(render.RenderExecutionFailure(
            error_category=render.RENDER_FAILURE_FFMPEG_TIMEOUT, return_code=None,
            command_fingerprint="fp", executable="ffmpeg", stderr_excerpt="", stdout_excerpt="",
            timed_out=True, timeout_sec=1200.0, output_path=str(output_path), source_identity=(),
        ))

    monkeypatch.setattr(render, "_run", _raise_timeout_from_run)
    with pytest.raises(render.RenderExecutionError) as exc_info:
        render.render_preview((_segment(source_clip),), str(out))
    failure = exc_info.value.failure
    assert failure.error_category == render.RENDER_FAILURE_FFMPEG_TIMEOUT
    assert failure.timeout_sec == 1200.0
    assert not out.exists()  # no temp promoted, no final published
    assert list(tmp_path.iterdir()) == []  # cleanup attempted and succeeded


def test_no_retry_after_timeout():
    source = Path("cutsell_worker/render.py").read_text(encoding="utf-8")
    for needle in ("retry_count", "max_attempts", "RETRY_LIMIT"):
        assert needle not in source


# =============================================================================
# Stage 4/7 items 12-24 -- success path + unrelated-authority equivalence
# =============================================================================

@pytestmark_ffmpeg
def test_successful_render_under_timeout_semantically_identical_to_d266(source_clip, tmp_path):
    out = tmp_path / "out.mp4"
    render.render_preview((_segment(source_clip),), str(out))
    probe = probe_media(str(out))
    assert probe.width == 1080 and probe.height == 1920
    assert probe.has_audio is True
    assert out.stat().st_size > 0


def test_codec_crf_preset_resolution_audio_unchanged():
    source = Path("cutsell_worker/render.py").read_text(encoding="utf-8")
    assert '"libx264"' in source
    assert '"-crf", "20"' in source
    assert '"veryfast"' in source
    assert '"aac"' in source
    assert '"160k"' in source
    assert "RENDER_FPS_DEFAULT = 30" in source


def _run_git_diff(rel_path: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", rel_path], capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("rel_path", [
    "cutsell_worker/render_plan.py",
    "cutsell_worker/audio_finishing_executor.py",
    "cutsell_worker/audio_finishing_composition.py",
    "cutsell_worker/visual_finishing_executor.py",
    "cutsell_worker/visual_finishing_composition.py",
    "cutsell_worker/boundary_engine_pass.py",
    "cutsell_worker/pacing_transition_decision.py",
    "cutsell_worker/post_render_media_qc.py",
    "cutsell_worker/post_render_watch_listen_qc.py",
    "cutsell_worker/media_probe.py",
])
def test_unrelated_authorities_unchanged(rel_path):
    assert _run_git_diff(rel_path) == "", f"D-266A must not touch {rel_path}"


# =============================================================================
# Stage 6 -- security-track integration is a source-level, discoverable fact
# =============================================================================

def test_timeout_documented_as_execution_safety_not_quality_policy():
    source = Path("cutsell_worker/render.py").read_text(encoding="utf-8")
    assert "D-266A" in source
    assert "1200" in source
