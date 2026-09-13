"""D-266 -- RENDER EXECUTION SAFETY FOUNDATION.

Offline hardening only, closing exactly the three P0 findings D-265's
forensic audit named in `render.py`'s live execution path:

1. OBSERVABILITY -- `_run()` used to discard ffmpeg's stderr/command and
   raise a bare `RuntimeError("ffmpeg_render_failed")`. It now raises a
   structured `RenderExecutionError` carrying a bounded stderr/stdout
   excerpt, return code, timeout flag, a deterministic command fingerprint,
   and output/source identity.
2. HUNG PROCESS / DOS -- the live render subprocess had no timeout. A
   `timeout_sec` seam now exists on every subprocess invocation in this
   module; per D-266 Stage 1/6 no existing repository convention was found
   sized for a full multi-segment render encode, so the seam is left at
   `None` (today's exact prior behavior) pending a Product Owner decision
   -- see docs/CUTSELL_DECISIONS.md D-266
   (TIMEOUT_POLICY_PENDING_PRODUCT_OWNER). `subprocess.TimeoutExpired` is
   still caught and reported through the SAME structured failure path.
3. PARTIAL FINAL OUTPUT -- the renderer used to write directly to the
   caller's final `output_path`. It now always writes to a unique,
   job-local temp file in the SAME directory (required for `os.replace`
   atomicity), validates it (exists, non-empty), and atomically promotes
   it. `output_path` is written exactly once, only on confirmed success.

No codec, filter, resolution, fps, audio parameter, retry policy, or
selection/boundary/pacing/finishing behavior is changed. Synthetic media
only -- no Video00 wording, timestamps or clip ids, no RAW, no provider.
"""
from __future__ import annotations

import ast
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutsell_worker import render
from cutsell_worker.media_probe import probe_media
from cutsell_worker.render_plan import RenderSegment

pytestmark_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def source_clip(tmp_path_factory):
    directory = tmp_path_factory.mktemp("d266_render")
    path = str(directory / "source.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000,volume=0.3,aformat=channel_layouts=stereo",
        "-t", "6", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "96k", path,
    ])
    return path


def _segment(source_path: str, clip_id: str = "c1", start: float = 0.0, end: float = 2.0, **kwargs) -> RenderSegment:
    return RenderSegment(
        clip_id=clip_id, source_asset_id="s1", source_path=source_path, start=start, end=end, **kwargs,
    )


def _source_without_docstrings(path: str) -> str:
    """D-262/D-263's own established false-positive-proofing technique: a
    module's scope-discipline PROSE legitimately names a forbidden word
    while describing that the code does NOT do that thing (e.g. this
    file's own render.py docstring saying "no boto3 ... anywhere in this
    file"). Strip every module/function/class docstring via `ast` before
    scanning for forbidden vocabulary, so only real code is checked."""
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


def _synthetic_failure(*, category=render.RENDER_FAILURE_FFMPEG_NONZERO_EXIT, return_code=1, timed_out=False, timeout_sec=None):
    return render.RenderExecutionFailure(
        error_category=category, return_code=return_code, command_fingerprint="deadbeefdeadbeefdeadbeef",
        executable="ffmpeg", stderr_excerpt="synthetic failure", stdout_excerpt="",
        timed_out=timed_out, timeout_sec=timeout_sec, output_path="", source_identity=(),
    )


# =============================================================================
# Stage 16/25/26 -- success path equivalence (real ffmpeg, real filtergraph)
# =============================================================================

@pytestmark_ffmpeg
def test_success_path_produces_expected_geometry(source_clip, tmp_path):
    out = tmp_path / "out.mp4"
    result = render.render_preview((_segment(source_clip),), str(out))
    assert result == str(out)
    probe = probe_media(str(out))
    assert probe.width == 1080 and probe.height == 1920
    assert probe.has_audio is True
    assert probe.duration_sec == pytest.approx(2.0, abs=0.15)


@pytestmark_ffmpeg
def test_success_path_multi_segment_concat_unchanged(source_clip, tmp_path):
    out = tmp_path / "out.mp4"
    segs = (_segment(source_clip, "a", 0.0, 1.0), _segment(source_clip, "b", 1.0, 2.5))
    render.render_preview(segs, str(out))
    probe = probe_media(str(out))
    assert probe.duration_sec == pytest.approx(1.0 + 1.5, abs=0.15)


@pytestmark_ffmpeg
def test_caption_render_path_still_works(source_clip, tmp_path):
    seg = _segment(source_clip, caption_text="hello world", caption_preset="clean")
    out = tmp_path / "out.mp4"
    render.render_preview((seg,), str(out))
    assert out.exists() and out.stat().st_size > 0


@pytestmark_ffmpeg
def test_audio_volume_and_mute_paths_still_work(source_clip, tmp_path):
    muted = _segment(source_clip, "m1", audio_muted=True)
    out = tmp_path / "out.mp4"
    render.render_preview((muted,), str(out))
    probe = probe_media(str(out))
    assert probe.has_audio is True  # muted segment still carries a (silent) audio track


@pytestmark_ffmpeg
def test_visual_transform_spec_path_still_produces_valid_output(source_clip, tmp_path):
    from cutsell_worker.visual_finishing_executor import VisualTransformSpec
    transform = VisualTransformSpec(
        action="PUNCH_IN", source_width=160, source_height=120, scale_factor=1.25,
        scaled_width=200, scaled_height=150, crop_x=20, crop_y=15, crop_width=160, crop_height=120,
    )
    seg = _segment(source_clip, visual_transform=transform)
    out = tmp_path / "out.mp4"
    render.render_preview((seg,), str(out))
    probe = probe_media(str(out))
    assert probe.width == 1080 and probe.height == 1920
    assert out.stat().st_size > 0


@pytestmark_ffmpeg
def test_no_change_path_visual_transform_none_by_default(source_clip, tmp_path):
    seg = _segment(source_clip)
    assert seg.visual_transform is None  # D-262's own unchanged live-production contract
    out = tmp_path / "out.mp4"
    render.render_preview((seg,), str(out))
    assert out.exists() and out.stat().st_size > 0


@pytestmark_ffmpeg
def test_shell_never_used_and_commands_are_argv_lists(source_clip, tmp_path, monkeypatch):
    calls = []
    real_run = subprocess.run

    def _spy(command, *a, **k):
        calls.append((command, k.get("shell", False)))
        return real_run(command, *a, **k)

    monkeypatch.setattr(subprocess, "run", _spy)
    out = tmp_path / "out.mp4"
    render.render_preview((_segment(source_clip),), str(out))
    assert calls, "expected at least one subprocess.run call"
    for command, shell in calls:
        assert shell is False
        assert isinstance(command, list)


@pytestmark_ffmpeg
def test_output_directory_contains_only_final_file_after_success(source_clip, tmp_path):
    out = tmp_path / "out.mp4"
    render.render_preview((_segment(source_clip),), str(out))
    assert list(tmp_path.iterdir()) == [out]


@pytestmark_ffmpeg
@pytest.mark.parametrize("name", ["out with spaces.mp4", "sortie-éè.mp4", "it's-a-take.mp4"])
def test_success_path_handles_special_characters_in_output_path(source_clip, tmp_path, name):
    out = tmp_path / name
    render.render_preview((_segment(source_clip),), str(out))
    assert out.exists() and out.stat().st_size > 0


@pytestmark_ffmpeg
def test_existing_final_file_is_overwritten_same_as_before(source_clip, tmp_path):
    out = tmp_path / "out.mp4"
    out.write_bytes(b"stale-content-from-a-previous-run")
    render.render_preview((_segment(source_clip),), str(out))
    assert out.stat().st_size > len(b"stale-content-from-a-previous-run")


# =============================================================================
# Stage 2/3/4/5 -- structured failure primitives (unit-level, no ffmpeg needed)
# =============================================================================

def test_run_raises_structured_error_on_nonzero_exit(monkeypatch):
    class FakeCompleted:
        returncode = 7
        stdout = "some stdout"
        stderr = "ffmpeg: invalid argument"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeCompleted())
    with pytest.raises(render.RenderExecutionError) as exc_info:
        render._run(["ffmpeg", "-i", "bad"], output_path="/tmp/x.mp4", source_identity=("c1",))
    failure = exc_info.value.failure
    assert failure.error_category == render.RENDER_FAILURE_FFMPEG_NONZERO_EXIT
    assert failure.return_code == 7
    assert failure.stderr_excerpt == "ffmpeg: invalid argument"
    assert failure.command_fingerprint
    assert failure.timed_out is False
    assert failure.source_identity == ("c1",)
    assert isinstance(exc_info.value, RuntimeError)  # existing blanket `except RuntimeError` callers stay compatible


def test_run_raises_structured_error_on_timeout(monkeypatch):
    def _raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=k.get("timeout"))

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    with pytest.raises(render.RenderExecutionError) as exc_info:
        render._run(["ffmpeg", "-i", "x"], output_path="/tmp/x.mp4", timeout_sec=5.0)
    failure = exc_info.value.failure
    assert failure.error_category == render.RENDER_FAILURE_FFMPEG_TIMEOUT
    assert failure.timed_out is True
    assert failure.timeout_sec == 5.0


def test_run_success_leaves_diagnostics_row(monkeypatch):
    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeCompleted())
    diagnostics: list = []
    render._run(["ffmpeg", "-i", "ok"], output_path="/tmp/ok.mp4", diagnostics=diagnostics)
    assert len(diagnostics) == 1
    assert diagnostics[0]["execution_status"] == "SUCCESS"
    assert diagnostics[0]["return_code"] == 0


def test_run_failure_leaves_diagnostics_row(monkeypatch):
    class FakeCompleted:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeCompleted())
    diagnostics: list = []
    with pytest.raises(render.RenderExecutionError):
        render._run(["ffmpeg", "-i", "bad"], output_path="/tmp/bad.mp4", diagnostics=diagnostics)
    assert len(diagnostics) == 1
    assert diagnostics[0]["execution_status"] == "FAILED"
    assert diagnostics[0]["error_category"] == render.RENDER_FAILURE_FFMPEG_NONZERO_EXIT


def test_command_fingerprint_deterministic():
    cmd = ["ffmpeg", "-i", "a.mp4", "-y", "out.mp4"]
    assert render._command_fingerprint(cmd) == render._command_fingerprint(list(cmd))


def test_command_fingerprint_changes_with_command():
    a = render._command_fingerprint(["ffmpeg", "-i", "a.mp4"])
    b = render._command_fingerprint(["ffmpeg", "-i", "b.mp4"])
    assert a != b


def test_stderr_excerpt_bounded():
    huge = "x" * 10_000
    assert len(render._bounded_excerpt(huge)) == render._STDERR_EXCERPT_MAX_CHARS


def test_stderr_excerpt_handles_none_and_bytes():
    assert render._bounded_excerpt(None) == ""
    assert render._bounded_excerpt(b"abc") == "abc"


# =============================================================================
# Stage 7-12 -- atomic output publication primitives
# =============================================================================

def test_finalize_render_output_missing_temp_raises(tmp_path):
    dest = tmp_path / "out.mp4"
    missing_temp = tmp_path / ".out.mp4.deadbeef.rendering.mp4"
    with pytest.raises(render.RenderExecutionError) as exc_info:
        render._finalize_render_output(missing_temp, dest, output_path=str(dest), source_identity=())
    assert exc_info.value.failure.error_category == render.RENDER_FAILURE_OUTPUT_MISSING
    assert not dest.exists()


def test_finalize_render_output_empty_temp_raises(tmp_path):
    dest = tmp_path / "out.mp4"
    temp = tmp_path / ".out.mp4.deadbeef.rendering.mp4"
    temp.write_bytes(b"")
    with pytest.raises(render.RenderExecutionError) as exc_info:
        render._finalize_render_output(temp, dest, output_path=str(dest), source_identity=())
    assert exc_info.value.failure.error_category == render.RENDER_FAILURE_OUTPUT_EMPTY
    assert not dest.exists()


def test_finalize_render_output_promotes_atomically(tmp_path):
    dest = tmp_path / "out.mp4"
    temp = tmp_path / ".out.mp4.deadbeef.rendering.mp4"
    temp.write_bytes(b"real-bytes")
    render._finalize_render_output(temp, dest, output_path=str(dest), source_identity=())
    assert dest.read_bytes() == b"real-bytes"
    assert not temp.exists()


def test_finalize_render_output_overwrites_existing_final(tmp_path):
    dest = tmp_path / "out.mp4"
    dest.write_bytes(b"old-content")
    temp = tmp_path / ".out.mp4.deadbeef.rendering.mp4"
    temp.write_bytes(b"new-content")
    render._finalize_render_output(temp, dest, output_path=str(dest), source_identity=())
    assert dest.read_bytes() == b"new-content"


def test_validate_output_path_rejects_existing_directory(tmp_path):
    dest = tmp_path / "out.mp4"
    dest.mkdir()
    with pytest.raises(render.RenderExecutionError) as exc_info:
        render._validate_output_path(dest)
    assert exc_info.value.failure.error_category == render.RENDER_FAILURE_INVALID_OUTPUT_PATH


def test_cleanup_temp_output_removes_leftover(tmp_path):
    temp = tmp_path / ".out.mp4.deadbeef.rendering.mp4"
    temp.write_bytes(b"x")
    render._cleanup_temp_output(temp)
    assert not temp.exists()


def test_cleanup_temp_output_never_raises_on_missing_file(tmp_path):
    temp = tmp_path / ".missing.rendering.mp4"
    render._cleanup_temp_output(temp)  # must not raise


def test_cleanup_never_touches_unrelated_files(tmp_path):
    other = tmp_path / "unrelated-source.mp4"
    other.write_bytes(b"source-bytes")
    temp = tmp_path / ".out.mp4.deadbeef.rendering.mp4"
    temp.write_bytes(b"x")
    render._cleanup_temp_output(temp)
    assert other.exists() and other.read_bytes() == b"source-bytes"


def test_job_local_temp_output_path_preserves_extension_and_directory():
    dest = Path("/tmp/some/dir/preview.mp4")
    temp = render._job_local_temp_output_path(dest, "abc123def456")
    assert temp.parent == dest.parent
    assert temp.suffix == ".mp4"
    assert "abc123def456" in temp.name
    assert temp != dest


def test_job_local_temp_output_path_not_pid_only():
    dest = Path("/tmp/dir/preview.mp4")
    a = render._job_local_temp_output_path(dest, "id-one")
    b = render._job_local_temp_output_path(dest, "id-two")
    assert a != b


# =============================================================================
# Stage 13/14/15 -- P0 regression proofs (partial file / timeout / nonzero)
# =============================================================================

@pytestmark_ffmpeg
def test_partial_temp_never_published_as_final(source_clip, tmp_path, monkeypatch):
    """P0 regression test: simulate ffmpeg writing SOME bytes and then
    failing (a crash mid-encode). The temp file may exist momentarily, but
    the final delivery path must never be published, and cleanup must
    remove the partial temp artifact."""
    out = tmp_path / "out.mp4"

    def _fake_run(command, *, output_path, **kwargs):
        Path(output_path).write_bytes(b"partial-bytes-not-a-real-mp4")
        raise render.RenderExecutionError(_synthetic_failure())

    monkeypatch.setattr(render, "_run", _fake_run)
    with pytest.raises(render.RenderExecutionError):
        render.render_preview((_segment(source_clip),), str(out))
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []  # temp cleaned up, no partial final


@pytestmark_ffmpeg
def test_timeout_leaves_no_final_output_and_cleans_temp(source_clip, tmp_path, monkeypatch):
    out = tmp_path / "out.mp4"

    def _raise_timeout(command, *, output_path, **kwargs):
        raise render.RenderExecutionError(_synthetic_failure(
            category=render.RENDER_FAILURE_FFMPEG_TIMEOUT, return_code=None, timed_out=True, timeout_sec=1.0,
        ))

    monkeypatch.setattr(render, "_run", _raise_timeout)
    with pytest.raises(render.RenderExecutionError) as exc_info:
        render.render_preview((_segment(source_clip),), str(out))
    assert exc_info.value.failure.error_category == render.RENDER_FAILURE_FFMPEG_TIMEOUT
    assert exc_info.value.failure.timed_out is True
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


@pytestmark_ffmpeg
def test_nonzero_exit_propagates_structured_evidence_and_leaves_no_output(source_clip, tmp_path, monkeypatch):
    out = tmp_path / "out.mp4"

    def _fake_run(command, *, output_path, **kwargs):
        raise render.RenderExecutionError(render.RenderExecutionFailure(
            error_category=render.RENDER_FAILURE_FFMPEG_NONZERO_EXIT,
            return_code=234, command_fingerprint="fp123", executable="ffmpeg",
            stderr_excerpt="bad filter", stdout_excerpt="",
            timed_out=False, timeout_sec=None, output_path=str(output_path), source_identity=("c1",),
        ))

    monkeypatch.setattr(render, "_run", _fake_run)
    with pytest.raises(render.RenderExecutionError) as exc_info:
        render.render_preview((_segment(source_clip),), str(out))
    f = exc_info.value.failure
    assert f.return_code == 234
    assert f.stderr_excerpt == "bad filter"
    assert f.command_fingerprint == "fp123"
    assert not out.exists()


def test_malformed_source_leaves_no_output_even_though_probe_fails_first(tmp_path):
    """Honest boundary note (newly observed during D-266, not previously
    named in D-265): a malformed source fails inside `media_probe.probe_
    media` (ffprobe, `check=True`) BEFORE this gate's own `_run`/temp-
    output machinery is ever reached, so the exception surfaced here is
    `probe_media`'s own unstructured `subprocess.CalledProcessError`, not
    a `RenderExecutionError`. D-266's own P0 scope was the render ENCODE
    subprocess's failure handling; hardening `probe_media`'s own error
    path (a `media_probe.py` change) is a separate, smaller, out-of-scope
    gap for a future gate. The safety property that DOES hold regardless:
    no temp or final output file is ever created, because the probe fails
    before any ffmpeg write begins."""
    bad_source = tmp_path / "not_a_video.mp4"
    bad_source.write_bytes(b"this is not a real media file")
    seg = _segment(str(bad_source))
    out = tmp_path / "out.mp4"
    with pytest.raises(Exception):
        render.render_preview((seg,), str(out))
    assert not out.exists()
    assert {p.name for p in tmp_path.iterdir()} == {"not_a_video.mp4"}


# =============================================================================
# Stage 20 -- concurrency / job isolation
# =============================================================================

@pytestmark_ffmpeg
def test_concurrent_renders_to_distinct_outputs_do_not_collide(source_clip, tmp_path):
    outs = [tmp_path / f"out{i}.mp4" for i in range(3)]
    errors: list = []

    def _render(out):
        try:
            render.render_preview((_segment(source_clip),), str(out))
        except Exception as exc:  # pragma: no cover -- diagnostic only
            errors.append(exc)

    threads = [threading.Thread(target=_render, args=(out,)) for out in outs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, errors
    for out in outs:
        assert out.exists() and out.stat().st_size > 0
    assert set(tmp_path.iterdir()) == set(outs)  # no stray temp files from either job


def test_temp_paths_never_collide_across_independent_jobs(tmp_path):
    dest_a = tmp_path / "jobA" / "out.mp4"
    dest_b = tmp_path / "jobB" / "out.mp4"
    dest_a.parent.mkdir()
    dest_b.parent.mkdir()
    temp_a = render._job_local_temp_output_path(dest_a, "execA")
    temp_b = render._job_local_temp_output_path(dest_b, "execB")
    assert temp_a != temp_b
    assert temp_a.parent == dest_a.parent
    assert temp_b.parent == dest_b.parent


# =============================================================================
# Stage 21/22 -- shell/path safety, extension/mux preservation
# =============================================================================

def test_no_shell_true_anywhere_in_render_module():
    source = Path("cutsell_worker/render.py").read_text(encoding="utf-8")
    assert "shell=True" not in source


@pytestmark_ffmpeg
def test_temp_output_extension_matches_final_and_ffmpeg_infers_format(source_clip, tmp_path, monkeypatch):
    out = tmp_path / "out.mp4"
    seen_temp_paths: list[str] = []
    real_run = render._run

    def _spy(command, *, output_path, **kwargs):
        seen_temp_paths.append(str(output_path))
        return real_run(command, output_path=output_path, **kwargs)

    monkeypatch.setattr(render, "_run", _spy)
    render.render_preview((_segment(source_clip),), str(out))
    assert seen_temp_paths
    assert all(path.endswith(".mp4") for path in seen_temp_paths)


@pytestmark_ffmpeg
def test_audio_join_treatment_preview_atomic_promotion(source_clip, tmp_path):
    """D-233's audio-only test-capability renderer gets the same temp+atomic
    treatment for consistency (Stage 25 item 2: an audio-only render path)."""
    plan = SimpleNamespace(
        timing_status="SUPPORTED", treatment="SHORT_CROSSFADE", chosen_duration=0.1,
        left_source_audio_start=0.0, left_source_audio_end=1.0, left_output_audio_start=0.0,
        right_source_audio_start=1.0, right_source_audio_end=2.0, right_output_audio_start=0.95,
    )
    out = tmp_path / "join.wav"
    result = render.render_audio_join_treatment_preview(
        plan, str(out), left_source_path=source_clip, right_source_path=source_clip,
    )
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0
    assert list(tmp_path.iterdir()) == [out]


# =============================================================================
# Stage 17/18 -- Audio/Visual Finishing + adjacent-authority firewalls
# =============================================================================

def _run_git_diff(rel_path: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", rel_path], capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("rel_path", [
    "cutsell_worker/audio_finishing_executor.py",
    "cutsell_worker/audio_finishing_composition.py",
    "cutsell_worker/audio_finishing_measurement.py",
    "cutsell_worker/audio_finishing_policy.py",
    "cutsell_worker/audio_finishing_outcome.py",
    "cutsell_worker/visual_finishing_executor.py",
    "cutsell_worker/visual_finishing_composition.py",
    "cutsell_worker/visual_finishing_measurement.py",
    "cutsell_worker/visual_finishing_policy.py",
    "cutsell_worker/boundary_engine_pass.py",
    "cutsell_worker/pacing_transition_decision.py",
    "cutsell_worker/post_render_media_qc.py",
    "cutsell_worker/post_render_watch_listen_qc.py",
    "cutsell_worker/media_probe.py",
    "cutsell_worker/finishing_contract.py",
])
def test_unrelated_authorities_unchanged(rel_path):
    assert _run_git_diff(rel_path) == "", f"D-266 must not touch {rel_path}"


# =============================================================================
# Stage 19/26 -- no retry policy, no secrets, codec/geometry unchanged
# =============================================================================

def test_no_retry_loop_introduced():
    source = Path("cutsell_worker/render.py").read_text(encoding="utf-8")
    for needle in ("retry_count", "max_attempts", "RETRY_LIMIT"):
        assert needle not in source


def test_render_module_has_no_network_or_credential_construction():
    source = _source_without_docstrings("cutsell_worker/render.py")
    for needle in ("boto3", "s3_client", "AWS_SECRET", "signed_url", "Authorization"):
        assert needle not in source


def test_codec_and_geometry_constants_unchanged():
    source = Path("cutsell_worker/render.py").read_text(encoding="utf-8")
    assert '"libx264"' in source
    assert '"-crf", "20"' in source
    assert '"aac"' in source
    assert '"160k"' in source
    assert "RENDER_FPS_DEFAULT = 30" in source


def test_render_execution_error_is_a_runtime_error_subclass():
    failure = _synthetic_failure()
    exc = render.RenderExecutionError(failure)
    assert isinstance(exc, RuntimeError)
    assert exc.failure is failure


def test_render_execution_failure_is_frozen():
    failure = _synthetic_failure()
    with pytest.raises(Exception):
        failure.return_code = 0  # frozen dataclass -- matches this codebase's own convention


# =============================================================================
# Stage 6 -- timeout seam, unbounded today, wired end to end
# =============================================================================

def test_timeout_seam_defaults_to_none_today():
    assert render.RENDER_SUBPROCESS_TIMEOUT_SEC is None


def test_timeout_expired_is_caught_and_never_propagates_raw(monkeypatch):
    def _raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=k.get("timeout"))

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    with pytest.raises(render.RenderExecutionError):
        render._run(["ffmpeg"], output_path="/tmp/x.mp4", timeout_sec=0.01)
    # never a bare, uncaught subprocess.TimeoutExpired escaping this module
