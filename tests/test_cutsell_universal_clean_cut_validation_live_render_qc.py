"""D-035: the Video00 RAW validation harness must render/QC through the SAME
live PostRenderWatchListenQC + bounded physical repair service the real
mobile-app export job uses (`export_job.run_export_job` ->
`live_render_qc.render_with_post_render_qc`) -- never a separate, bare
`render.render_preview()` call and never a second, independent QC
implementation.

Real ffmpeg rendering is exercised for real, unmocked, throughout (skipped
if ffmpeg is absent). The physical-defect DETECTION layer is injected for
the repair-loop tests, exactly as `tests/test_cutsell_live_render_qc.py`
already does for `render_with_post_render_qc` itself -- this file's job is
proving the HARNESS actually reaches that one shared service, not
re-proving the service's own internal correctness.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from cutsell_worker import export_job, live_render_qc, universal_clean_cut_validation as harness
from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.post_render_watch_listen_qc import (
    LINGERING_ACCIDENTAL_SILENCE,
    PostRenderFinding,
    PostRenderQCResult,
)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("validation_live_render_qc")


@pytest.fixture(scope="module")
def source_video(media_dir):
    path = str(media_dir / "source.mp4")
    _ffmpeg([
        "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:duration=8",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-y", path,
    ])
    return path


def _clip(clip_id, start, end, text, *, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=True,
    )


def _draft(clips, diagnostics=None):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="validation-live-qc-test", strategy=EditStrategy.STORYTELLING,
        selected=tuple(clips), alternates=(), discarded=(), diagnostics=diagnostics or {},
    )


@pytest.fixture
def two_clip_draft_and_paths(source_video):
    clip_a = _clip("a", 0.0, 3.0, "first idea")
    clip_b = _clip("b", 3.0, 6.0, "second idea")
    draft = _draft([clip_a, clip_b])
    return draft, {"src": source_video}


# ---------------------------------------------------------------------------
# 1. The harness invokes the canonical live render/QC service, not a bare render
# ---------------------------------------------------------------------------

def test_harness_invokes_canonical_render_qc_service(monkeypatch, tmp_path, two_clip_draft_and_paths):
    draft, local_paths = two_clip_draft_and_paths
    calls = []
    real = harness.render_with_post_render_qc

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(harness, "render_with_post_render_qc", spy)

    output = str(tmp_path / "preview.mp4")
    preview_path, reason, qc_result = harness._render_validation_preview(
        draft, local_paths, preview_output=output, preview_captions=False,
    )

    assert len(calls) == 1
    assert preview_path == output
    assert reason is None
    assert qc_result.status == "PASS"


# ---------------------------------------------------------------------------
# 2. Mobile export and Video00 validation share the exact same entry point
# ---------------------------------------------------------------------------

def test_export_job_and_validation_harness_share_one_render_qc_implementation():
    # Not two independent "Video00RenderQC"/"AppRenderQC" implementations --
    # both modules import the identical function object from live_render_qc.
    assert export_job.render_with_post_render_qc is live_render_qc.render_with_post_render_qc
    assert harness.render_with_post_render_qc is live_render_qc.render_with_post_render_qc


# ---------------------------------------------------------------------------
# 3. A clean synthetic candidate renders and QC passes
# ---------------------------------------------------------------------------

def test_clean_candidate_renders_and_qc_passes(tmp_path, two_clip_draft_and_paths):
    draft, local_paths = two_clip_draft_and_paths
    output = str(tmp_path / "preview.mp4")

    preview_path, reason, qc_result = harness._render_validation_preview(
        draft, local_paths, preview_output=output, preview_captions=False,
    )

    assert preview_path == output
    assert reason is None
    assert qc_result.status == "PASS"
    assert qc_result.output_path == output


# ---------------------------------------------------------------------------
# 4/5. A physical failure causes bounded Boundary repair; the repaired
#      candidate is rendered/QC'd again through this same harness call
# ---------------------------------------------------------------------------

def test_physical_failure_triggers_repair_and_repaired_candidate_is_re_qcd(
    monkeypatch, tmp_path, two_clip_draft_and_paths
):
    draft, local_paths = two_clip_draft_and_paths
    output = str(tmp_path / "preview.mp4")

    defect = PostRenderFinding(
        kind=LINGERING_ACCIDENTAL_SILENCE, start=2.7, end=3.0,
        detail={"duration_sec": 0.3}, routes_to="BoundaryEngine",
    )
    qc_calls = []
    real_qc = live_render_qc.run_post_render_media_qc

    def fake_qc(media_path, **kwargs):
        qc_calls.append(media_path)
        if len(qc_calls) == 1:
            return PostRenderQCResult(status="FAIL", findings=(defect,))
        return real_qc(media_path, **kwargs)

    monkeypatch.setattr(live_render_qc, "run_post_render_media_qc", fake_qc)

    preview_path, reason, qc_result = harness._render_validation_preview(
        draft, local_paths, preview_output=output, preview_captions=False,
    )

    assert qc_result.status == "PASS"
    assert preview_path == output
    assert len(qc_calls) == 2  # repaired candidate really was QC'd again
    assert len(qc_result.attempts) == 2
    assert qc_result.attempts[0].status == "PHYSICAL_FAIL_REPAIRED"
    assert qc_result.attempts[0].repair_applied["clip_id"] == "a"
    assert qc_result.attempts[1].status == "PASS"


# ---------------------------------------------------------------------------
# 6. Semantic mismatch cannot be repaired by Boundary through this harness
# ---------------------------------------------------------------------------

def test_semantic_mismatch_is_never_repaired_by_boundary(monkeypatch, tmp_path, two_clip_draft_and_paths):
    draft, local_paths = two_clip_draft_and_paths
    output = str(tmp_path / "preview.mp4")

    # Simulate a broken render plan (a duplicated segment) -- a real
    # structural/semantic mismatch caught before any media decode.
    real_build_render_plan = harness.build_render_plan

    def broken_plan(draft, local_paths):
        plan = real_build_render_plan(draft, local_paths)
        return (plan[0], plan[0])

    monkeypatch.setattr(harness, "build_render_plan", broken_plan)

    repair_calls = []
    monkeypatch.setattr(
        live_render_qc, "repair_segment_for_finding",
        lambda *a, **k: repair_calls.append(1) or None,
    )

    preview_path, reason, qc_result = harness._render_validation_preview(
        draft, local_paths, preview_output=output, preview_captions=False,
    )

    assert preview_path is None
    assert qc_result.status == "SEMANTIC_MISMATCH_INVALIDATED"
    assert reason == "post_render_qc_semantic_mismatch_invalidated"
    assert repair_calls == []  # Boundary was never asked to "fix" a semantic mismatch


# ---------------------------------------------------------------------------
# 7. A failed semantic plan (freeze_blocked) never renders at all
# ---------------------------------------------------------------------------

def test_freeze_blocked_semantic_plan_never_reaches_render_qc(monkeypatch, tmp_path, two_clip_draft_and_paths):
    draft, local_paths = two_clip_draft_and_paths
    output = str(tmp_path / "preview.mp4")

    calls = []
    monkeypatch.setattr(
        harness, "render_with_post_render_qc",
        lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(AssertionError("must not render")),
    )

    preview_path, reason, qc_result = harness._render_validation_preview(
        draft, local_paths, preview_output=output, preview_captions=False, freeze_blocked=True,
    )

    assert preview_path is None
    assert reason == "freeze_blocked_no_render"
    assert qc_result is None
    assert calls == []  # render/QC was never even attempted


# ---------------------------------------------------------------------------
# 8. The final artifact corresponds to the exact frozen plan id/version/hash
# ---------------------------------------------------------------------------

def test_final_artifact_identifies_the_exact_frozen_plan(tmp_path, two_clip_draft_and_paths):
    draft, local_paths = two_clip_draft_and_paths
    output = str(tmp_path / "preview.mp4")

    preview_path, reason, qc_result = harness._render_validation_preview(
        draft, local_paths, preview_output=output, preview_captions=False,
    )

    assert qc_result.status == "PASS"
    expected_plan = build_canonical_edit_plan(draft)
    assert qc_result.plan_id == expected_plan.plan_id
    assert qc_result.semantic_hash == expected_plan.semantic_hash

    diag = harness._live_render_qc_diagnostics(qc_result, skipped_reason=reason)
    assert diag["status"] == "PASS"
    assert diag["plan_id"] == expected_plan.plan_id
    assert diag["plan_version"] == expected_plan.plan_version
    assert diag["semantic_hash"] == expected_plan.semantic_hash
    assert diag["render_attempt_count"] == 1
