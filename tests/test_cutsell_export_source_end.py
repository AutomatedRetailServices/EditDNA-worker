"""Export of a cut made from the voice file (audio first): the voice file can be a few hundredths of a second longer
than the video, so the last clip can end just past the video's end. Real case 2026-10-19 (job d629f0b0):
audio_window_exceeds_source_availability:se-000-0013:requested_end=60.070:source_duration=60.012."""
import subprocess

import pytest

from cutsell_worker.media_probe import probe_media
from cutsell_worker.render import render_preview
from cutsell_worker.render_plan import build_render_plan
from cutsell_worker.serde import draft_from_dict


def _video(path, seconds=3.0):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=360x640:r=30:d={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency=440:d={seconds}", "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)
    return str(path)


def _draft(end):
    clip = lambda cid, a, b, pre, post: {"clip_id": cid, "source_asset_id": "s1", "source_order": 0, "start": a, "end": b,
                                         "text": "hola", "caption_text": "hola", "selected": True,
                                         "audio_pre_roll": pre, "audio_post_roll": post}
    return draft_from_dict({"schema_version": "cutsell.v1", "project_id": "p1", "strategy": "mixed",
                            "selected": [clip("c1", 0.0, 1.0, 0.0, 0.1), clip("c2", 1.6, end, 0.1, 0.3)],
                            "alternates": [], "discarded": [], "diagnostics": {"engine": "simple"},
                            "audio_overlap_enabled": True})


def test_a_last_clip_just_past_the_video_end_exports(tmp_path):
    source = _video(tmp_path / "v.mp4")
    real = probe_media(source).duration_sec
    plan = build_render_plan(_draft(real + 0.058), {"s1": source})
    assert plan[-1].end == pytest.approx(real)
    out = render_preview(plan, str(tmp_path / "out.mp4"), width=360, height=640)
    assert probe_media(out).duration_sec > 1.5


def test_a_real_mismatch_is_not_hidden(tmp_path):
    source = _video(tmp_path / "v.mp4")
    real = probe_media(source).duration_sec
    plan = build_render_plan(_draft(real + 2.0), {"s1": source})
    assert plan[-1].end == pytest.approx(real + 2.0)      # left as is: the renderer's own checks decide


def test_an_export_failure_names_its_step_and_reason():
    from cutsell_worker.export_job import failure_code
    exc = ValueError("audio_window_exceeds_source_availability:se-000-0013:requested_end=60.070:source_duration=60.012")
    timings = {"download_sources": 3.6, "download_overlays": 0, "download_audio": 0, "clean_audio": 0, "plan": 0.09}
    assert failure_code(timings, exc) == "render_and_qc:valueerror_audio_window_exceeds_source_availability"
    assert failure_code({}, RuntimeError("s3://bucket/secret/key")) == "download_sources:runtimeerror_s3"


def test_the_export_job_path_bounds_the_draft_itself(tmp_path):
    """The post-render QC compares the render with the draft: the bound must be on the draft, not only the plan."""
    from cutsell_worker.live_render_qc import render_with_post_render_qc
    from cutsell_worker.render_plan import bound_draft_to_sources
    source = _video(tmp_path / "v.mp4")
    real = probe_media(source).duration_sec
    draft = bound_draft_to_sources(_draft(real + 0.058), {"s1": source})
    assert draft.selected[-1].end == pytest.approx(real)
    plan = build_render_plan(draft, {"s1": source})
    qc = render_with_post_render_qc(draft, plan, str(tmp_path / "e.mp4"))
    assert qc.status != "SEMANTIC_MISMATCH_INVALIDATED", qc.attempts[-1].findings
