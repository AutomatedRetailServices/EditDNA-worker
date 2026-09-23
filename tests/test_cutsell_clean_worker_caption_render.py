from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import cutsell_app.main as api
import cutsell_worker.render as renderer
from cutsell_worker.caption_settings import patch_caption_settings
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.render_plan import RenderSegment, build_render_plan
from cutsell_worker.serde import draft_from_dict


def _draft_json():
    return {
        "schema_version": "cutsell.v1",
        "project_id": "p1",
        "strategy": "mixed",
        "selected": [{
            "clip_id": "c1",
            "source_asset_id": "s1",
            "source_order": 0,
            "start": 1.0,
            "end": 3.0,
            "text": "original transcript",
            "caption_text": "Edited caption!",
            "selected": True,
        }],
        "alternates": [],
        "discarded": [],
        "diagnostics": {},
    }


def test_caption_settings_are_stateless_and_validate_presets():
    original = _draft_json()
    updated = patch_caption_settings(original, enabled=False, preset="clean")
    assert updated["captions_enabled"] is False
    assert updated["caption_preset"] == "clean"
    assert "captions_enabled" not in original
    try:
        patch_caption_settings(original, preset="giant-neon")
    except ValueError as exc:
        assert "classic or clean" in str(exc)
    else:
        raise AssertionError("unknown caption preset must be rejected")


def test_mobile_caption_settings_endpoint():
    response = TestClient(api.app).post("/v1/draft-edits/caption-settings", json={
        "draft": _draft_json(),
        "enabled": True,
        "preset": "clean",
    })
    assert response.status_code == 200
    assert response.json()["captions_enabled"] is True
    assert response.json()["caption_preset"] == "clean"


def test_render_plan_uses_edited_caption_and_can_disable_all_captions():
    payload = _draft_json()
    payload["captions_enabled"] = True
    payload["caption_preset"] = "clean"
    plan = build_render_plan(draft_from_dict(payload), {"s1": "/tmp/source.mov"})
    assert plan[0].caption_text == "Edited caption!"
    assert plan[0].caption_preset == "clean"

    payload["captions_enabled"] = False
    disabled = build_render_plan(draft_from_dict(payload), {"s1": "/tmp/source.mov"})
    assert disabled[0].caption_text == ""


def test_renderer_writes_srt_and_adds_subtitles_filter(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"x")
    monkeypatch.setattr(renderer, "probe_media", lambda _path: SimpleNamespace(has_audio=True))
    part = tmp_path / "part.mp4"
    command = renderer._segment_command(
        RenderSegment(
            "c1", "s1", str(source), 0.0, 2.0,
            caption_text="Edited caption!",
            caption_preset="classic",
        ),
        part,
        vf="scale=1080:1920",
    )
    vf = command[command.index("-vf") + 1]
    assert "subtitles=" in vf
    subtitle = part.with_suffix(".srt")
    assert subtitle.exists()
    assert "Edited caption!" in subtitle.read_text(encoding="utf-8")
    assert "00:00:02,000" in subtitle.read_text(encoding="utf-8")


def test_renderer_skips_subtitle_filter_when_caption_empty(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"x")
    monkeypatch.setattr(renderer, "probe_media", lambda _path: SimpleNamespace(has_audio=True))
    command = renderer._segment_command(
        RenderSegment("c1", "s1", str(source), 0.0, 1.0, caption_text=""),
        tmp_path / "part.mp4",
        vf="scale=1080:1920",
    )
    vf = command[command.index("-vf") + 1]
    assert "subtitles=" not in vf


# ---------------------------------------------------------------------------
# CutSell.ai — Canonical Captions Reconciliation Gate: line-break/cue-
# injection hardening, Unicode/Spanish preservation, and a real local
# visible-pixel burn-in proof. No selection/timeline/overlay logic touched;
# only cutsell_worker/render.py's own `_caption_filter` sanitization.
# ---------------------------------------------------------------------------

def test_embedded_blank_line_cannot_inject_a_second_srt_cue(tmp_path):
    """A caller-supplied caption containing an embedded blank line plus a
    forged SRT cue header must never survive as a second, attacker-
    controlled cue -- this implementation only ever intends exactly one
    cue per clip. All whitespace (including newlines) collapses to single
    spaces instead."""
    hostile = "Real caption\n\n2\n00:00:00,000 --> 99:99:99,999\nFAKE INJECTED CUE"
    segment = RenderSegment("c1", "s1", "/tmp/source.mp4", 0.0, 2.0, caption_text=hostile)
    part = tmp_path / "part.mp4"
    result = renderer._caption_filter(segment, part)
    assert result is not None
    srt_text = part.with_suffix(".srt").read_text(encoding="utf-8")
    # Exactly one SRT block: index line "1", one timing line, one text
    # line, trailing blank line -- never a structurally distinct second
    # cue (which would require its own blank-line-preceded index line).
    lines = srt_text.split("\n")
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:02,000"
    # The forged "2 / timing / text" header survives only as literal words
    # inside the single collapsed text line -- never as its own cue.
    assert lines[2].startswith("Real caption 2 00:00:00,000 --> 99:99:99,999 FAKE INJECTED CUE")
    assert lines[2].count("-->") == 1  # the attacker's own literal text, not a real cue boundary
    assert len(lines) == 4  # index, timing, one text line, trailing blank -- no more cues


def test_embedded_newlines_and_carriage_returns_collapse_to_single_spaces(tmp_path):
    segment = RenderSegment(
        "c1", "s1", "/tmp/source.mp4", 0.0, 1.0,
        caption_text="line one\r\nline two\n\n\nline three   with    gaps",
    )
    part = tmp_path / "part.mp4"
    renderer._caption_filter(segment, part)
    srt_text = part.with_suffix(".srt").read_text(encoding="utf-8")
    body = srt_text.split("\n", 2)[2]
    assert "\n" not in body.strip("\n")
    assert "line one line two line three with gaps" in srt_text


def test_unicode_and_spanish_caption_text_preserved(tmp_path):
    spanish = "¡Vámonos! Esto es una prueba de subtítulos con eñes, tildes y signos: ¿cómo estás?"
    segment = RenderSegment("c1", "s1", "/tmp/source.mp4", 0.0, 1.5, caption_text=spanish)
    part = tmp_path / "part.mp4"
    result = renderer._caption_filter(segment, part)
    assert result is not None
    srt_text = part.with_suffix(".srt").read_text(encoding="utf-8")
    assert spanish in srt_text
    # Round-trips through UTF-8 exactly (no mojibake / lossy encoding).
    assert srt_text.encode("utf-8").decode("utf-8") == srt_text


def test_temp_caption_files_are_job_scoped_and_cleaned(monkeypatch, tmp_path):
    """render()'s TemporaryDirectory context manager owns the .srt sidecar's
    lifetime -- prove the srt path always lives inside the caller-supplied
    per-job workdir (never a shared/global location), and that once that
    directory is removed, the caption artifact is gone with it."""
    monkeypatch.setattr(renderer, "probe_media", lambda _path: SimpleNamespace(has_audio=True))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"x")
    job_workdir = tmp_path / "cutsell-render-job-scoped"
    job_workdir.mkdir()
    part = job_workdir / "part-0000.mp4"
    renderer._segment_command(
        RenderSegment("c1", "s1", str(source), 0.0, 1.0, caption_text="hello"),
        part,
        vf="scale=1080:1920",
    )
    srt = part.with_suffix(".srt")
    assert srt.exists()
    assert srt.parent == job_workdir
    import shutil
    shutil.rmtree(job_workdir)
    assert not srt.exists()


def test_captions_disabled_path_never_writes_an_srt_file(monkeypatch, tmp_path):
    monkeypatch.setattr(renderer, "probe_media", lambda _path: SimpleNamespace(has_audio=True))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"x")
    part = tmp_path / "part.mp4"
    renderer._segment_command(
        RenderSegment("c1", "s1", str(source), 0.0, 1.0, caption_text=""),
        part,
        vf="scale=1080:1920",
    )
    assert not part.with_suffix(".srt").exists()


def test_missing_captions_enabled_field_defaults_to_current_true_behavior():
    """Documents the LIVE current default rather than asserting the task's
    requested false-by-default: DraftTimeline.captions_enabled defaults to
    True (contracts.py) and serde.py's draft_from_dict does the same for a
    payload that omits the field entirely. Flipping this default is a
    product-behavior change to an already-shipped, already-tested feature
    and is intentionally NOT made by this gate -- see the audit report."""
    payload = _draft_json()
    assert "captions_enabled" not in payload
    from cutsell_worker.serde import draft_from_dict
    draft = draft_from_dict(payload)
    assert draft.captions_enabled is True


def test_visible_burned_caption_pixels_present_with_real_local_ffmpeg(tmp_path):
    """Real, local, no-paid-compute proof: render a tiny synthetic clip
    through the ACTUAL _segment_command ffmpeg invocation with a caption
    set, then through an identical invocation with no caption, and prove
    the two outputs' pixels differ in the caption-safe region (bottom-
    center, matching _caption_filter's fixed MarginV=120/Alignment=2) --
    i.e. something was actually burned into the frame, not just planned."""
    import shutil as _shutil
    import subprocess
    if _shutil.which("ffmpeg") is None:
        import pytest as _pytest
        _pytest.skip("ffmpeg not available on this runner")
    from PIL import Image

    width, height, fps = 320, 240, 10
    source = tmp_path / "src.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=gray:s={width}x{height}:r={fps}:d=1",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(source),
    ], check=True)

    def _render(caption_text: str, out_name: str) -> Path:
        part = tmp_path / out_name
        command = renderer._segment_command(
            RenderSegment("c1", "s1", str(source), 0.0, 1.0, caption_text=caption_text),
            part,
            vf=f"scale={width}:{height}",
        )
        # _segment_command already appends the output path as its final
        # argument -- run it as-is.
        subprocess.run(command, check=True, capture_output=True)
        return part

    plain = _render("", "plain.mp4")
    captioned = _render("HELLO CAPTIONS", "captioned.mp4")
    assert plain.exists() and plain.stat().st_size > 0
    assert captioned.exists() and captioned.stat().st_size > 0

    def _extract_frame(video: Path, png: Path) -> Image.Image:
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video), "-frames:v", "1", str(png),
        ], check=True)
        return Image.open(png).convert("RGB")

    plain_frame = _extract_frame(plain, tmp_path / "plain.png")
    captioned_frame = _extract_frame(captioned, tmp_path / "captioned.png")
    assert plain_frame.size == captioned_frame.size

    # `_caption_filter`'s MarginV=120 is a FIXED pixel margin, not scaled to
    # frame height -- on this deliberately tiny 240px-tall fixture that can
    # land well above any fixed bottom band, so compare the whole frame
    # rather than guessing where the fixed-size preset places the text at
    # this resolution. The two renders are otherwise byte-for-byte
    # identical inputs/filters, so ANY pixel difference is the burned
    # caption -- nothing else could differ.
    import numpy as np
    diff = np.abs(
        np.asarray(plain_frame, dtype=np.int16) - np.asarray(captioned_frame, dtype=np.int16)
    )
    changed_pixels = int((diff.sum(axis=-1) > 20).sum())
    assert changed_pixels > 0, "no visible pixel difference found anywhere in the frame"
