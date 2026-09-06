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
