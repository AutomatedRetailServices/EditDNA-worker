from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import cutsell_app.main as api
import cutsell_worker.render as renderer
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.draft_edits import DraftEditError, patch_audio
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
            "end": 2.0,
            "text": "hello",
            "caption_text": "hello",
            "selected": True,
        }],
        "alternates": [],
        "discarded": [],
        "diagnostics": {},
    }


def test_patch_audio_is_stateless_and_bounded():
    original = _draft_json()
    updated = patch_audio(original, "c1", muted=True, volume=0.35)
    assert updated["selected"][0]["audio_muted"] is True
    assert updated["selected"][0]["audio_volume"] == 0.35
    assert "audio_muted" not in original["selected"][0]
    try:
        patch_audio(original, "c1", volume=2.1)
    except DraftEditError as exc:
        assert "between 0.0 and 2.0" in str(exc)
    else:
        raise AssertionError("unsafe audio volume must be rejected")


def test_mobile_audio_edit_api_updates_selected_clip():
    response = TestClient(api.app).post("/v1/draft-edits/audio", json={
        "draft": _draft_json(),
        "clip_id": "c1",
        "muted": False,
        "volume": 0.5,
    })
    assert response.status_code == 200
    item = response.json()["selected"][0]
    assert item["audio_muted"] is False
    assert item["audio_volume"] == 0.5

    bad = TestClient(api.app).post("/v1/draft-edits/audio", json={
        "draft": _draft_json(),
        "clip_id": "c1",
        "volume": 3.0,
    })
    assert bad.status_code == 409


def test_serde_and_render_plan_preserve_audio_controls():
    payload = _draft_json()
    payload["selected"][0]["audio_muted"] = True
    payload["selected"][0]["audio_volume"] = 0.25
    draft = draft_from_dict(payload)
    plan = build_render_plan(draft, {"s1": "/tmp/source.mov"})
    assert plan[0].audio_muted is True
    assert plan[0].audio_volume == 0.25


def test_renderer_applies_mute_volume_and_silent_track_for_broll(monkeypatch, tmp_path):
    audio_source = tmp_path / "audio.mp4"
    silent_source = tmp_path / "silent.mp4"
    audio_source.write_bytes(b"x")
    silent_source.write_bytes(b"x")

    def fake_probe(path):
        return SimpleNamespace(has_audio=Path(path).name == "audio.mp4")

    monkeypatch.setattr(renderer, "probe_media", fake_probe)
    muted = renderer._segment_command(
        RenderSegment("c1", "s1", str(audio_source), 0.0, 1.0, audio_muted=True, audio_volume=1.0),
        tmp_path / "one.mp4",
        vf="scale=1080:1920",
    )
    assert "-af" in muted
    assert "volume=0.000" in muted

    lowered = renderer._segment_command(
        RenderSegment("c2", "s1", str(audio_source), 0.0, 1.0, audio_muted=False, audio_volume=0.5),
        tmp_path / "two.mp4",
        vf="scale=1080:1920",
    )
    assert "volume=0.500" in lowered

    broll = renderer._segment_command(
        RenderSegment("c3", "s2", str(silent_source), 0.0, 1.0),
        tmp_path / "three.mp4",
        vf="scale=1080:1920",
    )
    assert "anullsrc=channel_layout=stereo:sample_rate=48000" in broll
    assert broll.count("-map") == 2
