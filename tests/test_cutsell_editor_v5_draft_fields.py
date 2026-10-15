"""Editor v5 draft fields: clip speed, overlay layer/layout/edge/fill/rotation, the audio lane,
whole-video volume and voice cleanup. Every field is optional; an old draft parses and serializes
exactly as before."""
from __future__ import annotations

import json

import pytest

from cutsell_worker.contracts import AudioClip
from cutsell_worker.serde import draft_from_dict, result_to_dict, timeline_duration_sec


def _draft(**extra):
    payload = {
        "project_id": "prj_test",
        "selected": [
            {"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 4.0, "text": "hi"},
            {"clip_id": "c2", "source_asset_id": "s1", "start": 5.0, "end": 7.0, "text": "there"},
        ],
        "captions_enabled": False,
    }
    payload.update(extra)
    return payload


def _overlay(**extra):
    item = {"overlay_id": "o1", "kind": "photo", "uri": "s3://b/k.jpg", "start": 0.0, "end": 1.0}
    item.update(extra)
    return item


def _audio(**extra):
    item = {"audio_id": "a1", "kind": "voice", "asset_id": "asset_1", "start": 1.0, "source_in": 0.0,
            "source_out": 2.0}
    item.update(extra)
    return item


def test_old_draft_parses_with_defaults_and_serializes_unchanged():
    draft = draft_from_dict(_draft(media_overlays=[_overlay()]))
    assert all(clip.speed == 1.0 for clip in draft.selected)
    overlay = draft.media_overlays[0]
    assert (overlay.layer, overlay.layout, overlay.edge, overlay.fill, overlay.rotation) == (0, "float", "sharp", False, 0.0)
    assert overlay.mute_audio is True
    assert draft.audio_clips == () and draft.video_volume == 1.0 and draft.video_clean_voice is False
    data = result_to_dict(draft)
    for key in ("audio_clips", "video_volume", "video_clean_voice"):
        assert key not in data
    assert all("speed" not in clip for clip in data["selected"])
    assert set(data["media_overlays"][0]) == {
        "overlay_id", "kind", "uri", "start", "end", "x", "y", "width", "source_start", "source_end", "mute_audio"}


@pytest.mark.parametrize("speed", [0.5, 1, 1.5, 2, 3, "2.0"])
def test_allowed_speeds(speed):
    draft = draft_from_dict(_draft(selected=[
        {"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 4.0, "speed": speed}]))
    assert draft.selected[0].speed == float(speed)


@pytest.mark.parametrize("speed", [0.25, 1.25, 4, 0, -1, "fast"])
def test_other_speeds_are_rejected(speed):
    with pytest.raises(ValueError):
        draft_from_dict(_draft(selected=[
            {"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 4.0, "speed": speed}]))


def test_speed_round_trips_and_is_omitted_at_one():
    draft = draft_from_dict(_draft(selected=[
        {"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 4.0, "speed": 2},
        {"clip_id": "c2", "source_asset_id": "s1", "start": 5.0, "end": 7.0, "speed": 1}]))
    data = result_to_dict(draft)
    assert data["selected"][0]["speed"] == 2.0 and "speed" not in data["selected"][1]
    assert draft_from_dict(json.loads(json.dumps(data))).selected[0].speed == 2.0


def test_timeline_duration_accounts_for_speed():
    draft = draft_from_dict(_draft(selected=[
        {"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 4.0, "speed": 2},
        {"clip_id": "c2", "source_asset_id": "s1", "start": 5.0, "end": 7.0, "speed": 0.5}]))
    assert timeline_duration_sec(draft.selected) == pytest.approx(2.0 + 4.0)


def test_overlay_limits_use_the_speed_aware_duration():
    fast = [{"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 4.0, "speed": 2}]
    # 4 s of source at 2x is a 2 s timeline: an overlay to 3 s no longer fits.
    with pytest.raises(ValueError, match="media overlay exceeds"):
        draft_from_dict(_draft(selected=fast, media_overlays=[_overlay(end=3.0)]))
    with pytest.raises(ValueError, match="text overlay exceeds"):
        draft_from_dict(_draft(selected=fast, text_overlays=[
            {"overlay_id": "t1", "text": "x", "start": 0.0, "end": 3.0}]))
    slow = [{"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 2.0, "speed": 0.5}]
    # 2 s of source at 0.5x is a 4 s timeline: an overlay to 3.5 s fits.
    assert draft_from_dict(_draft(selected=slow, media_overlays=[_overlay(end=3.5)])).media_overlays


def test_overlay_edits_duration_is_speed_aware():
    from cutsell_worker.overlay_edits import add_media_overlay
    draft = _draft(selected=[{"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 2.0, "speed": 0.5}])
    out = add_media_overlay(draft, kind="photo", uri="s3://b/k.jpg", start=0.0, end=3.5)
    assert out["media_overlays"][0]["end"] == 3.5


def test_media_overlay_new_fields_parse_and_round_trip():
    draft = draft_from_dict(_draft(media_overlays=[
        _overlay(layer=7, layout="stacked", edge="soft", fill=True, rotation=-45.5)]))
    overlay = draft.media_overlays[0]
    assert (overlay.layer, overlay.layout, overlay.edge, overlay.fill, overlay.rotation) == (7, "stacked", "soft", True, -45.5)
    data = result_to_dict(draft)["media_overlays"][0]
    assert data["layer"] == 7 and data["layout"] == "stacked" and data["edge"] == "soft"
    assert data["fill"] is True and data["rotation"] == -45.5


@pytest.mark.parametrize("layout", ["float", "side_by_side", "stacked", "left", "right", "top", "bottom", "full"])
def test_every_layout_is_accepted(layout):
    assert draft_from_dict(_draft(media_overlays=[_overlay(layout=layout)])).media_overlays[0].layout == layout


@pytest.mark.parametrize("bad", [
    {"layer": -1}, {"layer": 51}, {"layer": 2.5}, {"layout": "corner"}, {"edge": "blurry"},
    {"rotation": 361}, {"rotation": -360.5},
])
def test_bad_overlay_fields_are_rejected(bad):
    with pytest.raises(ValueError):
        draft_from_dict(_draft(media_overlays=[_overlay(**bad)]))


def test_audio_clips_parse_with_defaults():
    draft = draft_from_dict(_draft(
        audio_clips=[_audio(), _audio(audio_id="a2", kind="sfx", sfx_id="tap", asset_id=None, start=0.5,
                                      source_out=0.3, volume=1.5, duck_video=True, clean=True)],
        video_volume=0.5, video_clean_voice=True))
    voice, sfx = draft.audio_clips
    assert voice == AudioClip(audio_id="a1", kind="voice", start=1.0, source_in=0.0, source_out=2.0, asset_id="asset_1")
    assert voice.end == 3.0
    assert sfx.sfx_id == "tap" and sfx.asset_id is None and sfx.volume == 1.5 and sfx.duck_video and sfx.clean
    assert draft.video_volume == 0.5 and draft.video_clean_voice is True
    data = result_to_dict(draft)
    assert data["video_volume"] == 0.5 and data["video_clean_voice"] is True
    assert data["audio_clips"][0]["asset_id"] == "asset_1" and "sfx_id" not in data["audio_clips"][0]
    assert data["audio_clips"][1]["sfx_id"] == "tap" and "asset_id" not in data["audio_clips"][1]
    again = draft_from_dict(json.loads(json.dumps(data)))
    assert again.audio_clips == draft.audio_clips


@pytest.mark.parametrize("bad", [
    {"audio_id": ""}, {"kind": "podcast"}, {"asset_id": None}, {"kind": "music", "asset_id": ""},
    {"kind": "sfx", "asset_id": None, "sfx_id": "boom"}, {"kind": "sfx", "asset_id": None},
    {"start": -0.1}, {"source_in": -1}, {"source_out": 0.0}, {"source_in": 2.0, "source_out": 2.0},
    {"volume": 2.5}, {"volume": -0.1},
])
def test_bad_audio_clips_are_rejected(bad):
    with pytest.raises(ValueError):
        draft_from_dict(_draft(audio_clips=[_audio(**bad)]))


def test_audio_clip_start_must_be_inside_the_timeline_but_may_run_past_the_end():
    # Timeline is 4 + 2 = 6 s.
    assert draft_from_dict(_draft(audio_clips=[_audio(start=5.5, source_out=10.0)])).audio_clips
    assert draft_from_dict(_draft(audio_clips=[_audio(start=6.04)])).audio_clips
    with pytest.raises(ValueError, match="after the end"):
        draft_from_dict(_draft(audio_clips=[_audio(start=6.2)]))
    # Speed-aware: 4 s at 2x + 2 s = 4 s timeline.
    fast = [{"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 4.0, "speed": 2},
            {"clip_id": "c2", "source_asset_id": "s1", "start": 5.0, "end": 7.0}]
    with pytest.raises(ValueError, match="after the end"):
        draft_from_dict(_draft(selected=fast, audio_clips=[_audio(start=5.0)]))


def test_audio_ids_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        draft_from_dict(_draft(audio_clips=[_audio(), _audio()]))


@pytest.mark.parametrize("volume", [-0.1, 2.1, "loud"])
def test_video_volume_range(volume):
    with pytest.raises(ValueError):
        draft_from_dict(_draft(video_volume=volume))


def test_video_volume_zero_is_kept():
    draft = draft_from_dict(_draft(video_volume=0))
    assert draft.video_volume == 0.0
    assert result_to_dict(draft)["video_volume"] == 0.0


# --- second wave: clip rotation, overlay volume, enhanced_at_record, zoom_removed ---------------

def test_second_wave_fields_default_and_are_omitted():
    draft = draft_from_dict(_draft(media_overlays=[_overlay()], audio_clips=[_audio()]))
    assert all(clip.rotation == 0 for clip in draft.selected)
    assert draft.media_overlays[0].volume == 1.0
    assert draft.audio_clips[0].enhanced_at_record is False
    assert draft.zoom_removed == ()
    data = result_to_dict(draft)
    assert "zoom_removed" not in data and all("rotation" not in clip for clip in data["selected"])
    assert "volume" not in data["media_overlays"][0] and "enhanced_at_record" not in data["audio_clips"][0]


def test_second_wave_fields_round_trip():
    draft = draft_from_dict(_draft(
        selected=[{"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 4.0, "rotation": 270}],
        media_overlays=[_overlay(kind="video", uri="s3://b/k.mp4", volume=1.5, mute_audio=False)],
        audio_clips=[_audio(enhanced_at_record=True)],
        zoom_removed=[{"source_asset_id": "s1", "source_time": 1.25}]))
    assert draft.selected[0].rotation == 270 and draft.media_overlays[0].volume == 1.5
    assert draft.audio_clips[0].enhanced_at_record is True
    assert draft.zoom_removed[0].source_asset_id == "s1" and draft.zoom_removed[0].source_time == 1.25
    data = json.loads(json.dumps(result_to_dict(draft)))
    assert data["selected"][0]["rotation"] == 270 and data["media_overlays"][0]["volume"] == 1.5
    assert data["audio_clips"][0]["enhanced_at_record"] is True
    assert data["zoom_removed"] == [{"source_asset_id": "s1", "source_time": 1.25}]
    assert result_to_dict(draft_from_dict(data)) == result_to_dict(draft)


@pytest.mark.parametrize("bad", [
    {"selected": [{"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 1.0, "rotation": 45}]},
    {"selected": [{"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 1.0, "rotation": "up"}]},
    {"media_overlays": [_overlay(volume=2.5)]},
    {"media_overlays": [_overlay(volume=-1)]},
    {"zoom_removed": [{"source_asset_id": "s1"}]},
    {"zoom_removed": [{"source_asset_id": "s1", "source_time": -1}]},
    {"zoom_removed": {"source_time": 1}},
])
def test_second_wave_bad_values_rejected(bad):
    with pytest.raises(ValueError):
        draft_from_dict(_draft(**bad))


def test_clip_rotation_adds_to_the_draft_rotation_and_splits_segments():
    from cutsell_worker.render_plan import build_render_plan
    draft = draft_from_dict(_draft(rotation=90, selected=[
        {"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 2.0, "rotation": 270},
        {"clip_id": "c2", "source_asset_id": "s1", "start": 2.0, "end": 4.0},
        {"clip_id": "c3", "source_asset_id": "s1", "start": 4.0, "end": 6.0, "rotation": 90}]))
    plan = build_render_plan(draft, {"s1": "/a.mp4"})
    assert [(seg.clip_id, seg.rotation) for seg in plan] == [("c1", 0), ("c2", 90), ("c3", 180)]


def test_tombstoned_ai_zoom_is_not_drawn_but_user_zoom_is():
    from cutsell_worker.render_plan import build_render_plan
    draft = draft_from_dict(_draft(auto_zoom_enabled=True, zoom_keyframes=[
        {"keyframe_id": "z1", "source_asset_id": "s1", "source_time": 1.0, "duration": 1.0, "scale": 1.1},
        {"keyframe_id": "z2", "source_asset_id": "s1", "source_time": 2.0, "duration": 1.0, "scale": 1.1, "origin": "user"},
        {"keyframe_id": "z3", "source_asset_id": "s1", "source_time": 3.0, "duration": 1.0, "scale": 1.1},
    ], zoom_removed=[{"source_asset_id": "s1", "source_time": 1.2}, {"source_asset_id": "s1", "source_time": 2.0},
                     {"source_asset_id": "other", "source_time": 3.0}]))
    plan = build_render_plan(draft, {"s1": "/a.mp4"})
    assert [k[0] for k in plan[0].zoom_keyframes] == [2.0, 3.0]


def test_overlay_volume_only_changes_the_command_when_set(monkeypatch):
    from cutsell_worker import media_overlay_render
    from cutsell_worker.contracts import MediaOverlay
    from cutsell_worker.media_probe import MediaProbe
    monkeypatch.setattr(media_overlay_render, "probe_media",
                        lambda path: MediaProbe(duration_sec=5, width=640, height=360, fps=30, has_audio=True))
    overlay = MediaOverlay(overlay_id="v", kind="video", uri="s3://b/v.mp4", start=1.0, end=2.0, mute_audio=False)

    def graph(o):
        command = media_overlay_render.build_final_overlay_command(
            "/j.mp4", "/o.mp4", media_overlays=(media_overlay_render.LocalMediaOverlay(o, "/v.mp4"),))
        return command[command.index("-filter_complex") + 1]

    from dataclasses import replace
    assert "volume=" not in graph(overlay)
    assert "asetpts=PTS-STARTPTS,volume=1.500,adelay=1000|1000[ova1]" in graph(replace(overlay, volume=1.5))


def test_edge_feather_is_pixels_at_1080_scaled_to_the_output():
    from cutsell_worker.media_overlay_render import edge_feather_px
    assert edge_feather_px("soft", 1080, 1920) == 12.0
    assert edge_feather_px("extra_soft", 1080, 1920) == 28.0
    assert edge_feather_px("sharp", 1080, 1920) == 0.0
    assert edge_feather_px("soft", 360, 640) == pytest.approx(4.0)
    assert edge_feather_px("extra_soft", 2160, 3840) == pytest.approx(56.0)
