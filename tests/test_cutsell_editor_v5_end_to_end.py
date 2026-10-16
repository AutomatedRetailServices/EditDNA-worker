"""Editor v5 end to end: a draft dict written the way the iPhone app writes it -> draft_from_dict ->
render plan -> the export's own audio-lane resolution (export_job.resolve_audio_mix, with the
registry and the sound-library download pointed at local files) -> render_with_post_render_qc (the
export's render + technical QC) -> measurements on the real MP4.

All media is generated with lavfi into tmp_path (tiny 360x640 sources, a few seconds); the export
renders at its real 1080x1920 output.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import cutsell_worker.export_job as export_job
from cutsell_worker import timeline_asset_registry as reg
from cutsell_worker.live_render_qc import render_with_post_render_qc
from cutsell_worker.media_overlay_render import LocalMediaOverlay
from cutsell_worker.render_plan import build_render_plan
from cutsell_worker.serde import draft_from_dict
from cutsell_worker.keyframes import overlay_track
from cutsell_worker.timeline_composition_executor import ResolvedTimelineAsset

W, H = 1080, 1920
NOISE = "noise=alls=12:allf=t"  # keeps solid test pictures moving (the QC rejects a frozen picture)


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True, capture_output=True)


def _video(path: Path, video: str, audio: str, seconds: float, vf: str | None = None) -> str:
    args = ["-f", "lavfi", "-i", video, "-f", "lavfi", "-i", audio]
    if vf:
        args += ["-vf", vf]
    _ff(*args, "-t", f"{seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path))
    return str(path)


def _audio(path: Path, source: str, seconds: float, af: str | None = None) -> str:
    _ff("-f", "lavfi", "-i", source, *(["-af", af] if af else []), "-t", f"{seconds}", "-c:a", "aac", str(path))
    return str(path)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    m = {"root": root}
    tone = "sine=frequency=440:sample_rate=48000:duration=8"
    m["moving"] = _video(root / "moving.mp4", "testsrc2=s=360x640:r=30:d=8", tone, 8, vf=None)
    m["moving_quiet"] = _video(root / "moving_quiet.mp4", "testsrc2=s=360x640:r=30:d=8", tone, 8)
    _ff("-i", m["moving_quiet"], "-c:v", "copy", "-af", "volume=0.5", "-c:a", "aac", str(root / "mq.mp4"))
    m["moving_quiet"] = str(root / "mq.mp4")
    m["split"] = _video(root / "split.mp4", "color=c=red:s=360x640:r=30:d=4", tone, 4,
                        vf=f"drawbox=x=180:y=0:w=180:h=640:color=blue:t=fill,{NOISE}")
    m["dark"] = _video(root / "dark.mp4", "color=c=0x202020:s=360x640:r=30:d=6", tone, 6, vf=NOISE)
    m["voice"] = _audio(root / "voice.m4a", "sine=frequency=1000:sample_rate=48000:duration=4", 4, af="volume=0.3")
    m["noisy_voice"] = _audio(root / "noisy_voice.m4a",
                              "anoisesrc=color=white:amplitude=0.08:sample_rate=48000:duration=3", 3)
    m["tap"] = _audio(root / "tap.m4a", "sine=frequency=2000:sample_rate=48000:duration=0.4", 0.4, af="volume=0.5")
    for name, color, size in (("red", "red", "200x200"), ("green", "lime", "200x200")):
        _ff("-f", "lavfi", "-i", f"color=c={color}:s={size}:d=1", "-frames:v", "1", str(root / f"{name}.png"))
        m[name] = str(root / f"{name}.png")
    # A photo whose left half is red and right half green, to prove overlays are never mirrored.
    _ff("-f", "lavfi", "-i", "color=c=red:s=200x100:d=1", "-vf", "drawbox=x=100:y=0:w=100:h=100:color=lime:t=fill",
        "-frames:v", "1", str(root / "lr.png"))
    m["lr"] = str(root / "lr.png")
    return m


@pytest.fixture
def export_files(monkeypatch, media):
    """The export's own resolution, pointed at local files: voice/music asset ids -> local audio,
    the official `tap` sound -> the local tap file."""
    assets = {"asset-voice": media["voice"], "asset-noisy": media["noisy_voice"]}

    def resolve(*, user_id, project_id, asset_id, required_media_kind, materialize=None, **_):
        assert required_media_kind == reg.TimelineMediaKind.AUDIO and user_id == "u1" and project_id == "p1"
        if asset_id not in assets:
            return reg.AssetResolutionResult(outcome=reg.ASSET_NOT_FOUND, reason_codes=(reg.ASSET_NOT_FOUND,))
        return reg.AssetResolutionResult(outcome=reg.ASSET_RESOLVED, resolved=ResolvedTimelineAsset(
            asset_id=asset_id, local_path=assets[asset_id], duration_sec=4.0, has_audio=True))

    def download(uri, destination):
        assert uri.endswith("assets/sfx/v1/tap_select.m4a"), uri
        shutil.copyfile(media["tap"], destination)
        return destination

    monkeypatch.setenv("S3_BUCKET", "cutsell-media-test")
    monkeypatch.setattr(export_job.timeline_store, "resolve_timeline_asset_live", resolve)
    monkeypatch.setattr(export_job, "download_source", download)
    # Build 17 Clean up: the cleaned copies live in a local stand-in for the bucket.
    store: dict[str, str] = {}
    root = Path(media["root"]) / "clean-store"
    root.mkdir(exist_ok=True)

    class Missing(Exception):
        response = {"Error": {"Code": "404"}}

    def head(bucket, key, **_):
        if key not in store:
            raise Missing(key)
        return {}

    def fetch(uri, destination, **_):
        key = uri.split("/", 3)[3]
        shutil.copyfile(store[key], destination)
        return destination

    class Uploader:
        def upload_file(self, local, bucket, key, ExtraArgs=None):
            target = root / f"{len(store):04d}.m4a"
            shutil.copyfile(local, target)
            store[key] = str(target)

    import cutsell_worker.storage as storage_module
    from cutsell_worker import audio_clean, storage_routing
    monkeypatch.setattr(storage_routing, "head", head)
    monkeypatch.setattr(storage_routing, "default_client", lambda **_: Uploader())
    monkeypatch.setattr(storage_module, "download_source", fetch)
    monkeypatch.setattr(audio_clean, "fetch_input", lambda *, ident, **_: assets[ident])
    return store


def _export(tmp_path, payload: dict, sources: dict[str, str], overlay_files: dict[str, str] | None = None,
            name: str = "out.mp4") -> str:
    """The export pipeline for one app-written draft; returns the delivered MP4."""
    draft = draft_from_dict(json.loads(json.dumps(payload)))
    work = tmp_path / name.replace(".mp4", "")
    work.mkdir()
    clean_paths = export_job.export_clean_audio_paths(draft, dict(sources), user_id="u1", project_id="p1",
                                                      directory=str(work))
    plan = build_render_plan(draft, sources, clean_paths)
    overlays = tuple(LocalMediaOverlay(
        overlay=o, path=(overlay_files or {})[o.overlay_id],
        keyframes=overlay_track(draft.keyframes, overlay_id=o.overlay_id) if o.layout == "float" else (),
    ) for o in draft.media_overlays)
    mix = export_job.resolve_audio_mix(draft, user_id="u1", project_id="p1", directory=str(work))
    result = render_with_post_render_qc(
        draft, plan, str(tmp_path / name), text_overlays=draft.text_overlays, media_overlays=overlays,
        **({"audio_mix": mix} if mix is not None else {}),
    )
    assert result.status == "PASS", result
    return result.output_path


def _draft(selected, **extra):
    payload = {"schema_version": "cutsell.v1", "project_id": "p1", "strategy": "mixed", "captions_enabled": False,
               "selected": selected, "alternates": [], "discarded": [], "diagnostics": {}}
    payload.update(extra)
    return payload


def _clip(clip_id, start, end, source="s1", **extra):
    item = {"clip_id": clip_id, "source_asset_id": source, "source_order": 0, "start": start, "end": end,
            "text": "", "caption_text": "", "semantic_role": "OTHER", "selected": True}
    item.update(extra)
    return item


def _durations(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", path],
                         check=True, capture_output=True, text=True).stdout
    return {s["codec_type"]: float(s["duration"]) for s in json.loads(out)["streams"]}


def _level(path, start, duration, band: int | None = None) -> float:
    """Mean level (dB) of the output's audio in a window, optionally one frequency band only."""
    chain = (f"bandpass=f={band}:width_type=h:w=100,bandpass=f={band}:width_type=h:w=100," if band else "") + "volumedetect"
    err = subprocess.run(["ffmpeg", "-hide_banner", "-ss", f"{start}", "-t", f"{duration}", "-i", path, "-vn",
                          "-af", chain, "-f", "null", "-"], capture_output=True, text=True).stderr
    match = re.search(r"mean_volume:\s*(-?[0-9.]+|-inf) dB", err)
    return -200.0 if match is None or match.group(1) == "-inf" else float(match.group(1))


def _frame(path, at) -> bytes:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{at}", "-i", path, "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    assert len(raw) == W * H * 3
    return raw


def _px(raw, x, y):
    o = (int(y) * W + int(x)) * 3
    return tuple(raw[o:o + 3])


def _close(actual, expected, tolerance=45):
    return all(abs(a - e) <= tolerance for a, e in zip(actual, expected))


# 1 -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("start,end,speed,expected", [(0.0, 4.0, 2, 2.0), (0.0, 2.0, 0.5, 4.0)])
def test_speed_changes_the_delivered_duration(tmp_path, media, export_files, start, end, speed, expected):
    out = _export(tmp_path, _draft([_clip("c1", start, end, speed=speed)]), {"s1": media["moving"]})
    streams = _durations(out)
    assert abs(streams["video"] - expected) <= 1 / 30 + 0.02
    assert abs(streams["audio"] - streams["video"]) <= 0.05


# 2 -------------------------------------------------------------------------------------------------

def test_mute_and_200_percent_volume(tmp_path, media, export_files):
    src = {"s1": media["moving_quiet"]}
    normal = _export(tmp_path, _draft([_clip("c1", 0.0, 3.0)]), src, name="normal.mp4")
    base = _level(normal, 0.5, 2.0)
    loud_video = _export(tmp_path, _draft([_clip("c1", 0.0, 3.0)], video_volume=2), src, name="v200.mp4")
    loud_clip = _export(tmp_path, _draft([_clip("c1", 0.0, 3.0, audio_volume=2)]), src, name="c200.mp4")
    assert _level(loud_video, 0.5, 2.0) == pytest.approx(base + 6.0, abs=1.0)
    assert _level(loud_clip, 0.5, 2.0) == pytest.approx(base + 6.0, abs=1.0)
    # A muted spoken clip (the creator's choice; the QC does not "repair" its silence).
    muted = _export(tmp_path, _draft([_clip("c1", 0.0, 3.0, audio_muted=True, text="hi there",
                                            words=[{"text": "hi", "start": 0.2, "end": 0.5}])]), src, name="mute.mp4")
    assert _level(muted, 0.5, 2.0) < -80
    zero = _export(tmp_path, _draft([_clip("c1", 0.0, 3.0)], video_volume=0), src, name="zero.mp4")
    assert _level(zero, 0.5, 2.0) < -80


# 3 -------------------------------------------------------------------------------------------------

def test_voice_ducks_the_video_by_about_12_db(tmp_path, media, export_files):
    voice = {"audio_id": "vo1", "kind": "voice", "asset_id": "asset-voice", "start": 2.0, "source_in": 0.0,
             "source_out": 1.5, "volume": 1.0, "duck_video": True, "clean": False}
    out = _export(tmp_path, _draft([_clip("c1", 0.0, 5.0)], audio_clips=[voice]), {"s1": media["moving_quiet"]})
    before = _level(out, 0.5, 1.0, band=440)
    during = _level(out, 2.3, 0.9, band=440)
    after = _level(out, 4.0, 0.8, band=440)
    assert during == pytest.approx(before - 12.0, abs=1.5)
    assert after == pytest.approx(before, abs=1.0)
    assert _level(out, 2.3, 0.9, band=1000) > _level(out, 0.5, 1.0, band=1000) + 15  # the voice is there


# 4 + 5 ---------------------------------------------------------------------------------------------

def test_layered_overlays_and_mirror_only_on_the_base_video(tmp_path, media, export_files):
    overlays = [
        # listed first but on the higher layer: must be on top where they overlap
        {"overlay_id": "red", "kind": "photo", "uri": "s3://b/red.png", "start": 0.0, "end": 3.0,
         "x": 0.5, "y": 0.3, "width": 0.4, "layer": 5},
        {"overlay_id": "green", "kind": "photo", "uri": "s3://b/green.png", "start": 0.0, "end": 3.0,
         "x": 0.6, "y": 0.35, "width": 0.4, "layer": 1},
        {"overlay_id": "lr", "kind": "photo", "uri": "s3://b/lr.png", "start": 0.0, "end": 3.0,
         "x": 0.5, "y": 0.8, "width": 0.5},
    ]
    files = {"red": media["red"], "green": media["green"], "lr": media["lr"]}
    plain = _export(tmp_path, _draft([_clip("c1", 0.0, 3.0)], media_overlays=overlays), {"s1": media["split"]},
                    files, name="plain.mp4")
    mirrored = _export(tmp_path, _draft([_clip("c1", 0.0, 3.0)], media_overlays=overlays, mirror=True),
                       {"s1": media["split"]}, files, name="mirrored.mp4")
    for path in (plain, mirrored):
        raw = _frame(path, 1.0)
        assert _close(_px(raw, W * 0.5, H * 0.3), (255, 0, 0))                    # overlap: higher layer (red)
        assert _close(_px(raw, W * 0.6 + 200, H * 0.35 + 200), (0, 225, 0))       # green alone
        # the left/right photo is never mirrored
        assert _close(_px(raw, W * 0.5 - 200, H * 0.8), (255, 0, 0))
        assert _close(_px(raw, W * 0.5 + 200, H * 0.8), (0, 225, 0))
    plain_raw, mirrored_raw = _frame(plain, 1.0), _frame(mirrored, 1.0)
    # the base video IS mirrored: red|blue becomes blue|red
    assert _close(_px(plain_raw, 100, H * 0.6), (255, 0, 0)) and _close(_px(plain_raw, W - 100, H * 0.6), (0, 0, 255))
    assert _close(_px(mirrored_raw, 100, H * 0.6), (0, 0, 255)) and _close(_px(mirrored_raw, W - 100, H * 0.6), (255, 0, 0))


def test_clip_rotation_turns_only_that_clip(tmp_path, media, export_files):
    # 180 degrees on the clip: red|blue becomes blue|red, overlays stay upright.
    out = _export(tmp_path, _draft([_clip("c1", 0.0, 3.0, rotation=180)], media_overlays=[
        {"overlay_id": "lr", "kind": "photo", "uri": "s3://b/lr.png", "start": 0.0, "end": 3.0,
         "x": 0.5, "y": 0.8, "width": 0.5}]), {"s1": media["split"]}, {"lr": media["lr"]})
    raw = _frame(out, 1.0)
    assert _close(_px(raw, 100, H * 0.4), (0, 0, 255)) and _close(_px(raw, W - 100, H * 0.4), (255, 0, 0))
    assert _close(_px(raw, W * 0.5 - 200, H * 0.8), (255, 0, 0))


# 6 -------------------------------------------------------------------------------------------------

def test_sfx_is_heard_at_its_time_only(tmp_path, media, export_files):
    tap = {"audio_id": "sfx1", "kind": "sfx", "sfx_id": "tap", "start": 1.5, "source_in": 0.0, "source_out": 0.3}
    out = _export(tmp_path, _draft([_clip("c1", 0.0, 4.0)], video_volume=0, audio_clips=[tap]),
                  {"s1": media["moving_quiet"]})
    assert _level(out, 1.55, 0.2) > -40
    assert _level(out, 0.3, 1.0) < -80
    assert _level(out, 2.2, 1.5) < -80


# 7 -------------------------------------------------------------------------------------------------

def _caption_pixels(raw) -> int:
    band = range(int(H * 0.70), int(H * 0.82), 4)
    return sum(1 for y in band for x in range(0, W, 4) if min(_px(raw, x, y)) > 200)


def test_captions_stay_on_their_words_after_a_speed_change(tmp_path, media, export_files):
    # Clip 1.0-5.0 s at 2x; the word is spoken at 3.0-3.6 s of the source -> output 1.0-1.3 s.
    clip = _clip("c1", 1.0, 5.0, speed=2, text="HELLO", caption_text="HELLO",
                 words=[{"text": "HELLO", "start": 3.0, "end": 3.6}])
    out = _export(tmp_path, _draft([clip], captions_enabled=True, diagnostics={"engine": "simple"}),
                  {"s1": media["dark"]})
    assert _caption_pixels(_frame(out, 0.8)) == 0          # not before its word (source 2.6 s)
    assert _caption_pixels(_frame(out, 1.15)) > 50         # on its word (source 3.3 s)


# 8 -------------------------------------------------------------------------------------------------

def test_cleanup_changes_the_sound_only_where_it_is_on(tmp_path, media, export_files):
    def noisy(clean):
        return {"audio_id": "vo1", "kind": "voice", "asset_id": "asset-noisy", "start": 1.0, "source_in": 0.0,
                "source_out": 2.0, "volume": 1.0, "clean": clean}

    src = {"s1": media["moving_quiet"]}
    raw = _export(tmp_path, _draft([_clip("c1", 0.0, 4.0)], audio_clips=[noisy(False)]), src, name="raw.mp4")
    clean = _export(tmp_path, _draft([_clip("c1", 0.0, 4.0)], audio_clips=[noisy(True)]), src, name="clean.mp4")
    # inside the clip's window the noise is treated (denoise + loudness): the high band changes
    assert abs(_level(clean, 1.3, 1.4, band=6000) - _level(raw, 1.3, 1.4, band=6000)) > 3
    # outside it, nothing changes
    assert _level(clean, 3.2, 0.7) == pytest.approx(_level(raw, 3.2, 0.7), abs=0.5)
    assert _level(clean, 0.1, 0.8) == pytest.approx(_level(raw, 0.1, 0.8), abs=0.5)
    # whole-video cleanup changes the video's own sound
    video_clean = _export(tmp_path, _draft([_clip("c1", 0.0, 4.0)], video_clean_voice=True), src, name="vclean.mp4")
    plain = _export(tmp_path, _draft([_clip("c1", 0.0, 4.0)]), src, name="plain.mp4")
    assert abs(_level(video_clean, 1.0, 2.0) - _level(plain, 1.0, 2.0)) > 3


# API round trip ------------------------------------------------------------------------------------

EVERY_FIELD_DRAFT = {
    "schema_version": "cutsell.v1", "project_id": "p1", "strategy": "mixed",
    "selected": [
        {"clip_id": "c1", "source_asset_id": "s1", "source_order": 0, "start": 0.0, "end": 4.0, "text": "hello there",
         "caption_text": "Hello!", "semantic_role": "OTHER", "selected": True,
         "words": [{"text": "hello", "start": 0.2, "end": 0.6}, {"text": "there", "start": 0.7, "end": 1.1}],
         "speed": 1.5, "rotation": 90, "audio_volume": 1.4, "audio_muted": False, "mirror": True, "manual_end": True},
        {"clip_id": "c2", "source_asset_id": "s1", "source_order": 0, "start": 5.0, "end": 7.0, "text": "",
         "caption_text": "", "semantic_role": "OTHER", "selected": True, "audio_muted": True, "speed": 0.5},
    ],
    "alternates": [], "discarded": [], "diagnostics": {"engine": "simple"},
    "captions_enabled": True, "caption_preset": "box", "caption_font": "anton",
    "caption_x": 0.5, "caption_y": 0.7, "caption_scale": 1.2,
    "mirror": True, "rotation": 180, "audio_overlap_enabled": True, "auto_zoom_enabled": True,
    "video_volume": 1.6, "video_clean_voice": True,
    "media_overlays": [
        {"overlay_id": "o1", "kind": "video", "uri": "s3://bucket/cutsell/overlay-assets/x/y/clip.mp4",
         "start": 0.5, "end": 2.0, "x": 0.4, "y": 0.6, "width": 0.5, "source_start": 1.0, "source_end": 2.5,
         "mute_audio": False, "volume": 0.7, "layer": 4, "layout": "side_by_side", "edge": "extra_soft",
         "fill": True, "rotation": -30.0},
        {"overlay_id": "o2", "kind": "photo", "uri": "s3://bucket/cutsell/overlay-assets/x/y/p.png",
         "start": 1.0, "end": 3.0, "x": 0.5, "y": 0.5, "width": 0.3, "source_start": 0.0, "source_end": None,
         "mute_audio": True, "layer": 2, "layout": "float", "edge": "soft", "rotation": 15.0},
    ],
    "text_overlays": [{"overlay_id": "t1", "text": "SALE", "start": 0.0, "end": 1.0, "x": 0.5, "y": 0.2, "scale": 1.0}],
    "audio_clips": [
        {"audio_id": "a1", "kind": "voice", "asset_id": "asset-voice", "start": 0.5, "source_in": 0.1,
         "source_out": 2.0, "volume": 1.3, "duck_video": True, "clean": True, "enhanced_at_record": True},
        {"audio_id": "a2", "kind": "music", "asset_id": "asset-music", "start": 0.0, "source_in": 3.0,
         "source_out": 30.0, "volume": 0.4, "duck_video": False, "clean": False},
        {"audio_id": "a3", "kind": "sfx", "sfx_id": "typing_long", "start": 2.2, "source_in": 0.0,
         "source_out": 1.0, "volume": 2.0, "duck_video": False, "clean": False},
    ],
    "zoom_keyframes": [
        {"keyframe_id": "z1", "source_asset_id": "s1", "source_time": 1.0, "duration": 1.5, "scale": 1.1,
         "kind": "hook", "origin": "ai"},
        {"keyframe_id": "z2", "source_asset_id": "s1", "source_time": 2.5, "duration": 1.0, "scale": 1.2,
         "kind": "key", "origin": "user"},
    ],
    "zoom_removed": [{"source_asset_id": "s1", "source_time": 3.25}],
    "broll_sources": [{"source_asset_id": "br1", "uri": "s3://bucket/cutsell/uploads/x/y/br.mp4"}],
}


def test_api_round_trip_keeps_every_editor_field(monkeypatch):
    from fastapi.testclient import TestClient
    import cutsell_app.main as api
    from cutsell_worker import draft_store
    from cutsell_worker.serde import result_to_dict
    from tests.test_cutsell_clean_worker_draft_store import FakeRedis

    fake = FakeRedis()
    monkeypatch.setattr(draft_store, "_redis_client", lambda client=None: fake)
    first = dict(EVERY_FIELD_DRAFT, selected=[EVERY_FIELD_DRAFT["selected"][0]], media_overlays=[], audio_clips=[])
    draft_store.create_initial_draft(user_id="u1", project_id="p1", draft=first,
                                     sources=[{"source_asset_id": "s1", "uri": "s3://bucket/cutsell/uploads/x/y/s.mov"}])
    client = TestClient(api.app)
    put = client.put("/v1/projects/p1/draft", json={"user_id": "u1", "expected_revision": 1, "draft": EVERY_FIELD_DRAFT})
    assert put.status_code == 200, put.text
    got = client.get("/v1/projects/p1/draft", params={"user_id": "u1"})
    assert got.status_code == 200
    body = got.json()
    assert body["revision"] == 2

    def unsigned(draft):
        out = json.loads(json.dumps(draft))
        for item in out.get("media_overlays") or ():
            item.pop("preview_url", None)
        for item in out.get("broll_sources") or ():
            item.pop("playback_url", None)
            item.pop("playback_expires_in", None)
        return out

    assert unsigned(body["draft"]) == EVERY_FIELD_DRAFT
    # And the server reads every field back identically (validated contract, not just stored JSON).
    parsed = result_to_dict(draft_from_dict(unsigned(body["draft"])))
    again = result_to_dict(draft_from_dict(parsed))
    # (`user_media_source_ids` is derived from the stored `broll_sources`, which the serialized
    # contract does not carry -- a pre-existing Editor v4 asymmetry, unrelated to v5 fields.)
    assert {k: v for k, v in parsed.items() if k != "user_media_source_ids"} == again
    assert parsed["selected"][0]["speed"] == 1.5 and parsed["selected"][0]["rotation"] == 90
    assert parsed["media_overlays"][0]["volume"] == 0.7 and parsed["media_overlays"][0]["layout"] == "side_by_side"
    assert parsed["audio_clips"][0]["enhanced_at_record"] is True
    assert parsed["zoom_removed"] == [{"source_asset_id": "s1", "source_time": 3.25}]
    assert parsed["video_volume"] == 1.6 and parsed["video_clean_voice"] is True
