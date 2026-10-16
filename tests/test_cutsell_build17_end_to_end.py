"""Build 17 end to end (iPhone draft -> API -> export, real ffmpeg renders measured):

1. Clean up voice: the stored cleaned copy lowers the noise between words, keeps the voice, is
   made once and reused, and the export plays exactly that copy (the file the preview plays).
2. Volume / Mute per clip and for the whole video: a muted sound is silent, the others keep
   playing; "Whole video" volume / Mute changes everything.
3. Keyframes: a video clip zooms smoothly from 100 % to 130 % between two keyframes; a rotation
   keyframe turns it; an overlay moves and grows between its keyframes; AI keyframes are drawn only
   with Auto Zoom AI on; an AI push-in as four keyframes is the curve the export always drew.
4. API: clean-audio start / status, and every build-17 field survives the draft round trip.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import cutsell_worker.export_job as export_job
from cutsell_worker import audio_clean, keyframes as kf, storage_routing
from cutsell_worker import timeline_asset_registry as reg
from cutsell_worker.keyframes import overlay_track
from cutsell_worker.live_render_qc import render_with_post_render_qc
from cutsell_worker.media_overlay_render import LocalMediaOverlay
from cutsell_worker.render_plan import build_render_plan
from cutsell_worker.serde import draft_from_dict
from cutsell_worker.timeline_composition_executor import ResolvedTimelineAsset

W, H = 1080, 1920


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True, capture_output=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp("b17")
    m = {"root": root}
    # "Voice" bursts (1 s on / 1 s off) over pink noise: the gaps are noise only.
    voice = ("aevalsrc='if(lt(mod(t,2),1),0.3*sin(2*PI*220*t)*(0.6+0.4*sin(2*PI*3*t))+0.15*sin(2*PI*660*t),0)'"
             ":s=48000:d=8")
    _ff("-f", "lavfi", "-i", voice, "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.05:d=8:r=48000",
        "-filter_complex", "[0][1]amix=inputs=2:normalize=0", "-ac", "2", "-c:a", "aac", str(root / "noisy.m4a"))
    m["noisy_audio"] = str(root / "noisy.m4a")
    # A moving 9:16 video with that noisy voice.
    _ff("-f", "lavfi", "-i", "testsrc2=s=360x640:r=30:d=8", "-i", m["noisy_audio"], "-t", "8",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(root / "noisy.mp4"))
    m["noisy_video"] = str(root / "noisy.mp4")
    # A dark grainy video with a white square in the middle (100 px on a 360-wide source): its
    # width on screen measures the zoom.
    _ff("-f", "lavfi", "-i", "color=c=0x101010:s=360x640:r=30:d=6", "-f", "lavfi",
        "-i", "sine=frequency=440:sample_rate=48000:duration=6",
        "-vf", "drawbox=x=130:y=270:w=100:h=100:color=white:t=fill,noise=alls=8:allf=t",
        "-t", "6", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(root / "square.mp4"))
    m["square"] = str(root / "square.mp4")
    _ff("-f", "lavfi", "-i", "sine=frequency=2000:sample_rate=48000:duration=0.6", "-af", "volume=0.5",
        "-c:a", "aac", str(root / "good.m4a"))
    m["good"] = str(root / "good.m4a")
    _ff("-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=4", "-af", "volume=0.3",
        "-c:a", "aac", str(root / "voice.m4a"))
    m["voice"] = str(root / "voice.m4a")
    _ff("-f", "lavfi", "-i", "color=c=lime:s=200x200:d=1", "-frames:v", "1", str(root / "green.png"))
    m["green"] = str(root / "green.png")
    return m


class _Missing(Exception):
    response = {"Error": {"Code": "404"}}


@pytest.fixture
def store(monkeypatch, media):
    """The bucket as a local folder; registry / sound library pointed at local files."""
    objects: dict[str, str] = {}
    folder = Path(media["root"]) / "bucket"
    folder.mkdir(exist_ok=True)
    assets = {"asset-voice": media["voice"], "asset-noisy": media["noisy_audio"]}

    def head(bucket, key, **_):
        if key not in objects:
            raise _Missing(key)
        return {}

    def fetch(uri, destination, **_):
        key = uri.split("/", 3)[3]
        if key.endswith("assets/sfx/v1/good.m4a"):
            shutil.copyfile(media["good"], destination)
            return destination
        shutil.copyfile(objects[key], destination)
        return destination

    class Uploader:
        uploads = 0

        def upload_file(self, local, bucket, key, ExtraArgs=None):
            Uploader.uploads += 1
            target = folder / f"{len(objects):04d}.m4a"
            shutil.copyfile(local, target)
            objects[key] = str(target)

    def resolve(*, user_id, project_id, asset_id, required_media_kind, materialize=None, **_):
        if asset_id not in assets:
            return reg.AssetResolutionResult(outcome=reg.ASSET_NOT_FOUND, reason_codes=(reg.ASSET_NOT_FOUND,))
        return reg.AssetResolutionResult(outcome=reg.ASSET_RESOLVED, resolved=ResolvedTimelineAsset(
            asset_id=asset_id, local_path=assets[asset_id], duration_sec=8.0, has_audio=True))

    import cutsell_worker.storage as storage_module
    monkeypatch.setenv("S3_BUCKET", "cutsell-videos")
    monkeypatch.setattr(storage_routing, "head", head)
    monkeypatch.setattr(storage_routing, "default_client", lambda **_: Uploader())
    monkeypatch.setattr(storage_module, "download_source", fetch)
    monkeypatch.setattr(export_job, "download_source", fetch)
    monkeypatch.setattr(export_job.timeline_store, "resolve_timeline_asset_live", resolve)
    monkeypatch.setattr(audio_clean, "fetch_input", lambda *, ident, **_: assets[ident])
    return {"objects": objects, "uploader": Uploader}


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


def _export(tmp_path, payload, sources, overlay_files=None, name="out.mp4"):
    """The export's own steps for one app-written draft (export_job.run_export_job minus the queue
    and the upload)."""
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
    result = render_with_post_render_qc(draft, plan, str(tmp_path / name), text_overlays=draft.text_overlays,
                                        media_overlays=overlays, **({"audio_mix": mix} if mix is not None else {}))
    assert result.status == "PASS", result
    return result.output_path


def _rms(path, start, duration) -> float:
    err = subprocess.run(["ffmpeg", "-hide_banner", "-ss", f"{start}", "-t", f"{duration}", "-i", path, "-vn",
                          "-af", "astats", "-f", "null", "-"], capture_output=True, text=True).stderr
    overall = err.split("Overall", 1)[-1]
    match = re.search(r"RMS level dB:\s*(-?[0-9.]+|-inf)", overall)
    return -200.0 if match is None or match.group(1) == "-inf" else float(match.group(1))


def _frame(path, at):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{at}", "-i", path, "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], check=True, capture_output=True).stdout
    assert len(raw) == W * H
    return raw


def _bright_width(raw, row: int) -> int:
    return sum(1 for x in range(W) if raw[row * W + x] > 200)


def _bright_box(raw, *, threshold=200):
    xs, ys = [], []
    for y in range(0, H, 4):
        for x in range(0, W, 4):
            if raw[y * W + x] > threshold:
                xs.append(x)
                ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


# 1. Clean up voice ---------------------------------------------------------------------------------

def test_clean_copy_lowers_the_noise_keeps_the_voice_and_the_length(tmp_path, media):
    out = str(tmp_path / "clean.m4a")
    facts = audio_clean.make_clean_audio(media["noisy_audio"], out)
    gap_before, gap_after = _rms(media["noisy_audio"], 5.2, 0.6), _rms(out, 5.2, 0.6)
    voice_before, voice_after = _rms(media["noisy_audio"], 4.2, 0.6), _rms(out, 4.2, 0.6)
    assert gap_after < gap_before - 12           # the noise between words is much lower
    assert abs(voice_after - voice_before) < 6   # the voice stays at about the same level
    assert facts["duration_sec"] == pytest.approx(8.0, abs=0.05)
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", out],
                                      capture_output=True, text=True).stdout)
    assert float(probe["format"]["duration"]) == pytest.approx(8.0, abs=0.06)  # same timeline: no shift


def test_whole_video_clean_up_exports_the_stored_copy_made_once(tmp_path, media, store):
    src = {"s1": media["noisy_video"]}
    plain = _export(tmp_path, _draft([_clip("c1", 0.0, 8.0)]), src, name="plain.mp4")
    cleaned = _export(tmp_path, _draft([_clip("c1", 0.0, 8.0)], video_clean_voice=True), src, name="cleaned.mp4")
    assert _rms(cleaned, 5.2, 0.6) < _rms(plain, 5.2, 0.6) - 12   # gap noise down in the video
    assert store["uploader"].uploads == 1
    # Exporting again reuses the stored copy (the one the iPhone preview downloaded): no new cleaning.
    _export(tmp_path, _draft([_clip("c1", 0.0, 8.0)], video_clean_voice=True), src, name="again.mp4")
    assert store["uploader"].uploads == 1
    key = audio_clean.clean_key(user_id="u1", project_id="p1", target="source", ident="s1")
    stored = store["objects"][key]
    # The export's sound in a voice window is the stored copy's sound (same level).
    assert _rms(cleaned, 4.2, 0.6) == pytest.approx(_rms(stored, 4.2, 0.6), abs=1.5)


def test_one_clip_clean_up_changes_only_that_clip(tmp_path, media, store):
    src = {"s1": media["noisy_video"]}
    payload = _draft([_clip("c1", 0.0, 4.0), _clip("c2", 4.0, 8.0, clean_voice=True)])
    out = _export(tmp_path, payload, src, name="one.mp4")
    plain = _export(tmp_path, _draft([_clip("c1", 0.0, 4.0), _clip("c2", 4.0, 8.0)]), src, name="plain2.mp4")
    assert _rms(out, 1.2, 0.6) == pytest.approx(_rms(plain, 1.2, 0.6), abs=1.0)   # clip 1: untouched gap
    assert _rms(out, 5.2, 0.6) < _rms(plain, 5.2, 0.6) - 12                        # clip 2: cleaned gap


def test_voice_over_clean_up_plays_the_cleaned_recording(tmp_path, media, store):
    def vo(clean):
        return {"audio_id": "vo1", "kind": "voice", "asset_id": "asset-noisy", "start": 0.0, "source_in": 4.0,
                "source_out": 8.0, "volume": 1.0, "clean": clean}
    src = {"s1": media["square"]}
    muted = [_clip("c1", 0.0, 4.0, audio_muted=True, audio_volume=0.0)]
    raw = _export(tmp_path, _draft(muted, audio_clips=[vo(False)]), src, name="vo_raw.mp4")
    clean = _export(tmp_path, _draft(muted, audio_clips=[vo(True)]), src, name="vo_clean.mp4")
    assert _rms(clean, 1.2, 0.6) < _rms(raw, 1.2, 0.6) - 12   # source 5.2 s: a gap
    assert _rms(clean, 0.2, 0.6) > -40                         # source 4.2 s: the voice is there


def test_missing_clean_copy_never_exports_the_uncleaned_sound():
    draft = draft_from_dict(_draft([_clip("c1", 0.0, 2.0, clean_voice=True)]))
    with pytest.raises(ValueError, match="clean_audio_missing"):
        build_render_plan(draft, {"s1": "/tmp/x.mp4"}, {})


# 2. Volume / Mute ------------------------------------------------------------------------------------

def _sound(audio_id, start, **extra):
    item = {"audio_id": audio_id, "kind": "sfx", "sfx_id": "good", "start": start, "source_in": 0.0,
            "source_out": 0.5}
    item.update(extra)
    return item


def test_a_muted_sound_is_silent_and_the_others_still_play(tmp_path, media, store):
    src = {"s1": media["square"]}
    # A spoken clip the creator muted (its words are still there), plus two sounds.
    base = [_clip("c1", 0.0, 4.0, audio_muted=True, audio_volume=0.0, text="hello there",
                  words=[{"text": "hello", "start": 0.2, "end": 1.5}, {"text": "there", "start": 2.0, "end": 3.8}])]
    out = _export(tmp_path, _draft(base, audio_clips=[_sound("a", 0.5, muted=True), _sound("b", 2.5)]), src,
                  name="mute.mp4")
    assert _rms(out, 0.6, 0.3) < -60     # the muted one
    assert _rms(out, 2.6, 0.3) > -40     # the other one plays


def test_whole_video_volume_and_mute(tmp_path, media, store):
    src = {"s1": media["square"]}
    full = _export(tmp_path, _draft([_clip("c1", 0.0, 4.0)]), src, name="full.mp4")
    half = _export(tmp_path, _draft([_clip("c1", 0.0, 4.0)], master_volume=0.5), src, name="half.mp4")
    mute = _export(tmp_path, _draft([_clip("c1", 0.0, 4.0)], audio_clips=[_sound("a", 1.0)], master_muted=True),
                   src, name="allmute.mp4")
    assert _rms(half, 1.0, 1.0) == pytest.approx(_rms(full, 1.0, 1.0) - 6.0, abs=1.0)
    assert _rms(mute, 0.2, 3.0) < -60


# 3. Keyframes ----------------------------------------------------------------------------------------

def _vk(kid, source_time, scale=1.0, **extra):
    item = {"keyframe_id": kid, "target": "video", "source_asset_id": "s1", "source_time": source_time,
            "scale": scale, "origin": "user"}
    item.update(extra)
    return item


def test_two_manual_keyframes_zoom_smoothly_from_100_to_130_percent(tmp_path, media, store):
    src = {"s1": media["square"]}
    out = _export(tmp_path, _draft([_clip("c1", 0.0, 4.0)], keyframes=[_vk("k1", 1.0, 1.0), _vk("k2", 3.0, 1.3)]),
                  src, name="kf.mp4")
    row = H // 2
    before, middle, after = (_bright_width(_frame(out, t), row) for t in (0.5, 2.0, 3.5))
    assert before == pytest.approx(300, abs=14)          # 100 px of 360 -> 300 px of 1080
    assert after == pytest.approx(390, abs=16)           # 130 %
    assert before + 20 < middle < after - 20             # in between at the midpoint (smooth)
    one = _export(tmp_path, _draft([_clip("c1", 0.0, 4.0)], keyframes=[_vk("k1", 1.0, 1.0)]), src, name="kf1.mp4")
    assert _bright_width(_frame(one, 3.5), row) == pytest.approx(300, abs=14)  # one deleted: no zoom left


def test_rotation_and_pan_keyframes(tmp_path, media, store):
    src = {"s1": media["square"]}
    rotated = _export(tmp_path, _draft([_clip("c1", 0.0, 3.0)], keyframes=[_vk("k1", 0.0, 1.0, rotation=45)]),
                      src, name="rot.mp4")
    # A 45° turned square is wider across its middle row than its side (300 px -> ~424 px).
    assert _bright_width(_frame(rotated, 1.0), H // 2) > 380
    panned = _export(tmp_path, _draft([_clip("c1", 0.0, 3.0)], keyframes=[_vk("k1", 0.0, 2.0, x=0.25)]),
                     src, name="pan.mp4")
    box = _bright_box(_frame(panned, 1.0))
    assert box is not None and (box[0] + box[2]) / 2 == pytest.approx(W / 2 + 0.25 * W, abs=24)
    # Panning past what the zoom allows is held at the limit (no empty edge), as in the preview.
    assert kf.clamp_offset(1.2, 0.5) == pytest.approx(0.1)


def test_ai_keyframes_follow_auto_zoom_and_user_ones_always_draw(tmp_path, media, store):
    src = {"s1": media["square"]}
    ai = [_vk("a1", 1.0, 1.0, origin="ai"), _vk("a2", 1.4, 1.25, origin="ai"), _vk("a3", 3.0, 1.25, origin="ai")]
    on = _export(tmp_path, _draft([_clip("c1", 0.0, 4.0)], keyframes=ai, auto_zoom_enabled=True), src, name="ai_on.mp4")
    off = _export(tmp_path, _draft([_clip("c1", 0.0, 4.0)], keyframes=ai, auto_zoom_enabled=False), src,
                  name="ai_off.mp4")
    assert _bright_width(_frame(on, 2.0), H // 2) > 350
    assert _bright_width(_frame(off, 2.0), H // 2) == pytest.approx(300, abs=14)


def test_ai_push_in_as_four_keyframes_is_the_curve_the_export_always_drew():
    from cutsell_worker.render import _zoom_expression
    from cutsell_worker.render_plan import RenderSegment

    zoom = {"keyframe_id": "z1", "source_asset_id": "s1", "source_time": 2.0, "duration": 1.5, "scale": 1.12,
            "origin": "ai"}
    points = kf.video_track(kf.zoom_to_keyframes(zoom), source_asset_id="s1", start=0.0, end=10.0)
    old = _zoom_expression(RenderSegment(clip_id="c", source_asset_id="s1", source_path="x", start=0.0, end=10.0,
                                         zoom_keyframes=((2.0, 1.5, 1.12),)))
    for t in (1.9, 2.0, 2.1, 2.2, 2.35, 2.9, 3.2, 3.35, 3.45, 3.6):
        legacy = eval(old.replace("if(", "_if(").replace("lt(", "_lt("), {  # noqa: S307 -- our own expression
            "_if": lambda c, a, b: a if c else b, "_lt": lambda a, b: a < b, "t": t})
        assert kf.value_at(points, t, 1) == pytest.approx(legacy, abs=1e-3), t


def test_overlay_keyframes_move_and_grow_the_overlay(tmp_path, media, store):
    src = {"s1": media["square"]}
    overlay = {"overlay_id": "o1", "kind": "photo", "uri": "s3://cutsell-videos/cutsell/overlay-assets/x/y/g.png",
               "start": 0.0, "end": 3.0, "x": 0.3, "y": 0.3, "width": 0.2, "source_start": 0.0, "layout": "float"}
    keys = [{"keyframe_id": "o-1", "target": "overlay", "overlay_id": "o1", "time": 0.5, "scale": 1.0},
            {"keyframe_id": "o-2", "target": "overlay", "overlay_id": "o1", "time": 2.5, "scale": 2.0, "x": 0.3}]
    out = _export(tmp_path, _draft([_clip("c1", 0.0, 3.0)], media_overlays=[overlay], keyframes=keys), src,
                  overlay_files={"o1": media["green"]}, name="ovkf.mp4")

    def green_box(at):
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{at}", "-i", out, "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
        xs = [x for y in range(0, H, 6) for x in range(0, W, 6)
              if raw[(y * W + x) * 3 + 1] > 180 and raw[(y * W + x) * 3] < 90 and raw[(y * W + x) * 3 + 2] < 90]
        return (min(xs), max(xs)) if xs else None

    start_box, end_box = green_box(0.2), green_box(2.6)
    assert start_box is not None and end_box is not None
    start_w, end_w = start_box[1] - start_box[0], end_box[1] - end_box[0]
    assert start_w == pytest.approx(0.2 * W, abs=20)
    assert end_w == pytest.approx(0.4 * W, abs=30)                       # 2x
    centre_start, centre_end = sum(start_box) / 2, sum(end_box) / 2
    assert centre_start == pytest.approx(0.3 * W, abs=20)
    assert centre_end == pytest.approx(0.6 * W, abs=30)                   # moved right by 0.3 of the frame


# 4. API ----------------------------------------------------------------------------------------------

def test_clean_audio_api_starts_once_reports_progress_and_hands_out_the_link(monkeypatch, store):
    from fastapi.testclient import TestClient
    import cutsell_app.clean_audio_routes as routes
    import cutsell_app.main as api
    from cutsell_worker import draft_store, queueing
    from tests.test_cutsell_clean_worker_draft_store import FakeRedis

    fake = FakeRedis()
    monkeypatch.setattr(draft_store, "_redis_client", lambda client=None: fake)
    monkeypatch.setattr(routes, "_redis", lambda: fake)
    draft_store.create_initial_draft(user_id="u1", project_id="p1", draft=_draft([_clip("c1", 0.0, 2.0)]),
                                     sources=[{"source_asset_id": "s1", "uri": "s3://cutsell-videos/cutsell/uploads/x/y/s.mp4"}])
    jobs = []
    monkeypatch.setattr(queueing, "enqueue_clean_audio",
                        lambda payload, **_: jobs.append(payload) or queueing.QueueSubmission(job_id="j1", queue_name="cutsell"))
    monkeypatch.setattr(storage_routing, "presigned_get", lambda bucket, key, **_: f"https://signed/{key}")
    client = TestClient(api.app)
    body = {"user_id": "u1", "target": "source", "id": "s1"}
    first = client.post("/v1/projects/p1/clean-audio", json=body).json()
    second = client.post("/v1/projects/p1/clean-audio", json=body).json()
    assert first["status"] == "cleaning" and second["status"] == "cleaning" and len(jobs) == 1
    assert client.post("/v1/projects/p1/clean-audio", json=dict(body, id="nope")).status_code == 404
    audio_clean.write_status(fake, user_id="u1", project_id="p1", target="source", ident="s1", state="cleaning",
                             progress=40)
    status = client.get("/v1/projects/p1/clean-audio", params={"user_id": "u1", "target": "source", "id": "s1"}).json()
    assert status == {"status": "cleaning", "progress": 40}
    key = audio_clean.clean_key(user_id="u1", project_id="p1", target="source", ident="s1")
    store["objects"][key] = "/dev/null"
    ready = client.get("/v1/projects/p1/clean-audio", params={"user_id": "u1", "target": "source", "id": "s1"}).json()
    assert ready["status"] == "ready" and ready["url"].endswith(key)
    # The copy lives inside the project's own upload folder (deleted with the project / account).
    from cutsell_worker.uploads import scoped_upload_prefix
    assert key.startswith(scoped_upload_prefix(user_id="u1", project_id="p1"))


def test_api_round_trip_keeps_every_build17_field(monkeypatch):
    from fastapi.testclient import TestClient
    import cutsell_app.main as api
    from cutsell_worker import draft_store
    from cutsell_worker.serde import result_to_dict
    from tests.test_cutsell_clean_worker_draft_store import FakeRedis

    fake = FakeRedis()
    monkeypatch.setattr(draft_store, "_redis_client", lambda client=None: fake)
    draft = _draft(
        [_clip("c1", 0.0, 4.0, clean_voice=True, audio_volume=0.6), _clip("c2", 4.0, 6.0, audio_muted=True)],
        master_volume=1.4, master_muted=True,
        audio_clips=[_sound("a", 0.5, muted=True, volume=0.8)],
        media_overlays=[{"overlay_id": "o1", "kind": "photo", "uri": "s3://bucket/cutsell/overlay-assets/x/y/p.png",
                         "start": 0.0, "end": 2.0, "x": 0.5, "y": 0.5, "width": 0.3, "source_start": 0.0,
                         "source_end": None, "mute_audio": True}],
        keyframes=[{"keyframe_id": "k1", "target": "video", "source_asset_id": "s1", "source_time": 1.0, "scale": 1.3,
                    "x": 0.1, "y": -0.05, "rotation": 10.0, "origin": "user", "kind": "key"},
                   {"keyframe_id": "k2", "target": "overlay", "overlay_id": "o1", "time": 0.5, "scale": 0.5,
                    "x": 0.0, "y": 0.0, "rotation": 0.0, "origin": "user", "kind": "key"}],
        auto_zoom_enabled=True,
    )
    draft_store.create_initial_draft(user_id="u1", project_id="p1", draft=_draft([_clip("c1", 0.0, 4.0)]),
                                     sources=[{"source_asset_id": "s1", "uri": "s3://bucket/cutsell/uploads/x/y/s.mov"}])
    client = TestClient(api.app)
    put = client.put("/v1/projects/p1/draft", json={"user_id": "u1", "expected_revision": 1, "draft": draft})
    assert put.status_code == 200, put.text
    got = client.get("/v1/projects/p1/draft", params={"user_id": "u1"}).json()["draft"]
    for item in got.get("media_overlays") or ():
        item.pop("preview_url", None)
    assert got == draft
    parsed = result_to_dict(draft_from_dict(got))
    assert parsed["master_volume"] == 1.4 and parsed["master_muted"] is True
    assert parsed["selected"][0]["clean_voice"] is True and parsed["audio_clips"][0]["muted"] is True
    assert [k["keyframe_id"] for k in parsed["keyframes"]] == ["k1", "k2"]
    assert result_to_dict(draft_from_dict(parsed))["keyframes"] == parsed["keyframes"]
    bad = dict(draft, keyframes=[{"keyframe_id": "x", "target": "video", "source_time": 1, "scale": 9}])
    assert client.put("/v1/projects/p1/draft", json={"user_id": "u1", "expected_revision": 2, "draft": bad}).status_code == 422
