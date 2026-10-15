"""Editor v5 final audio mix (voice / music / sound effects, video volume, voice cleanup, ducking)
and the QC / delivery-review exemptions for what the creator chose (speed, muted speech, volume 0)."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from cutsell_worker import render
from cutsell_worker.audio_mix_render import AudioMix, LocalAudioClip, build_audio_mix_command, duck_volume_filter
from cutsell_worker.live_boundary_repair import segment_output_windows
from cutsell_worker.live_render_qc import user_placed_windows
from cutsell_worker.render_plan import RenderSegment, build_render_plan
from cutsell_worker.serde import draft_from_dict


def _make(path: Path, *inputs: str, seconds: float, extra=()) -> str:
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for item in inputs:
        command += ["-f", "lavfi", "-i", item]
    command += [*extra, "-t", f"{seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)]
    subprocess.run(command, check=True, capture_output=True)
    return str(path)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp("mix")
    base = _make(root / "base.mp4", "testsrc2=s=360x640:r=30:d=3",
                 "sine=frequency=440:sample_rate=48000:duration=3", seconds=3, extra=["-af", "volume=0.3"])
    voice = root / "voice.m4a"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=1000:sample_rate=48000:duration=4", "-af", "volume=0.3",
                    "-c:a", "aac", str(voice)], check=True, capture_output=True)
    silent = _make(root / "silent.mp4", "testsrc2=s=360x640:r=30:d=4", "anullsrc=r=48000:cl=stereo", seconds=4)
    return root, base, str(voice), silent


def _band_level(path: str, start: float, duration: float, frequency: int) -> float:
    """Mean level (dB) of one frequency band inside a time window."""
    err = subprocess.run([
        "ffmpeg", "-hide_banner", "-ss", f"{start}", "-t", f"{duration}", "-i", path, "-vn",
        "-af", f"bandpass=f={frequency}:width_type=h:w=100,bandpass=f={frequency}:width_type=h:w=100,volumedetect",
        "-f", "null", "-",
    ], capture_output=True, text=True).stderr
    return float(re.search(r"mean_volume:\s*(-?[0-9.]+|-inf) dB", err).group(1).replace("-inf", "-200"))


def _durations(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration,codec_name",
                          "-of", "json", path], check=True, capture_output=True, text=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def _render(tmp_path, base, name, mix=None, diagnostics=None):
    seg = RenderSegment(clip_id="c1", source_asset_id="s1", source_path=base, start=0.0, end=3.0)
    return render.render_preview((seg,), str(tmp_path / name), width=360, height=640, audio_mix=mix,
                                 execution_diagnostics=diagnostics)


def test_noop_mix_never_runs_a_pass(tmp_path, media):
    _root, base, voice, _silent = media
    assert AudioMix().is_noop and not AudioMix(video_volume=0.5).is_noop
    assert not AudioMix(video_clean=True).is_noop
    plain, noop = [], []
    _render(tmp_path, base, "a.mp4", None, plain)
    _render(tmp_path, base, "b.mp4", AudioMix(), noop)
    runs = lambda rows: [r for r in rows if r.get("executable") == "ffmpeg"]
    assert len(runs(plain)) == len(runs(noop)) == 1
    mixed = []
    _render(tmp_path, base, "c.mp4", AudioMix(video_volume=0.5), mixed)
    assert len(runs(mixed)) == 2


def test_voice_clip_is_heard_where_it_is_placed_and_ducks_the_video(tmp_path, media):
    _root, base, voice, _silent = media
    clip = dict(path=voice, start=1.0, source_in=0.5, source_out=1.5)
    flat = _render(tmp_path, base, "flat.mp4", AudioMix(clips=(LocalAudioClip(**clip),)))
    ducked = _render(tmp_path, base, "ducked.mp4", AudioMix(clips=(LocalAudioClip(**clip, duck_video=True),)))
    # The voice (1 kHz) is audible inside its window and absent before it.
    assert _band_level(flat, 1.15, 0.7, 1000) > _band_level(flat, 0.1, 0.7, 1000) + 25
    # The video's own sound (440 Hz) is lower under a ducking clip, unchanged when not ducking.
    assert _band_level(ducked, 1.15, 0.7, 440) < _band_level(flat, 1.15, 0.7, 440) - 6
    assert _band_level(ducked, 0.1, 0.7, 440) == pytest.approx(_band_level(flat, 0.1, 0.7, 440), abs=1.0)
    # Picture stream copied, same length; audio ends with the video.
    streams = _durations(ducked)
    assert streams["video"]["codec_name"] == "h264"
    assert abs(float(streams["audio"]["duration"]) - float(streams["video"]["duration"])) <= 0.05


def test_clip_running_past_the_end_is_cut_there(tmp_path, media):
    _root, base, voice, _silent = media
    out = _render(tmp_path, base, "past.mp4", AudioMix(clips=(LocalAudioClip(voice, 2.0, 0.0, 4.0),)))
    streams = _durations(out)
    assert abs(float(streams["audio"]["duration"]) - float(streams["video"]["duration"])) <= 0.05
    assert _band_level(out, 2.2, 0.6, 1000) > -60


def test_video_volume_and_cleanup(tmp_path, media):
    _root, base, voice, _silent = media
    full = _render(tmp_path, base, "full.mp4")
    half = _render(tmp_path, base, "half.mp4", AudioMix(video_volume=0.5))
    assert _band_level(half, 0.5, 2.0, 440) == pytest.approx(_band_level(full, 0.5, 2.0, 440) - 6.0, abs=1.0)
    muted = _render(tmp_path, base, "zero.mp4", AudioMix(video_volume=0.0))
    assert _band_level(muted, 0.5, 2.0, 440) < -90
    cleaned = _render(tmp_path, base, "clean.mp4", AudioMix(video_clean=True,
                                                              clips=(LocalAudioClip(voice, 0.5, 0, 1, clean=True),)))
    assert abs(float(_durations(cleaned)["audio"]["duration"]) - 3.0) <= 0.1


def test_mix_command_shape():
    mix = AudioMix(video_volume=1.5, video_clean=True, clips=(
        LocalAudioClip("/v.m4a", 1.0, 0.25, 2.0, volume=0.8, duck_video=True, clean=True),
        LocalAudioClip("/sfx.m4a", 2.5, 0.0, 0.3)))
    command = build_audio_mix_command("/in.mp4", "/out.mp4", mix)
    graph = command[command.index("-filter_complex") + 1]
    assert command[command.index("-c:v") + 1] == "copy"
    assert "[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,afftdn,highpass=f=80,loudnorm=I=-16" in graph
    assert graph.index("loudnorm") < graph.index("volume=1.500")
    assert "[1:a]atrim=start=0.250:end=2.000,asetpts=PTS-STARTPTS" in graph and "adelay=1000|1000[mixclip1]" in graph
    assert "volume=0.800,adelay" in graph
    assert "amix=inputs=3:duration=first:dropout_transition=0:normalize=0,alimiter" in graph
    assert "apad" not in graph  # no length given: no pad/trim
    timed = build_audio_mix_command("/in.mp4", "/out.mp4", mix, duration_sec=3.0)
    assert "apad=whole_dur=3.000000,atrim=duration=3.000000[mixout]" in timed[timed.index("-filter_complex") + 1]
    assert duck_volume_filter(mix.clips) == ("volume='1-0.684*min(1,clip(min((t-0.920)/0.080,(2.830-t)/0.080),0,1))'"
                                             ":eval=frame")
    assert duck_volume_filter((LocalAudioClip("/a", 0, 0, 1),)) is None


def test_render_with_post_render_qc_passes_the_mix_through(monkeypatch, tmp_path):
    from cutsell_worker import live_render_qc
    seen = {}

    def spy(segments, output_path, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(live_render_qc, "render_preview", spy)
    draft = draft_from_dict({"project_id": "p", "captions_enabled": False,
                             "selected": [{"clip_id": "a", "source_asset_id": "s1", "start": 0.0, "end": 1.0}]})
    plan = build_render_plan(draft, {"s1": "/a.mp4"})
    mix = AudioMix(video_volume=0.5)
    with pytest.raises(RuntimeError):
        live_render_qc.render_with_post_render_qc(draft, plan, str(tmp_path / "o.mp4"), audio_mix=mix)
    assert seen["audio_mix"] is mix
    seen.clear()
    with pytest.raises(RuntimeError):
        live_render_qc.render_with_post_render_qc(draft, plan, str(tmp_path / "o.mp4"))
    assert "audio_mix" not in seen  # old call shape unchanged


# --- QC / delivery review exemptions -------------------------------------------------------------

def _windows(payload, paths=None):
    draft = draft_from_dict(payload)
    plan = build_render_plan(draft, paths or {"s1": "/a.mp4"})
    return user_placed_windows(draft, plan, segment_output_windows(plan))


def _clip(clip_id, start, end, **extra):
    item = {"clip_id": clip_id, "source_asset_id": "s1", "start": start, "end": end, "text": "hi",
            "words": [{"text": "hi", "start": start + 0.1, "end": start + 0.5}]}
    item.update(extra)
    return item


@pytest.fixture
def no_tighten(monkeypatch):
    monkeypatch.setattr(render, "tighten_trailing_silence", lambda seg, **_: seg)


def test_creator_choices_are_placed_windows(no_tighten):
    base = {"project_id": "p", "captions_enabled": False}
    # Engine material, speed 1, normal volume: nothing exempt (as before).
    assert _windows({**base, "selected": [_clip("a", 0, 2), _clip("b", 3, 5)]}) == ()
    # A sped-up clip: its whole output window.
    assert _windows({**base, "selected": [_clip("a", 0, 2), _clip("b", 3, 7, speed=2)]}) == ((2.0, 4.0),)
    # A spoken clip the creator muted: its whole window.
    assert _windows({**base, "selected": [_clip("a", 0, 2, audio_muted=True), _clip("b", 3, 5)]}) == ((0.0, 2.0),)
    # A wordless muted clip is not a creator mute of speech (frozen visual rules still apply).
    assert _windows({**base, "selected": [_clip("a", 0, 2, audio_muted=True, text="", words=[]), _clip("b", 3, 5)]}) == ()
    # Whole video at volume 0: the whole timeline.
    assert _windows({**base, "video_volume": 0, "selected": [_clip("a", 0, 2), _clip("b", 3, 5)]}) == ((0.0, 4.0),)
    assert _windows({**base, "video_volume": 0.5, "selected": [_clip("a", 0, 2), _clip("b", 3, 5)]}) == ()


def _review(tmp_path, payload, paths, name, mix=None):
    from cutsell_worker.perceptual_watch_listen import review_rendered_candidate
    draft = draft_from_dict(payload)
    plan = build_render_plan(draft, paths)
    out = tmp_path / name
    render.render_preview(plan, str(out), width=360, height=640, audio_mix=mix)
    return review_rendered_candidate(str(out), draft, plan, segment_output_windows(plan))


def _spoken(clip_id, source, start, end, **extra):
    words = [{"text": "w", "start": start + 0.1 * i, "end": start + 0.1 * i + 0.08} for i in range(int((end - start) * 10) - 1)]
    texts = {"a": "first we open the box", "b": "then the serum goes on", "c": "and that is the result"}
    item = {"clip_id": clip_id, "source_asset_id": source, "start": start, "end": end, "text": texts[clip_id],
            "words": words}
    item.update(extra)
    return item


def _dead_air(review) -> bool:
    return any(c.capability == "interior_dead_air_mp4" and c.status == "EVALUATED_FAIL" for c in review.capabilities)


def test_delivery_review_exempts_creator_choices_and_still_blocks_engine_silence(tmp_path, media):
    _root, base, _voice, silent = media
    paths = {"s1": base, "s2": silent}
    head = {"project_id": "p", "captions_enabled": False}
    # Engine material: a 3 s silent stretch at speed 1 and normal volume is dead air.
    engine = _review(tmp_path, {**head, "selected": [_spoken("a", "s1", 0, 1.5), _spoken("b", "s2", 0, 3.0),
                                                     _spoken("c", "s1", 1.5, 3.0)]}, paths, "engine.mp4")
    assert engine.has_confirmed_blocking_defect and _dead_air(engine)
    # The same silent stretch sped up 1.5x by the creator is the creator's choice.
    sped = _review(tmp_path, {**head, "selected": [_spoken("a", "s1", 0, 1.5), _spoken("b", "s2", 0, 3.0, speed=1.5),
                                                   _spoken("c", "s1", 1.5, 3.0)]}, paths, "sped.mp4")
    assert not sped.has_confirmed_blocking_defect
    # A spoken clip the creator muted.
    muted = _review(tmp_path, {**head, "selected": [_spoken("a", "s1", 0, 1.0), _spoken("b", "s1", 1.0, 2.8, audio_muted=True),
                                                    _spoken("c", "s1", 2.8, 3.0)]}, {"s1": base}, "muted.mp4")
    assert not muted.has_confirmed_blocking_defect
    unmuted_silence = _review(tmp_path, {**head, "selected": [_spoken("a", "s1", 0, 1.0), _spoken("b", "s2", 0, 1.8),
                                                              _spoken("c", "s1", 2.8, 3.0)]}, paths, "unmuted.mp4")
    assert unmuted_silence.has_confirmed_blocking_defect and _dead_air(unmuted_silence)
    # Whole video at volume 0.
    zero = _review(tmp_path, {**head, "video_volume": 0, "selected": [_spoken("a", "s1", 0, 3.0)]}, {"s1": base},
                   "zero.mp4", mix=AudioMix(video_volume=0.0))
    assert not zero.has_confirmed_blocking_defect
