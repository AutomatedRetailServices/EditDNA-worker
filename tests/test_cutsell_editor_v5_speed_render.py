"""Editor v5 clip speed in the export: setpts/atempo after captions/zoom/orientation, output windows
divided by speed, no Overlap audio rolls on a sped clip, never coalesced across speeds, and every
speed-1 plan renders with byte-identical ffmpeg commands."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cutsell_worker import live_boundary_repair, media_overlay_render, render
from cutsell_worker.render_plan import RenderSegment, build_render_plan
from cutsell_worker.serde import draft_from_dict
from tests import editor_v5_command_samples as samples

GOLDEN = Path(__file__).parent / "fixtures" / "editor_v5_head_render_commands.json"


def test_speed_one_commands_are_byte_identical_to_head():
    assert samples.build_commands() == json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("speed,expected", [
    (1.0, []), (0.5, ["atempo=0.5"]), (1.5, ["atempo=1.5"]), (2.0, ["atempo=2"]),
    (3.0, ["atempo=2.0", "atempo=1.5"]), (0.25, ["atempo=0.5", "atempo=0.5"]),
])
def test_atempo_chain_stays_in_range(speed, expected):
    assert render.atempo_filters(speed) == expected


def _seg(clip_id="c1", start=0.0, end=4.0, **kw):
    return RenderSegment(clip_id=clip_id, source_asset_id="s1", source_path="/src/a.mp4", start=start, end=end, **kw)


def test_output_duration_divides_by_speed():
    assert render.segment_output_duration_sec(_seg(speed=2.0)) == pytest.approx(2.0)
    assert render.segment_output_duration_sec(_seg(end=2.0, speed=0.5)) == pytest.approx(4.0)
    assert render.segment_output_duration_sec(_seg(end=1.0, speed=3.0)) == pytest.approx(10 / 30)  # 0.333 -> 10 frames
    assert render.segment_output_duration_sec(_seg(end=1.01)) == render.rendered_segment_duration_sec(1.01)


def test_speed_filters_come_after_captions_and_zoom(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "probe_media", samples.fake_probe)
    seg = _seg(start=1.0, end=3.0, speed=3.0, caption_text="hello", mirror=True, zoom_keyframes=((1.2, 1.0, 1.1),))
    command = render._concat_render_command((seg,), tmp_path / "o.mp4", width=360, height=640, fps=30, workdir=tmp_path)
    graph = command[command.index("-filter_complex") + 1]
    video, audio = graph.split(";")[:2]
    assert video.index("hflip") < video.index("eval=frame") < video.index("subtitles=") < video.index("setpts=(PTS-STARTPTS)/3")
    assert "setpts=(PTS-STARTPTS)/3,fps=30,trim=duration=0.666667" in video
    assert "afade=t=out:st=1.988:d=0.012,atempo=2.0,atempo=1.5,aformat" in audio
    assert "apad=whole_dur=0.666667,atrim=duration=0.666667" in audio


def test_audio_window_path_applies_speed_too(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "probe_media", samples.fake_probe)
    rolled = _seg("c1", 0.0, 2.0, audio_start=0.0, audio_end=2.05, audio_post_roll=0.05)
    fast = _seg("c2", 3.0, 5.0, speed=2.0)
    command = render._concat_render_command_with_audio_windows(
        (rolled, fast), tmp_path / "o.mp4", width=360, height=640, fps=30, workdir=tmp_path)
    graph = command[command.index("-filter_complex") + 1]
    assert "setpts=(PTS-STARTPTS)/2,fps=30,trim=duration=1.000000" in graph
    assert "atempo=2,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,atrim=duration=1.000000" in graph
    assert "adelay=2000|2000[a1]" in graph


def test_render_plan_copies_speed_and_drops_rolls_on_sped_clips():
    draft = draft_from_dict({
        "project_id": "p", "captions_enabled": False, "audio_overlap_enabled": True,
        "selected": [
            {"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 2.0, "audio_post_roll": 0.05},
            {"clip_id": "c2", "source_asset_id": "s1", "start": 2.0, "end": 4.0, "speed": 2,
             "audio_pre_roll": 0.05, "audio_post_roll": 0.05},
            {"clip_id": "c3", "source_asset_id": "s1", "start": 4.0, "end": 6.0, "speed": 2},
        ],
    })
    plan = build_render_plan(draft, {"s1": "/src/a.mp4"})
    # c1 (1x) and c2 (2x) touch in the source but never coalesce; c2 and c3 (both 2x) do.
    assert [(seg.clip_id, seg.speed, seg.start, seg.end) for seg in plan] == [("c1", 1.0, 0.0, 2.0), ("c2", 2.0, 2.0, 6.0)]
    assert plan[0].audio_post_roll == 0.05
    assert plan[1].audio_pre_roll == 0.0 and plan[1].audio_post_roll == 0.0


def test_with_audio_rolls_ignores_sped_segments(monkeypatch):
    monkeypatch.setattr(render, "probe_media", samples.fake_probe)
    out = render._with_audio_rolls((_seg("c1", 0.0, 2.0, audio_post_roll=0.05),
                                    _seg("c2", 3.0, 5.0, speed=2.0, audio_pre_roll=0.05)))
    assert out[0].audio_end == pytest.approx(2.05)
    assert out[1].audio_start is None and out[1].audio_end is None


def test_output_windows_divide_by_speed(monkeypatch):
    monkeypatch.setattr(render, "tighten_trailing_silence", lambda seg, **_: seg)
    windows = live_boundary_repair.segment_output_windows(
        (_seg("c1", 0.0, 4.0, speed=2.0), _seg("c2", 5.0, 7.0), _seg("c3", 8.0, 10.0, speed=0.5)))
    assert windows == [pytest.approx((0.0, 2.0)), pytest.approx((2.0, 4.0)), pytest.approx((4.0, 8.0))]


# --- real tiny renders --------------------------------------------------------------------------

def _make_source(path: Path, seconds: float) -> str:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=s=360x640:r=30:d={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)
    return str(path)


def _durations(path: str) -> tuple[float, float]:
    out = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", path,
    ], check=True, capture_output=True, text=True).stdout
    streams = {s["codec_type"]: float(s["duration"]) for s in json.loads(out)["streams"]}
    return streams["video"], streams["audio"]


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return _make_source(tmp_path_factory.mktemp("speed") / "src.mp4", 4.0)


@pytest.mark.parametrize("start,end,speed,expected", [(0.0, 4.0, 2.0, 2.0), (0.0, 2.0, 0.5, 4.0), (0.0, 3.0, 3.0, 1.0)])
def test_real_render_duration_follows_speed(tmp_path, source, start, end, speed, expected):
    seg = RenderSegment(clip_id="c1", source_asset_id="s1", source_path=source, start=start, end=end, speed=speed)
    output = render.render_preview((seg,), str(tmp_path / "out.mp4"), width=360, height=640)
    video, audio = _durations(output)
    frame = 1 / 30
    assert abs(video - expected) <= frame + 0.02
    # audio stays in sync with the picture (AAC frame padding only)
    assert abs(audio - video) <= 0.05


def test_real_render_mixed_speeds_stay_in_sync(tmp_path, source):
    plan = (
        RenderSegment(clip_id="c1", source_asset_id="s1", source_path=source, start=0.0, end=2.0, speed=2.0),
        RenderSegment(clip_id="c2", source_asset_id="s1", source_path=source, start=2.0, end=3.0),
        RenderSegment(clip_id="c3", source_asset_id="s1", source_path=source, start=3.0, end=4.0, speed=0.5),
    )
    output = render.render_preview(plan, str(tmp_path / "out.mp4"), width=360, height=640)
    video, audio = _durations(output)
    assert abs(video - 4.0) <= 1 / 30 + 0.02
    assert abs(audio - video) <= 0.05
    windows = live_boundary_repair.segment_output_windows(plan)
    assert windows[-1][1] == pytest.approx(4.0)
