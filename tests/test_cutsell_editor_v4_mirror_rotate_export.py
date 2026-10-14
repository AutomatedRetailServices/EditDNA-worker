"""Editor v4 Mirror / Rotate reach the export.

The iPhone saves `mirror` (bool) and `rotation` (0/90/180/270) on the draft; the export applies them
to the source picture before it is fitted into the frame. A draft without them renders exactly as
before (no filter added, no key written)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cutsell_worker import render
from cutsell_worker.render_plan import RenderSegment, build_render_plan
from cutsell_worker.serde import draft_from_dict, result_to_dict


def _draft(**extra):
    payload = {
        "project_id": "prj_test",
        "selected": [{"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 1.0, "text": "hi"}],
        "captions_enabled": False,
    }
    payload.update(extra)
    return payload


def test_mirror_and_rotation_parse_and_default_off():
    plain = draft_from_dict(_draft())
    assert plain.mirror is False and plain.rotation == 0
    turned = draft_from_dict(_draft(mirror=True, rotation=90))
    assert turned.mirror is True and turned.rotation == 90
    assert draft_from_dict(_draft(rotation=450)).rotation == 90


@pytest.mark.parametrize("bad", [45, "sideways"])
def test_rotation_must_be_a_quarter_turn(bad):
    with pytest.raises(ValueError):
        draft_from_dict(_draft(rotation=bad))


def test_serialized_draft_is_unchanged_when_not_set():
    data = result_to_dict(draft_from_dict(_draft()))
    assert "mirror" not in data and "rotation" not in data
    data = result_to_dict(draft_from_dict(_draft(mirror=True, rotation=180)))
    assert data["mirror"] is True and data["rotation"] == 180


def _segment(**kwargs):
    return RenderSegment(clip_id="c1", source_asset_id="s1", source_path="/x.mp4", start=0.0, end=1.0, **kwargs)


def test_orientation_filters():
    assert render._orientation_filters(_segment()) == []
    assert render._orientation_filters(_segment(mirror=True)) == ["hflip"]
    assert render._orientation_filters(_segment(rotation=90)) == ["transpose=1"]
    assert render._orientation_filters(_segment(rotation=180)) == ["hflip", "vflip"]
    assert render._orientation_filters(_segment(rotation=270)) == ["transpose=2"]
    assert render._orientation_filters(_segment(mirror=True, rotation=90)) == ["hflip", "transpose=1"]


def test_render_plan_carries_the_draft_orientation():
    draft = draft_from_dict(_draft(mirror=True, rotation=270))
    segments = build_render_plan(draft, {"s1": "/x.mp4"})
    assert all(seg.mirror and seg.rotation == 270 for seg in segments)


@pytest.fixture(scope="module")
def half_red_clip(tmp_path_factory):
    """Left half red, right half blue, 1 s, with sound."""
    path = tmp_path_factory.mktemp("orient") / "src.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=red:s=180x320:d=1:r=30",
        "-f", "lavfi", "-i", "color=c=blue:s=180x320:d=1:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-filter_complex", "[0:v]crop=90:320:0:0[l];[1:v]crop=90:320:0:0[r];[l][r]hstack[v]",
        "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)
    return str(path)


def _left_pixel(path: str) -> tuple[int, int, int]:
    raw = subprocess.run([
        "ffmpeg", "-loglevel", "error", "-ss", "0.5", "-i", path, "-frames:v", "1",
        "-vf", "scale=180:320,crop=1:1:30:160", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ], check=True, capture_output=True).stdout
    return raw[0], raw[1], raw[2]


@pytest.mark.parametrize("mirror", [False, True])
def test_real_export_mirrors_the_picture(tmp_path, half_red_clip, mirror):
    draft = draft_from_dict(_draft(mirror=mirror))
    segments = build_render_plan(draft, {"s1": half_red_clip})
    output = tmp_path / "out.mp4"
    render.render_preview(segments, str(output), width=180, height=320)
    r, g, b = _left_pixel(str(output))
    if mirror:
        assert b > 150 and r < 100, (r, g, b)   # left side is now the blue half
    else:
        assert r > 150 and b < 100, (r, g, b)


# --- Mirror on one clip (Editor v4: the selected clip, or every clip when none is selected) ---

def _two_clips(**first):
    return {
        "project_id": "prj_test", "captions_enabled": False,
        "selected": [
            {"clip_id": "c1", "source_asset_id": "s1", "start": 0.0, "end": 0.5, "text": "a", **first},
            {"clip_id": "c2", "source_asset_id": "s1", "start": 0.5, "end": 1.0, "text": "b"},
        ],
    }


def test_clip_mirror_parses_serializes_and_survives_a_split():
    from cutsell_worker.draft_edits import split_clip
    draft = draft_from_dict(_two_clips(mirror=True))
    assert draft.selected[0].mirror is True and draft.selected[1].mirror is False
    data = result_to_dict(draft)
    assert data["selected"][0]["mirror"] is True and "mirror" not in data["selected"][1]
    words = [{"text": "x", "start": 0.0, "end": 0.2}, {"text": "y", "start": 0.3, "end": 0.45}]
    payload = _two_clips(mirror=True)
    payload["selected"][0]["words"] = words
    payload["alternates"] = []
    payload["discarded"] = []
    halves = split_clip(payload, "c1", split_time=0.25, min_duration_sec=0.05)["selected"][:2]
    assert all(half["mirror"] is True for half in halves)


def test_plan_flips_only_the_mirrored_clip_and_never_merges_it():
    segments = build_render_plan(draft_from_dict(_two_clips(mirror=True)), {"s1": "/x.mp4"})
    assert [seg.mirror for seg in segments] == [True, False]          # contiguous, but not coalesced
    plain = build_render_plan(draft_from_dict(_two_clips()), {"s1": "/x.mp4"})
    assert len(plain) == 1 and plain[0].mirror is False                # unchanged behaviour


def test_real_export_mirrors_only_that_clip(tmp_path, half_red_clip):
    draft = draft_from_dict(_two_clips(mirror=True))
    output = tmp_path / "out.mp4"
    render.render_preview(build_render_plan(draft, {"s1": half_red_clip}), str(output), width=180, height=320)

    def left_at(t):
        raw = subprocess.run([
            "ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(output), "-frames:v", "1",
            "-vf", "scale=180:320,crop=1:1:30:160", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ], check=True, capture_output=True).stdout
        return raw[0], raw[2]

    r, b = left_at(0.25)
    assert b > 150 and r < 100, (r, b)       # first clip flipped: blue on the left
    r, b = left_at(0.75)
    assert r > 150 and b < 100, (r, b)       # second clip as recorded


# --- Trim by the edges (Editor v4): a clip is a window on its original ---------------------------
# The iPhone saves the same `source_asset_id` + `start`/`end` (never renamed) and, when the end was
# dragged by hand, `manual_end: true`. The export cuts the high-quality original at exactly those
# times and never tightens away the silence the creator placed by hand.

def _duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


@pytest.fixture(scope="module")
def tone_then_silence(tmp_path_factory):
    """4 s original: tone for 1.5 s, then 2.5 s of silence (a stretched clip's tail)."""
    path = tmp_path_factory.mktemp("trim") / "orig.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=green:s=180x320:d=4:r=30",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=1.5",
        "-filter_complex", "[1:a]apad=whole_dur=4[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", "4", str(path),
    ], check=True, capture_output=True)
    return str(path)


def _trim_draft(**clip):
    base = {"clip_id": "c1", "source_asset_id": "orig", "start": 0.0, "end": 3.5, "text": "hello there"}
    base.update(clip)
    return {"project_id": "prj_test", "captions_enabled": False, "selected": [base]}


def test_manual_end_parses_serializes_and_defaults_off():
    plain = draft_from_dict(_trim_draft())
    assert plain.selected[0].manual_end is False
    assert "manual_end" not in result_to_dict(plain)["selected"][0]          # stored drafts unchanged
    held = draft_from_dict(_trim_draft(manual_end=True))
    assert held.selected[0].manual_end is True
    assert result_to_dict(held)["selected"][0]["manual_end"] is True
    # The field names stay the same: a clip is still source_asset_id + start/end.
    data = result_to_dict(held)["selected"][0]
    assert (data["source_asset_id"], data["start"], data["end"]) == ("orig", 0.0, 3.5)


def test_plan_holds_a_hand_dragged_end():
    held = build_render_plan(draft_from_dict(_trim_draft(manual_end=True)), {"orig": "/o.mp4"})[0]
    assert held.trailing_trim_floor == 3.5 and held.end == 3.5 and held.source_asset_id == "orig"
    free = build_render_plan(draft_from_dict(_trim_draft()), {"orig": "/o.mp4"})[0]
    assert free.trailing_trim_floor is None


def test_tighten_keeps_the_silence_of_a_hand_dragged_end(tone_then_silence):
    free = build_render_plan(draft_from_dict(_trim_draft()), {"orig": tone_then_silence})[0]
    held = build_render_plan(draft_from_dict(_trim_draft(manual_end=True)), {"orig": tone_then_silence})[0]
    assert render.tighten_trailing_silence(free).end < 3.0                     # engine clip: tail tightened
    assert render.tighten_trailing_silence(held).end == pytest.approx(3.5)     # by hand: kept as placed


def test_real_export_cuts_the_original_at_the_dragged_window(tmp_path, tone_then_silence):
    # Stretched outward over material the cut had hidden: 0.5 s -> 3.8 s of the 4 s original.
    draft = draft_from_dict(_trim_draft(start=0.5, end=3.8, manual_end=True))
    output = tmp_path / "out.mp4"
    render.render_preview(build_render_plan(draft, {"orig": tone_then_silence}), str(output), width=180, height=320)
    assert _duration(str(output)) == pytest.approx(3.3, abs=0.12)


def test_split_keeps_the_hand_dragged_end_on_the_right_piece():
    from cutsell_worker.draft_edits import split_clip
    payload = _trim_draft(manual_end=True, words=[
        {"text": "hello", "start": 0.1, "end": 0.6}, {"text": "there", "start": 0.9, "end": 1.4}])
    payload["alternates"] = []
    payload["discarded"] = []
    left, right = split_clip(payload, "c1", split_time=0.75, min_duration_sec=0.05)["selected"][:2]
    assert "manual_end" not in left and right["manual_end"] is True
