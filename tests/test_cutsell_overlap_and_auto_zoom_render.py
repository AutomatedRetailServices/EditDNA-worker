"""Overlap audio + Auto Zoom AI in the export (render side).

- A draft with `audio_overlap_enabled` crossfades every join using each clip's own pre/post roll;
  without it the export is the exact command it always was.
- A draft with `auto_zoom_enabled` draws its zoom keyframes (source-anchored) on the segments
  that contain them; turned off, the keyframes stay in the draft but are not drawn.
"""
from __future__ import annotations

import subprocess

import pytest

from cutsell_worker import render
from cutsell_worker.render_plan import RenderSegment, build_render_plan
from cutsell_worker.serde import draft_from_dict, result_to_dict


def _draft(**extra):
    payload = {
        "project_id": "prj_test",
        "captions_enabled": False,
        "selected": [
            {"clip_id": "c1", "source_asset_id": "s1", "start": 0.2, "end": 1.2, "text": "a",
             "audio_pre_roll": 0.05, "audio_post_roll": 0.05},
            {"clip_id": "c2", "source_asset_id": "s1", "start": 2.0, "end": 3.0, "text": "b",
             "audio_pre_roll": 0.05, "audio_post_roll": 0.05},
        ],
    }
    payload.update(extra)
    return payload


@pytest.fixture(scope="module")
def tone_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("ov") / "src.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=s=180x320:d=4:r=30",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=4",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)
    return str(path)


def test_rolls_and_zooms_parse_and_stay_out_of_plain_drafts():
    plain = draft_from_dict({"project_id": "p", "selected": [{"clip_id": "c", "source_asset_id": "s", "start": 0, "end": 1}]})
    data = result_to_dict(plain)
    assert not {"audio_overlap_enabled", "auto_zoom_enabled", "zoom_keyframes"} & set(data)
    assert not {"audio_pre_roll", "audio_post_roll"} & set(data["selected"][0])
    rolled = draft_from_dict(_draft(audio_overlap_enabled=True))
    assert rolled.selected[0].audio_pre_roll == 0.05
    assert draft_from_dict(_draft(selected=[{"clip_id": "c", "source_asset_id": "s", "start": 0, "end": 1,
                                             "audio_post_roll": 0.5}])).selected[0].audio_post_roll == 0.06
    with pytest.raises(ValueError):
        draft_from_dict(_draft(zoom_keyframes=[{"source_asset_id": "s1", "source_time": 1, "duration": 1, "scale": 2.0}]))


def test_overlap_flag_decides_the_rolls_in_the_plan(tone_clip):
    on = build_render_plan(draft_from_dict(_draft(audio_overlap_enabled=True)), {"s1": tone_clip})
    off = build_render_plan(draft_from_dict(_draft()), {"s1": tone_clip})
    assert [s.audio_post_roll for s in on] == [0.05, 0.05]
    assert [s.audio_pre_roll for s in off] == [0.0, 0.0]


def test_rolls_become_crossfaded_audio_windows(tone_clip):
    segments = build_render_plan(draft_from_dict(_draft(audio_overlap_enabled=True)), {"s1": tone_clip})
    windows = render._with_audio_rolls(tuple(segments))
    first, second = windows
    # no lead on the first clip, no tail on the last one
    assert first.effective_audio_start == pytest.approx(0.2) and first.effective_audio_end == pytest.approx(1.25)
    assert second.effective_audio_start == pytest.approx(1.95) and second.effective_audio_end == pytest.approx(3.0)
    assert render._audio_join_fade_filters(1.05, fade_out=0.05)[1].endswith("d=0.050")
    # without overlap nothing changes
    plain = tuple(build_render_plan(draft_from_dict(_draft()), {"s1": tone_clip}))
    assert render._with_audio_rolls(plain) == plain


def test_zoom_only_when_enabled_and_inside_the_segment(tone_clip):
    keyframes = [{"source_asset_id": "s1", "source_time": 2.2, "duration": 0.7, "scale": 1.1}]
    on = build_render_plan(draft_from_dict(_draft(auto_zoom_enabled=True, zoom_keyframes=keyframes)), {"s1": tone_clip})
    off = build_render_plan(draft_from_dict(_draft(zoom_keyframes=keyframes)), {"s1": tone_clip})
    assert render._zoom_expression(on[0]) is None          # 2.2 s is not in the first clip (0.2-1.2)
    assert render._zoom_expression(on[1]) is not None      # it is in the second (2.0-3.0)
    assert all(render._zoom_expression(seg) is None for seg in off)
    filters = render._zoom_filters(on[1], width=180, height=320)
    assert filters[0].startswith("scale=w='2*trunc(iw*(1+0.1000*") and filters[1] == "crop=180:320"


def _audio_rms(path: str, start: float, end: float) -> float:
    raw = subprocess.run([
        "ffmpeg", "-loglevel", "error", "-ss", str(start), "-to", str(end), "-i", path,
        "-ac", "1", "-ar", "8000", "-f", "s16le", "-",
    ], check=True, capture_output=True).stdout
    import array
    samples = array.array("h", raw)
    return (sum(x * x for x in samples) / max(1, len(samples))) ** 0.5


@pytest.mark.parametrize("overlap,zoom", [(False, False), (True, True)])
def test_real_export_with_overlap_and_zoom(tmp_path, tone_clip, overlap, zoom):
    keyframes = [{"source_asset_id": "s1", "source_time": 2.1, "duration": 0.8, "scale": 1.12}]
    draft = draft_from_dict(_draft(audio_overlap_enabled=overlap, auto_zoom_enabled=zoom, zoom_keyframes=keyframes))
    output = tmp_path / "out.mp4"
    render.render_preview(build_render_plan(draft, {"s1": tone_clip}), str(output), width=180, height=320)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(output)],
                           check=True, capture_output=True, text=True).stdout
    assert float(probe) == pytest.approx(2.0, abs=0.12)    # video length is the clips' length either way
    assert _audio_rms(str(output), 0.3, 0.7) > 1000         # sound present


def test_unchanged_command_without_overlap_or_zoom(tmp_path, tone_clip):
    segments = tuple(build_render_plan(draft_from_dict(_draft()), {"s1": tone_clip}))
    a = render._concat_render_command_with_audio_windows(segments, tmp_path / "a.mp4", width=180, height=320, fps=30, workdir=tmp_path)
    b = render._concat_render_command(segments, tmp_path / "a.mp4", width=180, height=320, fps=30, workdir=tmp_path)
    assert a == b
    assert not any("eval=frame" in part for part in a)


def test_api_forwards_auto_zoom_to_the_worker(monkeypatch):
    from fastapi.testclient import TestClient
    from cutsell_app import main
    from cutsell_worker.serde import request_from_dict

    sent = []

    class Submission:
        job_id = "job_1"
        queue_name = "cutsell"

    monkeypatch.setattr(main, "enqueue_flow_b", lambda payload: sent.append(payload) or Submission())
    monkeypatch.setattr(main, "validate_product_source_uri", lambda *a, **k: None)
    client = TestClient(main.app)
    body = {"project_id": "p1", "user_id": "u1", "dialogue_overlap_enabled": True,
            "sources": [{"original_name": "a.mp4", "uri": "s3://b/k", "source_order": 0, "duration_sec": 5}]}
    for extra, expected in (({}, True), ({"auto_zoom_enabled": False}, False)):
        sent.clear()
        response = client.post("/v1/flow-b/jobs", json={**body, **extra})
        assert response.status_code == 202, response.text
        assert sent[0]["auto_zoom_enabled"] is expected
        request = request_from_dict(sent[0])
        assert request.auto_zoom_enabled is expected and request.dialogue_overlap_enabled is True
