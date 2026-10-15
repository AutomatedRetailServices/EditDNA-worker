"""Sample render plans whose ffmpeg commands are frozen in
`tests/fixtures/editor_v5_head_render_commands.json` (captured from the renderer before the Editor
v5 changes). Editor v5 must leave every command for a draft without the new fields byte-identical.

Regenerate (only when an intentional change to the speed-1 / default-overlay path is made):
    python -m tests.editor_v5_command_samples > tests/fixtures/editor_v5_head_render_commands.json
"""
from __future__ import annotations

import json
from pathlib import Path

from cutsell_worker import media_overlay_render, render
from cutsell_worker.contracts import MediaOverlay, TextOverlay
from cutsell_worker.media_probe import MediaProbe
from cutsell_worker.render_plan import RenderSegment

WORKDIR = "/tmp/editor-v5-golden-workdir"
_PROBES = {
    "/src/a.mp4": MediaProbe(duration_sec=30.0, width=1080, height=1920, fps=30.0, has_audio=True),
    "/src/silent.mp4": MediaProbe(duration_sec=30.0, width=720, height=1280, fps=30.0, has_audio=False),
    "/ov/v.mp4": MediaProbe(duration_sec=10.0, width=640, height=360, fps=30.0, has_audio=True),
}


def fake_probe(path, *args, **kwargs):
    return _PROBES[str(path)]


def _seg(clip_id, start, end, path="/src/a.mp4", **kw):
    return RenderSegment(clip_id=clip_id, source_asset_id="s1", source_path=path, start=start, end=end, **kw)


def sample_plans() -> dict[str, tuple[RenderSegment, ...]]:
    return {
        "plain": (_seg("c1", 0.0, 2.0), _seg("c2", 3.5, 5.25, audio_volume=0.7)),
        "captions_mirror_zoom": (
            _seg("c1", 1.0, 3.0, caption_text="hello world", mirror=True, rotation=90,
                 zoom_keyframes=((1.2, 1.0, 1.1),)),
            _seg("c2", 4.0, 6.0, audio_muted=True, caption_text="again", caption_preset="box"),
        ),
        "silent_source": (_seg("c1", 0.0, 1.5, path="/src/silent.mp4"), _seg("c2", 2.0, 3.0)),
        "audio_rolls": (
            _seg("c1", 0.0, 2.0, audio_start=0.0, audio_end=2.05, audio_post_roll=0.05),
            _seg("c2", 3.0, 5.0, audio_start=2.95, audio_end=5.0, audio_pre_roll=0.05),
        ),
    }


def overlay_samples() -> dict[str, dict]:
    photo = media_overlay_render.LocalMediaOverlay(
        overlay=MediaOverlay(overlay_id="p", kind="photo", uri="s3://b/p.jpg", start=0.5, end=2.0, x=0.3, y=0.4, width=0.5),
        path="/ov/p.jpg")
    video = media_overlay_render.LocalMediaOverlay(
        overlay=MediaOverlay(overlay_id="v", kind="video", uri="s3://b/v.mp4", start=1.0, end=3.0,
                             source_start=0.5, mute_audio=False),
        path="/ov/v.mp4")
    return {
        "photo_and_video": {"media_overlays": (photo, video), "text_overlays": ()},
        "photo_and_text": {"media_overlays": (photo,),
                           "text_overlays": (TextOverlay("t", "Hi", 0.0, 1.0),), "ass_path": "/w/t.ass"},
    }


def build_commands() -> dict[str, list[str]]:
    original = render.probe_media
    original_overlay = media_overlay_render.probe_media
    render.probe_media = fake_probe
    media_overlay_render.probe_media = fake_probe
    try:
        Path(WORKDIR).mkdir(parents=True, exist_ok=True)
        out: dict[str, list[str]] = {}
        for name, plan in sample_plans().items():
            out[f"concat:{name}"] = render._concat_render_command_with_audio_windows(
                plan, Path("/out/joined.mp4"), width=360, height=640, fps=30, workdir=Path(WORKDIR))
        for name, kwargs in overlay_samples().items():
            out[f"overlay:{name}"] = media_overlay_render.build_final_overlay_command(
                "/w/joined.mp4", "/w/out.mp4", width=360, height=640, **kwargs)
        return out
    finally:
        render.probe_media = original
        media_overlay_render.probe_media = original_overlay


if __name__ == "__main__":
    print(json.dumps(build_commands(), indent=1, sort_keys=True))
