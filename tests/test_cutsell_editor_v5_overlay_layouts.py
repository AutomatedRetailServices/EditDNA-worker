"""Editor v5 overlay layers, layouts, edges and rotation in the final overlay pass, checked on
pixels of a real exported frame. Drafts without the new fields keep the exact pre-v5 command
(see test_cutsell_editor_v5_speed_render.test_speed_one_commands_are_byte_identical_to_head)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cutsell_worker.contracts import MediaOverlay
from cutsell_worker.media_overlay_render import LocalMediaOverlay, build_final_overlay_command, overlay_region

W, H = 360, 640
# yuv420p round trip: pure colours come back within ~40 per channel (lime -> ~(0, 214, 0)).
BLUE, YELLOW, RED, GREEN = (0, 0, 255), (255, 255, 0), (255, 0, 0), (0, 225, 0)


def _run(command):
    subprocess.run(command, check=True, capture_output=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp("ov5")
    base = root / "base.mp4"  # top half yellow, bottom half blue, 2 s, with sound
    _run(["ffmpeg", "-y", "-loglevel", "error",
          "-f", "lavfi", "-i", f"color=c=yellow:s={W}x{H // 2}:d=2:r=30",
          "-f", "lavfi", "-i", f"color=c=blue:s={W}x{H // 2}:d=2:r=30",
          "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
          "-filter_complex", "[0:v][1:v]vstack[v]", "-map", "[v]", "-map", "2:a",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(base)])
    photos = {}
    for name, color, size in (("red", "red", "200x200"), ("green", "lime", "200x200"), ("redtall", "red", "100x400")):
        path = root / f"{name}.png"
        _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={color}:s={size}:d=1",
              "-frames:v", "1", str(path)])
        photos[name] = str(path)
    return root, str(base), photos


def _overlay(path, overlay_id="o", **kw):
    fields = {"start": 0.0, "end": 2.0}
    fields.update(kw)
    return LocalMediaOverlay(overlay=MediaOverlay(overlay_id=overlay_id, kind="photo", uri=f"s3://b/{overlay_id}.png",
                                                  **fields), path=path)


def _frame(root: Path, base: str, overlays, name: str, at: float = 1.0) -> bytes:
    output = root / f"{name}.mp4"
    _run(build_final_overlay_command(base, str(output), media_overlays=overlays, width=W, height=H))
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{at}", "-i", str(output), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    assert len(raw) == W * H * 3
    return raw


def _pixel(raw: bytes, x: int, y: int) -> tuple[int, int, int]:
    offset = (y * W + x) * 3
    return tuple(raw[offset:offset + 3])


def _close(actual, expected, tolerance=40):
    return all(abs(a - e) <= tolerance for a, e in zip(actual, expected))


def test_regions():
    assert overlay_region("full", W, H) == (0, 0, W, H)
    assert overlay_region("top", W, H) == (0, 0, W, H // 2)
    assert overlay_region("bottom", W, H) == (0, H // 2, W, H // 2)
    assert overlay_region("left", W, H) == (0, 0, W // 2, H)
    assert overlay_region("right", W, H) == (W // 2, 0, W // 2, H)
    assert overlay_region("stacked", W, H) == overlay_region("bottom", W, H)
    assert overlay_region("side_by_side", W, H) == overlay_region("right", W, H)


def test_higher_layer_is_drawn_on_top(media):
    root, base, photos = media
    # Red is listed first but on layer 2; green is listed second on layer 1: red must win.
    red = _overlay(photos["red"], "r", x=0.5, y=0.5, width=0.5, layer=2)
    green = _overlay(photos["green"], "g", x=0.6, y=0.55, width=0.5, layer=1)
    raw = _frame(root, base, (red, green), "layers")
    assert _close(_pixel(raw, W // 2, H // 2), RED)            # overlap: higher layer
    assert _close(_pixel(raw, int(W * 0.6) + 80, int(H * 0.55) + 80), GREEN)  # green only
    # Same layer: draft order decides (the later one on top), exactly as before v5.
    raw = _frame(root, base, (_overlay(photos["red"], "r", width=0.5), _overlay(photos["green"], "g", width=0.5)), "ties")
    assert _close(_pixel(raw, W // 2, H // 2), GREEN)


def test_stacked_shows_the_video_itself_on_top_and_the_overlay_below(media):
    root, base, photos = media
    raw = _frame(root, base, (_overlay(photos["red"], layout="stacked", fill=True),), "stacked")
    # Bottom half: the overlay.
    assert _close(_pixel(raw, W // 2, int(H * 0.75)), RED)
    # Top half: the base center-cropped (rows H/4..3H/4 of the base): yellow above, blue below
    # its middle -- without the crop the whole top half would be yellow.
    assert _close(_pixel(raw, W // 2, int(H * 0.1)), YELLOW)
    assert _close(_pixel(raw, W // 2, int(H * 0.4)), BLUE)
    # Outside the overlay window the frame is the plain base again.
    raw = _frame(root, base, (_overlay(photos["red"], layout="stacked", fill=True, start=0.0, end=0.5),), "stacked_after")
    assert _close(_pixel(raw, W // 2, int(H * 0.75)), BLUE)
    assert _close(_pixel(raw, W // 2, int(H * 0.4)), YELLOW)


def test_side_by_side_shows_the_video_left_and_the_overlay_right(media):
    root, base, photos = media
    raw = _frame(root, base, (_overlay(photos["green"], layout="side_by_side", fill=True),), "sbs")
    assert _close(_pixel(raw, int(W * 0.75), H // 2 - 50), GREEN)
    assert _close(_pixel(raw, W // 4, int(H * 0.25)), YELLOW)
    assert _close(_pixel(raw, W // 4, int(H * 0.75)), BLUE)


def test_left_layout_fit_keeps_the_base_around_and_beside_the_overlay(media):
    root, base, photos = media
    # A square photo fitted into the left half (180 x 640): 180 x 180 in its middle on a
    # transparent pad -- the base shows above/below it and in the whole right half.
    raw = _frame(root, base, (_overlay(photos["red"], layout="left"),), "left")
    assert _close(_pixel(raw, W // 4, H // 2 + 40), RED)
    assert _close(_pixel(raw, int(W * 0.75), H // 2 + 40), BLUE)
    assert _close(_pixel(raw, W // 4, int(H * 0.1)), YELLOW)
    # fill=True covers the whole left half.
    raw = _frame(root, base, (_overlay(photos["red"], layout="left", fill=True),), "left_fill")
    assert _close(_pixel(raw, W // 4, int(H * 0.1)), RED)
    assert _close(_pixel(raw, int(W * 0.75), int(H * 0.1)), YELLOW)


def test_full_fit_pads_with_black(media):
    root, base, photos = media
    raw = _frame(root, base, (_overlay(photos["redtall"], layout="full"),), "full")
    assert _close(_pixel(raw, W // 2, H // 2), RED)
    assert _close(_pixel(raw, 10, H // 2), (0, 0, 0))


def test_soft_edges_blend_into_the_video(media):
    root, base, photos = media
    sharp = _frame(root, base, (_overlay(photos["red"], layout="bottom", fill=True),), "sharp")
    soft = _frame(root, base, (_overlay(photos["red"], layout="bottom", fill=True, edge="extra_soft"),), "soft")
    edge = (2, int(H * 0.75))
    assert _close(_pixel(sharp, *edge), RED)
    assert _close(_pixel(soft, *edge), BLUE, tolerance=90)  # feathered edge: mostly the video below
    assert _close(_pixel(soft, W // 2, int(H * 0.75)), RED)  # the middle stays opaque


def test_float_rotation_leaves_transparent_corners(media):
    root, base, photos = media
    straight = _frame(root, base, (_overlay(photos["red"], x=0.5, y=0.75, width=0.5),), "straight")
    turned = _frame(root, base, (_overlay(photos["red"], x=0.5, y=0.75, width=0.5, rotation=45),), "turned")
    corner = (W // 2 - 85, int(H * 0.75) - 85)  # inside the square's corner, outside the rotated diamond
    assert _close(_pixel(straight, *corner), RED)
    assert _close(_pixel(turned, *corner), BLUE)
    assert _close(_pixel(turned, W // 2, int(H * 0.75)), RED)


def test_styled_commands_have_no_per_frame_geq(media):
    _root, base, photos = media
    command = build_final_overlay_command(base, "/tmp/x.mp4", width=W, height=H, media_overlays=(
        _overlay(photos["red"], layout="left", edge="soft"), _overlay(photos["green"], "g", edge="extra_soft")))
    graph = command[command.index("-filter_complex") + 1]
    assert "geq" not in graph
    assert graph.count("loop=loop=-1:size=1") == 2 and graph.count("alphamerge") == 2
