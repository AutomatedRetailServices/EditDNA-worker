"""Editor v5 transform keyframes (CapCut-style): a framing -- scale, position, rotation -- at a moment
of a video clip or of an overlay, smoothly interpolated between keyframes.

Draft field `keyframes` (a list; when the key is present it replaces the legacy `zoom_keyframes`):

    keyframe_id   unique id
    target        "video" | "overlay"
    source_asset_id, source_time   (video) the SOURCE moment, so splits / trims / reorders never move it
    overlay_id, time               (overlay) seconds after the overlay's start on the timeline
    scale         video 1.0..5.0 (zoom in); overlay 0.2..5.0 (a multiplier on its size)
    x, y          offset of the picture's centre as a fraction of the frame (-1..1)
    rotation      degrees (-720..720), on top of the clip's / overlay's own rotation
    origin        "ai" (Auto Zoom AI: drawn only while auto_zoom_enabled) | "user"
    kind, group   informational (the AI moment, the four keyframes of one AI zoom)
    curve         how the value moves from this keyframe to the next: "ease_in_out" (default),
                  "linear", "ease_in", "ease_out"

Interpolation (the iPhone preview uses the same rule) -- 21 Oct, CapCut: the VIDEO keyframes form ONE
track over the whole edited video (each keyframe sits where its source moment plays on the timeline),
so the picture moves smoothly across the cut's joins and never jumps. Before the first keyframe and
after the last the value is held; between two keyframes each value moves with the first one's curve
(ease in-out = smoothstep v = a + (b - a) * u*u*(3 - 2u)). An overlay's keyframes are its own track.
Video offsets are limited to what the zoom allows (|x| <= (scale - 1) / 2), so the picture never
shows an empty edge from panning.

An Auto Zoom AI push-in (legacy `zoom_keyframes`: source_time, duration, scale) is four keyframes --
zoom in, hold, zoom out: 1.0 at t, scale at t + r, scale at t + d - r, 1.0 at t + d, with
r = min(0.4, d / 3).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

MAX_KEYFRAMES = 400
VIDEO_SCALE_RANGE = (1.0, 5.0)
OVERLAY_SCALE_RANGE = (0.2, 5.0)
OFFSET_RANGE = (-1.0, 1.0)
ROTATION_RANGE = (-720.0, 720.0)
ZOOM_RAMP_SEC = 0.4
CURVES = ("ease_in_out", "linear", "ease_in", "ease_out")
#: At most this many keyframes are turned into one ffmpeg expression (per clip / overlay).
MAX_EXPRESSION_KEYFRAMES = 48


@dataclass(frozen=True)
class TransformKeyframe:
    keyframe_id: str
    target: str
    scale: float = 1.0
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    source_asset_id: str = ""
    source_time: float = 0.0
    overlay_id: str = ""
    time: float = 0.0
    origin: str = "user"
    kind: str = "key"
    group: str = ""
    curve: str = "ease_in_out"


#: One point of a track: (time, scale, x, y, rotation[, curve]) -- curve index into CURVES, default 0.
Point = tuple


def _number(item: dict, key: str, default: float, low: float, high: float) -> float:
    raw = item.get(key, default)
    try:
        value = float(default if raw is None else raw)
    except (TypeError, ValueError):
        raise ValueError(f"keyframe {key} must be a number") from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"keyframe {key} must be between {low} and {high}")
    return value


def keyframe_from_dict(item) -> TransformKeyframe:
    if not isinstance(item, dict):
        raise ValueError("keyframes items must be objects")
    keyframe_id = str(item.get("keyframe_id") or "").strip()
    if not keyframe_id or len(keyframe_id) > 120:
        raise ValueError("keyframe keyframe_id is required")
    target = str(item.get("target") or "video")
    if target not in ("video", "overlay"):
        raise ValueError("keyframe target must be video or overlay")
    scale_range = VIDEO_SCALE_RANGE if target == "video" else OVERLAY_SCALE_RANGE
    common = dict(
        keyframe_id=keyframe_id, target=target,
        scale=_number(item, "scale", 1.0, *scale_range),
        x=_number(item, "x", 0.0, *OFFSET_RANGE),
        y=_number(item, "y", 0.0, *OFFSET_RANGE),
        rotation=_number(item, "rotation", 0.0, *ROTATION_RANGE),
        origin="ai" if str(item.get("origin") or "user") == "ai" else "user",
        kind=str(item.get("kind") or "key")[:32],
        group=str(item.get("group") or "")[:120],
        curve=str(item.get("curve") or "ease_in_out") if str(item.get("curve") or "ease_in_out") in CURVES else "ease_in_out",
    )
    if target == "video":
        source = str(item.get("source_asset_id") or "")
        if len(source) > 200:
            raise ValueError("keyframe source_asset_id is too long")
        return TransformKeyframe(source_asset_id=source,
                                 source_time=_number(item, "source_time", 0.0, 0.0, 86400.0), **common)
    overlay_id = str(item.get("overlay_id") or "").strip()
    if not overlay_id or len(overlay_id) > 200:
        raise ValueError("overlay keyframe needs overlay_id")
    return TransformKeyframe(overlay_id=overlay_id, time=_number(item, "time", 0.0, 0.0, 86400.0), **common)


def keyframes_from_dict(raw) -> tuple[TransformKeyframe, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError("keyframes must be a list")
    if len(raw) > MAX_KEYFRAMES:
        raise ValueError(f"at most {MAX_KEYFRAMES} keyframes are allowed")
    items = tuple(keyframe_from_dict(item) for item in raw)
    if len({k.keyframe_id for k in items}) != len(items):
        raise ValueError("keyframe keyframe_id values must be unique")
    return items


def keyframe_to_dict(keyframe: TransformKeyframe) -> dict:
    data = {"keyframe_id": keyframe.keyframe_id, "target": keyframe.target, "scale": round(keyframe.scale, 4),
            "x": round(keyframe.x, 4), "y": round(keyframe.y, 4), "rotation": round(keyframe.rotation, 2),
            "origin": keyframe.origin, "kind": keyframe.kind}
    if keyframe.group:
        data["group"] = keyframe.group
    if keyframe.curve != "ease_in_out":
        data["curve"] = keyframe.curve
    if keyframe.target == "video":
        data.update(source_asset_id=keyframe.source_asset_id, source_time=round(keyframe.source_time, 3))
    else:
        data.update(overlay_id=keyframe.overlay_id, time=round(keyframe.time, 3))
    return data


def zoom_to_keyframes(zoom) -> tuple[TransformKeyframe, ...]:
    """One Auto Zoom AI push-in (legacy `zoom_keyframes` item) as its four keyframes."""
    get = (lambda key, default=None: zoom.get(key, default)) if isinstance(zoom, dict) else \
        (lambda key, default=None: getattr(zoom, key, default))
    t = float(get("source_time"))
    d = float(get("duration", 1.5) or 1.5)
    s = float(get("scale", 1.08) or 1.08)
    base_id = str(get("keyframe_id") or f"z{int(t * 1000)}")
    origin = "ai" if str(get("origin") or "ai") == "ai" else "user"
    r = min(ZOOM_RAMP_SEC, d / 3.0)
    times_scales = ((t, 1.0), (t + r, s), (t + d - r, s), (t + d, 1.0))
    return tuple(
        TransformKeyframe(keyframe_id=f"{base_id}-{n}", target="video", scale=round(scale, 4),
                          source_asset_id=str(get("source_asset_id") or ""), source_time=round(time, 3),
                          origin=origin, kind=str(get("kind") or "key")[:32], group=base_id)
        for n, (time, scale) in enumerate(times_scales, start=1)
    )


def clamp_offset(scale: float, offset: float) -> float:
    """Pan limit at this zoom: the picture never leaves an empty edge."""
    room = max(0.0, (float(scale) - 1.0) / 2.0)
    return max(-room, min(room, float(offset)))


def active(keyframes: Iterable[TransformKeyframe], *, auto_zoom_enabled: bool) -> tuple[TransformKeyframe, ...]:
    """The keyframes that are drawn: the creator's always, the AI's only while Auto Zoom AI is on."""
    return tuple(k for k in keyframes if k.origin != "ai" or auto_zoom_enabled)


def video_track(keyframes: Iterable[TransformKeyframe], *, source_asset_id: str, start: float,
                end: float) -> tuple[Point, ...]:
    """The keyframes of one video clip (source window start..end), as points in SOURCE seconds."""
    points = sorted(
        (float(k.source_time), float(k.scale), float(k.x), float(k.y), float(k.rotation), _curve_index(k.curve))
        for k in keyframes
        if k.target == "video" and k.source_asset_id in ("", source_asset_id)
        and float(start) - 1e-6 <= float(k.source_time) <= float(end) + 1e-6
    )
    return tuple(points[:MAX_EXPRESSION_KEYFRAMES])


def _curve_index(curve: str) -> int:
    return CURVES.index(curve) if curve in CURVES else 0


@dataclass(frozen=True)
class ClipWindow:
    """One kept clip on the edited timeline: its source window and where it starts on the timeline."""
    source_asset_id: str
    start: float
    end: float
    timeline_start: float
    speed: float = 1.0


def clip_windows(clips) -> tuple[ClipWindow, ...]:
    """The kept clips in order, each with its place on the edited timeline (duration = (end - start) / speed)."""
    out, t = [], 0.0
    for clip in clips:
        speed = float(getattr(clip, "speed", 1.0) or 1.0)
        start, end = float(clip.start), float(clip.end)
        out.append(ClipWindow(str(clip.source_asset_id), start, end, t, speed))
        t += max(0.0, end - start) / speed
    return tuple(out)


def timeline_track(keyframes: Iterable[TransformKeyframe], windows: Sequence[ClipWindow]) -> tuple[Point, ...]:
    """All video keyframes as ONE track on the edited timeline: (timeline time, scale, x, y, rotation, curve).
    A keyframe whose source moment is not played (cut out) is left out."""
    points = []
    for k in keyframes:
        if k.target != "video":
            continue
        for w in windows:
            if k.source_asset_id in ("", w.source_asset_id) and w.start - 1e-6 <= float(k.source_time) <= w.end + 1e-6:
                at = w.timeline_start + (float(k.source_time) - w.start) / w.speed
                points.append((at, float(k.scale), float(k.x), float(k.y), float(k.rotation), _curve_index(k.curve)))
                break
    return tuple(sorted(points)[:MAX_KEYFRAMES])


def clip_track(track: Sequence[Point], window: ClipWindow) -> tuple[Point, ...]:
    """The timeline track seen from one clip, in that clip's SOURCE seconds: the keyframes inside it plus the
    nearest one on each side (outside its window). Interpolating these inside the window gives exactly the
    timeline curve -- so the export moves across the clip's joins like the preview, never jumps."""
    if not track:
        return ()
    end_t = window.timeline_start + (window.end - window.start) / window.speed
    first = max([i for i, p in enumerate(track) if p[0] <= window.timeline_start + 1e-9] or [0])
    last = min([i for i, p in enumerate(track) if p[0] >= end_t - 1e-9] or [len(track) - 1])
    chosen = track[first:last + 1][:MAX_EXPRESSION_KEYFRAMES]
    return tuple((window.start + (p[0] - window.timeline_start) * window.speed,) + tuple(p[1:]) for p in chosen)


def overlay_track(keyframes: Iterable[TransformKeyframe], *, overlay_id: str) -> tuple[Point, ...]:
    points = sorted(
        (float(k.time), float(k.scale), float(k.x), float(k.y), float(k.rotation), _curve_index(k.curve))
        for k in keyframes if k.target == "overlay" and k.overlay_id == overlay_id
    )
    return tuple(points[:MAX_EXPRESSION_KEYFRAMES])


def _ease(u: float, curve: int = 0) -> float:
    u = max(0.0, min(1.0, u))
    if curve == 1:
        return u
    if curve == 2:
        return u * u
    if curve == 3:
        return u * (2.0 - u)
    return u * u * (3.0 - 2.0 * u)


def _curve_of(point) -> int:
    return int(point[5]) if len(point) > 5 else 0


def value_at(points: Sequence[Point], time: float, field: int) -> float:
    """Interpolated value of `field` (1 scale, 2 x, 3 y, 4 rotation) at `time`."""
    if not points:
        return 1.0 if field == 1 else 0.0
    if time <= points[0][0]:
        return points[0][field]
    if time >= points[-1][0]:
        return points[-1][field]
    for a, b in zip(points, points[1:]):
        if a[0] <= time <= b[0]:
            span = b[0] - a[0]
            u = 0.0 if span <= 1e-9 else (time - a[0]) / span
            return a[field] + (b[field] - a[field]) * _ease(u, _curve_of(a))
    return points[-1][field]


def is_constant(points: Sequence[Point], field: int, value: float) -> bool:
    return all(abs(p[field] - value) < 1e-6 for p in points)


def expression(points: Sequence[Point], field: int, *, offset: float = 0.0, var: str = "t") -> str:
    """ffmpeg expression of `field` over `var`, where `var + offset` is the points' time base.
    Smoothstep between keyframes, held before the first and after the last."""
    if not points:
        return "1" if field == 1 else "0"
    if is_constant(points, field, points[0][field]):
        return f"{points[0][field]:.5f}"
    time = f"({var}+{offset:.4f})" if offset else var
    expr = f"{points[-1][field]:.5f}"
    for a, b in reversed(list(zip(points, points[1:]))):
        span = max(1e-6, b[0] - a[0])
        u = f"clip(({time}-{a[0]:.4f})/{span:.4f},0,1)"
        curve = _curve_of(a)
        shaped = (u if curve == 1 else f"{u}*{u}" if curve == 2 else f"{u}*(2-{u})" if curve == 3
                  else f"{u}*{u}*(3-2*{u})")
        piece = f"({a[field]:.5f}+({b[field] - a[field]:.5f})*{shaped})"
        expr = f"if(lt({time},{b[0]:.4f}),{piece},{expr})"
    return f"if(lt({time},{points[0][0]:.4f}),{points[0][field]:.5f},{expr})"
