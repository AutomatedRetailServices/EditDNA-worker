"""FFmpeg final-pass compositor for CutSell photo/video overlays and text track."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

from .contracts import MediaOverlay, TextOverlay
from .media_probe import probe_media

# D-274E-A: canonical BT.709 SDR output color-metadata tags -- METADATA
# ONLY, no pixel/color transformation, no filter added. `render.py`'s
# final renderer already receives canonical SDR media from the D-274D
# source-normalization executor; this module's own real-render-audited
# encode commands (`build_final_overlay_command` here, and `render.py`'s
# own concat commands, which import this same tuple to avoid a second,
# independently-drifting copy) simply now EXPLICITLY TAG their already-
# correct output as what it already is -- closing the D-274E-disclosed
# gap (docs/CUTSELL_DECISIONS.md D-274E/D-274E-A) where `verify_output_
# format` against `FINAL_RENDER_OUTPUT_CONTRACT_V1` reported `PARTIAL`
# (missing evidence) rather than `PASS`, because no encode command
# anywhere in this render family wrote these tags. Defined here (the
# module `render.py` already imports FROM) rather than in `render.py`
# itself, to avoid a circular import.
CANONICAL_OUTPUT_COLOR_METADATA_FLAGS: tuple[str, ...] = (
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
    "-colorspace", "bt709",
    "-color_range", "tv",
)


@dataclass(frozen=True)
class LocalMediaOverlay:
    overlay: MediaOverlay
    path: str


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(float(seconds) * 100)))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _ass_text(value: str) -> str:
    return str(value).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")[:500]


def write_text_overlay_ass(overlays: tuple[TextOverlay, ...], path: Path, *, width: int, height: int) -> None:
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Overlay,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H66000000,-1,0,0,0,100,100,0,0,1,2,0,5,20,20,20,1\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    events = []
    for overlay in overlays:
        x = round(float(overlay.x) * width)
        y = round(float(overlay.y) * height)
        font_size = max(24, min(144, round(48 * float(overlay.scale))))
        events.append(
            f"Dialogue: 0,{_ass_timestamp(overlay.start)},{_ass_timestamp(overlay.end)},Overlay,,0,0,0,,"
            f"{{\\pos({x},{y})\\fs{font_size}}}{_ass_text(overlay.text)}\n"
        )
    path.write_text(header + "".join(events), encoding="utf-8")


def build_final_overlay_command(
    joined_path: str,
    output_path: str,
    *,
    media_overlays: Iterable[LocalMediaOverlay] = (),
    text_overlays: Iterable[TextOverlay] = (),
    width: int = 1080,
    height: int = 1920,
    ass_path: str | None = None,
) -> list[str]:
    media = tuple(media_overlays)
    text = tuple(text_overlays)
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", joined_path]

    # Add overlay inputs in deterministic draft order.
    probes = []
    for local in media:
        overlay = local.overlay
        path = str(local.path)
        if overlay.kind == "photo":
            command += ["-loop", "1", "-framerate", "30", "-i", path]
            probes.append(None)
        else:
            command += ["-ss", f"{overlay.source_start:.3f}", "-i", path]
            probes.append(probe_media(path))

    filters = ["[0:v]setpts=PTS-STARTPTS[v0]"]
    current_video = "v0"
    audio_labels = ["[0:a]"]
    # Editor v5: draw in ascending `layer` (a higher layer is on top); equal layers keep draft order.
    # Input indices stay in draft order. With every layer at 0 this is the draft order, as before.
    order = sorted(range(len(media)), key=lambda position: (int(getattr(media[position].overlay, "layer", 0) or 0), position))
    for position in order:
        index = position + 1
        local = media[position]
        overlay = local.overlay
        display_duration = max(0.001, overlay.end - overlay.start)
        if _is_default_overlay(overlay):
            pixel_width = max(2, round(float(overlay.width) * width))
            x_expr = f"W*{float(overlay.x):.6f}-w/2"
            y_expr = f"H*{float(overlay.y):.6f}-h/2"
            filters.append(
                f"[{index}:v]trim=duration={display_duration:.3f},scale={pixel_width}:-1,"
                f"setpts=PTS-STARTPTS+{overlay.start:.3f}/TB[ov{index}]"
            )
            next_video = f"v{index}"
            filters.append(
                f"[{current_video}][ov{index}]overlay=x='{x_expr}':y='{y_expr}':"
                f"enable='between(t,{overlay.start:.3f},{overlay.end:.3f})':eof_action=pass[{next_video}]"
            )
            current_video = next_video
        else:
            current_video = _append_styled_overlay(
                filters, local, index=index, current_video=current_video, width=width, height=height,
                probe=probes[position], display_duration=display_duration,
            )

        probe = probes[position]
        if overlay.kind == "video" and not overlay.mute_audio and probe is not None and probe.has_audio:
            delay_ms = max(0, round(overlay.start * 1000))
            # Editor v5: the overlay's own audio gain; 1.0 adds no filter (pre-v5 command).
            gain = float(getattr(overlay, "volume", 1.0))
            gain_filter = f"volume={gain:.3f}," if gain != 1.0 else ""
            filters.append(
                f"[{index}:a]atrim=duration={display_duration:.3f},asetpts=PTS-STARTPTS,"
                f"{gain_filter}adelay={delay_ms}|{delay_ms}[ova{index}]"
            )
            audio_labels.append(f"[ova{index}]")

    if text:
        if not ass_path:
            raise ValueError("ass_path is required when text overlays exist")
        escaped = Path(ass_path).as_posix().replace("'", "\\'")
        filters.append(f"[{current_video}]ass='{escaped}'[vout]")
        final_video = "[vout]"
    else:
        final_video = f"[{current_video}]"

    if len(audio_labels) > 1:
        filters.append("".join(audio_labels) + f"amix=inputs={len(audio_labels)}:duration=first:dropout_transition=0[aout]")
        final_audio = "[aout]"
    else:
        filters.append("[0:a]anull[aout]")
        final_audio = "[aout]"

    command += [
        "-filter_complex", ";".join(filters),
        "-map", final_video,
        "-map", final_audio,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        *CANONICAL_OUTPUT_COLOR_METADATA_FLAGS,
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        "-shortest",
        output_path,
    ]
    return command


# --- Editor v5 overlay layers, layouts, edges and rotation ------------------------------------

#: Region layouts: the overlay fills a part of the screen instead of floating at x/y/width.
REGION_LAYOUTS = ("full", "top", "bottom", "left", "right", "stacked", "side_by_side")
#: Layouts whose unused part of the region shows the video below (fit leaves a transparent pad).
_TRANSPARENT_PAD_LAYOUTS = ("left", "right", "top", "bottom")
#: Feather width in pixels at the 1080x1920 output; scaled with the output size.
EDGE_FEATHER_PX_1080 = {"sharp": 0.0, "soft": 12.0, "extra_soft": 28.0}


def edge_feather_px(edge: str, width: int, height: int) -> float:
    """Feather ramp width for `edge` on a width x height output (12 px soft / 28 px extra soft
    at 1080x1920, proportionally smaller or larger on other sizes)."""
    scale = min(float(width) / 1080.0, float(height) / 1920.0)
    return EDGE_FEATHER_PX_1080.get(str(edge), 0.0) * scale


def _is_default_overlay(overlay) -> bool:
    """True when the overlay uses none of the Editor v5 drawing options -- it is then drawn by the
    exact pre-v5 filter chain (layer only changes the drawing order)."""
    return (
        str(getattr(overlay, "layout", "float") or "float") == "float"
        and str(getattr(overlay, "edge", "sharp") or "sharp") == "sharp"
        and float(getattr(overlay, "rotation", 0.0) or 0.0) == 0.0
    )


def overlay_region(layout: str, width: int, height: int) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the overlay's region on a width x height frame. Stacked draws the overlay in
    the bottom half (the video itself fills the top half); side by side in the right half (the
    video itself fills the left half)."""
    half_w, half_h = width // 2, height // 2
    return {
        "full": (0, 0, width, height),
        "top": (0, 0, width, half_h),
        "bottom": (0, half_h, width, height - half_h),
        "left": (0, 0, half_w, height),
        "right": (half_w, 0, width - half_w, height),
        "stacked": (0, half_h, width, height - half_h),
        "side_by_side": (half_w, 0, width - half_w, height),
    }[layout]


def _base_half(layout: str, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Where the video itself is shown (center-cropped to fill) beside a stacked / side-by-side
    overlay, as (x, y, w, h); None for the other layouts."""
    if layout == "stacked":
        return (0, 0, width, height // 2)
    if layout == "side_by_side":
        return (0, 0, width // 2, height)
    return None


def _even(value: float) -> int:
    return max(2, int(round(float(value) / 2.0)) * 2)


def _media_size(local: LocalMediaOverlay, probe) -> tuple[int, int] | None:
    """Pixel size of the overlay media (video probe, or a probe of the photo)."""
    if probe is None:
        try:
            probe = probe_media(str(local.path))
        except Exception:
            return None
    w, h = int(getattr(probe, "width", 0) or 0), int(getattr(probe, "height", 0) or 0)
    return (w, h) if w > 0 and h > 0 else None


def _feather_filters(source: str, output: str, *, index: int, size: tuple[int, int],
                     rect: tuple[int, int, int, int], ramp: float) -> list[str] | None:
    """Multiply the overlay's alpha by a soft-edged rectangle mask. The mask is ONE frame (drawbox +
    gaussian blur, computed once) looped for the overlay's length, so no per-frame pixel math."""
    w, h = size
    rx, ry, rw, rh = rect
    inset = int(round(ramp / 2.0))
    if inset < 1 or rw - 2 * inset < 2 or rh - 2 * inset < 2:
        return None
    sigma = max(0.5, ramp / 4.0)
    return [
        f"color=c=black:s={w}x{h}:r=30:d=0.1,format=gray,"
        f"drawbox=x={rx + inset}:y={ry + inset}:w={rw - 2 * inset}:h={rh - 2 * inset}:color=white:t=fill,"
        f"lut=y='clip((val-16)*255/219,0,255)',gblur=sigma={sigma:.3f},"
        f"trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/30/TB[ovmask{index}]",
        f"[{source}]split[ovcol{index}][ovsrc{index}]",
        f"[ovsrc{index}]alphaextract[ovalpha{index}]",
        f"[ovalpha{index}][ovmask{index}]blend=all_mode=multiply:shortest=1[ovsoft{index}]",
        f"[ovcol{index}][ovsoft{index}]alphamerge[{output}]",
    ]


def _append_styled_overlay(filters: list[str], local: LocalMediaOverlay, *, index: int, current_video: str,
                           width: int, height: int, probe, display_duration: float) -> str:
    """Filters for one overlay that uses an Editor v5 layout / edge / rotation. Returns the label
    of the composed video after it."""
    overlay = local.overlay
    layout = str(getattr(overlay, "layout", "float") or "float")
    edge = str(getattr(overlay, "edge", "sharp") or "sharp")
    feather = edge_feather_px(edge, width, height)
    start, end = float(overlay.start), float(overlay.end)
    enable = f"enable='between(t,{start:.3f},{end:.3f})'"
    head = f"[{index}:v]trim=duration={display_duration:.3f},setpts=PTS-STARTPTS"
    shaped = f"ovshape{index}"

    if layout == "float":
        pixel_width = _even(float(overlay.width) * width)
        size = _media_size(local, probe) if feather > 0 else None
        if size is not None:
            pixel_height = _even(pixel_width * size[1] / size[0])
            filters.append(f"{head},scale={pixel_width}:{pixel_height},setsar=1,format=rgba[{shaped}]")
            soft = _feather_filters(shaped, f"ovfeather{index}", index=index, size=(pixel_width, pixel_height),
                                    rect=(0, 0, pixel_width, pixel_height), ramp=feather)
            if soft:
                filters.extend(soft)
                shaped = f"ovfeather{index}"
        else:
            filters.append(f"{head},scale={pixel_width}:-2,setsar=1,format=rgba[{shaped}]")
        tail = []
        rotation = float(getattr(overlay, "rotation", 0.0) or 0.0)
        if rotation:
            radians = math.radians(rotation)
            tail.append(f"rotate=a={radians:.6f}:ow=rotw({radians:.6f}):oh=roth({radians:.6f}):c=none")
        tail.append(f"setpts=PTS-STARTPTS+{start:.3f}/TB")
        filters.append(f"[{shaped}]" + ",".join(tail) + f"[ov{index}]")
        x_expr = f"W*{float(overlay.x):.6f}-w/2"
        y_expr = f"H*{float(overlay.y):.6f}-h/2"
        filters.append(f"[{current_video}][ov{index}]overlay=x='{x_expr}':y='{y_expr}':{enable}:eof_action=pass[v{index}]")
        return f"v{index}"

    rx, ry, rw, rh = overlay_region(layout, width, height)
    transparent = layout in _TRANSPARENT_PAD_LAYOUTS
    content = (0, 0, rw, rh)
    if bool(getattr(overlay, "fill", False)):
        filters.append(f"{head},scale={rw}:{rh}:force_original_aspect_ratio=increase,crop={rw}:{rh},"
                       f"setsar=1,format=rgba[{shaped}]")
    else:
        pad_color = "black@0" if transparent else "black"
        size = _media_size(local, probe)
        if size is not None:
            scale = min(rw / size[0], rh / size[1])
            cw, ch = min(rw, _even(size[0] * scale)), min(rh, _even(size[1] * scale))
            ox, oy = (rw - cw) // 2, (rh - ch) // 2
            content = (ox, oy, cw, ch)
            filters.append(f"{head},scale={cw}:{ch},setsar=1,format=rgba,pad={rw}:{rh}:{ox}:{oy}:color={pad_color}[{shaped}]")
        else:
            filters.append(f"{head},scale={rw}:{rh}:force_original_aspect_ratio=decrease,setsar=1,format=rgba,"
                           f"pad={rw}:{rh}:(ow-iw)/2:(oh-ih)/2:color={pad_color}[{shaped}]")
    if feather > 0:
        # A transparent pad shows the video below, so only the picture itself is feathered; an
        # opaque (black) region is feathered as a whole.
        rect = content if transparent else (0, 0, rw, rh)
        soft = _feather_filters(shaped, f"ovfeather{index}", index=index, size=(rw, rh), rect=rect,
                                ramp=feather)
        if soft:
            filters.extend(soft)
            shaped = f"ovfeather{index}"
    filters.append(f"[{shaped}]setpts=PTS-STARTPTS+{start:.3f}/TB[ov{index}]")

    base = current_video
    half = _base_half(layout, width, height)
    if half is not None:
        bx, by, bw, bh = half
        filters.append(f"[{base}]split[ovunder{index}][ovbase{index}]")
        filters.append(f"[ovbase{index}]crop={bw}:{bh}:{(width - bw) // 2}:{(height - bh) // 2}[ovhalf{index}]")
        filters.append(f"[ovunder{index}][ovhalf{index}]overlay=x={bx}:y={by}:{enable}[ovwith{index}]")
        base = f"ovwith{index}"
    filters.append(f"[{base}][ov{index}]overlay=x={rx}:y={ry}:{enable}:eof_action=pass[v{index}]")
    return f"v{index}"
