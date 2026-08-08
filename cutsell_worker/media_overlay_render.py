"""FFmpeg final-pass compositor for CutSell photo/video overlays and text track."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .contracts import MediaOverlay, TextOverlay
from .media_probe import probe_media


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
    for index, local in enumerate(media, start=1):
        overlay = local.overlay
        display_duration = max(0.001, overlay.end - overlay.start)
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

        probe = probes[index - 1]
        if overlay.kind == "video" and not overlay.mute_audio and probe is not None and probe.has_audio:
            delay_ms = max(0, round(overlay.start * 1000))
            filters.append(
                f"[{index}:a]atrim=duration={display_duration:.3f},asetpts=PTS-STARTPTS,"
                f"adelay={delay_ms}|{delay_ms}[ova{index}]"
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
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        "-shortest",
        output_path,
    ]
    return command
