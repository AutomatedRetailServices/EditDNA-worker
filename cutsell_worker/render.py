"""FFmpeg renderer for the clean CutSell draft timeline."""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

from .contracts import TextOverlay
from .media_probe import probe_media
from .render_plan import RenderSegment


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError("ffmpeg_render_failed")


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _caption_filter(segment: RenderSegment, part: Path) -> str | None:
    text = str(segment.caption_text or "").replace("\x00", "").strip()
    if not text:
        return None
    text = text[:500]
    subtitle = part.with_suffix(".srt")
    subtitle.write_text(
        f"1\n00:00:00,000 --> {_srt_timestamp(segment.duration_sec)}\n{text}\n",
        encoding="utf-8",
    )
    preset = str(segment.caption_preset or "classic")
    if preset == "clean":
        style = "Fontsize=24,Alignment=2,MarginV=120,BorderStyle=3,Outline=0,Shadow=0,BackColour=&H66000000,PrimaryColour=&H00FFFFFF"
    else:
        style = "Fontsize=24,Alignment=2,MarginV=120,BorderStyle=1,Outline=2,Shadow=0,OutlineColour=&H00000000,PrimaryColour=&H00FFFFFF"
    path = subtitle.as_posix().replace("'", "\\'")
    return f"subtitles='{path}':force_style='{style}'"


def _segment_command(segment: RenderSegment, part: Path, *, vf: str) -> list[str]:
    probe = probe_media(segment.source_path)
    effective_volume = 0.0 if segment.audio_muted else float(segment.audio_volume)
    caption = _caption_filter(segment, part)
    video_filter = f"{vf},{caption}" if caption else vf
    common_video = [
        "-vf", video_filter,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
    ]
    base = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{segment.start:.3f}",
        "-to", f"{segment.end:.3f}",
        "-i", segment.source_path,
    ]
    if probe.has_audio:
        return base + ["-af", f"volume={effective_volume:.3f}"] + common_video + [str(part)]
    return base + [
        "-f", "lavfi",
        "-t", f"{segment.duration_sec:.3f}",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-map", "0:v:0",
        "-map", "1:a:0",
    ] + common_video + ["-shortest", str(part)]


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(float(seconds) * 100)))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _ass_text(value: str) -> str:
    return str(value).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")[:500]


def _write_text_overlay_ass(overlays: tuple[TextOverlay, ...], path: Path, *, width: int, height: int) -> None:
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
        text = _ass_text(overlay.text)
        events.append(
            f"Dialogue: 0,{_ass_timestamp(overlay.start)},{_ass_timestamp(overlay.end)},Overlay,,0,0,0,,"
            f"{{\\pos({x},{y})\\fs{font_size}}}{text}\n"
        )
    path.write_text(header + "".join(events), encoding="utf-8")


def render_preview(
    segments: Iterable[RenderSegment],
    output_path: str,
    *,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    text_overlays: Iterable[TextOverlay] = (),
) -> str:
    """Render selected clips, captions and optional final-timeline text overlays."""
    segment_tuple = tuple(segments)
    overlay_tuple = tuple(text_overlays)
    if not segment_tuple:
        raise ValueError("render requires at least one segment")
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("invalid render geometry")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cutsell-render-") as directory:
        normalized = []
        for index, segment in enumerate(segment_tuple):
            if segment.end <= segment.start:
                raise ValueError(f"invalid render segment {segment.clip_id}")
            part = Path(directory) / f"part-{index:04d}.mp4"
            vf = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}"
            )
            _run(_segment_command(segment, part, vf=vf))
            normalized.append(part)

        concat_file = Path(directory) / "concat.txt"
        concat_file.write_text("".join(f"file '{part.as_posix()}'\n" for part in normalized), encoding="utf-8")
        joined = destination if not overlay_tuple else Path(directory) / "joined.mp4"
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", "-movflags", "+faststart", str(joined),
        ])

        if overlay_tuple:
            ass = Path(directory) / "text-overlays.ass"
            _write_text_overlay_ass(overlay_tuple, ass, width=width, height=height)
            ass_path = ass.as_posix().replace("'", "\\'")
            _run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(joined),
                "-vf", f"ass='{ass_path}'",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "copy", "-movflags", "+faststart", str(destination),
            ])

    if not destination.exists() or destination.stat().st_size <= 0:
        raise RuntimeError("ffmpeg_render_missing_output")
    return str(destination)
