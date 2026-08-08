"""FFmpeg renderer for the clean CutSell draft timeline."""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

from .contracts import TextOverlay
from .media_overlay_render import (
    LocalMediaOverlay,
    build_final_overlay_command,
    write_text_overlay_ass as _write_text_overlay_ass,
)
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


def render_preview(
    segments: Iterable[RenderSegment],
    output_path: str,
    *,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    text_overlays: Iterable[TextOverlay] = (),
    media_overlays: Iterable[LocalMediaOverlay] = (),
) -> str:
    """Render clips, captions, text and photo/video overlay lanes."""
    segment_tuple = tuple(segments)
    text_tuple = tuple(text_overlays)
    media_tuple = tuple(media_overlays)
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
        has_final_overlays = bool(text_tuple or media_tuple)
        joined = destination if not has_final_overlays else Path(directory) / "joined.mp4"
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", "-movflags", "+faststart", str(joined),
        ])

        if has_final_overlays:
            ass_path = None
            if text_tuple:
                ass = Path(directory) / "text-overlays.ass"
                _write_text_overlay_ass(text_tuple, ass, width=width, height=height)
                ass_path = str(ass)
            _run(build_final_overlay_command(
                str(joined), str(destination),
                media_overlays=media_tuple,
                text_overlays=text_tuple,
                width=width,
                height=height,
                ass_path=ass_path,
            ))

    if not destination.exists() or destination.stat().st_size <= 0:
        raise RuntimeError("ffmpeg_render_missing_output")
    return str(destination)
