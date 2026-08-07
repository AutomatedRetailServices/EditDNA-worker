"""FFmpeg preview renderer for the clean CutSell draft timeline.

The renderer consumes a source-safe RenderPlan. It never invents or stitches text;
it only trims selected source intervals and concatenates them in draft order.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

from .render_plan import RenderSegment


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError("ffmpeg_render_failed")


def render_preview(
    segments: Iterable[RenderSegment],
    output_path: str,
    *,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
) -> str:
    """Render selected clips to one H.264/AAC vertical MP4 preview."""
    segment_tuple = tuple(segments)
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
            # scale+pad preserves the creator's frame without stretching while
            # producing a deterministic TikTok-ready 9:16 canvas.
            vf = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}"
            )
            _run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{segment.start:.3f}",
                "-to", f"{segment.end:.3f}",
                "-i", segment.source_path,
                "-vf", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
                "-movflags", "+faststart",
                str(part),
            ])
            normalized.append(part)

        concat_file = Path(directory) / "concat.txt"
        concat_file.write_text(
            "".join(f"file '{part.as_posix()}'\n" for part in normalized),
            encoding="utf-8",
        )
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", "-movflags", "+faststart", str(destination),
        ])

    if not destination.exists() or destination.stat().st_size <= 0:
        raise RuntimeError("ffmpeg_render_missing_output")
    return str(destination)
