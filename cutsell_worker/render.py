"""FFmpeg renderer for the clean CutSell draft timeline."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
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

_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError("ffmpeg_render_failed")


def tighten_trailing_silence(
    segment: RenderSegment,
    *,
    minimum_silence_sec: float = 0.28,
    maximum_trim_sec: float = 12.0,
    edge_tolerance_sec: float = 0.16,
    speech_tail_pad_sec: float = 0.04,
) -> RenderSegment:
    """Remove proven silent post-roll from one selected source segment.

    Human Gold for Video 00 repeatedly marks a finished sentence followed by visible
    pause/mueca/reset before the next useful idea. Earlier stages can miss those visual
    boundaries. The final renderer has one objective signal available for every clip:
    the real source audio. We therefore trim only a silence interval that reaches the
    segment's trailing edge. Internal pauses are untouched and spoken audio is never
    removed.

    The ceiling is deliberately generous. Round 4 proved that long-form raw takes can
    contain more than three seconds of genuine trailing recording-process dead air; the
    previous 3 s guard rejected those objectively silent tails and left them visible in
    the preview. A 12 s cap still prevents an unbounded trim while allowing real creator
    post-roll to be removed.

    Exposed (D-030, no longer private) so `live_boundary_repair.py` can compute the
    SAME per-segment output-timeline durations `render_preview` actually produces --
    one implementation, not a second guess that could silently drift from it.
    """
    if segment.duration_sec < minimum_silence_sec + 0.35:
        return segment
    probe = probe_media(segment.source_path)
    if not probe.has_audio:
        return segment

    duration = segment.duration_sec
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "info",
        "-ss", f"{segment.start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", segment.source_path,
        "-vn",
        "-af", f"silencedetect=noise=-35dB:d={minimum_silence_sec:.3f}",
        "-f", "null", "-",
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        return segment

    intervals: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in completed.stderr.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            pending_start = float(start_match.group(1))
        end_match = _SILENCE_END_RE.search(line)
        if end_match and pending_start is not None:
            intervals.append((pending_start, float(end_match.group(1))))
            pending_start = None
    if pending_start is not None:
        intervals.append((pending_start, duration))

    trailing = None
    for silence_start, silence_end in intervals:
        silence_duration = max(0.0, silence_end - silence_start)
        reaches_edge = silence_end >= duration - edge_tolerance_sec
        if not reaches_edge or silence_duration < minimum_silence_sec:
            continue
        if trailing is None or silence_start > trailing[0]:
            trailing = (silence_start, silence_end)
    if trailing is None:
        return segment

    silence_start, _ = trailing
    trim_amount = duration - silence_start
    if trim_amount <= 0.0 or trim_amount > maximum_trim_sec:
        return segment

    new_end = segment.start + silence_start + speech_tail_pad_sec
    new_end = min(segment.end, new_end)
    if new_end - segment.start < 0.35 or segment.end - new_end < minimum_silence_sec - 0.05:
        return segment
    return replace(segment, end=new_end)


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
    segment_tuple = tuple(tighten_trailing_silence(segment) for segment in segments)
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
