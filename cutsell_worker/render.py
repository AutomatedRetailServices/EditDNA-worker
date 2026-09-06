"""FFmpeg renderer for the clean CutSell draft timeline."""
from __future__ import annotations

from dataclasses import replace
import math
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


# D-094.3 (F14): every hard cut between two source segments is a step in the
# audio waveform. Run 33995806350's PostRenderWatchListenQC flagged 8 of 22
# joins as ABRUPT_AUDIO_DISCONTINUITY (peak sample jumps 573-3719 vs a
# typical 41-153) because the concat step copies streams verbatim with no
# edge treatment. A 12 ms fade-in/fade-out on each segment's OWN edges is a
# purely physical join treatment: it changes no boundary, no timing, no
# selection -- the segment still starts and ends exactly where Boundary put
# it -- it only takes the waveform to zero across the join so the splice is
# click-free. Segments shorter than `_AUDIO_JOIN_FADE_MIN_SEGMENT_SEC` are
# left untouched (a fade would cover a material share of them).
_AUDIO_JOIN_FADE_SEC = 0.012
_AUDIO_JOIN_FADE_MIN_SEGMENT_SEC = 0.20


def _audio_join_fade_filters(duration_sec: float) -> list[str]:
    fade = float(_AUDIO_JOIN_FADE_SEC)
    if fade <= 0.0 or duration_sec < _AUDIO_JOIN_FADE_MIN_SEGMENT_SEC:
        return []
    fade_out_start = max(0.0, float(duration_sec) - fade)
    return [f"afade=t=in:st=0:d={fade:.3f}", f"afade=t=out:st={fade_out_start:.3f}:d={fade:.3f}"]


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
        audio_filter = ",".join([f"volume={effective_volume:.3f}", *_audio_join_fade_filters(segment.duration_sec)])
        return base + ["-af", audio_filter] + common_video + [str(part)]
    return base + [
        "-f", "lavfi",
        "-t", f"{segment.duration_sec:.3f}",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-map", "0:v:0",
        "-map", "1:a:0",
    ] + common_video + ["-shortest", str(part)]


# D-097.2 (Renderer owns the output timeline): the previous renderer encoded
# every segment to its own MP4 part and joined the parts with the concat
# DEMUXER in stream-copy mode. Each part carried its own AAC priming frame
# and frame/packet padding, so every join advanced the real output timeline
# by ~20-60 ms more than the segment's duration (measured +41 ms per part on
# a synthetic 12-part render: +450 ms after 11 joins) -- inserted silence at
# every cut, an output 2 % longer than the frozen plan, and, because
# `segment_output_windows` assumed the plan timeline, the post-render
# discontinuity check probing speech tens or hundreds of ms away from the
# real joins (runs 34008386434 / 34029861712: 8-9 false
# ABRUPT_AUDIO_DISCONTINUITY findings per attempt, three wasted 50 ms
# "repairs", NEEDS_HUMAN_REVIEW, no deliverable MP4). One ffmpeg pass with
# the concat FILTER fixes both: every segment is trimmed to an exact,
# frame-aligned duration (video `trim`, audio `apad`+`atrim` to the same
# length) before the join, so the output timeline is exactly the sum of
# `rendered_segment_duration_sec` values and the same function serves the
# QC/perceptual/ladder window mapping. Segment boundaries, selection and
# order are untouched -- this is a purely physical join treatment.
RENDER_FPS_DEFAULT = 30


def rendered_segment_duration_sec(duration_sec: float, *, fps: int = RENDER_FPS_DEFAULT) -> float:
    """The exact length one segment occupies on the output timeline: its
    (already tightened) duration rounded UP to whole output frames, which is
    what the `fps` filter emits for a cut of that length. Audio is padded/
    trimmed to the same value inside the render command."""
    fps = max(1, int(fps))
    frames = max(1, int(math.ceil(float(duration_sec) * fps - 1e-6)))
    return frames / fps


def _concat_render_command(
    segments: tuple[RenderSegment, ...],
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    workdir: Path,
) -> list[str]:
    """One ffmpeg invocation: per-input seek + normalize + exact-duration
    trim, then the concat filter, then one encode. `segments` are already
    trailing-silence tightened."""
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    filters: list[str] = []
    input_index = 0
    for index, segment in enumerate(segments):
        probe = probe_media(segment.source_path)
        exact = rendered_segment_duration_sec(segment.duration_sec, fps=fps)
        command += ["-ss", f"{segment.start:.3f}", "-to", f"{segment.end:.3f}", "-i", segment.source_path]
        video_input = input_index
        input_index += 1
        video_chain = [
            f"[{video_input}:v]scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "setsar=1",
            f"fps={fps}",
        ]
        caption = _caption_filter(segment, workdir / f"part-{index:04d}.mp4")
        if caption:
            video_chain.append(caption)
        video_chain += [f"trim=duration={exact:.6f}", "setpts=PTS-STARTPTS", f"format=yuv420p[v{index}]"]
        filters.append(",".join(video_chain))
        audio_format = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
        if probe.has_audio:
            effective_volume = 0.0 if segment.audio_muted else float(segment.audio_volume)
            audio_chain = [
                f"[{video_input}:a]volume={effective_volume:.3f}",
                *_audio_join_fade_filters(segment.duration_sec),
                audio_format,
                f"apad=whole_dur={exact:.6f}",
                f"atrim=duration={exact:.6f}",
                f"asetpts=PTS-STARTPTS[a{index}]",
            ]
        else:
            command += ["-f", "lavfi", "-t", f"{exact:.6f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
            audio_input = input_index
            input_index += 1
            audio_chain = [
                f"[{audio_input}:a]{audio_format}",
                f"atrim=duration={exact:.6f}",
                f"asetpts=PTS-STARTPTS[a{index}]",
            ]
        filters.append(",".join(audio_chain))
    filters.append(
        "".join(f"[v{index}][a{index}]" for index in range(len(segments)))
        + f"concat=n={len(segments)}:v=1:a=1[vout][aout]"
    )
    command += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
        str(output),
    ]
    return command


def render_preview(
    segments: Iterable[RenderSegment],
    output_path: str,
    *,
    width: int = 1080,
    height: int = 1920,
    fps: int = RENDER_FPS_DEFAULT,
    text_overlays: Iterable[TextOverlay] = (),
    media_overlays: Iterable[LocalMediaOverlay] = (),
    trim_report: list[dict] | None = None,
) -> str:
    """Render clips, captions, text and photo/video overlay lanes.

    ``trim_report`` (D-097.E, optional, mutated): one row per segment whose
    trailing edge `tighten_trailing_silence` actually moved -- the renderer's
    last mechanical op is recorded, never silent, so a RAW can attribute
    every exit to its owner (see boundary_engine_pass.py's ownership table).
    """
    segment_tuple = []
    for segment in segments:
        tightened = tighten_trailing_silence(segment)
        if trim_report is not None and abs(float(tightened.end) - float(segment.end)) > 1e-6:
            trim_report.append({
                "clip_id": segment.clip_id,
                "render_fragment_id": getattr(segment, "render_fragment_id", None),
                "original_end": round(float(segment.end), 3),
                "tightened_end": round(float(tightened.end), 3),
                "trim_sec": round(float(segment.end) - float(tightened.end), 3),
                "owner": "render.tighten_trailing_silence",
            })
        segment_tuple.append(tightened)
    segment_tuple = tuple(segment_tuple)
    text_tuple = tuple(text_overlays)
    media_tuple = tuple(media_overlays)
    if not segment_tuple:
        raise ValueError("render requires at least one segment")
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("invalid render geometry")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cutsell-render-") as directory:
        for segment in segment_tuple:
            if segment.end <= segment.start:
                raise ValueError(f"invalid render segment {segment.clip_id}")
        has_final_overlays = bool(text_tuple or media_tuple)
        joined = destination if not has_final_overlays else Path(directory) / "joined.mp4"
        # D-097.2: one pass, exact frame-aligned per-segment durations, gapless
        # concat filter -- see the module comment above render_preview.
        _run(_concat_render_command(
            segment_tuple, joined, width=width, height=height, fps=fps, workdir=Path(directory),
        ))

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
