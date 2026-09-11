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
from .media_probe import MediaProbe, probe_media
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


# =============================================================================
# D-214 -- PACING V2 RENDERER / TIMELINE CONTRACT EXTENSION (OFFLINE ONLY)
# =============================================================================
#
# D-213's own forensic (docs/CUTSELL_DECISIONS.md) named the ONE renderer gap
# blocking J_CUT/L_CUT/MICRO_AUDIO_OVERLAP: `_concat_render_command` trims
# every segment's audio to the SAME exact duration as its video before the
# `concat` filter, so audio and video can never diverge. This section closes
# that gap at the EXECUTION layer only.
#
# ## Ownership contract (binding, structurally enforced below)
#
# Pacing decides transition semantics; the renderer only ever realizes
# whatever `RenderSegment.audio_start`/`.audio_end` geometry it is handed --
# it never inspects a `mode` string, never picks a transition type, never
# runs any eligibility/safety check. `dialogue_pacing_transition_execution_
# diagnostics` below is READ-ONLY reporting (it infers a descriptive mode
# label from the geometry it is given, purely for observability) and can
# never influence what `render_timeline_with_audio_windows` actually
# builds -- the two functions never call each other.
#
# ## NOT live-wired
#
# Nothing in this section is imported or called by `render_preview`,
# `universal_clean_cut.py`, `pipeline.py`, or any other production call
# site. Production's only two live modes remain HARD_CUT/TIGHT_CUT via the
# existing, completely unmodified `render_preview`/`_concat_render_command`
# above. This is test-only execution CAPABILITY, per the D-214 directive's
# own "no live wiring" instruction.
#
# ## Backward compatibility (structural, not incidental)
#
# `RenderSegment.audio_start`/`.audio_end` default to `None` on every
# existing call site (`render_plan.build_render_plan` never sets them).
# `_concat_render_command_with_audio_windows` below detects this (`not
# any(segment.has_independent_audio_window for segment in segments)`) and
# delegates to the EXISTING, byte-for-byte-unchanged `_concat_render_
# command` in that case -- there is no parallel reimplementation of the
# HARD_CUT/TIGHT_CUT path to silently drift from the live one.

AUDIO_TIMELINE_EPSILON_SEC = 1e-6


def validate_audio_window(segment: RenderSegment, *, probe: MediaProbe | None = None) -> None:
    """Fail-closed validation of one segment's own AUDIO source window.
    Raises `ValueError` with an explicit reason; never silently clamps a
    window into range (per D-214's own "no silent semantic clamp"
    instruction). A segment with a default (non-divergent) audio window
    always passes trivially -- this only ever constrains an EXPLICIT
    `audio_start`/`audio_end`."""
    if not segment.has_independent_audio_window:
        return
    audio_start = segment.effective_audio_start
    audio_end = segment.effective_audio_end
    if audio_start < -AUDIO_TIMELINE_EPSILON_SEC:
        raise ValueError(f"invalid_audio_window_negative_start:{segment.clip_id}")
    if audio_end <= audio_start + AUDIO_TIMELINE_EPSILON_SEC:
        raise ValueError(f"malformed_audio_window_end_not_after_start:{segment.clip_id}")
    probe = probe if probe is not None else probe_media(segment.source_path)
    if not probe.has_audio:
        raise ValueError(f"independent_audio_window_requires_source_audio_track:{segment.clip_id}")
    if audio_end > probe.duration_sec + AUDIO_TIMELINE_EPSILON_SEC:
        raise ValueError(
            f"audio_window_exceeds_source_availability:{segment.clip_id}:"
            f"requested_end={audio_end:.3f}:source_duration={probe.duration_sec:.3f}"
        )


def _video_timeline_positions(segments: tuple[RenderSegment, ...], *, fps: int) -> tuple[float, ...]:
    """Cumulative OUTPUT-timeline start position of each segment's own VIDEO
    window -- a pure function of the (already frame-rounded) video
    durations, never of any audio window. This is the one authoritative
    timeline reference both the video graph and the audio graph below are
    placed against; no audio decision on join N can ever move where segment
    N+1's own video (or audio placement baseline) begins (D-214's own
    "multi-join isolation" requirement, satisfied structurally)."""
    positions = [0.0]
    for segment in segments[:-1]:
        positions.append(positions[-1] + rendered_segment_duration_sec(segment.duration_sec, fps=fps))
    return tuple(positions)


def _audio_placement_sec(segment: RenderSegment, video_position_sec: float) -> float:
    """Where this segment's own AUDIO window begins on the output timeline.
    Identical to `video_position_sec` for every segment with a default
    (non-divergent) audio window -- i.e. today's only behavior. A LEADING
    audio window (`audio_start < start`, the per-segment J-cut primitive)
    shifts this segment's own audio earlier by exactly that lead amount; a
    TRAILING audio window (`audio_end > end`, the per-segment L-cut
    primitive) never changes this segment's OWN placement, only how far its
    audio continues past it. Micro-overlap is simply a small lead and/or
    trail on the same join."""
    lead = max(0.0, segment.start - segment.effective_audio_start)
    return video_position_sec - lead


def _validate_audio_placements(segments: tuple[RenderSegment, ...], *, fps: int) -> tuple[float, ...]:
    """Validate every segment's audio window, then return each segment's
    output-timeline audio placement. Fails closed (raises) if any
    segment's own lead would need to start before timeline zero -- i.e. it
    claims more pre-roll than the entire preceding timeline provides."""
    for segment in segments:
        validate_audio_window(segment)
    video_positions = _video_timeline_positions(segments, fps=fps)
    placements = []
    for segment, video_position in zip(segments, video_positions):
        placement = _audio_placement_sec(segment, video_position)
        if placement < -AUDIO_TIMELINE_EPSILON_SEC:
            raise ValueError(
                f"audio_lead_exceeds_available_timeline:{segment.clip_id}:"
                f"requested_placement={placement:.3f}"
            )
        placements.append(max(0.0, placement))
    return tuple(placements)


def _concat_render_command_with_audio_windows(
    segments: tuple[RenderSegment, ...],
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    workdir: Path,
) -> list[str]:
    """D-214: like `_concat_render_command`, but each segment's AUDIO window
    may diverge from its VIDEO window. VIDEO is built EXACTLY as the
    existing function (same `concat` filter, same frame-exact per-segment
    trim) -- visual cut instants are never affected by any audio decision.
    AUDIO is built as an independent per-segment `adelay`-placed stream,
    combined with `amix(normalize=0)`: `normalize=0` is the load-bearing
    correctness property here -- with it, a segment that shares no overlap
    with any neighbor plays at its own unmodified volume for its own full
    duration, mathematically identical to a plain sequential concat; only
    the seconds two segments' own windows actually overlap are ever summed.
    This is why a HARD_CUT/TIGHT_CUT-only plan produces the exact same
    audible result either way, and why one join's overlap can never bleed
    into a join it does not touch."""
    if not any(segment.has_independent_audio_window for segment in segments):
        return _concat_render_command(segments, output, width=width, height=height, fps=fps, workdir=workdir)

    audio_placements = _validate_audio_placements(segments, fps=fps)

    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    video_filters: list[str] = []
    audio_filters: list[str] = []
    input_index = 0

    for index, segment in enumerate(segments):
        probe = probe_media(segment.source_path)
        exact_video = rendered_segment_duration_sec(segment.duration_sec, fps=fps)

        # --- video input + chain (identical shape to _concat_render_command) ---
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
        video_chain += [f"trim=duration={exact_video:.6f}", "setpts=PTS-STARTPTS", f"format=yuv420p[v{index}]"]
        video_filters.append(",".join(video_chain))

        # --- audio input + chain (independent window, adelay-placed) ---
        audio_start = segment.effective_audio_start
        audio_end = segment.effective_audio_end
        audio_duration = max(AUDIO_TIMELINE_EPSILON_SEC, audio_end - audio_start)
        placement_ms = max(0, round(audio_placements[index] * 1000.0))
        effective_volume = 0.0 if segment.audio_muted else float(segment.audio_volume)
        audio_format = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
        if probe.has_audio:
            command += ["-ss", f"{audio_start:.3f}", "-to", f"{audio_end:.3f}", "-i", segment.source_path]
            audio_input = input_index
            input_index += 1
            audio_chain = [
                f"[{audio_input}:a]volume={effective_volume:.3f}",
                *_audio_join_fade_filters(audio_duration),
                audio_format,
                f"atrim=duration={audio_duration:.6f}",
                "asetpts=PTS-STARTPTS",
                f"adelay={placement_ms}|{placement_ms}[a{index}]",
            ]
        else:
            command += ["-f", "lavfi", "-t", f"{audio_duration:.6f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
            audio_input = input_index
            input_index += 1
            audio_chain = [
                f"[{audio_input}:a]{audio_format}",
                f"atrim=duration={audio_duration:.6f}",
                "asetpts=PTS-STARTPTS",
                f"adelay={placement_ms}|{placement_ms}[a{index}]",
            ]
        audio_filters.append(",".join(audio_chain))

    video_filters.append(
        "".join(f"[v{index}]" for index in range(len(segments))) + f"concat=n={len(segments)}:v=1:a=0[vout]"
    )
    audio_filters.append(
        "".join(f"[a{index}]" for index in range(len(segments)))
        + f"amix=inputs={len(segments)}:duration=longest:dropout_transition=0:normalize=0[aout]"
    )
    command += [
        "-filter_complex", ";".join(video_filters + audio_filters),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
        str(output),
    ]
    return command


def render_timeline_with_audio_windows(
    segments: Iterable[RenderSegment],
    output_path: str,
    *,
    width: int = 1080,
    height: int = 1920,
    fps: int = RENDER_FPS_DEFAULT,
) -> str:
    """D-214's own render entrypoint -- realizes independent per-segment
    audio/video windows (HARD_CUT/TIGHT_CUT/J_CUT/L_CUT/MICRO_AUDIO_OVERLAP
    geometry alike). NOT called by `render_preview` or any production call
    site (see this section's own module-level docstring); test-only
    execution capability. No captions/overlays -- out of this offline
    mechanism-proof task's scope; use `render_preview` for the live,
    fully-featured path."""
    segment_tuple = tuple(segments)
    if not segment_tuple:
        raise ValueError("render requires at least one segment")
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("invalid render geometry")
    for segment in segment_tuple:
        if segment.end <= segment.start:
            raise ValueError(f"invalid render segment {segment.clip_id}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cutsell-render-timeline-") as directory:
        _run(_concat_render_command_with_audio_windows(
            segment_tuple, destination, width=width, height=height, fps=fps, workdir=Path(directory),
        ))
    if not destination.exists() or destination.stat().st_size <= 0:
        raise RuntimeError("ffmpeg_render_missing_output")
    return str(destination)


# --- D-214 execution diagnostics (read-only, never consumed for a decision) --

TRANSITION_HARD_CUT = "HARD_CUT"
TRANSITION_TIGHT_CUT = "TIGHT_CUT"
TRANSITION_J_CUT = "J_CUT"
TRANSITION_L_CUT = "L_CUT"
TRANSITION_MICRO_AUDIO_OVERLAP = "MICRO_AUDIO_OVERLAP"

EXECUTION_STATUS_EXECUTABLE = "EXECUTABLE"
EXECUTION_STATUS_REJECTED = "REJECTED"


def _infer_transition_mode(left: RenderSegment, right: RenderSegment) -> str:
    """A purely DESCRIPTIVE label inferred from geometry already present on
    the two segments -- never a decision. `TIGHT_CUT` vs `HARD_CUT` here is
    a placeholder distinction at the execution layer only (this module has
    no access to Boundary's own already-applied-trim diagnostics the way
    `dialogue_pacing_transition.py`'s live TIGHT_CUT attribution does); real
    Pacing V2 mode selection remains entirely out of this task's scope."""
    right_leads = right.start - right.effective_audio_start > AUDIO_TIMELINE_EPSILON_SEC
    left_trails = left.effective_audio_end - left.end > AUDIO_TIMELINE_EPSILON_SEC
    if right_leads and left_trails:
        return TRANSITION_MICRO_AUDIO_OVERLAP
    if right_leads:
        return TRANSITION_J_CUT
    if left_trails:
        return TRANSITION_L_CUT
    return TRANSITION_HARD_CUT


def dialogue_pacing_transition_execution_diagnostics(
    segments: Iterable[RenderSegment], *, fps: int = RENDER_FPS_DEFAULT,
) -> dict:
    """One row per adjacent pair describing what `render_timeline_with_
    audio_windows` would (or, on a validation failure, would NOT) actually
    execute for that join -- requested vs. actual timing, exact placements,
    and an explicit rejection reason where applicable. Read-only; building
    this diagnostic never renders anything and never mutates `segments`."""
    segment_tuple = tuple(segments)
    rows: list[dict] = []
    try:
        video_positions = _video_timeline_positions(segment_tuple, fps=fps)
        placements = _validate_audio_placements(segment_tuple, fps=fps)
        execution_status = EXECUTION_STATUS_EXECUTABLE
        rejection_reason = None
    except ValueError as exc:
        video_positions = tuple(
            sum(rendered_segment_duration_sec(s.duration_sec, fps=fps) for s in segment_tuple[:i])
            for i in range(len(segment_tuple))
        )
        placements = tuple(video_positions)
        execution_status = EXECUTION_STATUS_REJECTED
        rejection_reason = str(exc)

    for index in range(len(segment_tuple) - 1):
        left, right = segment_tuple[index], segment_tuple[index + 1]
        left_audio_end_placement = placements[index] + left.audio_duration_sec
        right_audio_start_placement = placements[index + 1]
        overlap_sec = max(0.0, left_audio_end_placement - right_audio_start_placement)
        rows.append({
            "transition_index": index,
            "left_clip_id": left.clip_id,
            "right_clip_id": right.clip_id,
            "mode": _infer_transition_mode(left, right),
            "video_switch_time": round(video_positions[index + 1], 3),
            "left_audio_end_source": round(left.effective_audio_end, 3),
            "right_audio_start_source": round(right.effective_audio_start, 3),
            "left_audio_end_timeline": round(left_audio_end_placement, 3),
            "right_audio_start_timeline": round(right_audio_start_placement, 3),
            "requested_overlap_sec": round(
                max(0.0, (right.start - right.effective_audio_start)) + max(0.0, (left.effective_audio_end - left.end)),
                3,
            ),
            "actual_overlap_sec": round(overlap_sec, 3),
            "timeline_gap_sec": round(max(0.0, right_audio_start_placement - left_audio_end_placement), 3),
            "execution_status": execution_status,
            "fallback_reason": rejection_reason,
        })
    return {
        "schema_version": "cutsell.render_timeline_execution_diagnostics.v1",
        "segment_count": len(segment_tuple),
        "execution_status": execution_status,
        "rejection_reason": rejection_reason,
        "transitions": rows,
    }


# =============================================================================
# D-233 -- AUDIO JOIN TREATMENT RENDERER/TIMING CONTRACT EXECUTION (OFFLINE
# ONLY, NOT LIVE-WIRED)
# =============================================================================
#
# Consumes exactly one `pacing_v2_audio_join_treatment_timing.
# AudioJoinTreatmentTimingPlan` (duck-typed here, never imported, matching
# this module's own existing D-214 execution-diagnostics precedent of
# defining local literal constants rather than a cross-module import) and
# realizes it as a bounded, two-source, AUDIO-ONLY ffmpeg render. This is a
# test/offline execution CAPABILITY -- like `render_timeline_with_audio_
# windows` above, it is NOT imported or called by `render_preview`,
# `universal_clean_cut.py`, `pipeline.py`, or any other production call
# site, has no feature flag, and is used by tests only.
#
# ## Video-timing immutability, structural
#
# This function has NO video-geometry parameter anywhere on its signature
# -- it cannot move a visual cut point by construction, not merely by
# convention. `plan.visual_join_time` is consumed only as a diagnostic
# label for waveform verification (see the test file), never as a video
# position to render.
#
# ## Filter strategy: afade + adelay + amix, never acrossfade
#
# `acrossfade` is a two-input-only filter that would force this function
# to always take exactly two inputs and would own its own internal timing
# model -- unwanted coupling for a shape (AMBIENCE_CARRY_LEFT/RIGHT can be
# ONE-input-only; AMBIENCE_BRIDGE and SHORT_CROSSFADE need two) that
# varies per treatment. `afade` (parameterized IN/OUT envelope, distinct
# constant from the existing fixed 12ms `_AUDIO_JOIN_FADE_SEC` click
# fade), `adelay` (independent per-source output placement, the EXACT
# same primitive `_concat_render_command_with_audio_windows` already uses
# for J_CUT/L_CUT/MICRO_AUDIO_OVERLAP geometry), and `amix(normalize=0)`
# (the same load-bearing "never re-scale, only sum" property that section
# already established) compose cleanly across all four treatment shapes
# with ONE small, deterministic, reused implementation.
#
# ## Technical 12ms click-fade interaction (explicit decision)
#
# The treatment's OWN envelope (a crossfade's fade-out/fade-in, sized to
# `plan.chosen_duration`) already reaches silence smoothly at the treated
# edge -- re-applying the fixed 12ms technical click fade AT THAT SAME
# EDGE would be a redundant, stacked envelope (a fade-within-a-fade,
# though harmless in practice since 12 ms is far shorter than any real
# `chosen_duration`, it is still an unnecessary second envelope). Decision
# (tested): the treatment's own envelope ABSORBS the technical click fade
# at the specific edge it covers; this function never calls `_audio_join_
# fade_filters` at all -- it is a bounded, isolated, two/one-source slice
# render, not a whole-segment render, so there is no "other, untreated
# edge" for it to protect in the first place (that remains the
# unmodified, whole-segment responsibility of `_concat_render_command`/
# `_concat_render_command_with_audio_windows` in any FUTURE live
# integration, entirely out of this task's own scope).

_TREATMENT_SHORT_CROSSFADE = "SHORT_CROSSFADE"
_TREATMENT_AMBIENCE_CARRY_LEFT = "AMBIENCE_CARRY_LEFT"
_TREATMENT_AMBIENCE_CARRY_RIGHT = "AMBIENCE_CARRY_RIGHT"
_TREATMENT_AMBIENCE_BRIDGE = "AMBIENCE_BRIDGE"
_TIMING_SUPPORTED = "SUPPORTED"


def _treatment_envelope_filters(kind: str, duration_sec: float) -> list[str]:
    """Parameterized fade envelope -- distinct constant/purpose from the
    existing fixed `_AUDIO_JOIN_FADE_SEC` (12 ms) technical click fade;
    never reuses that constant, never modifies it. `kind` is `"IN"` or
    `"OUT"`; the WHOLE clip fades across its own full `duration_sec` (a
    treatment slice IS the fade -- there is no untreated remainder inside
    this bounded slice)."""
    d = max(0.0, float(duration_sec))
    if d <= 0.0:
        return []
    if kind == "IN":
        return [f"afade=t=in:st=0:d={d:.6f}"]
    return [f"afade=t=out:st=0:d={d:.6f}"]


def render_audio_join_treatment_preview(
    plan: object,
    output_path: str,
    *,
    left_source_path: str | None = None,
    right_source_path: str | None = None,
    sample_rate: int = 48000,
) -> str:
    """D-233's own bounded, offline, NOT-live-wired executor for exactly
    ONE `AudioJoinTreatmentTimingPlan` (duck-typed -- see module docstring
    section above for why no cross-module import is taken). Produces an
    AUDIO-ONLY output file (a real ffmpeg-encoded audio stream, `.wav` or
    any ffmpeg-supported audio container the caller names via
    `output_path`'s own extension) -- video-timing immutability is
    structural, not merely tested (see above).

    Raises `ValueError` (fails closed, never silently substitutes a
    different treatment) when `plan.timing_status != "SUPPORTED"` or when
    the plan's own treatment requires a source path this call did not
    receive."""
    timing_status = getattr(plan, "timing_status", None)
    if timing_status != _TIMING_SUPPORTED:
        raise ValueError(f"audio_join_treatment_timing_not_supported:{timing_status}")
    treatment = getattr(plan, "treatment")
    chosen_duration = float(getattr(plan, "chosen_duration"))

    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    audio_filters: list[str] = []
    input_index = 0
    labels: list[str] = []

    left_start = getattr(plan, "left_source_audio_start", None)
    if left_start is not None:
        if not left_source_path:
            raise ValueError("audio_join_treatment_missing_left_source_path")
        left_end = float(getattr(plan, "left_source_audio_end"))
        command += ["-ss", f"{float(left_start):.6f}", "-to", f"{left_end:.6f}", "-i", left_source_path]
        filter_parts = []
        if treatment == _TREATMENT_SHORT_CROSSFADE:
            filter_parts += _treatment_envelope_filters("OUT", chosen_duration)
        filter_parts.append(f"aformat=sample_fmts=fltp:sample_rates={sample_rate}:channel_layouts=stereo")
        delay_ms = max(0, round(float(getattr(plan, "left_output_audio_start") or 0.0) * 1000.0))
        filter_parts.append(f"adelay={delay_ms}|{delay_ms}[left]")
        audio_filters.append(f"[{input_index}:a]" + ",".join(filter_parts))
        labels.append("[left]")
        input_index += 1

    right_start = getattr(plan, "right_source_audio_start", None)
    if right_start is not None:
        if not right_source_path:
            raise ValueError("audio_join_treatment_missing_right_source_path")
        right_end = float(getattr(plan, "right_source_audio_end"))
        command += ["-ss", f"{float(right_start):.6f}", "-to", f"{right_end:.6f}", "-i", right_source_path]
        filter_parts = []
        if treatment in (_TREATMENT_SHORT_CROSSFADE, _TREATMENT_AMBIENCE_CARRY_RIGHT):
            filter_parts += _treatment_envelope_filters("IN", chosen_duration)
        filter_parts.append(f"aformat=sample_fmts=fltp:sample_rates={sample_rate}:channel_layouts=stereo")
        delay_ms = max(0, round(float(getattr(plan, "right_output_audio_start") or 0.0) * 1000.0))
        filter_parts.append(f"adelay={delay_ms}|{delay_ms}[right]")
        audio_filters.append(f"[{input_index}:a]" + ",".join(filter_parts))
        labels.append("[right]")
        input_index += 1

    if not labels:
        raise ValueError("audio_join_treatment_no_source_window_in_plan")

    if len(labels) == 1:
        audio_filters.append(f"{labels[0]}anull[aout]")
    else:
        audio_filters.append("".join(labels) + f"amix=inputs={len(labels)}:duration=longest:dropout_transition=0:normalize=0[aout]")

    command += [
        "-filter_complex", ";".join(audio_filters),
        "-map", "[aout]",
        str(output_path),
    ]
    _run(command)
    destination = Path(output_path)
    if not destination.exists() or destination.stat().st_size <= 0:
        raise RuntimeError("ffmpeg_audio_join_treatment_missing_output")
    return str(destination)
