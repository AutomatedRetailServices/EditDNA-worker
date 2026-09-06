"""Objective audio dead-air intervals from the source media (D-095.2).

Why this exists: every silence signal the engine had BEFORE Selection Freeze
was derived from ASR word timing (`silence_analysis.word_silence_gaps`,
`take_segmentation._speech_units`, `post_selection_interior_gap_trim`'s
word-gap scan). Whisper-style word timestamps are routinely stretched over
real silence inside a hesitant delivery, so a kept take could carry seconds
of dead air that no word gap ever revealed -- and the first objective audio
measurement happened only AFTER render, in `post_render_media_qc`
(LINGERING_ACCIDENTAL_SILENCE, -35 dB for >= 1.2 s, unrepairable when it
sits mid-segment). Run 33995806350: a 12.5 s kept candidate carrying a
2.36 s interior silence invalidated the whole render.

This module measures the same thing the QC measures, on the SOURCE, once
per source, with ffmpeg's `silencedetect` (the exact filter
`render.tighten_trailing_silence` already trusts for trailing post-roll) and
publishes the intervals as `TemporalEvent`s of kind
``audio_silence_interval`` on the whole-video context -- the same channel
local-performance reset events already travel on -- so the existing
interior-gap trimmer can use them as evidence. Observability + evidence
only: nothing here changes semantic membership.
"""
from __future__ import annotations

from dataclasses import replace
import re
import subprocess
from typing import Iterable, Mapping

from .whole_video_analysis import TemporalEvent, WholeVideoContext

AUDIO_SILENCE_EVENT_KIND = "audio_silence_interval"
DEFAULT_NOISE_DB = -35.0
DEFAULT_MINIMUM_SILENCE_SEC = 0.60
_SUBPROCESS_TIMEOUT_SEC = 600

_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[0-9.]+)")


def detect_audio_silence_intervals(
    path: str,
    *,
    noise_db: float = DEFAULT_NOISE_DB,
    minimum_silence_sec: float = DEFAULT_MINIMUM_SILENCE_SEC,
    ffmpeg_bin: str = "ffmpeg",
) -> tuple[tuple[float, float], ...]:
    """Closed silence intervals (source seconds) at or below ``noise_db`` lasting
    at least ``minimum_silence_sec``. Never raises: an unreadable file, a
    missing ffmpeg or a timeout yields an empty tuple (the caller records
    the count, so an empty result is visible, never silent)."""
    command = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "info", "-nostats",
        "-i", str(path), "-vn",
        "-af", f"silencedetect=noise={noise_db:.1f}dB:d={minimum_silence_sec:.3f}",
        "-f", "null", "-",
    ]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=_SUBPROCESS_TIMEOUT_SEC,
        )
    except Exception:  # noqa: BLE001 -- evidence source unavailable; reported via count, never fatal
        return ()
    if completed.returncode != 0:
        return ()
    intervals: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in completed.stderr.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            pending_start = max(0.0, float(start_match.group(1)))
        end_match = _SILENCE_END_RE.search(line)
        if end_match and pending_start is not None:
            end = float(end_match.group(1))
            if end - pending_start >= minimum_silence_sec - 1e-3:
                intervals.append((pending_start, end))
            pending_start = None
    # A silence still open at end-of-file is trailing post-roll, not an
    # interior interval; tighten_trailing_silence owns that case.
    return tuple(sorted(intervals))


def audio_silence_events(
    local_paths: Mapping[str, str],
    *,
    noise_db: float = DEFAULT_NOISE_DB,
    minimum_silence_sec: float = DEFAULT_MINIMUM_SILENCE_SEC,
) -> dict[str, tuple[TemporalEvent, ...]]:
    out: dict[str, tuple[TemporalEvent, ...]] = {}
    for source_asset_id, path in sorted(local_paths.items()):
        intervals = detect_audio_silence_intervals(path, noise_db=noise_db, minimum_silence_sec=minimum_silence_sec)
        out[str(source_asset_id)] = tuple(
            TemporalEvent(
                source_asset_id=str(source_asset_id),
                start=float(start),
                end=float(end),
                kind=AUDIO_SILENCE_EVENT_KIND,
                confidence=1.0,
                description=f"ffmpeg silencedetect <= {noise_db:.0f} dB for {end - start:.2f}s",
            )
            for start, end in intervals
        )
    return out


def merge_audio_silence_into_context(
    context: WholeVideoContext,
    events_by_source: Mapping[str, Iterable[TemporalEvent]],
) -> WholeVideoContext:
    """Add the audio silence events to each matching source of the whole-video
    context (deduplicated on kind/start/end), mirroring
    ``local_performance.merge_local_events_into_context``."""
    merged = []
    for source in context.sources:
        additions_source = events_by_source.get(source.source_asset_id)
        if not additions_source:
            merged.append(source)
            continue
        known = {(e.kind, round(float(e.start), 3), round(float(e.end), 3)) for e in source.events}
        additions = tuple(
            e for e in additions_source
            if (e.kind, round(float(e.start), 3), round(float(e.end), 3)) not in known
        )
        merged.append(replace(source, events=tuple(sorted(
            tuple(source.events) + additions, key=lambda e: (float(e.start), float(e.end), e.kind),
        ))))
    return replace(context, sources=tuple(merged))
