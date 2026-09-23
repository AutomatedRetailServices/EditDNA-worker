"""Bounded whole-source temporal sampling for CutSell Watch + Listen context."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple


@dataclass(frozen=True)
class SourceFrameSample:
    source_asset_id: str
    timestamp: float
    path: str
    relative_position: float


def sample_source_frames(
    media_path: str,
    *,
    source_asset_id: str,
    duration_sec: float,
    output_dir: str,
    target_interval_sec: float = 0.75,
    min_frames: int = 12,
    max_frames: int = 120,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Tuple[SourceFrameSample, ...]:
    """Observe the complete source at useful temporal density.

    Whole-source sampling is deliberately denser than take-level scoring because
    its job is to catch short performance transitions: a facial reaction after a
    good line, body reset, glance away, retry setup, product fumble, visual hook,
    or a meaningful silent beat. The ceiling keeps API payload/cost bounded while
    still covering an ordinary short-form raw recording throughout its duration.
    """
    duration = max(0.0, float(duration_sec))
    if duration <= 0.0:
        return ()
    if min_frames < 1 or max_frames < min_frames:
        raise ValueError("invalid frame bounds")
    interval = max(0.25, float(target_interval_sec))
    desired = int(duration / interval) + 1
    count = max(min_frames, min(max_frames, desired))

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    edge = min(0.01, 0.15 / max(1, count))
    if count == 1:
        fractions = (0.5,)
    else:
        usable = 1.0 - 2.0 * edge
        fractions = tuple(edge + usable * i / (count - 1) for i in range(count))

    samples = []
    for index, fraction in enumerate(fractions):
        timestamp = duration * fraction
        destination = directory / f"{source_asset_id}-whole-{index:03d}.jpg"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{timestamp:.3f}", "-i", media_path,
            "-frames:v", "1", "-q:v", "3", str(destination),
        ]
        runner(command, capture_output=True, check=True)
        samples.append(SourceFrameSample(
            source_asset_id=source_asset_id,
            timestamp=timestamp,
            path=str(destination),
            relative_position=float(fraction),
        ))
    return tuple(samples)
