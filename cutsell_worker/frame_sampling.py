"""Source-safe temporal frame sampling for multimodal take analysis."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Tuple

from .contracts import CandidateTake


@dataclass(frozen=True)
class FrameSample:
    clip_id: str
    source_asset_id: str
    timestamp: float
    path: str


def sample_take_frames(
    media_path: str,
    take: CandidateTake,
    output_dir: str,
    *,
    frame_count: int = 3,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Tuple[FrameSample, ...]:
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    if take.end <= take.start:
        return ()

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    duration = take.end - take.start
    fractions = tuple((index + 1) / (frame_count + 1) for index in range(frame_count))
    samples = []
    for index, fraction in enumerate(fractions):
        timestamp = take.start + duration * fraction
        destination = directory / f"{take.clip_id}-{index:02d}.jpg"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{timestamp:.3f}", "-i", media_path,
            "-frames:v", "1", "-q:v", "3", str(destination),
        ]
        runner(command, capture_output=True, check=True)
        samples.append(FrameSample(
            clip_id=take.clip_id,
            source_asset_id=take.source_asset_id,
            timestamp=timestamp,
            path=str(destination),
        ))
    return tuple(samples)
