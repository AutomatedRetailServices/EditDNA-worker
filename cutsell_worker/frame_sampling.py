"""Source-safe adaptive temporal frame sampling for multimodal take analysis."""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

from .contracts import CandidateTake


@dataclass(frozen=True)
class FrameSample:
    clip_id: str
    source_asset_id: str
    timestamp: float
    path: str
    relative_position: float = 0.5


def adaptive_frame_count(
    take: CandidateTake,
    *,
    min_frames: int = 4,
    max_frames: int = 12,
    target_fps: float = 1.5,
) -> int:
    """Choose enough samples to observe delivery across the whole take.

    Short takes still get multiple observations. Longer takes scale up toward a
    bounded ceiling so visual QA is meaningfully temporal without exploding cost.
    """
    if min_frames < 1 or max_frames < min_frames:
        raise ValueError("invalid frame bounds")
    duration = max(0.0, take.duration_sec)
    desired = int(math.ceil(duration * max(0.1, target_fps)))
    return max(min_frames, min(max_frames, desired))


def sample_take_frames(
    media_path: str,
    take: CandidateTake,
    output_dir: str,
    *,
    frame_count: int | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Tuple[FrameSample, ...]:
    """Sample a take across its full temporal span, including near-edge delivery.

    The previous implementation used three fixed interior frames. The adaptive
    default observes more of the take and intentionally includes points near the
    beginning/end where false starts, glances and post-line fumbles often happen.
    """
    if take.end <= take.start:
        return ()
    count = adaptive_frame_count(take) if frame_count is None else int(frame_count)
    if count < 1:
        raise ValueError("frame_count must be at least 1")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    duration = take.end - take.start

    # Stay just inside source boundaries while covering almost the entire take.
    # Equal spacing gives stable, deterministic samples for regression QA.
    edge = min(0.08, 0.45 / max(1, count))
    if count == 1:
        fractions = (0.5,)
    else:
        usable = 1.0 - (2.0 * edge)
        fractions = tuple(edge + usable * index / (count - 1) for index in range(count))

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
            relative_position=float(fraction),
        ))
    return tuple(samples)
