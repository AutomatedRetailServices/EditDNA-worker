# overwrite the vision_sampler the worker actually imports
cat > /workspace/EditDNA-worker/worker/vision_sampler.py <<'PY'
"""
vision_sampler.py
Small helper for picking frame timestamps from a video for vision / OCR / scene checks.
Keep this file super simple so pipeline import never breaks.
"""

from __future__ import annotations
from typing import List


def sample_timestamps(
    duration_sec: float,
    interval_sec: float = 2.0,
    max_samples: int = 50,
) -> List[float]:
    """
    Return a list of timestamps (in seconds) where we want to sample the video.
    """
    if duration_sec <= 0:
        return []

    if interval_sec <= 0:
        interval_sec = 2.0

    ts: List[float] = []
    t = 0.0
    while t < duration_sec and len(ts) < max_samples:
        ts.append(round(t, 3))
        t += interval_sec

    if duration_sec not in ts and len(ts) < max_samples:
        ts.append(round(duration_sec, 3))

    return ts


if __name__ == "__main__":
    print(sample_timestamps(10.5, interval_sec=2.5))
PY
