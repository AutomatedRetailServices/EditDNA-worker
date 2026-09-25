from __future__ import annotations
from typing import List

def sample_timestamps(duration_sec: float, interval_sec: float = 2.0, max_samples: int = 50) -> List[float]:
    """Return a list of timestamps to sample from a video."""
    if duration_sec <= 0:
        return []
    if interval_sec <= 0:
        interval_sec = 2.0

    ts = []
    t = 0.0
    while t < duration_sec and len(ts) < max_samples:
        ts.append(round(t, 3))
        t += interval_sec

    if duration_sec not in ts and len(ts) < max_samples:
        ts.append(round(duration_sec, 3))

    return ts
