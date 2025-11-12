# worker/vision.py
# Minimal hooks for future face/vision scoring. Currently returns neutral scores.
from typing import Optional

def estimate_face_quality(local_path: str, start: float, end: float) -> Optional[float]:
    """
    Placeholder hook. Return a float [0..1] indicating face/visual stability.
    For now: return 1.0 so it doesn't block anything.
    """
    return 1.0
