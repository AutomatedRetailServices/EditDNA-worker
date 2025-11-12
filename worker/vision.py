import cv2
import os
import base64
from typing import List, Tuple, Optional
from worker import utils

def extract_midframe_b64(local_video_path: str, start: float, end: float) -> Optional[str]:
    """Grab a mid-frame between start/end, return base64 PNG data URI (for GPT-4o)."""
    try:
        cap = cv2.VideoCapture(local_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        t = (start + end) / 2.0
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        # encode PNG to memory
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None
