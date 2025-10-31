import os
from typing import List, Dict, Any


def run_asr_and_segments(local_video_path: str) -> List[Dict[str, Any]]:
    """
    This calls ASR (speech-to-text) + sentence boundary pass + micro-cut logic.

    You already had Whisper running in the pod (ASR_ENABLED=1 logs). In your pod
    logs we saw:
        [asr] segments: 21
        [seg] takes: 21
        [micro] input_takes=21 → micro_takes=21

    So here's what we do:
    - If ASR_ENABLED=1, we EXPECT you already have a working function in your
      old code that produced those segment dicts. Plug that here.
    - If ASR_ENABLED=0, we just return a dummy empty list.

    Each segment dict must look like:
    {
        "start": float,
        "end": float,
        "text": "spoken words",
        "face_q": 1.0,
        "scene_q": 1.0,
        "vtx_sim": 0.0
    }

    IMPORTANT:
    Right now this is a stub so you can boot. It raises if ASR is on but
    we didn't wire your Whisper code.
    """

    asr_enabled = os.getenv("ASR_ENABLED", "1").strip() in ("1", "true", "yes", "on")

    if not asr_enabled:
        # no ASR? return empty to avoid crash
        return []

    # ----- REAL IMPLEMENTATION NEEDED -----
    # You ALREADY have working ASR in the old pipeline (because we saw actual
    # timestamps, text, etc in your successful responses).
    #
    # Take that code (whatever generated "segments: 21") and paste it
    # below instead of raising.
    #
    raise RuntimeError(
        "run_asr_and_segments() needs to call your Whisper/ASR segmentation code "
        "that produced the [asr] segments / [seg] takes in your logs."
    )
