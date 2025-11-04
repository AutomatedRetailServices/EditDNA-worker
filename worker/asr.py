"""
worker/asr.py
SAFE VERSION – no CUDA, no faster-whisper, no onnxruntime.

This exists to keep the RQ worker from crashing the whole process when
the ASR stack tries to load cuDNN but the container doesn't have the right
GPU libs.

Later, when the image has the right cuDNN / onnxruntime, we can swap back
to the real ASR. For now, we just return one segment so the rest of the
pipeline (cut, concat, S3, JSON) can run.
"""

from __future__ import annotations
from typing import List, Dict, Any


def transcribe_segments(path: str) -> List[Dict[str, Any]]:
    # we ignore the actual file, this is just to keep the pipeline alive
    return [
        {
            "start": 0.0,
            "end": 10.0,
            "text": "TEMP PLACEHOLDER: ASR forced to CPU-safe stub (no cuDNN).",
        }
    ]
