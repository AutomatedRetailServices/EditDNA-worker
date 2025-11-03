# /workspace/worker/asr.py
"""
ASR helpers for EditDNA worker.

We try to use faster-whisper if it's installed.
If not, we fall back to a dummy segment so the pipeline can still run.
"""

from __future__ import annotations
from typing import List, Dict, Any
import os


def _fake_segments(path: str) -> List[Dict[str, Any]]:
    return [
        {
            "start": 0.0,
            "end": 10.0,
            "text": "TEMP PLACEHOLDER: real ASR not wired yet.",
        }
    ]


def _segments_faster_whisper(path: str) -> List[Dict[str, Any]]:
    # imported inside so we can catch ImportError
    from faster_whisper import WhisperModel  # type: ignore

    model_name = os.getenv("ASR_MODEL", "small")
    compute_type = os.getenv("ASR_COMPUTE_TYPE", "int8")

    model = WhisperModel(model_name, compute_type=compute_type)

    segments_out: List[Dict[str, Any]] = []

    segments, _info = model.transcribe(path, beam_size=5)
    for seg in segments:
        segments_out.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            }
        )
    return segments_out


def transcribe_segments(path: str) -> List[Dict[str, Any]]:
    """
    Public entrypoint used by pipeline.
    """
    try:
        return _segments_faster_whisper(path)
    except ModuleNotFoundError:
        # model not installed in this image → return placeholder
        return _fake_segments(path)
    except Exception:
        # any other ASR problem → return placeholder
        return _fake_segments(path)
