# /workspace/worker/asr.py
"""
ASR helpers for EditDNA worker.

We try to use faster-whisper if it's installed.
If not, we fall back to a dummy segment so the pipeline can still run
and the API gets a valid JSON instead of a 500.
"""

from __future__ import annotations
from typing import List, Dict, Any
import os


def _fake_segments(path: str) -> List[Dict[str, Any]]:
    # last-resort placeholder, same style you had before
    return [
        {
            "start": 0.0,
            "end": 10.0,
            "text": "TEMP PLACEHOLDER: real ASR not wired yet.",
        }
    ]


def _segments_faster_whisper(path: str) -> List[Dict[str, Any]]:
    # we keep the import inside so ImportError can be caught above
    from faster_whisper import WhisperModel  # type: ignore

    # model choice: allow override via env
    model_name = os.getenv("ASR_MODEL", "medium")
    compute_type = os.getenv("ASR_COMPUTE_TYPE", "int8")  # or "float16" if GPU
    model = WhisperModel(model_name, compute_type=compute_type)

    segments_out: List[Dict[str, Any]] = []
    # beam_size etc can be tuned later
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
        # faster_whisper not installed in this image → return placeholder
        return _fake_segments(path)
    except Exception:
        # any other ASR problem → also return placeholder
        return _fake_segments(path)
