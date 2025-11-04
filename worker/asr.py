"""
worker/asr.py

Default: SAFE stub ASR (no CUDA, no cuDNN, no onnxruntime).
Optional: set env ASR_BACKEND=faster to TRY faster-whisper.
If faster-whisper fails (missing libs), we fall back to the safe stub.
"""

from __future__ import annotations
from typing import List, Dict, Any
import os


# ---------- always-safe version ----------
def _safe_stub(path: str) -> List[Dict[str, Any]]:
    return [
        {
            "start": 0.0,
            "end": 10.0,
            "text": "TEMP PLACEHOLDER: ASR forced to CPU-safe stub (no cuDNN).",
        }
    ]


# ---------- optional faster-whisper version ----------
def _try_faster_whisper(path: str) -> List[Dict[str, Any]]:
    # import inside so if it explodes, we can catch it
    from faster_whisper import WhisperModel  # type: ignore

    model_name = os.getenv("ASR_MODEL", "small")
    compute_type = os.getenv("ASR_COMPUTE_TYPE", "int8")
    model = WhisperModel(model_name, compute_type=compute_type)

    out: List[Dict[str, Any]] = []
    segments, _info = model.transcribe(path, beam_size=5)
    for seg in segments:
        out.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            }
        )
    return out


def transcribe_segments(path: str) -> List[Dict[str, Any]]:
    backend = os.getenv("ASR_BACKEND", "safe").lower()

    # default: safe
    if backend != "faster":
        return _safe_stub(path)

    # if user explicitly asked for faster, try it
    try:
        return _try_faster_whisper(path)
    except Exception:
        # if the container doesn't have cuDNN / onnxruntime, don't crash the worker
        return _safe_stub(path)
