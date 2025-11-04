"""
worker/asr.py

ASR stack for EditDNA worker.

Plan:
1. Try GPU faster-whisper (we're on CUDA 12.4 + cuDNN 9 image now)
2. If that fails, try CPU faster-whisper
3. If THAT fails, return a safe placeholder so RQ never crashes
"""

from __future__ import annotations
from typing import List, Dict, Any
import os


def _placeholder(path: str) -> List[Dict[str, Any]]:
    # keeps the pipeline alive even if ASR is totally broken
    return [
        {
            "start": 0.0,
            "end": 10.0,
            "text": "TEMP PLACEHOLDER: ASR fallback.",
        }
    ]


def _gpu_asr(path: str) -> List[Dict[str, Any]]:
    from faster_whisper import WhisperModel

    model_name = os.getenv("ASR_MODEL", "small")
    # we are on a CUDA 12.4 + cuDNN 9 image, so we can use device="cuda"
    model = WhisperModel(model_name, device="cuda", compute_type="float16")

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


def _cpu_asr(path: str) -> List[Dict[str, Any]]:
    from faster_whisper import WhisperModel

    model_name = os.getenv("ASR_MODEL", "small")
    # CPU path, slower but doesn't need GPU libs
    model = WhisperModel(model_name, device="cpu", compute_type="int8")

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
    """
    Public entry the pipeline calls.

    Env knobs:
      ASR_BACKEND=gpu|cpu|auto (default: auto)
      ASR_MODEL=small|medium|large-v2 ...
    """
    backend = os.getenv("ASR_BACKEND", "auto").lower()

    # explicit GPU
    if backend == "gpu":
        try:
            return _gpu_asr(path)
        except Exception:
            return _placeholder(path)

    # explicit CPU
    if backend == "cpu":
        try:
            return _cpu_asr(path)
        except Exception:
            return _placeholder(path)

    # auto → try GPU first, then CPU
    try:
        return _gpu_asr(path)
    except Exception:
        try:
            return _cpu_asr(path)
        except Exception:
            return _placeholder(path)
