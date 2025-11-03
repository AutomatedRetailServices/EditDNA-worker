from __future__ import annotations
from typing import List, Dict, Any
import os

def _fake_segments(path: str) -> List[Dict[str, Any]]:
    return [{
        "start": 0.0,
        "end": 10.0,
        "text": "TEMP PLACEHOLDER: real ASR not wired yet.",
    }]

def _segments_faster_whisper(path: str) -> List[Dict[str, Any]]:
    from faster_whisper import WhisperModel
    model_name = os.getenv("ASR_MODEL", "small")
    compute_type = os.getenv("ASR_COMPUTE_TYPE", "int8")
    model = WhisperModel(model_name, compute_type=compute_type)
    out: List[Dict[str, Any]] = []
    segments, _info = model.transcribe(path, beam_size=5)
    for seg in segments:
        out.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
        })
    return out

def transcribe_segments(path: str) -> List[Dict[str, Any]]:
    try:
        return _segments_faster_whisper(path)
    except Exception:
        return _fake_segments(path)
