# /workspace/EditDNA-worker/worker/asr.py
from __future__ import annotations
import os
from typing import Dict, Any, List
from faster_whisper import WhisperModel

# you can change these with env vars
_MODEL_NAME = os.getenv("ASR_MODEL", "small")
_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "auto")

_model = None


def _get_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model

    device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") else "cpu"
    _model = WhisperModel(
        _MODEL_NAME,
        device=device,
        compute_type=_COMPUTE_TYPE,
    )
    return _model


def transcribe(local_video_path: str) -> Dict[str, Any]:
    """
    Standard shape the pipeline expects:
    {
      "text": "... full transcript ...",
      "segments": [
        {"start": 0.0, "end": 2.5, "text": "first line"},
        {"start": 2.5, "end": 5.0, "text": "second line"},
        ...
      ]
    }
    """
    model = _get_model()

    seg_gen, info = model.transcribe(local_video_path)

    segments: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

    for seg in seg_gen:
        txt = (seg.text or "").strip()
        full_text_parts.append(txt)
        segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": txt,
            }
        )

    return {
        "text": " ".join([p for p in full_text_parts if p]),
        "segments": segments,
    }
