"""
worker/asr.py
Simple ASR wrapper around faster-whisper so pipeline.py can call:

    from worker import asr
    result = asr.transcribe("/tmp/video.mp4")

and get back { "segments": [ { "start":..., "end":..., "text":... }, ... ] }
"""

from __future__ import annotations
from typing import List, Dict, Any
import os
import tempfile
import subprocess

from faster_whisper import WhisperModel


# you can tweak these
_MODEL_NAME = os.getenv("ASR_MODEL", "medium")   # or "small", "base", etc.
_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "auto")  # "auto", "float16", "int8"


# load once
_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            _MODEL_NAME,
            device="cuda" if os.path.exists("/dev/nvidia0") or os.path.exists("/dev/nvidiactl") else "cpu",
            compute_type=_COMPUTE_TYPE,
        )
    return _model


def _extract_audio(input_video: str) -> str:
    """
    If faster-whisper can read the video directly you can skip this, but
    this makes it explicit: we turn video -> wav
    """
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        tmp_wav,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_wav


def transcribe(video_path: str) -> Dict[str, Any]:
    """
    Main entrypoint the rest of the pipeline expects.
    Returns:
    {
      "segments": [
        {"start": float, "end": float, "text": str},
        ...
      ]
    }
    """
    model = _get_model()

    # you can let faster-whisper read video directly, but this is safer
    audio_path = _extract_audio(video_path)

    segments_out: List[Dict[str, Any]] = []
    # note: adjust language / task if you want
    segments, info = model.transcribe(audio_path, beam_size=5)

    for seg in segments:
        segments_out.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            }
        )

    return {
        "segments": segments_out,
        "duration": getattr(info, "duration", None),
        "language": getattr(info, "language", None),
    }
