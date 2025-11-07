# /workspace/EditDNA-worker/worker/asr.py

from __future__ import annotations
import os
import tempfile
from typing import List, Dict, Any

from faster_whisper import WhisperModel

# we use moviepy just to pull audio out of .mov/.mp4
from moviepy.editor import VideoFileClip

# ------------------------------------------------------------------
# Config knobs (can be overridden by env)
# ------------------------------------------------------------------
_MODEL_NAME = os.getenv("ASR_MODEL", "medium")
_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "auto")  # e.g. "int8", "int8_float16", "auto"

# we keep a global so we don’t reload the model every job
_model = None


def _get_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model

    print(f"[asr] loading Faster-Whisper model={_MODEL_NAME} compute_type={_COMPUTE_TYPE}", flush=True)
    _model = WhisperModel(
        _MODEL_NAME,
        device="cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
        compute_type=_COMPUTE_TYPE,
    )
    return _model


def _extract_audio(video_path: str) -> str:
    """
    Take a local video file, extract audio to a temp .wav, return path.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_audio_path = tmp.name
    tmp.close()

    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(tmp_audio_path, verbose=False, logger=None)
    clip.close()

    return tmp_audio_path


def transcribe(local_video_path: str) -> List[Dict[str, Any]]:
    """
    Main entry the pipeline calls:
        asr.transcribe("/tmp/IMG_03.mov")

    Returns list of segments:
    [
      {"text": "hello", "start": 0.0, "end": 2.3},
      ...
    ]
    """
    # 1) get model
    model = _get_model()

    # 2) pull audio from video
    audio_path = _extract_audio(local_video_path)

    print(f"[asr] transcribing audio={audio_path}", flush=True)

    # 3) run ASR
    segments, info = model.transcribe(
        audio_path,
        beam_size=1,
        best_of=1,
        vad_filter=True,
    )

    out: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        # seg is a faster-whisper Segment object
        text = (seg.text or "").strip()
        if not text:
            continue
        out.append(
            {
                "text": text,
                "start": float(seg.start),
                "end": float(seg.end),
            }
        )

    print(f"[asr] got {len(out)} segments", flush=True)
    return out
