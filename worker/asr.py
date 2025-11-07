# /workspace/EditDNA-worker/worker/asr.py

from __future__ import annotations
import os
import tempfile
import subprocess
from typing import List, Dict, Any, Optional

from faster_whisper import WhisperModel

# -------------------------------------------------
# config from env
# -------------------------------------------------
_MODEL_NAME = os.getenv("ASR_MODEL", "medium")
_ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "auto")

# keep model cached
_model: Optional[WhisperModel] = None


def _pick_compute_type(device: str) -> str:
    """
    If user asked for float16 but we are on CPU, fall back to int8.
    This is what your error was about.
    """
    ct = _ASR_COMPUTE_TYPE
    if device == "cpu" and ct.lower() in ("float16", "fp16", "float32"):
        print("[asr] requested float16 on CPU → falling back to int8", flush=True)
        return "int8"
    return ct


def _get_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model

    device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    compute_type = _pick_compute_type(device)

    print(
        f"[asr] loading Faster-Whisper model={_MODEL_NAME} device={device} compute_type={compute_type}",
        flush=True,
    )

    _model = WhisperModel(
        _MODEL_NAME,
        device=device,
        compute_type=compute_type,
    )
    return _model


# ---------- audio extraction helpers ----------

def _extract_audio_with_moviepy(video_path: str) -> str:
    # import inside so missing moviepy doesn't break import
    from moviepy.editor import VideoFileClip

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_audio_path = tmp.name
    tmp.close()

    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(tmp_audio_path, verbose=False, logger=None)
    clip.close()
    return tmp_audio_path


def _extract_audio_with_ffmpeg(video_path: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_audio_path = tmp.name
    tmp.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        tmp_audio_path,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_audio_path


def _extract_audio(video_path: str) -> str:
    try:
        return _extract_audio_with_moviepy(video_path)
    except Exception as e:
        print(f"[asr] moviepy failed ({e}) → trying ffmpeg", flush=True)
        return _extract_audio_with_ffmpeg(video_path)


# ---------- main API ----------

def transcribe(local_video_path: str) -> List[Dict[str, Any]]:
    """
    Return list of segments: [{"text":..., "start":..., "end":...}, ...]
    """
    model = _get_model()
    audio_path = _extract_audio(local_video_path)
    print(f"[asr] transcribing audio={audio_path}", flush=True)

    segments, info = model.transcribe(
        audio_path,
        beam_size=1,
        best_of=1,
        vad_filter=True,
    )

    out: List[Dict[str, Any]] = []
    for seg in segments:
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


# your pipeline was calling this name, so keep it
def transcribe_local(path: str):
    return transcribe(path)
