# worker/asr.py
import os
import subprocess
from typing import List, Dict, Any

from .utils import ffprobe_duration, ensure_float, MAX_TAKE_SEC

# Try optional Whisper (openai-whisper). If unavailable, we fallback.
try:
    import whisper  # pip install openai-whisper
    HAVE_WHISPER = True
except Exception:
    HAVE_WHISPER = False

TMP_WAV = "/tmp/editdna_audio.wav"

def _extract_audio(media_path: str) -> str:
    """
    Extract mono 16k wav for ASR. Requires ffmpeg in the container.
    """
    try:
        cmd = [
            "ffmpeg", "-nostdin", "-y",
            "-i", media_path,
            "-ac", "1", "-ar", "16000",
            TMP_WAV
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return TMP_WAV
    except Exception:
        return media_path  # best effort: return original, Whisper can read many formats

def _whisper_transcribe(media_path: str) -> List[Dict[str, Any]]:
    """
    Use Whisper (local) if installed. Returns list of segments:
      {id,start,end,text,slot?}
    """
    model_name = os.getenv("WHISPER_MODEL", "base")
    model = whisper.load_model(model_name)
    audio_in = _extract_audio(media_path)
    res = model.transcribe(audio_in, fp16=False)
    segs = []
    for i, s in enumerate(res.get("segments", []) or []):
        start = float(s.get("start", 0.0))
        end = float(s.get("end", start))
        text = (s.get("text") or "").strip()
        if end > start:
            segs.append({
                "id": f"ASR{i:04d}",
                "start": start,
                "end": end,
                "text": text,
                "slot": "STORY"
            })
    return segs

def transcribe(media_path: str) -> List[Dict[str, Any]]:
    """
    Primary ASR entry point expected by pipeline.py.
    If Whisper is not installed, fallback to a single full-length segment (no text).
    """
    if HAVE_WHISPER:
        try:
            segs = _whisper_transcribe(media_path)
            if segs:
                return segs
        except Exception as e:
            # Fall through to dummy if Whisper fails
            print(f"[ASR] Whisper error -> fallback: {e}")

    # Fallback: one segment spanning the full video duration (text empty)
    dur = float(ffprobe_duration(media_path))
    dur = max(0.0, dur)
    # keep a reasonable span if an extreme input arrives (assembly handles length)
    end_t = min(dur, float(MAX_TAKE_SEC) if MAX_TAKE_SEC else dur)
    return [{
        "id": "ASR0000",
        "start": 0.0,
        "end": end_t if end_t > 0 else 10.0,  # last resort min
        "text": "",
        "slot": "STORY"
    }]
