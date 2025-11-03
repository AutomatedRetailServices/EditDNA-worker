# worker/asr.py
import os
from typing import List, Dict, Any

# We prefer faster-whisper; falls back to openai-whisper if not available.
ASR_ENGINE = (os.getenv("ASR_ENGINE") or "faster").lower()
ASR_MODEL  = os.getenv("ASR_MODEL") or "medium"  # tiny/base/small/medium/large-v3
ASR_LANG   = os.getenv("ASR_LANG")  or "en"

def _segments_faster_whisper(path: str) -> List[Dict[str, Any]]:
    from faster_whisper import WhisperModel
    model = WhisperModel(ASR_MODEL, device="cuda" if os.getenv("CUDA_VISIBLE_DEVICES") not in (None,"","-1") else "cpu")
    segments, _ = model.transcribe(path, language=ASR_LANG, vad_filter=True, beam_size=5)
    out = []
    for i, s in enumerate(segments, start=1):
        out.append({"id": f"S{i:04d}", "start": float(s.start), "end": float(s.end), "text": s.text.strip()})
    return out

def _segments_openai_whisper(path: str) -> List[Dict[str, Any]]:
    import whisper
    model = whisper.load_model(ASR_MODEL)
    result = model.transcribe(path, language=ASR_LANG)
    out = []
    for i, s in enumerate(result.get("segments", []), start=1):
        out.append({"id": f"S{i:04d}", "start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()})
    return out

def transcribe_segments(path: str) -> List[Dict[str, Any]]:
    if ASR_ENGINE == "openai":
        return _segments_openai_whisper(path)
    return _segments_faster_whisper(path)
