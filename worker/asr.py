from typing import List, Dict, Any, Optional
import os
from faster_whisper import WhisperModel

# Light wrapper that returns segments with {start,end,text}
# Use local faster-whisper; no network dependency.
_MODEL_NAME = os.getenv("ASR_MODEL", "base.en")
_DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
_COMPUTE = "float16" if _DEVICE == "cuda" else "int8"

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel(_MODEL_NAME, device=_DEVICE, compute_type=_COMPUTE)
    return _model

def transcribe_segments(local_video_path: str) -> Optional[List[Dict[str, Any]]]:
    try:
        model = _get_model()
        segments, _ = model.transcribe(local_video_path, vad_filter=True)
        out: List[Dict[str, Any]] = []
        for seg in segments:
            txt = (seg.text or "").strip()
            if not txt:
                continue
            out.append({"start": float(seg.start), "end": float(seg.end), "text": txt})
        return out or None
    except Exception:
        return None
