# worker/asr.py
import os
from typing import List, Dict, Any, Optional

# Default = faster-whisper (ctranslate2) for speed on GPU/CPU.
# Fallback to openai-whisper if import fails.
def transcribe_segments(local_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Returns a list of segments:
      [{"start": float, "end": float, "text": str}, ...]
    """
    try:
        # Try faster-whisper
        from faster_whisper import WhisperModel

        model_size = os.getenv("WHISPER_MODEL", "base")
        compute_type = os.getenv("WHISPER_COMPUTE", "auto")
        device = os.getenv("WHISPER_DEVICE", "auto")  # "cuda" / "cpu" / "auto"

        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(local_path, beam_size=1, vad_filter=True)

        out: List[Dict[str, Any]] = []
        for seg in segments:
            out.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": (seg.text or "").strip()
            })
        return out if out else None

    except Exception:
        # Fallback to openai-whisper (PyTorch)
        try:
            import torch
            import whisper as openai_whisper

            model_size = os.getenv("WHISPER_MODEL", "base")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = openai_whisper.load_model(model_size, device=device)
            result = model.transcribe(local_path, verbose=False)
            out: List[Dict[str, Any]] = []
            for seg in result.get("segments", []):
                out.append({
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": (seg.get("text") or "").strip()
                })
            return out if out else None
        except Exception:
            return None
