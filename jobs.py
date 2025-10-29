# ===== ASR ROBUST LOADER (OpenAI-Whisper → fallback to Faster-Whisper) =====
import os, json, math
from typing import List, Dict, Any

ASR_MODEL  = os.getenv("ASR_MODEL", "small").strip()           # valid: tiny/base/small/medium/large (+ .en variants)
ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda").strip()           # "cuda" or "cpu"
ASR_ENABLED = os.getenv("ASR_ENABLED", "1").strip() in ("1","true","True")
ASR_FP16 = os.getenv("ASR_FP16", "1").strip() in ("1","true","True")

# Optional: centralize caches to avoid partial/corrupt leftovers
os.environ.setdefault("WHISPER_CACHE_DIR", "/root/.cache/whisper")
os.makedirs(os.environ["WHISPER_CACHE_DIR"], exist_ok=True)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/.cache/huggingface")
os.makedirs(os.environ["HUGGINGFACE_HUB_CACHE"], exist_ok=True)

def _asr_openai_whisper(local_path: str) -> List[Dict[str, Any]]:
    """
    Try OpenAI-Whisper (pip: openai-whisper). Returns list of segments:
    [{"start": float, "end": float, "text": "..."}, ...]
    """
    import whisper
    model_name = ASR_MODEL or "small"
    # Load model
    model = whisper.load_model(model_name, device=ASR_DEVICE)
    # fp16 only on cuda
    use_fp16 = ASR_FP16 and (ASR_DEVICE == "cuda")
    result = model.transcribe(local_path, fp16=use_fp16, language=os.getenv("ASR_LANGUAGE", None))
    segments = []
    for seg in result.get("segments", []) or []:
        segments.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": (seg.get("text") or "").strip()
        })
    return segments

def _asr_faster_whisper(local_path: str) -> List[Dict[str, Any]]:
    """
    Fallback using Faster-Whisper (pip: faster-whisper).
    Pulls models from HF Hub (different CDN) — avoids the Azure 404.
    """
    from faster_whisper import WhisperModel

    # Map OpenAI names → common Faster-Whisper repos
    # You can adjust to medium/large as you like.
    name = (ASR_MODEL or "small").lower()
    if name.startswith("tiny"):
        repo = "Systran/faster-whisper-tiny"
    elif name.startswith("base"):
        repo = "Systran/faster-whisper-base"
    elif name.startswith("small"):
        repo = "Systran/faster-whisper-small"
    elif name.startswith("medium"):
        repo = "Systran/faster-whisper-medium"
    else:
        # default to small if "large" not available in your pod
        repo = "Systran/faster-whisper-small"

    compute_type = "float16" if (ASR_DEVICE == "cuda" and ASR_FP16) else "int8"
    model = WhisperModel(
        repo,
        device=ASR_DEVICE if ASR_DEVICE in ("cuda","cpu") else "cpu",
        compute_type=compute_type,                       # very fast on GPU
        download_root=os.environ["HUGGINGFACE_HUB_CACHE"]
    )

    lang = os.getenv("ASR_LANGUAGE", None)
    segments_gen, _info = model.transcribe(
        local_path,
        beam_size=1,
        language=lang if lang not in ("", None) else None,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300}
    )
    segments = []
    for seg in segments_gen:
        segments.append({
            "start": float(seg.start or 0.0),
            "end": float(seg.end or 0.0),
            "text": (seg.text or "").strip()
        })
    return segments

def _clean_broken_openai_cache_on_404(err: Exception):
    """
    If the OpenAI CDN returns a 404 (or partial), clear the broken cache so a retry can succeed later.
    """
    import shutil
    msg = repr(err)
    if "HTTP Error 404" in msg or "URLError" in msg or "Connection timed out" in msg:
        cache_dir = os.environ.get("WHISPER_CACHE_DIR", "/root/.cache/whisper")
        try:
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)
        except Exception:
            pass  # ignore

def do_asr_or_fallback(local_path: str) -> List[Dict[str, Any]]:
    """
    SINGLE entry point your pipeline should call.
    Uses OpenAI-Whisper first; on any download error → Faster-Whisper.
    """
    if not ASR_ENABLED:
        return []  # ASR disabled
    # First try OpenAI-Whisper
    try:
        return _asr_openai_whisper(local_path)
    except Exception as e:
        # If the error is a 404/timeout/etc, clean cache and fall back
        _clean_broken_openai_cache_on_404(e)
        try:
            return _asr_faster_whisper(local_path)
        except Exception:
            # re-raise the original so the error message is meaningful to you
            raise e

# === Your pipeline hook ===
def _do_whisper_asr(local_path: str) -> List[Dict[str, Any]]:
    """
    Keep the function name your pipeline already calls.
    """
    return do_asr_or_fallback(local_path)
# ===== END ASR ROBUST LOADER =====
