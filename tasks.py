# tasks.py (at repo root or /workspace/EditDNA-worker/tasks.py)
import os
from typing import Dict, Any, List

# Ensure worker package is importable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE))
PKG  = os.path.abspath(os.path.join(HERE, "worker"))
for p in (ROOT, PKG):
    if p not in os.sys.path:
        os.sys.path.insert(0, p)

from worker import pipeline  # imports the ALWAYS-ON LLM pipeline

def _norm_urls(payload: Dict[str, Any]) -> List[str]:
    urls = payload.get("file_urls") or payload.get("files") or []
    if isinstance(urls, str):
        urls = [urls]
    return urls if isinstance(urls, list) else []

def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected payload:
      - session_id: str
      - files or file_urls: [str]  (we normalize to file_urls)
      - portrait: bool (optional)
      - funnel_counts: any (optional)
      - max_duration: float|None (optional)  # None = no cap
      - s3_prefix: str (optional)
    """
    session_id   = payload.get("session_id") or "session-unknown"
    file_urls    = _norm_urls(payload)
    portrait     = bool(payload.get("portrait", False))
    funnel_counts= payload.get("funnel_counts")
    # IMPORTANT: allow None (no cap) — if client sends 0 or "", treat as None
    raw_max_dur  = payload.get("max_duration", None)
    max_duration = None
    try:
        if raw_max_dur not in (None, "", 0, "0"):
            max_duration = float(raw_max_dur)
    except Exception:
        max_duration = None
    s3_prefix    = payload.get("s3_prefix")

    out = pipeline.run_pipeline(
        session_id=session_id,
        file_urls=file_urls,
        portrait=portrait,
        funnel_counts=funnel_counts,
        max_duration=max_duration,
        s3_prefix=s3_prefix,
    )
    return out
