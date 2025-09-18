# worker.py — RQ task definitions for EditDNA.ai
# This file is imported by the RQ worker process. The FastAPI web app enqueues
# jobs by *string path* like "worker.job_render" so these function names MUST match.

import os
import time
import socket
from typing import Any, Dict, List
import requests

# Import your job logic from jobs.py
# - analyze_session(session_id, product_link, features_csv, tone="casual")
# - render_from_files(session_id, files, output_prefix="editdna/outputs")
from jobs import analyze_session as jobs_analyze_session, render_from_files

# -------- Optional OpenAI health (v1 SDK) --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
try:
    from openai import OpenAI
    _openai_import_ok = True
except Exception:
    OpenAI = None  # type: ignore
    _openai_import_ok = False

_client = None
if _openai_import_ok and OPENAI_API_KEY:
    try:
        _client = OpenAI(api_key=OPENAI_API_KEY)
        _openai_client_ok = True
    except Exception:
        _openai_client_ok = False
else:
    _openai_client_ok = False


# =========================================================
# 0) Tiny test job
# =========================================================
def task_nop() -> Dict[str, Any]:
    """Minimal no-op to verify queuing/worker."""
    return {"echo": {"hello": "world"}}


# =========================================================
# 1) URL HEAD checker (S3/HTTP)
# =========================================================
def _head(url: str, timeout: float = 20.0) -> Dict[str, Any]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        size = int(r.headers.get("Content-Length", "0") or 0)
        return {
            "url": url,
            "status": "OK",
            "http": r.status_code,
            "size": size,
            "method": "HEAD",
        }
    except Exception as e:
        return {
            "url": url,
            "status": "ERROR",
            "http": 0,
            "size": 0,
            "method": "HEAD",
            "error": str(e),
        }


def check_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload: { "urls": [...], "session_id": "optional" }
    """
    urls = payload.get("urls") or []
    sess = payload.get("session_id") or f"sess-{int(time.time())}"
    checked = [_head(u) for u in urls]
    return {"session_id": sess, "checked": checked}


# =========================================================
# 2) Analyze session → promo script (wrapper around jobs.analyze_session)
# =========================================================
def analyze_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload: {
      "session_id": "sess-...",
      "product_link": "https://...",
      "features_csv": "durable, waterproof, lightweight",
      "tone": "casual"  # optional
    }
    """
    sess = str(payload.get("session_id") or f"sess-{int(time.time())}")
    product_link = str(payload.get("product_link") or "")
    features_csv = str(payload.get("features_csv") or "")
    tone = str(payload.get("tone") or "casual")
    # Call the real implementation in jobs.py
    return jobs_analyze_session(sess, product_link, features_csv, tone)


# =========================================================
# 3) Diagnostics
# =========================================================
def diag_openai() -> Dict[str, Any]:
    """
    Quick connectivity check to OpenAI using the v1.x SDK.
    """
    result = {
        "has_key": bool(OPENAI_API_KEY),
        "import_ok": _openai_import_ok,
        "client_ok": _openai_client_ok,
        "attempts": [],
        "last_error": None,
        "reply": None,
        "model": OPENAI_MODEL,
    }

    if not (_openai_import_ok and _openai_client_ok and _client and OPENAI_API_KEY):
        result["last_error"] = "SDK/client not initialized."
        return result

    try:
        result["attempts"].append({"api": "chat.completions", "ok": None, "error": None})
        r = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "health-check"},
                {"role": "user", "content": "Reply with the single word: ok"},
            ],
            max_tokens=3,
        )
        msg = (r.choices[0].message.content or "").strip()
        result["reply"] = msg
        result["attempts"][-1]["ok"] = True
        return result
    except Exception as e:
        result["attempts"][-1]["ok"] = False
        result["attempts"][-1]["error"] = str(e)
        result["last_error"] = str(e)
        return result


def net_probe() -> Dict[str, Any]:
    """Very small network sanity probe."""
    out = {"dns": None, "tls": None}
    # DNS
    try:
        socket.gethostbyname("api.openai.com")
        out["dns"] = "ok"
    except Exception as e:
        out["dns"] = f"fail: {e}"

    # TLS/HTTP (no auth endpoint)
    try:
        r = requests.get("https://api.openai.com/v1/models", timeout=10)
        out["tls"] = f"ok: TLSv{getattr(r.raw, 'version', 'N/A')}"
    except Exception as e:
        out["tls"] = f"fail: {e}"

    return out


# =========================================================
# 4) Render job wrappers
# =========================================================
def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload: {
      "session_id": "sess-...",
      "files": ["https://...","s3://bucket/key", ...],
      "output_prefix": "editdna/outputs"  # optional
    }
    """
    sess = str(payload.get("session_id") or f"sess-{int(time.time())}")
    files: List[str] = payload.get("files") or []
    output_prefix = str(payload.get("output_prefix") or "editdna/outputs")
    return render_from_files(sess, files, output_prefix=output_prefix)


def job_render_chunked(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same as job_render, kept for compatibility. Our jobs.render_from_files
    already uses a chunked low-memory pipeline, so we call the same function.
    """
    sess = str(payload.get("session_id") or f"sess-{int(time.time())}")
    files: List[str] = payload.get("files") or []
    output_prefix = str(payload.get("output_prefix") or "editdna/outputs")
    return render_from_files(sess, files, output_prefix=output_prefix)
