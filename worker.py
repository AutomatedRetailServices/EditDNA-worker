# worker.py — RQ task definitions (now includes the render job)
import os
import time
import socket
import json
from typing import Any, Dict, List
import requests

# -------- OpenAI (v1.x) --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
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

# -------- Render pipeline --------
# We import the ffmpeg-based renderer from jobs.py
from jobs import render_from_files


# =========================================================
# 0) Tiny test job
# =========================================================
def task_nop() -> Dict[str, Any]:
    return {"echo": {"hello": "world"}}


# =========================================================
# 1) Check S3/public URLs (HEAD request)
# payload: { "urls": [...], "session_id": "sess-..." }
# =========================================================
def _head(url: str, timeout: float = 20.0) -> Dict[str, Any]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        size = int(r.headers.get("Content-Length", "0") or 0)
        return {
            "url": url,
            "status": "OK" if r.status_code < 400 else "ERROR",
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
    urls = payload.get("urls") or []
    sess = payload.get("session_id") or f"sess-{int(time.time())}"
    checked = [_head(u) for u in urls]
    return {"session_id": sess, "checked": checked}


# =========================================================
# 2) Analyze session → make promo script
# payload: {
#   "session_id": "sess-...",
#   "tone": "casual",
#   "product_link": "https://...",
#   "features_csv": "durable, waterproof, lightweight"
# }
# Returns { session_id, engine, openai_diag?, script, product_link, features }
# =========================================================
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _make_prompt(session_id: str, tone: str, product_link: str, features: List[str]) -> str:
    feat_str = ", ".join(features) if features else "key features"
    return (
        f"You are a marketing writer. Create a short {tone or 'neutral'} promo script "
        f"for product {product_link or '(no link)'} based on these features: {feat_str}. "
        f"Keep it 3–5 sentences, engaging, and suitable for voiceover."
    )


def _parse_features(payload: Dict[str, Any]) -> List[str]:
    feats: List[str] = []
    csv = (payload.get("features_csv") or "").strip()
    if csv:
        feats = [p.strip() for p in csv.split(",") if p.strip()]
    # also accept array form
    if not feats and isinstance(payload.get("features"), list):
        feats = [str(x).strip() for x in payload["features"] if str(x).strip()]
    return feats


def analyze_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    sess = payload.get("session_id") or f"sess-{int(time.time())}"
    tone = str(payload.get("tone") or "neutral")
    product_link = str(payload.get("product_link") or "")
    features = _parse_features(payload)

    # default stub (in case OpenAI is missing/fails)
    stub = {
        "session_id": sess,
        "engine": "stub",
        "script": f"[DEV STUB] {tone} promo highlighting {', '.join(features) or 'features'}. "
                  f"Product: {product_link or '(no link)'}.",
        "product_link": product_link or None,
        "features": features,
    }

    # if OpenAI not configured, return stub
    if not (_openai_import_ok and _openai_client_ok and _client and OPENAI_API_KEY):
        return stub

    # try OpenAI
    diag = {"has_key": bool(OPENAI_API_KEY), "import_ok": _openai_import_ok,
            "client_ok": _openai_client_ok, "attempts": [], "last_error": None}

    prompt = _make_prompt(sess, tone, product_link, features)
    try:
        diag["attempts"].append({"api": "chat.completions", "ok": None, "error": None})
        resp = _client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You write concise, friendly promo scripts."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=220,
        )
        text = (resp.choices[0].message.content or "").strip()
        diag["attempts"][-1]["ok"] = True
        return {
            "session_id": sess,
            "engine": "openai",
            "openai_diag": diag,
            "script": text,
            "product_link": product_link or None,
            "features": features,
        }
    except Exception as e:
        diag["attempts"].append({"api": "chat.completions", "ok": False, "error": str(e)})
        # graceful fallback to stub
        out = dict(stub)
        out["engine"] = "stub"
        out["openai_diag"] = diag
        return out


# =========================================================
# 3) NEW — Render job (downloads → ffmpeg stitch → uploads)
# payload: {
#   "session_id": "sess-...",
#   "files": ["s3://bucket/raw/a.mov", "s3://bucket/raw/b.mov", ...],
#   "output_prefix": "editdna/outputs"   # optional, defaults in app layer
# }
# Returns:
#   { ok, session_id, inputs, output_s3 } or { ok: False, error: "..." }
# =========================================================
def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    files = payload.get("files") or []
    output_prefix = (payload.get("output_prefix") or "editdna/outputs").strip("/")

    try:
        return render_from_files(
            session_id=session_id,
            input_s3_urls=files,
            output_key_prefix=output_prefix,
        )
    except Exception as e:
        return {"ok": False, "session_id": session_id, "error": str(e)}


# =========================================================
# 4) Diagnostics
# =========================================================
def diag_openai() -> Dict[str, Any]:
    """
    Quick connectivity check to OpenAI using the v1.x SDK.
    Returns same shape you saw in Postman.
    """
    result = {
        "has_key": bool(OPENAI_API_KEY),
        "import_ok": _openai_import_ok,
        "client_ok": _openai_client_ok,
        "attempts": [],
        "last_error": None,
        "reply": None,
    }

    if not (_openai_import_ok and _openai_client_ok and _client and OPENAI_API_KEY):
        result["last_error"] = "SDK/client not initialized."
        return result

    try:
        result["attempts"].append({"api": "chat.completions", "ok": None, "error": None})
        r = _client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a health-check bot."},
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
        out["tls"] = f"ok: TLSv{getattr(r.raw, 'version', '1.2')}"
    except Exception as e:
        out["tls"] = f"fail: {e}"

    return out
