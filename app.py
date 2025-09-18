# worker.py — RQ task definitions
import os
import time
import socket
from typing import Any, Dict, List
import requests

from jobs import render_from_files, render_chunked

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


# =========================================================
# 0) Tiny test job
# =========================================================
def task_nop() -> Dict[str, Any]:
    return {"echo": {"hello": "world"}}


# =========================================================
# 1) Check S3/public URLs (HEAD)
# =========================================================
def _head(url: str, timeout: float = 20.0) -> Dict[str, Any]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        size = int(r.headers.get("Content-Length", "0") or 0)
        return {"url": url, "status": "OK", "http": r.status_code, "size": size, "method": "HEAD"}
    except Exception as e:
        return {"url": url, "status": "ERROR", "http": 0, "size": 0, "method": "HEAD", "error": str(e)}


def check_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    urls = payload.get("urls") or []
    sess = payload.get("session_id") or f"sess-{int(time.time())}"
    checked = [_head(u) for u in urls]
    return {"session_id": sess, "checked": checked}


# =========================================================
# 2) Analyze session → promo script (OpenAI with stub fallback)
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
    if not feats and isinstance(payload.get("features"), list):
        feats = [str(x).strip() for x in payload["features"] if str(x).strip()]
    return feats

def analyze_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    sess = payload.get("session_id") or f"sess-{int(time.time())}"
    tone = str(payload.get("tone") or "neutral")
    product_link = str(payload.get("product_link") or "")
    features = _parse_features(payload)

    stub = {
        "session_id": sess,
        "engine": "stub",
        "script": f"[DEV STUB] {tone} promo highlighting {', '.join(features) or 'features'}. Product: {product_link or '(no link)'}."
    }

    if not (_openai_import_ok and _openai_client_ok and _client and OPENAI_API_KEY):
        return stub

    try:
        resp = _client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You write concise, friendly promo scripts."},
                {"role": "user", "content": _make_prompt(sess, tone, product_link, features)},
            ],
            temperature=0.7,
            max_tokens=220,
        )
        text = (resp.choices[0].message.content or "").strip()
        return {"session_id": sess, "engine": "openai", "script": text}
    except Exception:
        return stub


# =========================================================
# 3) Diagnostics
# =========================================================
def diag_openai() -> Dict[str, Any]:
    if not (_openai_import_ok and _openai_client_ok and _client and OPENAI_API_KEY):
        return {"has_key": bool(OPENAI_API_KEY), "ok": False, "error": "SDK/client not initialized."}
    try:
        r = _client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[{"role": "system", "content": "health-check"}, {"role": "user", "content": "ok"}],
            max_tokens=3,
        )
        return {"ok": True, "reply": (r.choices[0].message.content or "").strip()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def net_probe() -> Dict[str, Any]:
    out = {"dns": None, "tls": None}
    try:
        socket.gethostbyname("api.openai.com"); out["dns"] = "ok"
    except Exception as e:
        out["dns"] = f"fail: {e}"
    try:
        r = requests.get("https://api.openai.com/v1/models", timeout=10)
        out["tls"] = f"ok: TLSv{getattr(r.raw,'version', 'N/A')}"
    except Exception as e:
        out["tls"] = f"fail: {e}"
    return out


# =========================================================
# 4) Render job wrappers
# =========================================================
def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    return render_from_files(
        session_id=payload["session_id"],
        input_s3_urls=payload["files"],
        output_key_prefix=payload.get("output_prefix") or "editdna/outputs",
    )

def job_render_chunked(payload: Dict[str, Any]) -> Dict[str, Any]:
    return render_chunked(
        session_id=payload["session_id"],
        input_s3_urls=payload["files"],
        output_key_prefix=payload.get("output_prefix") or "editdna/outputs",
    )
