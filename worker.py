# worker.py — RQ task definitions + wrappers that call jobs.py
import os
import time
import socket
from typing import Any, Dict, List
import requests

# Redis connection for RQ worker process
import redis
from rq import Connection
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")
# Keep bytes mode for RQ
conn = redis.from_url(REDIS_URL, decode_responses=False)

# Jobs module (ffmpeg work lives here)
import jobs

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
# 1) Check S3/public URLs (HEAD request)
# payload: { "urls": [...], "session_id": "sess-..." }
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
    urls = payload.get("urls") or []
    sess = payload.get("session_id") or f"sess-{int(time.time())}"
    checked = [_head(u) for u in urls]
    return {"session_id": sess, "checked": checked}


# =========================================================
# 2) Analyze session → make promo script (OpenAI or stub)
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
        "script": f"[DEV STUB] {tone} promo highlighting {', '.join(features) or 'features'}. "
                  f"Product: {product_link or '(no link)'}.",
        "product_link": product_link or None,
        "features": features,
    }

    if not (_openai_import_ok and _openai_client_ok and _client and OPENAI_API_KEY):
        return stub

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
        diag["attempts"][-1]["ok"] = False
        diag["attempts"][-1]["error"] = str(e)
        diag["last_error"] = str(e)
        out = dict(stub)
        out["engine"] = "stub"
        out["openai_diag"] = diag
        return out


# =========================================================
# 3) Render jobs — thin wrappers over jobs.py
# =========================================================
def job_render(session_id: str, files: List[str], output_prefix: str) -> Dict[str, Any]:
    return jobs.job_render(session_id, files, output_prefix)

def job_render_chunked(session_id: str, files: List[str], output_prefix: str) -> Dict[str, Any]:
    return jobs.job_render_chunked(session_id, files, output_prefix)


# =========================================================
# 4) Diagnostics
# =========================================================
def diag_openai() -> Dict[str, Any]:
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
    out = {"dns": None, "tls": None}
    try:
        socket.gethostbyname("api.openai.com")
        out["dns"] = "ok"
    except Exception as e:
        out["dns"] = f"fail: {e}"

    try:
        r = requests.get("https://api.openai.com/v1/models", timeout=10)
        out["tls"] = f"ok: TLSv{getattr(r.raw, 'version', '1.2')}"
    except Exception as e:
        out["tls"] = f"fail: {e}"

    return out
