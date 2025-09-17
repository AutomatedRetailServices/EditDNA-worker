# worker.py
import os
import json
import time
import socket
import ssl
from typing import List, Dict, Any, Optional

import requests

# -----------------------
# Config
# -----------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE", "").rstrip("/")  # optional, not required
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# -----------------------
# Utilities
# -----------------------
def _head_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    HEAD the URL and report http status and size. Falls back to GET if HEAD not allowed.
    """
    info: Dict[str, Any] = {"url": url, "status": "OK", "http": None, "size": 0, "method": "HEAD"}
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        info["http"] = r.status_code
        # Some S3 HEADs won’t include Content-Length for certain keys; try GET-if-403/405
        if r.status_code in (403, 405) or "Content-Length" not in r.headers:
            r_get = requests.get(url, stream=True, timeout=timeout)
            info["method"] = "GET"
            info["http"] = r_get.status_code
            if r_get.ok:
                size = 0
                for chunk in r_get.iter_content(chunk_size=8192):
                    if chunk:
                        size += len(chunk)
                        if size > 5_000_000:  # don’t read full file; we just want to know it’s public
                            break
                info["size"] = size
        else:
            try:
                info["size"] = int(r.headers.get("Content-Length", "0"))
            except Exception:
                info["size"] = 0
    except requests.RequestException as e:
        info["status"] = "ERR"
        info["http"] = None
        info["error"] = str(e)
    return info


def _make_session_id() -> str:
    # Epoch-ish session id like the ones you’ve been seeing
    return f"sess-{int(time.time())}"


# -----------------------
# Tasks exposed to RQ
# -----------------------
def task_nop() -> Dict[str, Any]:
    """Simple echo used by /enqueue_nop"""
    return {"echo": {"hello": "world"}}


def process_urls(urls: List[str], session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Validates external URLs (S3) are publicly reachable.
    Returns the canonical shape you’ve been using.
    """
    sess = session_id or _make_session_id()
    checked = []
    for u in urls:
        u = u.strip()
        if not u:
            continue
        # normalize accidental trailing ? (no query)
        if u.endswith("?"):
            u = u[:-1]
        checked.append(_head_url(u))
    return {
        "session_id": sess,
        "checked": checked,
    }


# -----------------------
# OpenAI integration (+ fallback)
# -----------------------
def _openai_client_diag() -> Dict[str, Any]:
    """
    Try to create a client and do a minimal call. Returns booleans + last error.
    We *don’t* fail the job if this can’t connect; analyze() will fall back.
    """
    out: Dict[str, Any] = {
        "has_key": bool(OPENAI_API_KEY),
        "import_ok": False,
        "client_ok": False,
        "attempts": [],
        "last_error": None,
    }
    try:
        from openai import OpenAI  # type: ignore
        out["import_ok"] = True
        client = OpenAI(api_key=OPENAI_API_KEY)
        out["client_ok"] = True

        # 1) Try legacy Chat Completions (works on openai>=1.0)
        try:
            out["attempts"].append({"api": "chat.completions"})
            _ = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                timeout=10,
            )
            out["attempts"][-1]["ok"] = True
            return out  # success
        except Exception as e:
            out["attempts"][-1]["ok"] = False
            out["attempts"][-1]["error"] = str(e)
            out["last_error"] = str(e)

        # 2) Try the Responses API (in case your account prefers it)
        try:
            out["attempts"].append({"api": "responses"})
            _ = client.responses.create(
                model="gpt-4o-mini",
                input="ping",
                timeout=10,
            )
            out["attempts"][-1]["ok"] = True
            return out
        except Exception as e:
            out["attempts"][-1]["ok"] = False
            out["attempts"][-1]["error"] = str(e)
            out["last_error"] = str(e)

    except Exception as e:
        out["last_error"] = str(e)

    return out


def analyze(session_id: str,
            tone: Optional[str] = None,
            product_link: Optional[str] = None,
            features_csv: Optional[str] = None) -> Dict[str, Any]:
    """
    Builds a promo script. If OpenAI works, we use it.
    If not, we return a deterministic stub (what you’ve been seeing).
    """
    # Parse features
    features = []
    if features_csv:
        features = [f.strip() for f in features_csv.split(",") if f.strip()]

    # Try OpenAI
    diag = _openai_client_diag()
    if diag.get("has_key") and diag.get("import_ok") and diag.get("client_ok"):
        # If our last attempt in diag succeeded, just do it “for real”.
        ok_attempt = next((a for a in diag.get("attempts", []) if a.get("ok")), None)
        if ok_attempt:
            try:
                from openai import OpenAI  # type: ignore
                client = OpenAI(api_key=OPENAI_API_KEY)

                prompt = [
                    {"role": "system", "content": "You write short, persuasive product promo scripts."},
                    {"role": "user", "content":
                        f"Write a {tone or 'neutral'} 60-second promo script. "
                        f"Product: {product_link or 'N/A'}. "
                        f"Key features: {', '.join(features) if features else 'N/A'}."
                     }
                ]
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=prompt,
                    max_tokens=220,
                    temperature=0.7,
                    timeout=15,
                )
                text = resp.choices[0].message.content if resp.choices else ""
                return {
                    "session_id": session_id,
                    "engine": "openai",
                    "openai_diag": {k: v for k, v in diag.items() if k != "attempts"},
                    "script": text or "",
                    "product_link": product_link,
                    "features": features,
                }
            except Exception as e:
                # fall through to stub with the error captured
                diag["last_error"] = f"chat_call: {e}"

    # Fallback stub
    summary = ", ".join(features) if features else ""
    stub = f"[DEV STUB] {tone or 'neutral'} promo highlighting {summary or 'your key features'}. " \
           f"Product: {product_link or 'N/A'}."
    return {
        "session_id": session_id,
        "engine": "stub",
        "openai_diag": {k: v for k, v in diag.items() if k != "attempts"},
        "script": stub,
        "product_link": product_link,
        "features": features,
    }


# -----------------------
# Deep connectivity diagnostic task
# -----------------------
def diag_openai() -> Dict[str, Any]:
    """
    Low-level network diagnostic to api.openai.com
    (DNS -> TLS -> Authenticated HTTP).
    """
    resultado: Dict[str, Any] = {
        "has_key": bool(OPENAI_API_KEY),
        "dns": None,
        "tls": None,
        "http": None,
    }

    # DNS
    try:
        socket.getaddrinfo("api.openai.com", 443)
        resultado["dns"] = "ok"
    except Exception as e:
        resultado["dns"] = f"fail: {e}"
        return resultado

    # TLS
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection(("api.openai.com", 443), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname="api.openai.com") as ssock:
                resultado["tls"] = f"ok: {ssock.version()}"
    except Exception as e:
        resultado["tls"] = f"fail: {e}"
        return resultado

    # HTTP
    try:
        r = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            timeout=10,
        )
        resultado["http"] = f"{r.status_code}"
    except Exception as e:
        resultado["http"] = f"fail: {e}"

    return resultado
