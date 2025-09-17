# worker.py — RQ task functions (OpenAI with retries)

import os
import time
import json
from typing import Dict, Any, List

import requests

# ------------- tiny util -------------
HTTP_TIMEOUT = 20  # seconds


def _safe_head(url: str) -> Dict[str, Any]:
    """HEAD first; if not allowed, try GET with stream to avoid download."""
    try:
        r = requests.head(url, timeout=HTTP_TIMEOUT, allow_redirects=True)
        size = int(r.headers.get("Content-Length") or 0)
        return {
            "url": url,
            "status": "OK",
            "http": r.status_code,
            "size": size,
            "method": "HEAD",
        }
    except Exception:
        # Fallback GET (some servers block HEAD)
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT, stream=True, allow_redirects=True)
            size = int(r.headers.get("Content-Length") or 0)
            return {
                "url": url,
                "status": "OK",
                "http": r.status_code,
                "size": size,
                "method": "GET",
            }
        except Exception as e:
            return {
                "url": url,
                "status": f"ERROR: {type(e).__name__}",
                "http": 0,
                "size": 0,
                "method": "HEAD",
            }


# ------------- tasks -------------

def task_nop() -> Dict[str, Any]:
    return {"echo": {"hello": "world"}}


def check_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload = { "session_id": "...", "urls": ["https://..."] }
    """
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    urls: List[str] = payload.get("urls") or []

    checked: List[Dict[str, Any]] = []
    for raw in urls:
        # trim trailing '?' if present
        url = raw.rstrip("?")
        info = _safe_head(url)

        # normalize some common S3 outcomes
        if info["http"] == 403 and info["size"] == 0:
            # public object not accessible; helpful hint in status
            info["status"] = "OK"  # keep API stable
            info["note"] = "403 from S3 (object likely not public)"
        checked.append(info)

    return {"session_id": session_id, "checked": checked}


# ---------- OpenAI helpers ----------

def _make_openai_client(diag: Dict[str, Any]):
    """
    Returns (client | None). Fills diag with has_key/import_ok/client_ok and errors.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    diag["has_key"] = bool(api_key)

    try:
        from openai import OpenAI  # type: ignore
        diag["import_ok"] = True
    except Exception as e:
        diag["import_ok"] = False
        diag["last_error"] = f"import: {e}"
        return None

    try:
        client = OpenAI(api_key=api_key)
        diag["client_ok"] = True
        return client
    except Exception as e:
        diag["client_ok"] = False
        diag["last_error"] = f"client: {e}"
        return None


def _chat_with_retries(client, prompt: str, diag: Dict[str, Any], attempts: int = 3) -> str:
    """
    Calls chat.completions with retries. Raises last Exception if all fail.
    Records each attempt in diag["attempts"].
    """
    diag.setdefault("attempts", [])
    last_err = None
    for i in range(attempts):
        try:
            diag["attempts"].append({"api": "chat.completions", "try": i + 1})
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a concise promo script generator."},
                    {"role": "user", "content": prompt},
                ],
                timeout=HTTP_TIMEOUT,
            )
            text = completion.choices[0].message.content
            diag["attempts"][-1]["ok"] = True
            return text
        except Exception as e:
            last_err = e
            diag["attempts"][-1]["ok"] = False
            diag["attempts"][-1]["error"] = str(e)
            # simple backoff: 1s, 2s, 4s
            time.sleep(2 ** i)

    diag["last_error"] = str(last_err) if last_err else "unknown"
    raise last_err if last_err else RuntimeError("OpenAI call failed")


def analyze_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload:
    {
      "session_id": "...",
      "tone": "casual",
      "product_link": "https://example.com",
      "features_csv": "durable, waterproof, lightweight"
    }
    """
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    tone = (payload.get("tone") or "").strip() or "casual"
    product_link = (payload.get("product_link") or "").strip() or None
    features_csv = (payload.get("features_csv") or "").strip()
    features = [f.strip() for f in features_csv.split(",") if f.strip()] if features_csv else []

    # Build the prompt
    prompt_lines = [
        f"Write a short product promo script.",
        f"Tone: {tone}.",
    ]
    if product_link:
        prompt_lines.append(f"Product page: {product_link}.")
    if features:
        prompt_lines.append(f"Key features: {', '.join(features)}.")
    prompt_lines.append("Return just the script text.")
    prompt = " ".join(prompt_lines)

    diag: Dict[str, Any] = {}
    engine = "stub"
    script_text = None

    client = _make_openai_client(diag)
    if client and diag.get("client_ok"):
        try:
            script_text = _chat_with_retries(client, prompt, diag)
            engine = "openai"
        except Exception:
            # fall back below
            pass

    if not script_text:
        # fallback stub so your pipeline keeps working
        feat = ", ".join(features) if features else "no specific features"
        script_text = f"[DEV STUB] {tone} promo highlighting {feat}."
        if product_link:
            script_text += f" Product: {product_link}."

    return {
        "session_id": session_id,
        "engine": engine,
        "openai_diag": diag if engine != "openai" else {"ok": True},
        "script": script_text,
        "product_link": product_link,
        "features": features,
    }


def diag_openai(_: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Lightweight diagnostic job to verify OpenAI connectivity from the worker.
    Returns booleans + per-attempt info.
    """
    diag: Dict[str, Any] = {}
    client = _make_openai_client(diag)

    if not client or not diag.get("client_ok"):
        return {**diag, "reply": None}

    reply = None
    try:
        reply = _chat_with_retries(client, "Reply with the single word: pong", diag)
    except Exception:
        pass

    return {
        "has_key": bool(diag.get("has_key")),
        "import_ok": bool(diag.get("import_ok")),
        "client_ok": bool(diag.get("client_ok")),
        "attempts": diag.get("attempts", []),
        "last_error": diag.get("last_error"),
        "reply": reply,
    }


# Note:
# - This file only defines functions for RQ to import.
# - On Render, start the worker with:
#     rq worker -u $REDIS_URL default
# - Do NOT run `python worker.py` as the start command.
