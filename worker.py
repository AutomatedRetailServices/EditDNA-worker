# worker.py — RQ worker tasks
import os, time, json
from typing import List, Dict, Any

import requests
import boto3
from botocore.config import Config

# Optional OpenAI import (we handle absence gracefully)
try:
    from openai import OpenAI
except Exception:  # nosec - best-effort import
    OpenAI = None  # type: ignore

# ----- Environment -----
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# S3 (only used if you later extend)
_s3 = boto3.client("s3", region_name=AWS_REGION, config=Config(retries={"max_attempts": 3}))

# -----------------------
# Utilities
# -----------------------
def _head_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        ok = r.status_code == 200
        size = int(r.headers.get("content-length", "0") or "0")
        return {"url": url, "status": "OK" if ok else "BAD", "http": r.status_code, "size": size, "method": "HEAD"}
    except Exception as e:
        return {"url": url, "status": "ERROR", "http": 0, "size": 0, "error": str(e), "method": "HEAD"}

# -----------------------
# Tasks
# -----------------------
def task_nop():
    return {"echo": {"hello": "world"}}

def check_urls(payload: Dict[str, Any]):
    """
    payload: { session_id, urls: [ ... ] }
    Returns: { session_id, checked: [ {url, http, size, status}, ... ] }
    """
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    urls: List[str] = payload.get("urls", [])
    checked = [_head_url(u) for u in urls]
    return {"session_id": session_id, "checked": checked}

def _analyze_stub(session_id: str, tone: str | None, product_link: str | None, features: List[str]) -> Dict[str, Any]:
    script = f"[DEV STUB] {tone or 'casual'} promo highlighting " + ", ".join(features or []) + "."
    if product_link:
        script += f" Product: {product_link}."
    return {"session_id": session_id, "engine": "stub", "script": script, "product_link": product_link, "features": features}

def _openai_try_make_script(tone: str | None, product_link: str | None, features: List[str]) -> str:
    """Tiny OpenAI call; returns text or raises."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"Write a short {tone or 'casual'} promo mentioning: {', '.join(features or [])}. Link: {product_link or 'N/A'}."
    # Try Responses API first
    try:
        r = client.responses.create(model="gpt-4o-mini", input=prompt, max_output_tokens=120)
        return r.output_text.strip()
    except Exception:
        # Fallback to Chat Completions
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
        )
        return r.choices[0].message.content.strip()

def analyze_session(payload: Dict[str, Any]):
    """
    Uses OpenAI if key present & reachable; otherwise returns stub.
    """
    session_id = payload["session_id"]
    tone = payload.get("tone")
    product_link = payload.get("product_link")
    features_csv = payload.get("features_csv", "")
    features = [s.strip() for s in features_csv.split(",") if s.strip()] if isinstance(features_csv, str) else list(features_csv or [])

    # If no key → stub
    if not OPENAI_API_KEY or OpenAI is None:
        return _analyze_stub(session_id, tone, product_link, features)

    # Try OpenAI; on any network error, fall back to stub but include diag
    try:
        script = _openai_try_make_script(tone, product_link, features)
        return {
            "session_id": session_id,
            "engine": "openai",
            "script": script,
            "product_link": product_link,
            "features": features,
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "engine": "stub",
            "openai_diag": {"last_error": str(e)},
            **_analyze_stub(session_id, tone, product_link, features),
        }

# -------- Diagnostics --------
def diag_openai():
    """
    Verify key, import, client init, and a tiny call (with fallback).
    """
    has_key = bool(OPENAI_API_KEY)
    import_ok = OpenAI is not None
    client_ok = False
    last_error = None
    reply = None
    attempts = []

    if import_ok and has_key:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            client_ok = True
            # Try chat first, then responses
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Say OK"}],
                    max_tokens=4,
                )
                reply = r.choices[0].message.content
                attempts.append({"api": "chat.completions", "ok": True})
            except Exception as e1:
                attempts.append({"api": "chat.completions", "ok": False, "error": str(e1)})
                try:
                    r = client.responses.create(model="gpt-4o-mini", input="Say OK", max_output_tokens=4)
                    reply = r.output_text
                    attempts.append({"api": "responses", "ok": True})
                except Exception as e2:
                    attempts.append({"api": "responses", "ok": False, "error": str(e2)})
                    last_error = str(e2)
        except Exception as e:
            last_error = f"client_init: {e!s}"

    return {
        "has_key": has_key,
        "import_ok": import_ok,
        "client_ok": client_ok,
        "attempts": attempts,
        "last_error": last_error,
        "reply": reply,
    }

def net_probe():
    """
    Raw HTTPS probe to api.openai.com to reveal DNS/TLS/egress problems.
    """
    url = "https://api.openai.com/v1/models"
    headers = {}
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        return {
            "ok": True,
            "url": url,
            "status": r.status_code,
            "reason": r.reason,
            "body_prefix": r.text[:200],
        }
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e)}
