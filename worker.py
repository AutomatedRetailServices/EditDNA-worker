# worker.py — RQ worker and tasks
import os
import json
import time
import redis
import requests
from rq import Queue, Worker, Connection

# Optional OpenAI imports (diag_openai handles absence gracefully)
try:
    from openai import OpenAI
except Exception:  # keep worker alive even if package missing
    OpenAI = None

REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

conn = redis.from_url(REDIS_URL, decode_responses=False)
q = Queue("default", connection=conn)

S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE", "").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------
# Helpers
# -------------------------
def head_size(url: str) -> dict:
    try:
        r = requests.head(url, timeout=10)
        size = int(r.headers.get("Content-Length", "0")) if r.status_code == 200 else 0
        return {"url": url, "status": "OK", "http": r.status_code, "size": size, "method": "HEAD"}
    except Exception as e:
        return {"url": url, "status": "ERR", "http": 0, "size": 0, "error": str(e)}

# -------------------------
# Tasks
# -------------------------
def task_nop():
    return {"echo": {"hello": "world"}}

def check_urls(payload: dict):
    """
    payload: { session_id, urls: [...] }
    """
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    urls = payload.get("urls", [])
    checked = [head_size(u) for u in urls]
    return {"session_id": session_id, "checked": checked}

def analyze_session(payload: dict):
    """
    Dummy/stub script generator. If OPENAI is available, you can later
    swap to a real model call.
    """
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    tone = payload.get("tone")
    product_link = payload.get("product_link")
    features_csv = payload.get("features_csv", "")
    features = [f.strip() for f in features_csv.split(",") if f.strip()]

    script = f"[DEV STUB] {tone or 'casual'} promo highlighting " \
             f"{', '.join(features) if features else 'key features'}. " \
             f"Product: {product_link or 'N/A'}."

    return {
        "session_id": session_id,
        "engine": "stub",
        "script": script,
        "product_link": product_link,
        "features": features,
    }

def diag_openai():
    """
    Diagnostic to verify OpenAI import, client init, and a tiny chat call.
    Returns a dict; never raises.
    """
    has_key = bool(OPENAI_API_KEY)
    import_ok = OpenAI is not None
    client_ok = False
    last_error = None
    reply = None

    if import_ok and has_key:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            client_ok = True
            # very small call
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=4,
            )
            reply = r.choices[0].message.content
        except Exception as e:
            last_error = f"chat_call: {e!s}"

    return {
        "has_key": has_key,
        "import_ok": import_ok,
        "client_ok": client_ok,
        "last_error": last_error,
        "reply": reply,
    }

# -------------------------
# Worker entry
# -------------------------
if __name__ == "__main__":
    with Connection(conn):
        worker = Worker([q])
        worker.work()
