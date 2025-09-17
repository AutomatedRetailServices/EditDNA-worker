# worker.py — RQ tasks used by editdna
import os, time, json
import requests

# RQ needs this file to be importable by name "worker"
# Functions defined here are referenced from app.py as strings like "worker.check_urls"

# -------------------------
# Helpers
# -------------------------

def _get_env(name: str, default=None):
    v = os.getenv(name)
    return v if (v is not None and v != "") else default

def _head(url: str, timeout=10):
    """HEAD a URL and return dict with http code and size (if provided)."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        size = 0
        # Some S3 objects only send size on GET, not HEAD; we keep 0 if missing
        if "Content-Length" in r.headers:
            try:
                size = int(r.headers["Content-Length"])
            except Exception:
                size = 0
        return {"url": url, "status": "OK", "http": r.status_code, "size": size, "method": "HEAD"}
    except requests.RequestException as e:
        return {"url": url, "status": "ERR", "http": 0, "size": 0, "error": str(e), "method": "HEAD"}

# -------------------------
# 0) Tiny test job
# -------------------------
def task_nop():
    # simple echo payload so you can confirm worker runs
    return {"echo": {"hello": "world"}}

# -------------------------
# 1) Check S3 URLs
# payload: { "session_id": "...", "urls": [ ... ] }
# returns: { "session_id": "...", "checked": [ {url,status,http,size}, ... ] }
# -------------------------
def check_urls(payload: dict):
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    urls = payload.get("urls") or []
    checked = [_head(u) for u in urls]
    return {"session_id": session_id, "checked": checked}

# -------------------------
# 2) Analyze: build a promo script (uses OpenAI if available, else a stub)
# payload:
# {
#   "session_id": "sess-...",
#   "tone": "casual",
#   "product_link": "https://example.com/p/123",
#   "features_csv": "durable, waterproof, lightweight"
# }
# -------------------------

def _openai_client():
    """
    Return an OpenAI client or None if missing/invalid.
    We ONLY use chat.completions (supported by your sdk).
    """
    api_key = _get_env("OPENAI_API_KEY")
    if not api_key:
        return None, "no_key"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # quick, cheap capability check (no network call yet)
        _ = client  # avoid lint warning
        return client, None
    except Exception as e:
        return None, f"import_error: {e}"

def _gpt_script(client, tone: str, product_link: str, features: list, session_id: str):
    # Build prompt
    feats = ", ".join([f.strip() for f in features if f.strip()])
    system = "You write short, punchy product promo scripts for social video."
    user = (
        f"Session: {session_id}\n"
        f"Tone: {tone or 'neutral'}\n"
        f"Product: {product_link or 'N/A'}\n"
        f"Key features: {feats or 'N/A'}\n\n"
        "Write a 60-90 second promo script. Keep it conversational, energetic, and specific."
    )

    # Call chat.completions (the correct path for your installed SDK)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()

def analyze_session(payload: dict):
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    tone = payload.get("tone") or None
    product_link = payload.get("product_link") or None
    features_csv = payload.get("features_csv") or ""
    features = [s.strip() for s in features_csv.split(",")] if features_csv else []

    # Try OpenAI
    client, diag_err = _openai_client()
    engine_used = "stub"
    script_text = None
    openai_diag = {
        "has_key": bool(_get_env("OPENAI_API_KEY")),
        "import_ok": diag_err is None,
        "client_ok": diag_err is None,
        "last_error": None,
    }

    if client is not None:
        try:
            script_text = _gpt_script(client, tone, product_link, features, session_id)
            engine_used = "openai.chat.completions"
        except Exception as e:
            # Fall back to stub but report the error
            openai_diag["last_error"] = f"chat_call: {e}"

    if not script_text:
        # Dev fallback so your flow still completes
        feats = ", ".join([f for f in features if f])
        script_text = f"[DEV STUB] {tone or 'neutral'} promo highlighting {feats or 'your key features'}. Product: {product_link or 'n/a'}."

    return {
        "session_id": session_id,
        "engine": engine_used,
        "openai_diag": openai_diag,
        "script": script_text,
        "product_link": product_link,
        "features": features,
    }

# -------------------------
# 3) OpenAI diagnostic task (used by POST /diag/openai)
# Returns whether the SDK is installed and a tiny real call to chat.completions
# -------------------------
def diag_openai(_payload: dict | None = None):
    client, diag_err = _openai_client()
    out = {
        "has_key": bool(_get_env("OPENAI_API_KEY")),
        "import_ok": diag_err is None,
        "client_ok": diag_err is None,
        "attempts": [],
        "last_error": None,
        "reply": None,
    }

    if client is None:
        out["last_error"] = diag_err
        return out

    # Single, correct attempt (no .responses)
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=4,
        )
        txt = r.choices[0].message.content.strip()
        out["attempts"].append({"api": "chat.completions", "ok": True, "error": None})
        out["reply"] = txt
    except Exception as e:
        out["attempts"].append({"api": "chat.completions", "ok": False, "error": str(e)})
        out["last_error"] = str(e)

    return out
