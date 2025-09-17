# worker.py — RQ tasks (OpenAI-powered analyze)
import os, time, requests
from typing import Dict, Any

# --- Optional: OpenAI (real script generation) ---
# If OPENAI_API_KEY is not set or the API errors, we fall back to a stub.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        _client = None


def task_nop() -> Dict[str, Any]:
    """Tiny test job."""
    return {"echo": {"hello": "world"}}


def _head_or_get(url: str, timeout: float = 20.0):
    """
    Try HEAD first. If we get a 403/405 with no size, try a small GET (range)
    to confirm public readability.
    """
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        size = int(r.headers.get("Content-Length") or 0)
        if r.status_code in (200, 206) and size > 0:
            return {"status": "OK", "http": r.status_code, "size": size, "method": "HEAD"}
        # Some S3 setups block HEAD; try a ranged GET for a quick check
        if r.status_code in (403, 405) or size == 0:
            g = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=timeout, allow_redirects=True)
            g_size = int(g.headers.get("Content-Range", "bytes 0-0/0").split("/")[-1] or 0)
            # If Content-Range not present, fall back to Content-Length
            if g_size == 0:
                g_size = int(g.headers.get("Content-Length") or 0)
            if g.status_code in (200, 206) and g_size >= 0:
                return {"status": "OK", "http": g.status_code, "size": g_size, "method": "GET(range)"}
            return {"status": "ERROR", "http": g.status_code, "size": 0, "method": "GET(range)"}
        return {"status": "OK", "http": r.status_code, "size": size, "method": "HEAD"}
    except Exception as e:
        return {"status": "ERROR", "http": 0, "size": 0, "error": str(e), "method": "HEAD"}


def check_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload = {"session_id": "sess-001", "urls": ["https://.../IMG_4856.mov", ...]}
    Returns per-URL HTTP status/size so you can see they’re reachable.
    """
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    out = []
    for url in payload.get("urls", []):
        result = _head_or_get(url, timeout=20.0)
        result["url"] = url
        out.append(result)
    return {"session_id": session_id, "checked": out}


def _generate_script_with_openai(session_id: str, tone: str, product_link: str, features: list) -> str:
    """
    Use OpenAI Chat Completions to produce a short promo script.
    """
    assert _client is not None, "OpenAI client not initialized"

    # Build a compact feature list for the prompt
    features_txt = ", ".join(features) if features else "key benefits"

    system_msg = (
        "You are a creative ad copywriter. Write tight, engaging short-form promo scripts "
        "for product videos. Keep it conversational, vivid, and concrete. Avoid fluff."
    )
    user_msg = (
        f"Create a ~90–130 word voiceover script for a short product promo.\n"
        f"Tone: {tone or 'neutral'}\n"
        f"Product page: {product_link or 'N/A'}\n"
        f"Features/benefits to highlight: {features_txt}\n\n"
        "Requirements:\n"
        "- Hook in the first sentence.\n"
        "- Mention 2–4 concrete benefits.\n"
        "- Natural language (no numbered lists).\n"
        "- End with a simple call-to-action.\n"
        "- Do NOT include hashtags or emojis.\n"
        f"- Add a subtle internal tag at the end: [session:{session_id}]\n"
    )

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.8,
        max_tokens=300,
    )
    text = (resp.choices[0].message.content or "").strip()
    return text


def analyze_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload = {
      "session_id": "...",
      "tone": "casual",
      "product_link": "https://example.com",
      "features_csv": "durable, waterproof, lightweight"
    }
    Returns a generated script + echoes structured inputs.
    """
    sess = payload["session_id"]
    tone = (payload.get("tone") or "").strip()
    product_link = (payload.get("product_link") or "").strip()

    # Normalize features
    features_csv = payload.get("features_csv", "")
    features = [s.strip() for s in features_csv.split(",") if s.strip()]

    # Default stub (if no API or an error occurs)
    script_text = f"[DEV STUB] {tone or 'neutral'} promo highlighting {', '.join(features) or 'your key benefits'}. Product: {product_link or 'N/A'}."

    # Try real OpenAI generation if possible
    if _client is not None:
        try:
            script_text = _generate_script_with_openai(sess, tone, product_link, features)
        except Exception as e:
            # Keep stub but include error note to help debugging
            script_text = script_text + f"  (openai_error='{str(e)[:160]}')"

    # Small delay to simulate longer processing if needed
    time.sleep(0.5)

    return {
        "session_id": sess,
        "script": script_text,
        "product_link": product_link or None,
        "features": features,
    }
