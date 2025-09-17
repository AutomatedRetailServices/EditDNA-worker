# worker.py — URL checks + real /analyze script via OpenAI
import os, time, requests
from typing import List, Dict, Any

# ---------- Tiny test ----------
def task_nop():
    return {"echo": {"hello": "world"}}

# ---------- URL checker (HEAD with tiny GET fallback) ----------
def check_urls(payload: Dict[str, Any]):
    """
    payload = {"session_id":"...", "urls":[...]}
    Returns per-URL status, http code, size, and method used.
    """
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    out = []
    for url in payload.get("urls", []):
        try:
            method_used = "HEAD"
            r = requests.head(url, timeout=20, allow_redirects=True)
            code = r.status_code
            size = int(r.headers.get("Content-Length") or 0)

            # Fallback for servers/buckets that block HEAD
            if code >= 400:
                method_used = "GET:bytes=0-0"
                rg = requests.get(
                    url,
                    timeout=25,
                    allow_redirects=True,
                    stream=True,
                    headers={"Range": "bytes=0-0"},
                )
                code = rg.status_code
                cr = rg.headers.get("Content-Range")  # e.g., "bytes 0-0/1234567"
                if cr and "/" in cr:
                    try:
                        size = int(cr.split("/")[-1])
                    except Exception:
                        size = int(rg.headers.get("Content-Length") or 0)

            status = "OK" if (200 <= code < 300 or code == 206) else "ERROR"
            out.append(
                {"url": url, "status": status, "http": code, "size": size, "method": method_used}
            )
        except Exception as e:
            out.append({"url": url, "status": "ERROR", "http": 0, "size": 0, "error": str(e)})
    return {"session_id": session_id, "checked": out}

# ---------- Analyze: real script generation ----------
def _features_from_payload(payload: Dict[str, Any]) -> List[str]:
    if "features" in payload and isinstance(payload["features"], list):
        return [str(x).strip() for x in payload["features"] if str(x).strip()]
    csv = payload.get("features_csv", "")
    return [s.strip() for s in csv.split(",") if s.strip()]

def _openai_client_or_none():
    # Safe: only use if API key present. Otherwise, fallback to stub text.
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def _generate_script_with_openai(tone: str, product_link: str, features: List[str]) -> str:
    """
    Uses a lightweight model for speed/cost. Change model if you prefer.
    """
    client = _openai_client_or_none()
    if client is None:
        # Fallback stub if no API key is configured
        feat = ", ".join(features) if features else "key benefits"
        return f"(DEV STUB) {tone or 'neutral'} promo highlighting {feat}. Product: {product_link or 'N/A'}."

    tone_txt = tone or "neutral"
    feats_txt = ", ".join(features) if features else "key benefits"
    user_prompt = (
        "Write a concise 60–90 second promo script for a product video. "
        "Use a clear hook, 3–5 bullets of value, and a simple CTA. "
        f"Tone: {tone_txt}. "
        f"Product link: {product_link or 'N/A'}. "
        f"Key features to weave in: {feats_txt}. "
        "Output plain text only (no markdown)."
    )

    # OpenAI SDK v1.x — chat.completions
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a sharp D2C copywriter."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=450,
    )
    return resp.choices[0].message.content.strip()

def analyze_session(payload: Dict[str, Any]):
    """
    payload = {
      "session_id": "...",
      "tone": "friendly",
      "product_link": "https://example.com",
      "features_csv": "fast, compact, lightweight"  # or features: [...]
    }
    """
    sess = payload["session_id"]
    tone = payload.get("tone")
    product_link = payload.get("product_link")
    features = _features_from_payload(payload)

    script = _generate_script_with_openai(tone=tone, product_link=product_link, features=features)

    return {
        "session_id": sess,
        "script": script,
        "product_link": product_link,
        "features": features,
    }
