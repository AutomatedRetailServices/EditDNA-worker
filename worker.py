# worker.py — RQ tasks (OpenAI-powered analyze) + diagnostics
import os, time, requests
from typing import Dict, Any

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_diag = {"has_key": bool(OPENAI_API_KEY), "import_ok": False, "client_ok": False, "last_error": None}

_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_diag["import_ok"] = True
        try:
            _client = OpenAI(api_key=OPENAI_API_KEY)
            openai_diag["client_ok"] = True
        except Exception as e:
            openai_diag["last_error"] = f"client_init: {e!s}"[:200]
    except Exception as e:
        openai_diag["last_error"] = f"import: {e!s}"[:200]


def task_nop() -> Dict[str, Any]:
    return {"echo": {"hello": "world"}}


def _head_or_get(url: str, timeout: float = 20.0):
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        size = int(r.headers.get("Content-Length") or 0)
        if r.status_code in (200, 206) and size > 0:
            return {"status": "OK", "http": r.status_code, "size": size, "method": "HEAD"}
        if r.status_code in (403, 405) or size == 0:
            g = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=timeout, allow_redirects=True)
            g_size = int(g.headers.get("Content-Range", "bytes 0-0/0").split("/")[-1] or 0)
            if g_size == 0:
                g_size = int(g.headers.get("Content-Length") or 0)
            if g.status_code in (200, 206) and g_size >= 0:
                return {"status": "OK", "http": g.status_code, "size": g_size, "method": "GET(range)"}
            return {"status": "ERROR", "http": g.status_code, "size": 0, "method": "GET(range)"}
        return {"status": "OK", "http": r.status_code, "size": size, "method": "HEAD"}
    except Exception as e:
        return {"status": "ERROR", "http": 0, "size": 0, "error": str(e), "method": "HEAD"}


def check_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    out = []
    for url in payload.get("urls", []):
        result = _head_or_get(url, timeout=20.0)
        result["url"] = url
        out.append(result)
    return {"session_id": session_id, "checked": out}


def _generate_script_with_openai(session_id: str, tone: str, product_link: str, features: list) -> str:
    assert _client is not None, "OpenAI client not initialized"
    features_txt = ", ".join(features) if features else "key benefits"

    system_msg = (
        "You are a creative ad copywriter. Write tight, engaging short-form promo scripts for product videos."
    )
    user_msg = (
        f"Create a ~90–130 word voiceover script.\n"
        f"Tone: {tone or 'neutral'}\n"
        f"Product page: {product_link or 'N/A'}\n"
        f"Features: {features_txt}\n"
        "- Hook first sentence; 2–4 concrete benefits; natural prose; end with CTA.\n"
        f"[session:{session_id}]"
    )

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.8,
        max_tokens=300,
    )
    return (resp.choices[0].message.content or "").strip()


def analyze_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    sess = payload["session_id"]
    tone = (payload.get("tone") or "").strip()
    product_link = (payload.get("product_link") or "").strip()
    features_csv = payload.get("features_csv", "")
    features = [s.strip() for s in features_csv.split(",") if s.strip()]

    engine = "stub"
    script_text = f"[DEV STUB] {tone or 'neutral'} promo highlighting {', '.join(features) or 'your key benefits'}. Product: {product_link or 'N/A'}."

    if _client is not None:
        try:
            script_text = _generate_script_with_openai(sess, tone, product_link, features)
            engine = "openai"
        except Exception as e:
            openai_diag["last_error"] = f"chat_call: {e!s}"[:200]
            # keep stub

    # tiny pause
    time.sleep(0.3)

    return {
        "session_id": sess,
        "engine": engine,
        "openai_diag": openai_diag,  # <- shows exactly why stub was used
        "script": script_text,
        "product_link": product_link or None,
        "features": features,
    }
