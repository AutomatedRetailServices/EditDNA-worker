# worker.py — RQ task implementations used by app.py
import os, time, socket, ssl
from typing import Dict, List, Any
import requests

# RQ will import functions by dotted path, e.g. "worker.analyze_session"
# Do NOT rename these symbols; app.py enqueues them by these names.

# ---------- Tiny NOP ----------
def task_nop() -> Dict[str, Any]:
    return {"echo": {"hello": "world"}}

# ---------- S3 URL checker ----------
def _head_ok(url: str, timeout: float = 15.0) -> Dict[str, Any]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        ok = (200 <= r.status_code < 300)
        return {
            "url": url,
            "status": "OK" if ok else "ERR",
            "http": r.status_code,
            "size": int(r.headers.get("Content-Length", "0") or 0),
            "method": "HEAD",
        }
    except Exception as e:
        return {"url": url, "status": "ERR", "http": 0, "size": 0, "error": str(e), "method": "HEAD"}

def check_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    urls: List[str] = payload.get("urls", [])
    session_id = payload.get("session_id", f"sess-{int(time.time())}")
    results = [_head_ok(u) for u in urls]
    return {"session_id": session_id, "checked": results}

# ---------- OpenAI (real) ----------
# We will use chat.completions (stable) and keep a clean fallback.
def _openai_client():
    from openai import OpenAI  # import inside so diag can report import errors
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=key)

def _features_list(csv: str) -> List[str]:
    parts = [p.strip() for p in (csv or "").split(",") if p.strip()]
    return parts

def analyze_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected payload:
      session_id, tone, product_link, features_csv
    Returns a dict consumed by /jobs in app.py.
    """
    session_id = payload["session_id"]
    tone = payload.get("tone", "casual")
    product_link = payload.get("product_link", "")
    features_csv = payload.get("features_csv", "")
    features = _features_list(features_csv)

    # Try real OpenAI first; on any exception, return a DEV STUB so the API never 500s.
    try:
        client = _openai_client()
        prompt = (
            f"Write a {tone} 25–35 second TikTok promo script. "
            f"Highlight these features: {', '.join(features) or 'none provided'}. "
            f"Product URL: {product_link or 'n/a'}. "
            f"Format it as short lines suitable for voiceover, no scene directions."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise promo scriptwriter."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        script = resp.choices[0].message.content
        return {
            "session_id": session_id,
            "engine": "openai.chat.completions",
            "script": script,
            "product_link": product_link,
            "features": features,
        }
    except Exception as e:
        # Soft fallback so your flow still completes
        return {
            "session_id": session_id,
            "engine": "stub",
            "openai_diag": {"last_error": str(e)},
            "script": f"[DEV STUB] {tone} promo highlighting {', '.join(features) or 'features'}. "
                      f"Product: {product_link or 'n/a'}.",
            "product_link": product_link,
            "features": features,
        }

# ---------- Diagnostics ----------
def diag_openai() -> Dict[str, Any]:
    """
    Verifies: env var present, SDK import, client creation, and a tiny call.
    Never raises; always returns a dict so /jobs shows a clean result.
    """
    out = {
        "has_key": bool(os.getenv("OPENAI_API_KEY")),
        "import_ok": False,
        "client_ok": False,
        "attempts": [],
        "last_error": None,
        "reply": None,
    }
    try:
        from openai import OpenAI
        out["import_ok"] = True
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            out["client_ok"] = True
            # Minimal call (no fancy APIs) to avoid SDK surface mismatches
            try:
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Say OK"}],
                    temperature=0
                )
                out["attempts"].append({"api": "chat.completions", "ok": True})
                out["reply"] = r.choices[0].message.content
                out["last_error"] = None
            except Exception as e:
                out["attempts"].append({"api": "chat.completions", "ok": False, "error": str(e)})
                out["last_error"] = str(e)
        except Exception as e:
            out["last_error"] = f"client: {e}"
    except Exception as e:
        out["last_error"] = f"import: {e}"
    return out

def net_probe() -> Dict[str, Any]:
    """
    Quick outbound checks for Render networking/TLS.
    """
    host = "api.openai.com"
    result = {"dns": "unknown", "tls": "unknown"}
    try:
        socket.gethostbyname(host)
        result["dns"] = "ok"
    except Exception as e:
        result["dns"] = f"fail: {e}"
        return result

    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=8) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                result["tls"] = f"ok: TLSv{ssock.version()}"
    except Exception as e:
        result["tls"] = f"fail: {e}"
    return result
