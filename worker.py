# worker.py — tasks for EditDNA.ai
import time, requests

# -------------------------
# 0) Tiny test job
# -------------------------
def task_nop():
    # tiny test job to prove the pipeline works
    return {"echo": {"hello": "world"}}

# -------------------------
# 1) Check URLs (HEAD request)
# -------------------------
def check_urls(payload: dict):
    """
    payload = {
      "session_id": "sess-123",
      "urls": ["https://example.com/file1.mov", ...]
    }
    Returns per-URL status and size.
    """
    session_id = payload.get("session_id", f"sess-{int(time.time())}")
    out = []

    for url in payload.get("urls", []):
        try:
            r = requests.head(url, timeout=20, allow_redirects=True)
            size = int(r.headers.get("Content-Length") or 0)
            out.append({
                "url": url,
                "status": "OK",
                "http": r.status_code,
                "size": size
            })
        except Exception as e:
            out.append({
                "url": url,
                "status": "ERROR",
                "http": 0,
                "error": str(e)
            })

    return {"session_id": session_id, "checked": out}

# -------------------------
# 2) Analyze session (dummy)
# -------------------------
def analyze_session(payload: dict):
    """
    payload = {
      "session_id": "sess-123",
      "tone": "friendly",
      "product_link": "https://example.com",
      "features_csv": "fast, compact, lightweight"
    }
    Returns dummy script result.
    """
    session_id = payload.get("session_id", f"sess-{int(time.time())}")
    tone = payload.get("tone", "neutral")
    product_link = payload.get("product_link")
    features_csv = payload.get("features_csv", "")

    features = [f.strip() for f in features_csv.split(",") if f.strip()]

    time.sleep(1)  # simulate some work

    return {
        "session_id": session_id,
        "script": f"Promo script for {session_id} with tone={tone}",
        "product_link": product_link,
        "features": features,
    }
