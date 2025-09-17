# worker.py
import os, json, time, requests

def task_nop():
    # tiny test job
    return {"echo": {"hello": "world"}}

def check_urls(payload: dict):
    """
    payload = {"session_id": "sess-001", "urls": ["https://.../IMG_4856.mov", ...]}
    Returns per-URL HTTP status/size so you can see they’re reachable.
    """
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    out = []
    for url in payload.get("urls", []):
        try:
            r = requests.head(url, timeout=20, allow_redirects=True)
            size = int(r.headers.get("Content-Length") or 0)
            out.append({"url": url, "status": "OK", "http": r.status_code, "size": size})
        except Exception as e:
            out.append({"url": url, "status": "ERROR", "http": 0, "error": str(e)})
    return {"session_id": session_id, "checked": out}

def analyze_session(payload: dict):
    """
    payload = {"session_id": "...", "tone": "...", "product_link": "...", "features_csv": "..."}
    Dummy analysis that just echos back what it received.
    Replace this with your real analysis later.
    """
    sess = payload["session_id"]
    # pretend work
    time.sleep(1)
    return {
        "session_id": sess,
        "script": f"Promo script for session {sess} with tone={payload.get('tone','neutral')}",
        "product_link": payload.get("product_link"),
        "features": [s.strip() for s in (payload.get("features_csv","").split(",")) if s.strip()],
    }
