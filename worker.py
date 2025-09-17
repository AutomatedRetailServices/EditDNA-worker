# worker.py
import os, json, time, requests
import redis
from rq import Worker, Queue, Connection

def task_nop():
    return {"echo": {"hello": "world"}}

def check_urls(payload: dict):
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
    sess = payload["session_id"]
    time.sleep(1)
    return {
        "session_id": sess,
        "script": f"Promo script for session {sess} with tone={payload.get('tone','neutral')}",
        "product_link": payload.get("product_link"),
        "features": [s.strip() for s in (payload.get("features_csv","").split(",")) if s.strip()],
    }

# --------------------------
# Entry point for Render worker
# --------------------------
if __name__ == "__main__":
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise RuntimeError("Missing REDIS_URL")

    conn = redis.from_url(redis_url, decode_responses=False)
    with Connection(conn):
        worker = Worker(["default"])
        worker.work()
