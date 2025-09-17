# worker.py — RQ task functions + entrypoint
import os, time, logging
import requests

# -------------------------
# Task functions
# -------------------------
def task_nop():
    # tiny test job to prove the pipeline works
    return {"echo": {"hello": "world"}}

def check_urls(payload: dict):
    """
    payload = {"session_id": "...", "urls": ["https://.../file.mov", ...]}
    Returns per-URL status and size.
    """
    session_id = payload.get("session_id", f"sess-{int(time.time())}")
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
    payload = {
      "session_id": "...",
      "tone": "friendly",
      "product_link": "https://example.com",
      "features_csv": "fast, compact, lightweight"
    }
    """
    session_id = payload.get("session_id", f"sess-{int(time.time())}")
    tone = payload.get("tone", "neutral")
    product_link = payload.get("product_link")
    features_csv = payload.get("features_csv", "")
    features = [f.strip() for f in features_csv.split(",") if f.strip()]
    time.sleep(1)  # simulate work
    return {
        "session_id": session_id,
        "script": f"Promo script for {session_id} with tone={tone}",
        "product_link": product_link,
        "features": features,
    }

# -------------------------
# RQ worker entrypoint
# -------------------------
if __name__ == "__main__":
    import redis
    from rq import Worker, Queue, Connection

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    REDIS_URL = os.getenv("REDIS_URL", "")
    if not REDIS_URL:
        raise RuntimeError("Missing REDIS_URL")

    conn = redis.from_url(REDIS_URL, decode_responses=False)
    listen = ["default"]

    logging.info(f"Worker connecting to {REDIS_URL}; queues={listen}")

    with Connection(conn):
        Worker([Queue(name) for name in listen]).work(with_scheduler=True)
