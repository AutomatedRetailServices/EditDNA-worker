# worker.py
import os
import time
from typing import List, Dict, Any

import httpx
from redis import from_url as redis_from_url
from rq import Worker, Queue

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# RQ requires a Redis connection without auto-decoding for safety
redis_conn = redis_from_url(REDIS_URL, decode_responses=False)
queue_name = "default"

# ------------- TASKS (called by the queue) -----------------
def hello_world() -> Dict[str, str]:
    # super small, finishes fast
    return {"echo": {"hello": "world"}}

def check_urls(urls: List[str]) -> Dict[str, Any]:
    """
    For each URL:
      - HEAD first, fallback to GET (streaming) if server doesn't support HEAD
      - collect http status and size (Content-Length or downloaded length)
    Returns { "checked": [...] }
    """
    results = []
    timeout = httpx.Timeout(15.0, connect=10.0)
    headers = {"User-Agent": "editdna-checker/1.0"}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        for url in urls:
            rec = {"url": url, "status": "ERR", "http": None, "size": None}
            try:
                # Try HEAD
                r = client.head(url)
                rec["http"] = r.status_code
                if r.status_code == 200:
                    size = r.headers.get("content-length")
                    rec["size"] = int(size) if size and size.isdigit() else None
                    rec["status"] = "OK"
                else:
                    # fallback to GET if not 200 on HEAD (some S3 links may require GET)
                    r = client.get(url, stream=True)
                    rec["http"] = r.status_code
                    if r.status_code == 200:
                        total = 0
                        for chunk in r.iter_bytes():
                            total += len(chunk)
                            if total > 5 * 1024 * 1024:  # don't download full movie; cap ~5MB
                                break
                        rec["size"] = total
                        rec["status"] = "OK"
            except Exception as e:
                rec["status"] = f"ERR: {e.__class__.__name__}"
            results.append(rec)
            # tiny pause to be nice to S3/CDN
            time.sleep(0.02)
    return {"checked": results}

# ------------- RQ Worker entrypoint (for Render “Background Worker”) -------------
if __name__ == "__main__":
    w = Worker([Queue(queue_name, connection=redis_conn)], connection=redis_conn)
    # This will block and listen forever
    w.work(with_scheduler=True)
