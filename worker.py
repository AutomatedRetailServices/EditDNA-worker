# worker.py — RQ 1.16 compatible worker functions
import os
import redis
import requests
from typing import List, Dict, Any

REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

# bytes mode
conn = redis.from_url(REDIS_URL, decode_responses=False)

# The worker process is launched by Render via: python worker.py
# This file both defines functions AND can run an RQ Worker when executed.
# (Render's startCommand should be: python worker.py)

def echo_nop(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {"echo": payload or {"hello": "world"}}

def process_urls(urls: List[str]) -> Dict[str, Any]:
    """
    Minimal 'real' work: validate that each URL is reachable.
    We use HEAD first (fast), then GET a tiny byte range if HEAD is not allowed.
    """
    out: List[Dict[str, Any]] = []
    for u in urls:
        entry: Dict[str, Any] = {"url": u}
        try:
            r = requests.head(u, timeout=30, allow_redirects=True)
            if r.status_code in (200, 206):
                size = r.headers.get("Content-Length")
                entry.update({"status": "ok", "http": r.status_code, "size": int(size) if size else None})
            else:
                # Some S3/CDNs disallow HEAD; try a quick GET of 1st bytes
                r2 = requests.get(u, timeout=45, stream=True)
                ok = r2.status_code in (200, 206)
                entry.update({"status": "ok" if ok else "error", "http": r2.status_code})
        except Exception as e:
            entry.update({"status": "error", "error": str(e)})
        out.append(entry)
    return {"checked": out}


# --- Run an RQ worker if executed directly ---
if __name__ == "__main__":
    from rq import Worker, Queue, Connection  # import only when running as worker

    with Connection(conn):
        w = Worker([Queue("default")])
        # Using with_scheduler=True keeps scheduled/cleanup working
        w.work(with_scheduler=True)
