# worker.py — minimal RQ worker that listens on the "default" queue
import os
import redis
from rq import Worker, Queue, Connection

REDIS_URL = os.environ.get("REDIS_URL", "").strip()
if not REDIS_URL:
    raise SystemExit("ERROR: REDIS_URL not set for worker")

# IMPORTANT: binary-safe (prevents utf-8 decode errors)
conn = redis.from_url(REDIS_URL, decode_responses=False)

if __name__ == "__main__":
    with Connection(conn):
        w = Worker(["default"])
        w.work(with_scheduler=True)
