# RQ 1.16-compatible worker
import os, redis
from rq import Worker, Queue, Connection

REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

# IMPORTANT: bytes, not strings
conn = redis.from_url(REDIS_URL, decode_responses=False)

if __name__ == "__main__":
    with Connection(conn):
        worker = Worker([Queue("default")])
        worker.work(with_scheduler=True)
