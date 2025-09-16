# worker.py — RQ 1.16 compatible
import os
import redis
from rq import Queue, Worker, Connection

REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

# IMPORTANT: bytes mode to play nicely with RQ internals
conn = redis.from_url(REDIS_URL, decode_responses=False)

if __name__ == "__main__":
    with Connection(conn):
        worker = Worker([Queue("default")])  # listen on "default"
        worker.work(with_scheduler=True)
