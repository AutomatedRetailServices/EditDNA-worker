# worker.py — minimal, RQ 1.16.x compatible
import os
import redis
from rq import Queue, Worker, Connection

REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

conn = redis.from_url(REDIS_URL, decode_responses=False)

if __name__ == "__main__":
    # Listen only on the "default" queue (matches app.py)
    with Connection(conn):
        worker = Worker([Queue("default")])
        # IMPORTANT: no extra kwargs like worker_ttl, no RQScheduler ctor
        worker.work(with_scheduler=True)
