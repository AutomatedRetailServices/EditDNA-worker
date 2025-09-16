# worker.py — EditDNA RQ worker (RQ 1.16.x compatible)

import os
import redis
from rq import Worker, Queue, Connection

REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

conn = redis.from_url(REDIS_URL, decode_responses=False)
QUEUE_NAMES = ["default"]  # must match app.py

if __name__ == "__main__":
    ttl = int(os.getenv("RQ_WORKER_TTL", "1200"))  # 20 minutes
    with Connection(conn):
        queues = [Queue(name) for name in QUEUE_NAMES]
        worker = Worker(queues)
        # Built-in scheduler loop; no manual RQScheduler object needed
        worker.work(with_scheduler=True, worker_ttl=ttl)
