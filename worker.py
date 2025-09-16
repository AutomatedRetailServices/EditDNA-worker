# worker.py — EditDNA RQ worker
import os
from rq import Worker, Queue, Connection
import redis

REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

# Binary-safe like in app.py
conn = redis.from_url(REDIS_URL, decode_responses=False)

# Match the same queue name used by app.py
QUEUES = ["default"]

if __name__ == "__main__":
    with Connection(conn):
        # --worker-ttl is controlled by env var here for Render flexibility
        ttl = int(os.getenv("RQ_WORKER_TTL", "1200"))  # 20 min
        # start the worker (same behavior as `rq worker --with-scheduler`)
        w = Worker(queues=[Queue(q) for q in QUEUES], name=None)
        # enable scheduler by setting env `RQ_WORKER_SCHEDULER=1`
        scheduler = os.getenv("RQ_WORKER_SCHEDULER", "1") == "1"
        if scheduler:
            from rq.scheduler import RQScheduler
            RQScheduler(connection=conn, queue=Queue("default")).run(burst=True)
        w.work(with_scheduler=True, worker_ttl=ttl)
