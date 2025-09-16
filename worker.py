# worker.py  — RQ 1.16 compatible
import os
import sys
import redis
from rq import Worker, Queue, Connection

# Use the same Redis that Render gives you
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    print("ERROR: REDIS_URL env var is missing", file=sys.stderr)
    sys.exit(1)

conn = redis.from_url(REDIS_URL, decode_responses=True)

if __name__ == "__main__":
    # Listen on the SAME queue name your app uses ("default")
    with Connection(conn):
        worker = Worker([Queue("default")])
        # RQ 1.16+: built-in scheduler; no extra kwargs
        worker.work(with_scheduler=True)
