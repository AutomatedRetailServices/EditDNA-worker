import os
import redis
from rq import Worker, Queue, Connection

LISTEN = ["default"]

REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

# NOTE: With redis-py v5+, 'rediss://' automatically enables TLS.
conn = redis.from_url(REDIS_URL)

if __name__ == "__main__":
    with Connection(conn):
        Worker([Queue(name) for name in LISTEN]).work()
