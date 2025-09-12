import os
import redis
from rq import Worker, Queue, Connection

LISTEN = ["default"]

REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

# No ssl= arg needed. 'rediss://' in the URL already enables TLS.
conn = redis.from_url(REDIS_URL)

if __name__ == "__main__":
    with Connection(conn):
        Worker([Queue(name) for name in LISTEN]).work()
