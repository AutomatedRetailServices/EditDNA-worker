import os
import redis
from rq import Worker, Queue, Connection

# Queues to listen on
LISTEN = ["default"]

# Grab Redis URL from environment
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL environment variable")

# Connect to Redis
conn = redis.from_url(REDIS_URL, ssl=REDIS_URL.startswith("rediss://"))

if __name__ == "__main__":
    with Connection(conn):
        Worker([Queue(name) for name in LISTEN]).work()
