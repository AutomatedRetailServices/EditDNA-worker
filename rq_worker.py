"""RunPod-friendly RQ worker launcher with stable Redis options."""

from __future__ import annotations

import os
import sys
from urllib.parse import urlsplit, urlunsplit

from redis import Redis
from redis.exceptions import RedisError
from rq import Worker

WORKER_TTL_SECONDS = 3600
REDIS_CONNECTION_OPTIONS = {
    "socket_keepalive": True,
    "health_check_interval": 30,
    "retry_on_timeout": True,
    "socket_connect_timeout": 10,
}


def mask_redis_url(redis_url: str) -> str:
    """Return a log-safe Redis URL that never exposes credentials."""
    parsed = urlsplit(redis_url)
    if not parsed.scheme or not parsed.netloc:
        return "<configured>"

    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    credentials = "***@" if parsed.username or parsed.password else ""
    netloc = f"{credentials}{host}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def create_connection(redis_url: str) -> Redis:
    """Create the Redis connection used by the worker."""
    return Redis.from_url(redis_url, **REDIS_CONNECTION_OPTIONS)


def run_worker() -> bool:
    """Validate Redis and start one RQ worker for the configured queue."""
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        print("ERROR: REDIS_URL is required", file=sys.stderr)
        return False

    queue_name = os.environ.get("QUEUE_NAME") or "default"
    safe_url = mask_redis_url(redis_url)

    print(f"Connecting to Redis: {safe_url}")
    print(f"Starting RQ worker for queue: {queue_name}")

    try:
        connection = create_connection(redis_url)
        connection.ping()
    except (RedisError, TimeoutError) as exc:
        print(f"ERROR: Could not connect to Redis before worker startup: {exc}", file=sys.stderr)
        return False

    worker = Worker([queue_name], connection=connection, worker_ttl=WORKER_TTL_SECONDS)
    return worker.work()


def main() -> int:
    return 0 if run_worker() else 1


if __name__ == "__main__":
    raise SystemExit(main())
