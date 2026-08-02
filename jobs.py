import os
from typing import List, Optional, Dict, Any

import redis
from rq import Queue, Retry

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "default")


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    """Read a non-negative integer setting and fail fast on bad configuration."""
    value = int(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def get_queue(name: Optional[str] = None) -> Queue:
    """
    Crea la cola RQ usando REDIS_URL de env.
    """
    qname = name or QUEUE_NAME
    conn = redis.from_url(REDIS_URL)
    return Queue(qname, connection=conn)


def enqueue_render(
    session_id: str,
    files: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
    mode: str = "human",
    meta: Optional[Dict[str, Any]] = None,
):
    """
    Helper para encolar el job principal de edición.
    """
    q = get_queue()
    payload: Dict[str, Any] = {
        "session_id": session_id,
        "files": files,
        "file_urls": file_urls,
        "mode": mode,
    }
    # RQ retries every exception raised by the job; it cannot distinguish a
    # transient infrastructure error from deterministic media failure here.
    # Default retries off until the pipeline exposes classified exceptions.
    max_retries = _env_int("RQ_MAX_RETRIES", 0)
    job = q.enqueue(
        "tasks.job_render",
        kwargs=payload,
        meta={**(meta or {}), "stage": "queued", "progress": 0, "message": "Render queued"},
        job_timeout=_env_int("RQ_JOB_TIMEOUT", 3600, minimum=1),
        result_ttl=_env_int("RQ_RESULT_TTL", 86400),
        failure_ttl=_env_int("RQ_FAILURE_TTL", 604800),
        retry=Retry(max=max_retries, interval=10) if max_retries else None,
    )
    return job
