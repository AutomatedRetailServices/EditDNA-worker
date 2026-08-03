import os
import hashlib
import json
from typing import List, Optional, Dict, Any

import redis
from rq import Queue, Retry
from rq.job import Job

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


def benchmark_fingerprint(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def benchmark_retry_count() -> int:
    attempts = _env_int("BENCHMARK_SESSION_MAX_ATTEMPTS", 3, minimum=1)
    required_retries = attempts - 1
    configured = os.getenv("BENCHMARK_RQ_MAX_RETRIES")
    if configured is None:
        return required_retries
    retries = _env_int("BENCHMARK_RQ_MAX_RETRIES", required_retries)
    if retries < required_retries:
        raise ValueError("BENCHMARK_RQ_MAX_RETRIES cannot provide BENCHMARK_SESSION_MAX_ATTEMPTS executions")
    return retries


def enqueue_benchmark(job_id: str, payload: Dict[str, Any]):
    """Enqueue a benchmark on the existing RQ/Redis infrastructure."""
    queue = get_queue(); connection = queue.connection
    lock_key = f"editdna:benchmark:active:{benchmark_fingerprint(payload)}"
    existing_id = connection.get(lock_key)
    if existing_id:
        existing_id = existing_id.decode() if isinstance(existing_id, bytes) else str(existing_id)
        try:
            existing = Job.fetch(existing_id, connection=connection)
            if getattr(existing.get_status(refresh=True), "value", str(existing.get_status())) in {
                    "queued", "started", "deferred", "scheduled"}:
                return existing, True
        except Exception:
            pass
        connection.delete(lock_key)
    if not connection.set(lock_key, job_id, nx=True, ex=_env_int("BENCHMARK_DUPLICATE_TTL", 86400, minimum=1)):
        winner = connection.get(lock_key)
        winner = winner.decode() if isinstance(winner, bytes) else str(winner)
        return Job.fetch(winner, connection=connection), True
    job = queue.enqueue(
        "tasks.job_benchmark", job_id, payload, job_id=job_id,
        meta={"stage": "queued", "total_sessions": 0, "processed_sessions": 0,
              "successful_sessions": 0, "failed_sessions": 0, "errors_count": 0},
        job_timeout=_env_int("BENCHMARK_JOB_TIMEOUT", 86400, minimum=1),
        result_ttl=_env_int("RQ_RESULT_TTL", 86400),
        failure_ttl=_env_int("RQ_FAILURE_TTL", 604800),
        retry=Retry(max=benchmark_retry_count(), interval=30),
    )
    return job, False
