import os
from typing import List, Optional, Dict, Any

import redis
from rq import Queue

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "default")


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
    meta: Optional[Dict[str, Any]] = None,
):
    """
    Helper para encolar el job principal de edición.

    Encola usando el *string* "tasks.job_render" como ya
    lo está esperando el worker.
    """
    q = get_queue()
    payload: Dict[str, Any] = {
        "session_id": session_id,
        "files": files,
        "file_urls": file_urls,
    }
    job = q.enqueue(
        "tasks.job_render",   # <- el mismo path que sale en el log
        payload,
        meta=meta or {},
    )
    return job
