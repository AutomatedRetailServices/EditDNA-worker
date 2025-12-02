import os
import json
from typing import List, Optional, Dict, Any

import redis
from rq import Queue

# -------------------------------------------------------------------
# Redis + Queue config
# -------------------------------------------------------------------

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "default")


def get_queue(name: Optional[str] = None) -> Queue:
    """
    Return an RQ queue connected to the Redis URL defined in env.
    """
    qname = name or QUEUE_NAME
    conn = redis.from_url(REDIS_URL)
    return Queue(qname, connection=conn)


# -------------------------------------------------------------------
# RQ PATH (used by Render Web API)
# -------------------------------------------------------------------

def enqueue_render(
    session_id: str,
    files: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
    mode: str = "human",
    meta: Optional[Dict[str, Any]] = None,
):
    """
    Convenience function to enqueue an EditDNA render job
    using the RQ worker system.

    The worker expects: tasks.job_render({session_id, files, file_urls, mode})
    """
    q = get_queue()

    # Normalize lists
    if files is None:
        files = []
    if file_urls is None:
        file_urls = []

    if not files and not file_urls:
        raise ValueError("enqueue_render: 'files' or 'file_urls' required")

    # Normalize mode
    mode = (mode or "human").lower()
    if mode not in ("human", "clean", "blooper"):
        mode = "human"

    payload: Dict[str, Any] = {
        "session_id": session_id,
        "files": files,
        "file_urls": file_urls,
        "mode": mode,
    }

    job = q.enqueue(
        "tasks.job_render",   # worker-side entrypoint
        payload,
        meta=meta or {},
    )
    return job
