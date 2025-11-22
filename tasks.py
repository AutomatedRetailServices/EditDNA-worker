# web/tasks.py
# Adapter entre la API y el Worker GPU
# Cuando el cliente llama: API → job_render() → encolamos tasks.job_render en Redis

import os
import uuid
from typing import Any, Dict, List
from redis import Redis
from rq import Queue

# Config Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(REDIS_URL)
q = Queue("default", connection=redis_conn)


def job_render(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Esta función la llama la API.
    Encola un trabajo para que el worker lo ejecute (tasks.job_render)
    """

    session_id = data.get("session_id", f"session-{uuid.uuid4().hex[:8]}")

    payload: Dict[str, Any] = {
        "session_id": session_id,
        "files": data.get("files", []),
        "portrait": bool(data.get("portrait", True)),
        "max_duration": float(data.get("max_duration", 120.0)),
        "s3_prefix": data.get("s3_prefix", "editdna/outputs/")
    }

    # Si vienen prompts en la API (más adelante)
    if "funnel_counts" in data:
        payload["funnel_counts"] = data["funnel_counts"]

    # 🚀 Aquí es la clave:
    job = q.enqueue("tasks.job_render", payload)

    return {
        "ok": True,
        "job_id": job.id,
        "session_id": session_id
    }
