# app.py — Web API for EditDNA (full replacement)

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from redis import Redis
from rq import Queue
from rq.job import Job


# ---------------------------------------------------------------------
# Redis / RQ setup (web and worker **must** share the exact same REDIS_URL)
# ---------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("RQ_QUEUE", "default")

redis = Redis.from_url(REDIS_URL)
queue = Queue(QUEUE_NAME, connection=redis, default_timeout=60 * 60)  # 60 min


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(title="editdna", version="1.1.0")


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class RenderRequest(BaseModel):
    # required inputs
    files: List[str | HttpUrl]

    # session/output controls
    session_id: Optional[str] = "session"
    output_prefix: Optional[str] = "editdna/outputs"
    portrait: Optional[bool] = True

    # (optional) “smart edit” knobs — passed through to worker
    mode: Optional[str] = "concat"         # "concat" | "split" | "best" (future)
    max_duration: Optional[int] = None     # seconds (for "best")
    take_top_k: Optional[int] = None       # pick k best clips (for "best")
    min_clip_seconds: Optional[float] = None
    max_clip_seconds: Optional[float] = None
    drop_silent: Optional[bool] = None
    drop_black: Optional[bool] = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _job_payload(job: Job) -> Dict[str, Any]:
    status = job.get_status(refresh=True)
    result = None
    error = None

    if status == "finished":
        result = job.result
    elif status == "failed":
        error = job.exc_info

    return {
        "job_id": job.id,
        "status": status,
        "result": result,
        "error": error,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
    }


def _get_job_or_404(job_id: str) -> Job:
    try:
        return Job.fetch(job_id, connection=redis)
    except Exception:
        raise HTTPException(status_code=404, detail="Not Found")


# ---------------------------------------------------------------------
# Core routes
# ---------------------------------------------------------------------
@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {"ok": True, "service": "editdna", "time": int(datetime.utcnow().timestamp())}
    )


@app.post("/enqueue_nop")
def enqueue_nop() -> JSONResponse:
    """Enqueue a tiny health-check task that the worker should finish immediately."""
    job = queue.enqueue("worker.task_nop")  # <- worker must be listening on same queue
    return JSONResponse({"job_id": job.id})


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = _get_job_or_404(job_id)
    return JSONResponse(_job_payload(job))


@app.post("/render")
def render(req: RenderRequest) -> JSONResponse:
    """
    Enqueue a render job. We send a single dict payload so both the new
    (dict-arg) and old (positional) worker signatures will work.
    """
    payload: Dict[str, Any] = {
        "session_id": req.session_id or "session",
        "files": [str(x) for x in req.files],
        "output_prefix": req.output_prefix or "editdna/outputs",
        "portrait": bool(req.portrait),
        # pass-through smart-edit knobs
        "mode": (req.mode or "concat").lower(),
        "max_duration": req.max_duration,
        "take_top_k": req.take_top_k,
        "min_clip_seconds": req.min_clip_seconds,
        "max_clip_seconds": req.max_clip_seconds,
        "drop_silent": req.drop_silent,
        "drop_black": req.drop_black,
    }

    job = queue.enqueue("worker.job_render", payload)
    return JSONResponse({"job_id": job.id, "session_id": payload["session_id"]})


# ---------------------------------------------------------------------
# Health & Debug (to avoid “queued forever” mysteries)
# ---------------------------------------------------------------------
@app.get("/health")
def health() -> JSONResponse:
    """
    Basic service/queue health snapshot.
    - redis_ping: True/False
    - queue: name + length (pending jobs)
    - same_redis_hint: helps ensure WEB and WORKER share the same DB index
    """
    try:
        redis_ping = bool(redis.ping())
    except Exception:
        redis_ping = False

    # VERY conservative: only expose the DB index portion to help debugging
    same_redis_hint = REDIS_URL.rsplit("/", 1)[-1] if "://" in REDIS_URL else REDIS_URL

    return JSONResponse(
        {
            "ok": redis_ping,
            "queue": {"name": QUEUE_NAME, "pending": len(queue)},
            "redis_db_hint": same_redis_hint,
            "web_default_timeout_sec": queue.default_timeout,
        }
    )


@app.get("/debug/queue_len")
def queue_len() -> JSONResponse:
    """Lightweight queue length for dashboards."""
    return JSONResponse({"queue": QUEUE_NAME, "pending": len(queue)})


# (Optional) a convenience endpoint to re-enqueue a stuck job’s payload if needed.
# Left out intentionally to keep the surface area small.
