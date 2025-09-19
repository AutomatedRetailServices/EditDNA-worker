from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from redis import Redis
from rq import Queue
from rq.job import Job

# -----------------------------------------------------------------------------
# Redis / RQ setup
# -----------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis = Redis.from_url(REDIS_URL)
queue = Queue("default", connection=redis, default_timeout=60 * 60)  # 60 min

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="editdna", version="1.1.0")

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
Mode = Literal["concat", "split"]

class RenderRequest(BaseModel):
    session_id: Optional[str] = "session"
    files: List[str | HttpUrl]
    output_prefix: Optional[str] = "editdna/outputs"
    portrait: Optional[bool] = True
    mode: Optional[Mode] = "concat"  # NEW: "concat" | "split"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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
    job = Job.fetch(job_id, connection=redis)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {"ok": True, "service": "editdna", "time": int(datetime.utcnow().timestamp())}
    )

@app.post("/enqueue_nop")
def enqueue_nop() -> JSONResponse:
    job = queue.enqueue("worker.task_nop")
    return JSONResponse({"job_id": job.id})

@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    try:
        job = _get_job_or_404(job_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Not Found")
    return JSONResponse(_job_payload(job))

@app.post("/render")
def render(req: RenderRequest) -> JSONResponse:
    """
    Enqueue a render job.
    mode="concat"  -> one stitched mp4
    mode="split"   -> per-clip mp4s (list)
    """
    payload = {
        "session_id": req.session_id or "session",
        "files": [str(x) for x in req.files],
        "output_prefix": req.output_prefix or "editdna/outputs",
        "portrait": bool(req.portrait),
        "mode": req.mode or "concat",
    }
    job = queue.enqueue("worker.job_render", payload)
    return JSONResponse({"job_id": job.id, "session_id": payload["session_id"]})

# Optional compatibility endpoint
@app.post("/render_chunked")
def render_chunked(req: RenderRequest) -> JSONResponse:
    payload = {
        "session_id": req.session_id or "session",
        "files": [str(x) for x in req.files],
        "output_prefix": req.output_prefix or "editdna/outputs",
        "portrait": bool(req.portrait),
        "mode": req.mode or "concat",
    }
    job = queue.enqueue("worker.job_render_chunked", payload)
    return JSONResponse({"job_id": job.id, "session_id": payload["session_id"]})
