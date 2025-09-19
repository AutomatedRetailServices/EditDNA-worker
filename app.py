# app.py

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

# -----------------------------------------------------------------------------
# Redis / RQ setup
# -----------------------------------------------------------------------------
APP_VERSION = "1.2.1"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis = Redis.from_url(REDIS_URL)
queue = Queue("default", connection=redis, default_timeout=60 * 60)  # 60 min

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="editdna", version=APP_VERSION)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class RenderRequest(BaseModel):
    session_id: Optional[str] = "session"
    files: List[str | HttpUrl]
    output_prefix: Optional[str] = "editdna/outputs"
    portrait: Optional[bool] = True

    # optional knobs (pass-through to worker)
    mode: Optional[str] = "concat"            # "concat" | "best" | "split"
    max_duration: Optional[int] = None        # seconds cap for final video
    take_top_k: Optional[int] = None          # keep best K clips (mode="best")
    min_clip_seconds: Optional[float] = None  # trim lower bound
    max_clip_seconds: Optional[float] = None  # trim upper bound
    drop_silent: Optional[bool] = True
    drop_black: Optional[bool] = True

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


# small built-in nop task so /enqueue_nop always works
def _nop() -> dict:
    return {"echo": {"hello": "world"}}

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "service": "editdna",
            "version": APP_VERSION,
            "time": int(datetime.utcnow().timestamp()),
        }
    )

@app.get("/version")
def version() -> JSONResponse:
    return JSONResponse({"ok": True, "version": APP_VERSION})

@app.get("/health")
def health() -> JSONResponse:
    try:
        redis_ok = redis.ping()
    except Exception:
        redis_ok = False

    try:
        pending = queue.count()  # <-- call the method
    except Exception:
        pending = None

    return JSONResponse(
        {
            "ok": bool(redis_ok),
            "queue": {"name": queue.name, "pending": pending},
            "redis_db_hint": os.getenv("REDIS_URL", "unknown").split("/")[-1],
            "web_default_timeout_sec": queue.default_timeout,
            "service": "editdna",
            "version": APP_VERSION,
        }
    )

@app.post("/enqueue_nop")
def enqueue_nop() -> JSONResponse:
    # enqueue the local function object (importable as app._nop)
    job = queue.enqueue(_nop)
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
    payload = {
        "session_id": req.session_id or "session",
        "files": [str(x) for x in req.files],
        "output_prefix": req.output_prefix or "editdna/outputs",
        "portrait": bool(req.portrait),
        # pass-through options
        "mode": (req.mode or "concat"),
        "max_duration": req.max_duration,
        "take_top_k": req.take_top_k,
        "min_clip_seconds": req.min_clip_seconds,
        "max_clip_seconds": req.max_clip_seconds,
        "drop_silent": req.drop_silent,
        "drop_black": req.drop_black,
    }
    # IMPORTANT: job_render is in jobs.py
    job = queue.enqueue("jobs.job_render", payload)
    return JSONResponse({"job_id": job.id, "session_id": payload["session_id"]})

@app.post("/render_chunked")
def render_chunked(req: RenderRequest) -> JSONResponse:
    payload = {
        "session_id": req.session_id or "session",
        "files": [str(x) for x in req.files],
        "output_prefix": req.output_prefix or "editdna/outputs",
        "portrait": bool(req.portrait),
        "mode": "concat",
    }
    job = queue.enqueue("jobs.job_render_chunked", payload)
    return JSONResponse({"job_id": job.id, "session_id": payload["session_id"]})
