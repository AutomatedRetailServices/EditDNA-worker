# app.py — web API for EditDNA (full file)

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

APP_VERSION = "1.2.6"

# -----------------------------------------------------------------------------
# Redis / RQ
# -----------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis = Redis.from_url(REDIS_URL)
# No default_timeout here; set per-job in enqueue
queue = Queue("default", connection=redis)

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

    # knobs passed through to worker (used by jobs.py “best clips” flow)
    mode: Optional[str] = "concat"            # "concat" | "best"
    max_duration: Optional[int] = None        # seconds cap for final video
    take_top_k: Optional[int] = None          # keep best K clips
    min_clip_seconds: Optional[float] = None  # lower bound per clip
    max_clip_seconds: Optional[float] = None  # upper bound per clip
    drop_silent: Optional[bool] = True
    drop_black: Optional[bool] = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _job_payload(job: Job) -> Dict[str, Any]:
    status = job.get_status(refresh=True)
    result = job.result if status == "finished" else None
    error = job.exc_info if status == "failed" else None
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
        redis_ok = bool(redis.ping())
    except Exception:
        redis_ok = False

    # Note: queue.count is a property that queries Redis lazily; it's cheap.
    return JSONResponse(
        {
            "ok": redis_ok,
            "service": "editdna",
            "version": APP_VERSION,
            "queue": {"name": queue.name, "pending": queue.count},
            # tail of REDIS_URL to visually verify the web/worker point to same instance
            "redis_url_tail": os.getenv("REDIS_URL", "unknown")[-32:],
        }
    )


@app.post("/enqueue_nop")
def enqueue_nop() -> JSONResponse:
    # Goes through the worker shim so "worker.task_nop" remains valid.
    job = queue.enqueue("worker.task_nop", result_ttl=300)
    return JSONResponse({"job_id": job.id})


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = _get_job_or_404(job_id)
    return JSONResponse(_job_payload(job))


@app.post("/render")
def render(req: RenderRequest) -> JSONResponse:
    # Build ONE payload dict; the worker expects a dict (jobs.job_render(payload))
    payload = {
        "session_id": req.session_id or "session",
        "files": [str(x) for x in req.files],
        "output_prefix": (req.output_prefix or "editdna/outputs").strip("/"),
        "portrait": bool(req.portrait),
        # pass-through knobs used by jobs.py
        "mode": (req.mode or "concat"),
        "max_duration": req.max_duration,
        "take_top_k": req.take_top_k,
        "min_clip_seconds": req.min_clip_seconds,
        "max_clip_seconds": req.max_clip_seconds,
        "drop_silent": req.drop_silent,
        "drop_black": req.drop_black,
    }

    job = queue.enqueue(
        "worker.job_render",          # call the shim; it forwards to jobs.job_render
        payload,                      # send ONE dict so mode/best is honored
        job_timeout=60 * 60,          # 60 min
        result_ttl=86400,             # keep result for 1 day
        ttl=7200,                     # allow up to 2h in the queue
    )
    return JSONResponse({"job_id": job.id, "session_id": payload["session_id"]})


@app.post("/render_chunked")
def render_chunked(req: RenderRequest) -> JSONResponse:
    # You can define this however you like; here we keep it similar but force a mode.
    payload = {
        "session_id": req.session_id or "session",
        "files": [str(x) for x in req.files],
        "output_prefix": (req.output_prefix or "editdna/outputs").strip("/"),
        "portrait": bool(req.portrait),
        "mode": (req.mode or "best"),  # example: default to "best" for this endpoint
        "max_duration": req.max_duration,
        "take_top_k": req.take_top_k,
        "min_clip_seconds": req.min_clip_seconds,
        "max_clip_seconds": req.max_clip_seconds,
        "drop_silent": req.drop_silent,
        "drop_black": req.drop_black,
    }

    job = queue.enqueue(
        "worker.job_render_chunked",
        payload,
        job_timeout=60 * 60,
        result_ttl=86400,
        ttl=7200,
    )
    return JSONResponse({"job_id": job.id, "session_id": payload["session_id"]})
