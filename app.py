# app.py — web API for EditDNA

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

APP_VERSION = "1.2.7"

# Redis / RQ setup
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis = Redis.from_url(REDIS_URL)
queue = Queue("default", connection=redis)

app = FastAPI(title="editdna", version=APP_VERSION)


# ---------- Models ----------
class RenderRequest(BaseModel):
    session_id: Optional[str] = "session"
    files: List[str | HttpUrl]
    output_prefix: Optional[str] = "editdna/outputs"
    portrait: Optional[bool] = True
    mode: Optional[str] = "concat"
    max_duration: Optional[int] = None
    take_top_k: Optional[int] = None
    min_clip_seconds: Optional[float] = None
    max_clip_seconds: Optional[float] = None
    drop_silent: Optional[bool] = True
    drop_black: Optional[bool] = True
    with_captions: Optional[bool] = False  # <- keep parity with jobs.py


# ---------- Helpers ----------
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


# ---------- Routes ----------
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
    try:
        q_count = queue.count
    except Exception:
        q_count = None
    return JSONResponse(
        {
            "ok": redis_ok,
            "service": "editdna",
            "version": APP_VERSION,
            "queue": {"name": queue.name, "pending": q_count},
            "redis_url_tail": os.getenv("REDIS_URL", "unknown")[-32:],
        }
    )


@app.post("/enqueue_nop")
def enqueue_nop() -> JSONResponse:
    # IMPORTANT: use "tasks.task_nop" (module.function) — matches tasks.py
    job = queue.enqueue("tasks.task_nop", result_ttl=300)
    return JSONResponse({"job_id": job.id})


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = _get_job_or_404(job_id)
    return JSONResponse(_job_payload(job))


@app.post("/render")
def render(req: RenderRequest) -> JSONResponse:
    session_id = req.session_id or "session"
    payload = req.dict()
    # IMPORTANT: use "tasks.job_render"
    job = queue.enqueue(
        "tasks.job_render",
        payload,
        job_timeout=60 * 60,
        result_ttl=86400,
        ttl=7200,
    )
    return JSONResponse({"job_id": job.id, "session_id": session_id})


@app.post("/render_chunked")
def render_chunked(req: RenderRequest) -> JSONResponse:
    session_id = req.session_id or "session"
    payload = req.dict()
    # IMPORTANT: use "tasks.job_render_chunked"
    job = queue.enqueue(
        "tasks.job_render_chunked",
        payload,
        job_timeout=60 * 60,
        result_ttl=86400,
        ttl=7200,
    )
    return JSONResponse({"job_id": job.id, "session_id": session_id})
