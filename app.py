# app.py — FastAPI front-end for enqueuing and tracking RQ jobs (full file)

import os
from datetime import datetime
from typing import Any, Dict

import redis
import rq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import callables directly so RQ can serialize them without string names
from worker import task_nop, job_render, job_render_chunked

# ---------- FastAPI ----------
app = FastAPI(title="editdna")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Redis / RQ ----------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = redis.from_url(REDIS_URL)
queue = rq.Queue("default", connection=redis_conn, default_timeout=60 * 20)  # 20 min


# ---------- Schemas ----------
class RenderPayload(BaseModel):
    session_id: str = Field(..., description="Arbitrary session id you choose")
    files: list[str] = Field(..., description="List of HTTP(S) or s3:// URLs")
    output_prefix: str = Field("editdna/outputs", description="Local prefix or s3://bucket/prefix")


# ---------- helpers ----------
def job_to_dict(job: rq.job.Job) -> Dict[str, Any]:
    status = job.get_status(refresh=True)
    result = None
    error = None
    if job.is_finished:
        result = job.result
    elif job.is_failed:
        error = job.exc_info or "failed"
    return {
        "job_id": job.id,
        "status": status,
        "result": result,
        "error": error,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
    }


# ---------- Routes ----------
@app.get("/")
def root():
    return {"ok": True, "service": "editdna", "time": int(datetime.utcnow().timestamp())}


@app.post("/enqueue_nop")
def enqueue_nop():
    job = queue.enqueue(task_nop)
    return {"job_id": job.id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    try:
        job = rq.job.Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_to_dict(job)


@app.post("/jobs/requeue/{job_id}")
def job_requeue(job_id: str):
    try:
        job = rq.job.Job.fetch(job_id, connection=redis_conn)
        job.requeue()
        return {"ok": True, "job_id": job.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"requeue failed: {e}")


@app.post("/render")
def enqueue_render(payload: RenderPayload):
    job = queue.enqueue(job_render, payload.model_dump())
    return {"job_id": job.id, "session_id": payload.session_id}


@app.post("/render_chunked")
def enqueue_render_chunked(payload: RenderPayload):
    job = queue.enqueue(job_render_chunked, payload.model_dump())
    return {"job_id": job.id, "session_id": payload.session_id}
