# app.py — FastAPI + RQ web service (full file)

import os
import time
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import redis
from rq import Queue
from rq.job import Job

import worker  # <- our task functions live here

# ---------------- Redis / RQ setup ----------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    rconn = redis.from_url(REDIS_URL)
    q = Queue("default", connection=rconn, default_timeout=60 * 20)  # 20 min job timeout
except Exception as e:
    rconn = None
    q = None
    print("Failed to connect to Redis:", e)

# ---------------- FastAPI ----------------
app = FastAPI(title="editdna", version="1.0.0")


@app.get("/")
def root():
    return {"ok": True, "service": "editdna", "time": int(time.time())}


@app.get("/admin/health")
def admin_health():
    return {"status": "ok"}


# ---------------- Pydantic models ----------------
class RenderBody(BaseModel):
    session_id: str = Field(..., description="Unique session id (any string)")
    files: List[str] = Field(..., description="List of S3/HTTP urls (or s3://bucket/key)")
    output_prefix: Optional[str] = Field("editdna/outputs", description="S3 key prefix for output")


# ---------------- Queue helpers ----------------
def _ensure_queue():
    if q is None:
        raise HTTPException(status_code=500, detail="Queue not available")
    return q


def _job_payload(j: Job) -> Dict[str, Any]:
    return {
        "job_id": j.id,
        "status": j.get_status(refresh=False),
        "result": j.result,
        "error": j.meta.get("exc_string") if isinstance(j.meta, dict) and j.meta.get("exc_string") else None,
        "enqueued_at": j.enqueued_at.isoformat() if j.enqueued_at else None,
        "ended_at": j.ended_at.isoformat() if j.ended_at else None,
    }


# ---------------- Test/NOP ----------------
@app.post("/enqueue_nop")
def enqueue_nop():
    qx = _ensure_queue()
    job = qx.enqueue(worker.task_nop, job_timeout=60)
    return {"job_id": job.id}


# ---------------- Render (downloads first, then ffmpeg) ----------------
@app.post("/render")
def render_video(body: RenderBody):
    """
    Enqueue render with ONE dict payload (matches worker.job_render signature).
    """
    qx = _ensure_queue()
    payload = {
        "session_id": body.session_id,
        "files": body.files,
        "output_prefix": body.output_prefix or "editdna/outputs",
    }
    job = qx.enqueue(worker.job_render, payload, job_timeout=60 * 30)  # 30 minutes
    return {"job_id": job.id, "session_id": body.session_id}


# Optional alias (same behavior)
@app.post("/render_chunked")
def render_chunked(body: RenderBody):
    qx = _ensure_queue()
    payload = {
        "session_id": body.session_id,
        "files": body.files,
        "output_prefix": body.output_prefix or "editdna/outputs",
    }
    job = qx.enqueue(worker.job_render_chunked, payload, job_timeout=60 * 30)
    return {"job_id": job.id, "session_id": body.session_id}


# ---------------- Jobs API ----------------
@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    qx = _ensure_queue()
    job = Job.fetch(job_id, connection=qx.connection)
    return _job_payload(job)


@app.post("/jobs/requeue/{job_id}")
def requeue_job(job_id: str):
    qx = _ensure_queue()
    job = Job.fetch(job_id, connection=qx.connection)
    ok = job.requeue()
    if not ok:
        raise HTTPException(status_code=400, detail="requeue failed")
    return {"job_id": job.id, "status": job.get_status(refresh=True)}
