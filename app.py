# app.py  — FastAPI + RQ 1.16.2 API
import os
import uuid
from typing import Any, Dict, List, Optional

import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rq import Queue
from rq.job import Job
from rq.registry import (
    StartedJobRegistry,
    FailedJobRegistry,
    FinishedJobRegistry,
    ScheduledJobRegistry,
    DeferredJobRegistry,
)

# ------------------------------
# Redis / RQ setup (match worker.py)
# ------------------------------
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

# IMPORTANT: keep decode_responses=False (bytes) to avoid decode errors seen in logs
redis_conn = redis.from_url(REDIS_URL, decode_responses=False)
q = Queue("default", connection=redis_conn)

# ------------------------------
# Tasks (NO LAMBDAS)
# ------------------------------
def nop_task() -> Dict[str, Any]:
    """A tiny task to verify enqueue/execute/result flow."""
    return {"ok": True, "msg": "nop done"}

# Example long-running placeholder you can hook up later
def analyze_core_from_session(session_id: str) -> Dict[str, Any]:
    # Do your real work here later; keep it trivial for now
    return {"ok": True, "session_id": session_id, "analysis": "placeholder"}

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="editdna-web API")

# ------------------------------
# Models
# ------------------------------
class ProcessUrlsIn(BaseModel):
    urls: List[str]
    tone: Optional[str] = None
    product_link: Optional[str] = None
    features_csv: Optional[str] = None

# ------------------------------
# Endpoints
# ------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "editdna-web", "queue": q.name}

@app.post("/enqueue_nop")
def enqueue_nop():
    job = q.enqueue(nop_task)
    return {"ok": True, "job_id": job.get_id()}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        # Unknown job id (not in this Redis / already cleaned)
        raise HTTPException(status_code=404, detail="unknown_job_id")

    status = job.get_status()
    result = None
    error = None

    if status == "finished":
        result = job.result
    elif status == "failed":
        # exc_info is bytes; make it string safely
        error = job.exc_info.decode("utf-8", errors="ignore") if isinstance(job.exc_info, (bytes, bytearray)) else str(job.exc_info)

    return {
        "job_id": job.id,
        "status": status,
        "result": result,
        "error": error,
        "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
        "ended_at": str(job.ended_at) if job.ended_at else None,
    }

@app.get("/admin/health")
def admin_health():
    registries = {
        "queued": q.count,
        "started": StartedJobRegistry(queue=q).count,
        "scheduled": ScheduledJobRegistry(queue=q).count,
        "deferred": DeferredJobRegistry(queue=q).count,
        "failed": FailedJobRegistry(queue=q).count,
        "finished": FinishedJobRegistry(queue=q).count,
    }
    return {"ok": True, "registries": registries}

# -------- Optional placeholders so your old Postman tabs don't 500 --------
@app.post("/process_urls")
def process_urls(body: ProcessUrlsIn):
    # Just echo + create a fake session id so you can continue your flow
    session_id = uuid.uuid4().hex
    return {"ok": True, "session_id": session_id, "files": [{"file_id": "demo", "source": "url"}]}

@app.post("/analyze_sync")
def analyze_sync(session_id: str):
    # Run the analyze job synchronously for now
    result = analyze_core_from_session(session_id)
    return {"ok": True, "result": result}

# ------------------------------
# (No if __name__ == "__main__":) — Render runs via uvicorn startCommand
# ------------------------------
