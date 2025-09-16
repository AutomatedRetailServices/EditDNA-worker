i# app.py — minimal health-check API for Redis/RQ worker

import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from redis import Redis
from rq import Queue
from rq.job import Job   # <-- use rq.job.Job (works with RQ 1.16)

# ----- FastAPI app -----
app = FastAPI(title="editdna diagnostic")

# ----- Redis / RQ setup -----
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL env var")

redis_conn = Redis.from_url(REDIS_URL, decode_responses=False)
q = Queue("default", connection=redis_conn)   # must match worker queue name

# ----- Small no-op job the worker can execute -----
def nop():
    # RQ cannot serialize lambdas; use a named function.
    return "ok"

# ----- Models -----
class EnqueueResponse(BaseModel):
    ok: bool
    job_id: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None

# ----- Routes -----
@app.get("/")
def root():
    return {"ok": True, "msg": "editdna diagnostic alive"}

@app.post("/enqueue_nop", response_model=EnqueueResponse)
def enqueue_nop():
    # Enqueue the named function (NOT a lambda)
    job = q.enqueue(nop)
    return {"ok": True, "job_id": job.id}

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception as e:
        return {"job_id": job_id, "status": "unknown", "error": str(e)}

    # job.get_status() -> queued | started | deferred | finished | failed | scheduled
    status = job.get_status()
    payload: JobStatusResponse = {
        "job_id": job.get_id(),
        "status": status
    }

    if status == "finished":
        # our nop() returns "ok"
        try:
            payload["result"] = job.result if isinstance(job.result, str) else str(job.result)
        except Exception:
            payload["result"] = None
    elif status == "failed":
        try:
            payload["error"] = str(job.exc_info or "")
        except Exception:
            payload["error"] = "failed (no traceback)"

    return payload
