import os
from typing import List, Optional
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from redis import Redis
from rq import Queue
from datetime import timedelta

# RQ / Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(REDIS_URL)
q = Queue("default", connection=redis_conn)

# job timeout (seconds)
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "3600"))  # 1 hour default
RESULT_TTL = int(os.getenv("RESULT_TTL", "500"))
FAIL_TTL = int(os.getenv("FAIL_TTL", str(7 * 24 * 3600)))  # keep failed jobs 7d

# import worker tasks
from worker import task_nop, job_render  # noqa: E402

app = FastAPI()

class RenderRequest(BaseModel):
    session_id: str = Field(default="sess-demo-001")
    files: List[str]
    output_prefix: str = Field(default="editdna/outputs")

@app.get("/")
def root():
    return {"ok": True, "service": "editdna", "time": int(os.getenv("RENDER_START_TIME", "0")) or __import__("time").time()}

@app.post("/enqueue_nop")
def enqueue_nop():
    job = q.enqueue(
        task_nop,
        job_timeout=JOB_TIMEOUT,
        result_ttl=RESULT_TTL,
        failure_ttl=FAIL_TTL,
        ttl=JOB_TIMEOUT + 60,
    )
    return {"job_id": job.id}

@app.post("/render")
def render_endpoint(payload: RenderRequest = Body(...)):
    if not payload.files:
        raise HTTPException(422, "files[] required")
    job = q.enqueue(
        job_render,
        payload.session_id,
        payload.files,
        payload.output_prefix,
        job_timeout=JOB_TIMEOUT,
        result_ttl=RESULT_TTL,
        failure_ttl=FAIL_TTL,
        ttl=JOB_TIMEOUT + 60,
    )
    return {"job_id": job.id, "session_id": payload.session_id}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    from rq.job import Job
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(404, "job not found")

    status_map = {
        "queued": "queued",
        "started": "started",
        "deferred": "queued",
        "finished": "finished",
        "failed": "failed",
        "stopped": "failed",
        "canceled": "failed",
        "scheduled": "queued",
    }
    status = status_map.get(job.get_status(), job.get_status())

    result = job._result if job.is_finished else None
    err = None
    if job.is_failed:
        try:
            err = job.exc_info or str(job._result)
        except Exception:
            err = "failed"
    return {
        "job_id": job.id,
        "status": status,
        "result": result,
        "error": err,
        "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
        "ended_at": str(job.ended_at) if job.ended_at else None,
    }
