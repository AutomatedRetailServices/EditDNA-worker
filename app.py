import os
from datetime import datetime
from typing import List, Optional

import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from rq import Queue
from rq.job import Job

# ---------------------------------------------------------------------
# Redis / RQ
# ---------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_r = redis.from_url(REDIS_URL)
_q = Queue("default", connection=_r)

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(title="editdna", version="1.0.0")


class NOPReq(BaseModel):
    pass


class RenderReq(BaseModel):
    session_id: str
    files: List[HttpUrl]
    output_prefix: Optional[str] = "editdna/outputs"  # currently unused (local /tmp output)


def _job_payload(job: Job):
    """Uniform job status payload."""
    status_map = {
        "queued": "queued",
        "started": "started",
        "deferred": "queued",
        "finished": "finished",
        "failed": "failed",
        "stopped": "failed",
        "canceled": "failed",
    }
    status = status_map.get(job.get_status(refresh=True) or "", "queued")

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


@app.get("/")
def health():
    return {"ok": True, "service": "editdna", "time": int(datetime.utcnow().timestamp())}


@app.post("/enqueue_nop")
def enqueue_nop(_: NOPReq | None = None):
    # Calls worker.task_nop (no args)
    job = _q.enqueue("worker.task_nop")
    return {"job_id": job.id}


@app.post("/render")
def render(req: RenderReq):
    # Enqueue worker.job_render(session_id, files, output_prefix)
    job = _q.enqueue(
        "worker.job_render",
        req.session_id,
        [str(u) for u in req.files],  # cast HttpUrl -> str so it's JSON-serializable in job meta
        req.output_prefix,
    )
    return {"job_id": job.id, "session_id": req.session_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=_r)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_payload(job)
