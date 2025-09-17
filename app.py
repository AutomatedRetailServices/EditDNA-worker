# app.py
import os
from typing import List, Dict, Any

from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from redis import from_url as redis_from_url
from rq import Queue
from rq.job import Job  # <-- import Job from rq.job (not from rq)
import datetime as dt

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# single Redis connection shared by API and worker
redis_conn = redis_from_url(REDIS_URL, decode_responses=False)  # keep bytes for RQ
q = Queue("default", connection=redis_conn)

app = FastAPI(title="editdna-web minimal queue API")

# ---------- Schemas ----------
class ProcessUrlsIn(BaseModel):
    urls: List[str]

# ---------- Small utilities ----------
def job_to_dict(job: Job) -> Dict[str, Any]:
    """Return a compact, safe JSON for a job."""
    status = job.get_status()
    meta = job.meta or {}
    enq = job.enqueued_at.isoformat() if job.enqueued_at else None
    end = job.ended_at.isoformat() if job.ended_at else None
    # When failed, job.exc_info holds traceback; we just surface the message briefly
    err = None
    if status == "failed":
        err = (job.exc_info or "").splitlines()[-1] if job.exc_info else "unknown error"
    return {
        "job_id": job.id,
        "status": status,
        "result": job.result,
        "error": err,
        "enqueued_at": enq,
        "ended_at": end,
        **({} if not meta else {"meta": meta}),
    }

# ---------- Endpoints ----------
@app.get("/admin/health")
def health():
    """Quick view of RQ registries."""
    from rq.registry import (
        StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry,
        DeferredJobRegistry, ScheduledJobRegistry
    )
    regs = {
        "queued": q.count,
        "started": StartedJobRegistry(queue=q).count,
        "scheduled": ScheduledJobRegistry(queue=q).count,
        "deferred": DeferredJobRegistry(queue=q).count,
        "failed": FailedJobRegistry(queue=q).count,
        "finished": FinishedJobRegistry(queue=q).count,
    }
    # evaluate the callables to ints
    regs = {k: (v() if callable(v) else v) for k, v in regs.items()}
    return {"ok": True, "registries": regs}

@app.post("/enqueue_nop")
def enqueue_nop():
    """Simple 'hello world' job to prove queue-worker works."""
    # import inside function so worker can import it too
    from worker import hello_world
    job = q.enqueue(hello_world)
    return {"ok": True, "job_id": job.id}

@app.post("/process_urls")
def process_urls(payload: ProcessUrlsIn = Body(...)):
    """Queue a job that validates the given URLs (HTTP 200 + size)."""
    if not payload.urls:
        raise HTTPException(status_code=400, detail="No urls provided")
    from worker import check_urls  # named function (no lambdas)
    job = q.enqueue(check_urls, payload.urls)
    return {"ok": True, "job_id": job.id}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="unknown_job_id")
    return job_to_dict(job)

@app.get("/")
def root():
    return {"ok": True, "at": dt.datetime.utcnow().isoformat() + "Z"}
