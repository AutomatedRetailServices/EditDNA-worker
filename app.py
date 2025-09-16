import os, json, uuid
import redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from rq import Queue, job as rq_job

REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

# IMPORTANT: bytes, not strings
redis_conn = redis.from_url(REDIS_URL, decode_responses=False)
q = Queue("default", connection=redis_conn)

app = FastAPI()

# ----- tiny worker task we can see run
def nop_task(payload: dict) -> dict:
    return {"echo": payload}

@app.post("/enqueue_nop")
def enqueue_nop():
    jid = str(uuid.uuid4())
    job = q.enqueue(nop_task, {"msg": "hello"}, job_id=jid)
    return {"ok": True, "job_id": job.id}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = rq_job.Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="unknown_job_id")
    data = {
        "job_id": job.id,
        "status": job.get_status(),
        "result": job.result if job.is_finished else None,
        "error": job.exc_info if job.is_failed else None,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
    }
    return JSONResponse(data)

@app.get("/admin/health")
def health():
    from rq.registry import (StartedJobRegistry, FinishedJobRegistry,
                             FailedJobRegistry, DeferredJobRegistry, ScheduledJobRegistry)
    r = {
        "queued": q.count,
        "started": StartedJobRegistry("default", connection=redis_conn).count,
        "scheduled": ScheduledJobRegistry("default", connection=redis_conn).count,
        "deferred": DeferredJobRegistry("default", connection=redis_conn).count,
        "failed": FailedJobRegistry("default", connection=redis_conn).count,
        "finished": FinishedJobRegistry("default", connection=redis_conn).count,
    }
    return {"ok": True, "registries": r}
