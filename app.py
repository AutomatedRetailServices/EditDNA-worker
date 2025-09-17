# app.py
import os, json, datetime as dt
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import redis
from rq import Queue, job as rq_job

# ---- Redis & RQ (MUST be identical for web & worker services) ----
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(REDIS_URL)
q = Queue("default", connection=r)  # use same queue name "default"

# ---- import the functions by name (callables) from worker.py ----
from worker import task_nop, check_urls, analyze_session

app = FastAPI()

# ------------ Pydantic payloads ------------
class ProcessURLsPayload(BaseModel):
    session_id: str | None = None
    urls: list[str]

class AnalyzePayload(BaseModel):
    session_id: str
    tone: str | None = None
    product_link: str | None = None
    features_csv: str | None = None

# ------------ Helpers ------------
def serialize_job(j: rq_job.Job):
    return {
        "job_id": j.id,
        "status": j.get_status(),
        "result": j.result,
        "error": j.meta.get("error") if hasattr(j, "meta") else None,
        "enqueued_at": j.enqueued_at.isoformat() if j.enqueued_at else None,
        "ended_at": j.ended_at.isoformat() if j.ended_at else None,
    }

# ------------ Routes ------------
@app.get("/")
def root():
    return {"ok": True, "service": "editdna-web", "queue": q.name}

@app.get("/admin/health")
def health():
    # show simple registry counts to know worker is connected to same Redis
    from rq.registry import FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry, ScheduledJobRegistry, DeferredJobRegistry
    def count(reg):
        try:
            return len(reg.get_job_ids())
        except Exception:
            return -1
    return {
        "ok": True,
        "registries": {
            "queued": q.count,
            "started": count(StartedJobRegistry(q)),
            "scheduled": count(ScheduledJobRegistry(q)),
            "deferred": count(DeferredJobRegistry(q)),
            "failed": count(FailedJobRegistry(q)),
            "finished": count(FinishedJobRegistry(q)),
        }
    }

@app.post("/enqueue_nop")
def enqueue_nop():
    job = q.enqueue(task_nop)          # <— callable, NOT a string
    return {"ok": True, "job_id": job.id}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    j = rq_job.Job.fetch(job_id, connection=r)
    if not j:
        raise HTTPException(status_code=404, detail="unknown_job_id")
    return serialize_job(j)

@app.post("/process_urls")
async def process_urls(req: Request):
    data = await req.json()
    payload = ProcessURLsPayload(**data)
    job = q.enqueue(check_urls, payload.model_dump())
    # return the session_id so you can reuse it in /analyze later
    return {"ok": True, "job_id": job.id, "session_id": payload.session_id or ""}

@app.post("/analyze")
async def analyze(req: Request):
    data = await req.json()
    payload = AnalyzePayload(**data)
    job = q.enqueue(analyze_session, payload.model_dump())
    return {"ok": True, "job_id": job.id}
