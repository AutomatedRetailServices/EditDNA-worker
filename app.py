# app.py
import os, uuid, json, datetime as dt
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from redis import Redis
from rq import Queue
from rq.job import Job
from rq.registry import FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry, ScheduledJobRegistry, DeferredJobRegistry

# ---- SINGLE SOURCE OF TRUTH: Redis connection & Queue ----
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# IMPORTANT: decode_responses=True so keys/values are str, not bytes, and
# BOTH web app and worker MUST use the SAME REDIS_URL (including the /db index).
redis_conn = Redis.from_url(REDIS_URL, decode_responses=True)
q = Queue("default", connection=redis_conn)  # queue name must match your worker

app = FastAPI(title="editdna-web", version="1.0.0")

# ---- Models ----
class EnqueueNopResp(BaseModel):
    ok: bool = True
    job_id: str

class ProcessUrlsReq(BaseModel):
    session_id: Optional[str] = Field(None, description="If omitted, server creates one")
    urls: List[str]
    tone: Optional[str] = None
    product_link: Optional[str] = None
    features_csv: Optional[str] = None

class ProcessUrlsResp(BaseModel):
    ok: bool = True
    job_id: str
    session_id: str

class AnalyzeReq(BaseModel):
    session_id: str

class AnalyzeResp(BaseModel):
    ok: bool = True
    job_id: str

def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat()

# ---- Dummy worker functions (these names are what the worker imports) ----
# In your worker.py you should import these via: from app import task_nop, task_process_urls, task_analyze
def task_nop() -> Dict[str, Any]:
    return {"echo": {"hello": "world"}, "ts": _now_iso()}

def task_process_urls(session_id: str, urls: List[str], meta: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal “check” so you can see something real in /jobs
    checked = []
    for u in urls:
        # we don't download; just return a fake OK with a pseudo size
        size = 4_000_000 + (hash(u) % 50_000_000)
        checked.append({"url": u, "status": "OK", "http": 200, "size": size})
    return {"session_id": session_id, "checked": checked, "meta": meta, "ts": _now_iso()}

def task_analyze(session_id: str) -> Dict[str, Any]:
    # Reads what process_urls stored (if you choose to persist per-session)
    # For now just echo the session id.
    return {"session_id": session_id, "analysis": "done", "ts": _now_iso()}

# ---- API ----
@app.get("/admin/health")
def health():
    failed = FailedJobRegistry("default", connection=redis_conn)
    finished = FinishedJobRegistry("default", connection=redis_conn)
    started = StartedJobRegistry("default", connection=redis_conn)
    scheduled = ScheduledJobRegistry("default", connection=redis_conn)
    deferred = DeferredJobRegistry("default", connection=redis_conn)
    return {
        "ok": True,
        "registries": {
            "queued": q.count,
            "started": len(list(started.get_job_ids())),
            "scheduled": len(list(scheduled.get_job_ids())),
            "deferred": len(list(deferred.get_job_ids())),
            "failed": len(list(failed.get_job_ids())),
            "finished": len(list(finished.get_job_ids())),
        },
    }

@app.post("/enqueue_nop", response_model=EnqueueNopResp)
def enqueue_nop():
    job = q.enqueue(task_nop, job_timeout=600, result_ttl=600, failure_ttl=86400)
    return {"ok": True, "job_id": job.get_id()}

@app.post("/process_urls", response_model=ProcessUrlsResp)
def process_urls(req: ProcessUrlsReq):
    # make/keep a session_id
    session_id = req.session_id or str(uuid.uuid4())
    meta = {
        "tone": req.tone,
        "product_link": req.product_link,
        "features_csv": req.features_csv,
    }
    job = q.enqueue(
        task_process_urls,
        session_id,
        req.urls,
        meta,
        job_timeout=60 * 30,
        result_ttl=60 * 60,
        failure_ttl=60 * 60 * 24,
    )
    return {"ok": True, "job_id": job.get_id(), "session_id": session_id}

@app.post("/analyze", response_model=AnalyzeResp)
def analyze(req: AnalyzeReq):
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    job = q.enqueue(
        task_analyze,
        req.session_id,
        job_timeout=60 * 30,
        result_ttl=60 * 60,
        failure_ttl=60 * 60 * 24,
    )
    return {"ok": True, "job_id": job.get_id()}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="unknown_job_id")

    data = {
        "job_id": job.get_id(),
        "status": job.get_status(),
        "result": job.result,
        "error": job.exc_info or None,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
    }
    return data

# Debug helper: confirms we can see the job in Redis using the same connection
@app.get("/admin/echo_job/{job_id}")
def echo_job(job_id: str):
    key = f"rq:job:{job_id}"
    exists = redis_conn.exists(key)
    raw = redis_conn.hgetall(key) if exists else {}
    return {"exists": bool(exists), "redis_db": REDIS_URL, "raw": raw}
