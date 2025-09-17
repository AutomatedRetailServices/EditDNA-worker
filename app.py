# app.py
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Literal, Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from redis import Redis
from rq import Queue
from rq.job import Job  # important: import Job from rq.job, not rq

# --------------------------------------------------------------------------------------
# Infrastructure
# --------------------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# decode_responses=True so we get strings out (prevents .decode() crashes you saw)
redis_conn: Redis = Redis.from_url(REDIS_URL, decode_responses=True)
q: Queue = Queue(connection=redis_conn)

app = FastAPI(title="editdna-web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Small helpers
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

def job_to_dict(job: Job) -> Dict[str, Any]:
    status = job.get_status(refresh=True)
    result = None
    if status == "finished":
        # worker should return JSON-serializable results
        result = job.result

    error_text = None
    if status == "failed" and job.exc_info:
        error_text = job.exc_info.splitlines()[-1].strip()

    enq = None
    if job.enqueued_at:
        enq = job.enqueued_at.replace(tzinfo=timezone.utc).isoformat()

    end = None
    if job.ended_at:
        end = job.ended_at.replace(tzinfo=timezone.utc).isoformat()

    return {
        "job_id": job.id,
        "status": status,
        "result": result,
        "error": error_text,
        "enqueued_at": enq,
        "ended_at": end,
    }

# --------------------------------------------------------------------------------------
# Pydantic payloads
# --------------------------------------------------------------------------------------

class EnqueueNopPayload(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict)

class ProcessUrlsPayload(BaseModel):
    # You can optionally pass your own session_id; if omitted, we generate one
    session_id: Optional[str] = None
    urls: List[str]
    tone: Optional[Literal["casual", "friendly", "energetic", "professional"]] = None
    product_link: Optional[str] = None
    features_csv: Optional[str] = None  # e.g. "durable, waterproof, lightweight"

class AnalyzePayload(BaseModel):
    session_id: str

# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------

@app.get("/admin/health")
def health():
    """Quick health endpoint + RQ registry counts."""
    try:
        from rq.registry import FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry
        failed = FailedJobRegistry(queue=q).count
        finished = FinishedJobRegistry(queue=q).count
        started = StartedJobRegistry(queue=q).count
        queued = q.count
        return {"ok": True, "registries": {
            "queued": queued,
            "started": started,
            "scheduled": 0,
            "deferred": 0,
            "failed": failed,
            "finished": finished
        }}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/enqueue_nop")
def enqueue_nop(body: EnqueueNopPayload):
    """
    Tiny test task so you can sanity-check the pipeline.
    The worker function is worker.nop (must exist in worker.py).
    """
    job = q.enqueue("worker.nop", body.payload)
    return {"ok": True, "job_id": job.id}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Poll a job by id."""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="unknown_job_id")
    return job_to_dict(job)

@app.post("/process_urls")
def process_urls(body: ProcessUrlsPayload):
    """
    Accepts a list of URLs and optional metadata. Creates/records a session,
    enqueues the process job, and returns both job_id and session_id.
    """
    if not body.urls:
        raise HTTPException(status_code=400, detail="urls_required")

    session_id = body.session_id or str(uuid.uuid4())

    # Store the session payload so /analyze can find it later
    # Keys:
    #   sess:{sid}:urls           -> JSON list of urls
    #   sess:{sid}:meta           -> JSON dict of extra fields
    #   sess:{sid}:last_process   -> last job id for process step (optional)
    redis_conn.hset(
        f"sess:{session_id}:meta",
        mapping={
            "tone": body.tone or "",
            "product_link": body.product_link or "",
            "features_csv": body.features_csv or "",
            "created_at": utcnow_iso(),
        },
    )
    # Use Redis JSON-ish storage via string; pydantic already validated list
    import json
    redis_conn.set(f"sess:{session_id}:urls", json.dumps(body.urls))

    # Enqueue worker process (function must exist in worker.py)
    job = q.enqueue(
        "worker.process_urls",
        body.urls,
        {
            "session_id": session_id,
            "tone": body.tone,
            "product_link": body.product_link,
            "features_csv": body.features_csv,
        },
    )
    redis_conn.set(f"sess:{session_id}:last_process", job.id)

    return {"ok": True, "job_id": job.id, "session_id": session_id}

@app.post("/analyze")
def analyze(body: AnalyzePayload):
    """
    Kicks off analysis for a previously created session (from /process_urls).
    """
    sid = body.session_id.strip()
    if not redis_conn.exists(f"sess:{sid}:urls"):
        raise HTTPException(status_code=404, detail="session_not_found")

    # Enqueue worker analysis (function must exist in worker.py)
    job = q.enqueue("worker.analyze_session", sid)
    # Optionally remember the last analyze job
    redis_conn.set(f"sess:{sid}:last_analyze", job.id)

    return {"ok": True, "job_id": job.id, "session_id": sid}

# Root is optional; returning 404 by default is fine, but a greeting helps
@app.get("/")
def root():
    return {"ok": True, "service": "editdna-web", "time": utcnow_iso()}
