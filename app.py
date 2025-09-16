# app.py
import os
from datetime import datetime, timezone
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, AnyHttpUrl, Field
import redis
from rq import Queue
from rq.job import Job
from rq.registry import (
    StartedJobRegistry,
    ScheduledJobRegistry,
    DeferredJobRegistry,
    FailedJobRegistry,
    FinishedJobRegistry,
)

# ---------- Redis / RQ setup ----------
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

# IMPORTANT: keep decode_responses=False (bytes) to avoid .decode errors in RQ internals
conn = redis.from_url(REDIS_URL, decode_responses=False)
q = Queue("default", connection=conn)

# ---------- FastAPI ----------
app = FastAPI(title="editdna-web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Job functions (importable by the worker) ----------
def nop_job() -> dict:
    """Tiny job to prove queue <-> worker works."""
    return {
        "echo": "hello",
        "ts": datetime.now(timezone.utc).isoformat(),
    }

def check_urls_job(urls: List[str]) -> dict:
    """HEAD each URL and return basic info."""
    out = {"checked": []}
    timeout = httpx.Timeout(10.0, connect=10.0)
    headers = {"User-Agent": "editdna/1.0"}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        for url in urls:
            rec = {"url": url}
            try:
                r = client.head(url)
                rec["status"] = "OK" if r.is_success else "HTTP_ERROR"
                rec["http"] = r.status_code
                # Content-Length may be missing
                size = r.headers.get("content-length")
                rec["size"] = int(size) if size and size.isdigit() else None
            except Exception as e:
                rec["status"] = "REQUEST_ERROR"
                rec["error"] = str(e)
            out["checked"].append(rec)
    return out

# ---------- Request/Response models ----------
class EnqueueResp(BaseModel):
    ok: bool = True
    job_id: str

class URLsIn(BaseModel):
    urls: List[AnyHttpUrl] = Field(..., min_items=1, description="List of video URLs")

class JobResp(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None
    enqueued_at: Optional[str] = None
    ended_at: Optional[str] = None

# ---------- Endpoints ----------
@app.post("/enqueue_nop", response_model=EnqueueResp)
def enqueue_nop():
    job = q.enqueue(nop_job)
    return EnqueueResp(job_id=job.id)

@app.post("/process_urls", response_model=EnqueueResp)
def process_urls(payload: URLsIn):
    job = q.enqueue(check_urls_job, [str(u) for u in payload.urls])
    return EnqueueResp(job_id=job.id)

@app.get("/jobs/{job_id}", response_model=JobResp)
def get_job(job_id: str):
    try:
        job = Job.fetch(job_id, connection=conn)
    except Exception:
        raise HTTPException(status_code=404, detail="unknown_job_id")

    # RQ returns bytes timestamps; guard for missing attrs
    status = job.get_status(refresh=True) or "unknown"
    result = job.return_value() if job.is_finished else None
    error = (job.exc_info.decode() if isinstance(job.exc_info, (bytes, bytearray)) else job.exc_info) if job.is_failed else None

    def ts_to_iso(ts):
        # RQ stores datetime or bytes; normalize to ISO
        if hasattr(ts, "isoformat"):
            return ts.isoformat()
        if isinstance(ts, (bytes, bytearray)):
            try:
                return ts.decode()
            except Exception:
                return None
        return str(ts) if ts else None

    return JobResp(
        job_id=job.id,
        status=status,
        result=result,
        error=error,
        enqueued_at=ts_to_iso(job.enqueued_at),
        ended_at=ts_to_iso(job.ended_at),
    )

@app.get("/admin/health")
def health():
    queue = q  # default queue
    regs = {
        "queued": queue.count,
        "started": StartedJobRegistry("default", connection=conn).count,
        "scheduled": ScheduledJobRegistry("default", connection=conn).count,
        "deferred": DeferredJobRegistry("default", connection=conn).count,
        "failed": FailedJobRegistry("default", connection=conn).count,
        "finished": FinishedJobRegistry("default", connection=conn).count,
    }
    # materialize counts to ints
    regs = {k: int(v) for k, v in regs.items()}
    return {"ok": True, "registries": regs}

