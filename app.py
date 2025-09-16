# app.py
import os, re
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import redis
from rq import Queue
from rq.job import Job
from rq.registry import (FailedJobRegistry, FinishedJobRegistry,
                         StartedJobRegistry, ScheduledJobRegistry, DeferredJobRegistry)

# ----- Redis / RQ wiring -----
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

# VERY IMPORTANT: bytes mode to avoid decode crashes
conn = redis.from_url(REDIS_URL, decode_responses=False)
q = Queue("default", connection=conn)

# Worker functions
from worker import echo_nop, process_urls as worker_process_urls  # local import

app = FastAPI()


# ----- helpers -----
def extract_urls(payload: Dict[str, Any]) -> List[str]:
    """
    Accepts any of:
      { "urls": ["https://...mov", ...] }
      { "files": [{ "url": "https://..."}, ...] }
      { "text": "one or more URLs in a blob" }
    """
    urls: List[str] = []

    if isinstance(payload.get("urls"), list):
        urls = [str(u).strip() for u in payload["urls"] if str(u).strip()]
    elif isinstance(payload.get("files"), list):
        for f in payload["files"]:
            u = (f or {}).get("url") or (f or {}).get("source")  # tolerate "url" or "source"
            if u:
                urls.append(str(u).strip())
    elif isinstance(payload.get("text"), str):
        urls = re.findall(r"https?://\S+", payload["text"])
    else:
        # last resort: try to find any url-looking tokens in the raw body
        for k, v in payload.items():
            if isinstance(v, str) and v.startswith("http"):
                urls.append(v.strip())

    # de-dup while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def job_dict(job: Job) -> Dict[str, Any]:
    status = job.get_status(refresh=True)
    result: Optional[Any] = None
    error: Optional[str] = None

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
        "ended_at": (job.ended_at or job._ended_at).isoformat() if getattr(job, "ended_at", None) or getattr(job, "_ended_at", None) else None,
    }


# ----- endpoints -----

@app.post("/enqueue_nop")
def enqueue_nop():
    job = q.enqueue(echo_nop, {"hello": "world"})
    return {"ok": True, "job_id": job.id}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = Job.fetch(job_id, connection=conn)
    except Exception:
        raise HTTPException(status_code=404, detail="unknown_job_id")
    return job_dict(job)

@app.post("/process_urls")
async def process_urls_ep(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_json")

    urls = extract_urls(payload)
    if not urls:
        raise HTTPException(status_code=400, detail="no_urls_found")

    # enqueue real work
    job = q.enqueue(worker_process_urls, urls)

    # echo back what we queued
    return {
        "ok": True,
        "job_id": job.id,
        "files": [{"source": "url", "url": u} for u in urls],
    }

@app.get("/admin/health")
def admin_health():
    regs = {
        "queued": q.count,
        "started": len(StartedJobRegistry(queue=q, connection=conn)),
        "scheduled": len(ScheduledJobRegistry(queue=q, connection=conn)),
        "deferred": len(DeferredJobRegistry(queue=q, connection=conn)),
        "failed": len(FailedJobRegistry(queue=q, connection=conn)),
        "finished": len(FinishedJobRegistry(queue=q, connection=conn)),
    }
    return {"ok": True, "registries": regs}

@app.get("/")
def root():
    return {"ok": True, "service": "editdna-web"}
