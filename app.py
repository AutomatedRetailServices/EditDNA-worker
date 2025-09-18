import os
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from rq import Queue
from rq.job import Job
from redis import Redis

# --- Redis / RQ ---
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(REDIS_URL)
q = Queue("default", connection=redis_conn)

# Import worker functions (module must be importable by both web & worker)
from worker import job_render, task_nop  # noqa: E402

# --- FastAPI app ---
app = FastAPI(title="editdna", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class RenderPayload(BaseModel):
    session_id: str = Field(..., example="sess-demo-001")
    files: list[HttpUrl] = Field(..., example=[
        "https://your-s3/clip1.mov",
        "https://your-s3/clip2.mov"
    ])
    output_prefix: str = Field("editdna/outputs")


# --- Routes ---

@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "editdna", "time": int(datetime.utcnow().timestamp())}


@app.post("/enqueue_nop")
def enqueue_nop() -> Dict[str, Any]:
    job = q.enqueue(task_nop)
    return {"job_id": job.id}


@app.post("/render")
def render(payload: RenderPayload = Body(...)) -> Dict[str, Any]:
    # Enqueue ONE dict arg so the worker signature matches
    job = q.enqueue(job_render, payload.dict())
    return {"job_id": job.id, "session_id": payload.session_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> Dict[str, Any]:
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get_status()  # queued | started | finished | failed | deferred
    result = job.result if status == "finished" else None
    error = None

    # For failed jobs, expose the traceback
    if status == "failed":
        try:
            error = job.exc_info or str(job.meta.get("error"))
        except Exception:
            error = "Unknown error"

    # Timestamps (may be None depending on state)
    enq = getattr(job, "enqueued_at", None)
    end = getattr(job, "ended_at", None)

    return {
        "job_id": job.id,
        "status": status,
        "result": result,
        "error": error,
        "enqueued_at": enq.isoformat() if enq else None,
        "ended_at": end.isoformat() if end else None,
    }
