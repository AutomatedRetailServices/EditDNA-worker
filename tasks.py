# app.py  — EditDNA web API (Render)
import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import redis
from rq import Queue, Job

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

redis_conn = redis.from_url(REDIS_URL)
queue = Queue("default", connection=redis_conn)

app = FastAPI(title="EditDNA Web API")


# ---------- Models ----------

class RenderRequest(BaseModel):
    session_id: str
    files: List[str]


class RenderEnqueueResponse(BaseModel):
    ok: bool
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    ok: bool
    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


# ---------- Routes ----------

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/render", response_model=RenderEnqueueResponse)
def render(req: RenderRequest):
    """
    Enqueue a render job into RQ.
    Worker will run tasks.job_render(session_id=..., files=[...])
    """
    job: Job = queue.enqueue(
        "tasks.job_render",  # resolved in the worker container
        kwargs={
            "session_id": req.session_id,
            "files": req.files,
        },
    )
    return RenderEnqueueResponse(ok=True, job_id=job.id, status=job.get_status())


@app.get("/job/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str):
    job = Job.fetch(job_id, connection=redis_conn)
    if job.is_failed:
        return JobStatusResponse(
            ok=False,
            job_id=job.id,
            status=job.get_status(),
            result=None,
            error=str(job.exc_info),
        )
    return JobStatusResponse(
        ok=True,
        job_id=job.id,
        status=job.get_status(),
        result=job.result,
        error=None,
    )
