import os
import logging
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis import Redis
from rq import Queue, Job

from pipeline import run_pipeline

# =====================
# LOGGING
# =====================

logger = logging.getLogger("editdna.tasks")
logger.setLevel(logging.INFO)

# =====================
# REDIS / RQ
# =====================

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
REDIS_QUEUE_NAME = os.environ.get("REDIS_QUEUE_NAME", "editdna")

redis_conn = Redis.from_url(REDIS_URL)
queue = Queue(REDIS_QUEUE_NAME, connection=redis_conn)

# =====================
# FASTAPI
# =====================

app = FastAPI()


class RenderRequest(BaseModel):
    session_id: str
    files: Optional[List[str]] = None
    file_urls: Optional[List[str]] = None
    mode: Literal["human", "clean", "blooper"] = "human"


class JobResponse(BaseModel):
    job_id: str
    status: str


@app.post("/render", response_model=JobResponse)
def create_render_job(req: RenderRequest):
    """
    Enqueue a render job.
    VERY IMPORTANT: we forward req.mode directly to the worker.
    """
    # Normalize mode, but DO NOT ignore what user sent
    mode = (req.mode or "human").lower()
    if mode not in ("human", "clean", "blooper"):
        mode = "human"

    files = req.files or req.file_urls
    if not files:
        raise HTTPException(
            status_code=400,
            detail="You must provide 'files' or 'file_urls'",
        )

    logger.info(
        f"/render called: session_id={req.session_id} "
        f"mode={mode} raw_body={req.dict()}"
    )

    job: Job = queue.enqueue(
        "tasks.run_pipeline_job",
        kwargs={
            "session_id": req.session_id,
            "files": files,
            "file_urls": None,
            "mode": mode,
        },
        job_timeout=60 * 30,
        result_ttl=60 * 60 * 24,
    )

    logger.info(
        f"Enqueued job id={job.id} "
        f"session_id={req.session_id} mode={mode}"
    )

    return JobResponse(job_id=job.id, status="queued")


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """
    Return job status + result.
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "id": job.id,
        "status": job.get_status(),
        "enqueued_at": job.enqueued_at,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
        "result": job.result,
        "meta": job.meta,
    }


# =====================
# WORKER ENTRYPOINT
# =====================

def run_pipeline_job(
    session_id: str,
    files: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
    mode: str = "human",
) -> Dict[str, Any]:
    """
    This is what the RQ worker actually runs.
    It MUST pass mode straight to pipeline.run_pipeline.
    """
    mode = (mode or "human").lower()
    if mode not in ("human", "clean", "blooper"):
        mode = "human"

    logger.info(
        f"run_pipeline_job: session_id={session_id} "
        f"mode={mode} files={files} file_urls={file_urls}"
    )

    result = run_pipeline(
        session_id=session_id,
        files=files,
        file_urls=file_urls,
        mode=mode,
    )

    logger.info(
        "run_pipeline_job finished: "
        f"composer_mode={result.get('composer_mode')} "
        f"composer.mode={result.get('composer', {}).get('mode')}"
    )

    return result
