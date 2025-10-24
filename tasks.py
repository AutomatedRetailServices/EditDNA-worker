# tasks.py — bridge between web API (Render) and worker job (RunPod)
import os, json, tempfile, requests
from rq import Queue
from redis import Redis
from jobs import job_render

# Connect to Redis (shared between web + worker)
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(redis_url)
q = Queue("default", connection=redis_conn)

def enqueue_render_job(payload: dict):
    """Enqueue a video render job."""
    s3_key = payload.get("s3_key") or payload.get("video_url")
    if not s3_key:
        raise ValueError("Missing s3_key or video_url in payload")

    # Enqueue background job
    job = q.enqueue(job_render, s3_key, job_timeout=1800)
    return {"job_id": job.id, "status": "queued"}

def job_status(job_id: str):
    """Fetch job status and results."""
    from rq.job import Job
    job = Job.fetch(job_id, connection=redis_conn)
    data = {"id": job.id, "status": job.get_status()}
    if job.is_finished:
        data["result"] = job.result
    elif job.is_failed:
        data["error"] = str(job.exc_info)
    return data
