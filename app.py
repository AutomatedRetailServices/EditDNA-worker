from fastapi import FastAPI, HTTPException
from redis import Redis
from rq import Queue
from rq.job import Job

app = FastAPI()

# Connect to Redis
redis_conn = Redis(host="redis", port=6379, decode_responses=True)
queue = Queue("default", connection=redis_conn)


# --- Healthcheck ---
@app.get("/admin/health")
def health():
    return {
        "ok": True,
        "registries": {
            "queued": len(queue.jobs),
            "started": len(queue.started_job_registry),
            "scheduled": len(queue.scheduled_job_registry),
            "deferred": len(queue.deferred_job_registry),
            "failed": len(queue.failed_job_registry),
            "finished": len(queue.finished_job_registry),
        },
    }


# --- Dummy Task ---
def nop_task():
    return "nop done"


# --- Enqueue a test job ---
@app.post("/enqueue_nop")
def enqueue_nop():
    job = queue.enqueue(nop_task)
    return {"ok": True, "job_id": job.id}


# --- Get job status ---
@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="unknown job_id")

    return {
        "job_id": job.id,
        "status": job.get_status(),
        "result": job.result,
        "error": str(job.exc_info) if job.exc_info else None,
    }
