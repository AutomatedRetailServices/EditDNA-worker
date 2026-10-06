"""CutSell production worker on Modal (CPU only, pay per use, scales to zero).

The API on Render keeps doing exactly what it does today: it puts jobs on the RQ queue "cutsell"
in Redis. Nothing has to be listening all day. This app runs the SAME job functions the RQ worker
always ran (cutsell_worker.worker_job.run_flow_b_job, cutsell_worker.export_job.run_export_job,
cutsell_worker.batch_job.run_batch_item) -- it only changes who starts them:

  wake         HTTPS endpoint the API calls right after it enqueues a job (cutsell_worker/worker_wake.py).
               It starts one `drain_queue` container and returns at once.
  drain_queue  takes ONE job from the queue, runs it, exits. Several run side by side when several
               videos arrive together.
  sweep        every few minutes, starts a `drain_queue` for anything still waiting (a missed wake,
               a retry). Safety net only.

Deploy (from the repository root, with the Modal CLI logged in):

    modal deploy modal_cutsell_worker.py

Secrets: one Modal secret named "cutsell-worker" with
    REDIS_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET,
    ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, CUTSELL_WORKER_WAKE_TOKEN
    (+ DATABASE_URL / SENTRY_DSN if the API uses them)
The values are the same ones the Render API service has; they are never written in this file.

See docs/CUTSELL_MODAL_WORKER.md.
"""
from __future__ import annotations

import os

import modal

APP_NAME = "cutsell-worker"
SECRET_NAME = "cutsell-worker"
QUEUE_NAME = "cutsell"
JOB_TIMEOUT_SEC = 3600            # the API enqueues every job with this timeout (queueing.py)
MAX_PARALLEL_JOBS = 8             # ceiling on videos processed at the same time
SWEEP_EVERY_MINUTES = 5

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .env({
        "CUTSELL_ENGINE": "simple",
        "CUTSELL_EXPORT_DELIVERY_GATE": "technical_qc",
    })
    # The whole package directory, not add_local_python_source: the simple engine reads its
    # prompts from .txt files next to the code, and add_local_python_source ships only .py files.
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)

secret = modal.Secret.from_name(SECRET_NAME)


def _queue():
    from redis import Redis
    from rq import Queue

    connection = Redis.from_url(os.environ["REDIS_URL"])
    return Queue(QUEUE_NAME, connection=connection), connection


@app.function(
    image=image, secrets=[secret], cpu=2.0, memory=4096,
    timeout=JOB_TIMEOUT_SEC + 600, max_containers=MAX_PARALLEL_JOBS, retries=0,
)
def drain_queue() -> dict:
    """Run at most one queued job, exactly as `rq worker --burst` would, then exit."""
    from rq import Worker

    queue, connection = _queue()
    waiting = len(queue)
    if waiting == 0:
        return {"ran": False, "waiting_before": 0}
    worker = Worker([queue], connection=connection)
    worker.work(burst=True, max_jobs=1, with_scheduler=False)
    remaining = len(queue)
    if remaining:
        # A job can enqueue the next one itself (batches) and a retry re-queues without going
        # through the API's wake call: hand the rest to a fresh container instead of waiting
        # for the sweep.
        drain_queue.spawn()
    return {"ran": True, "waiting_before": waiting, "waiting_after": remaining}


@app.function(image=image, secrets=[secret], cpu=0.25, memory=256, timeout=60)
@modal.fastapi_endpoint(method="POST")
def wake(payload: dict) -> dict:
    """Called by the API after it enqueues a job. Body: {"token": "<CUTSELL_WORKER_WAKE_TOKEN>"}."""
    import hmac

    from fastapi import HTTPException

    expected = os.environ.get("CUTSELL_WORKER_WAKE_TOKEN", "")
    supplied = str((payload or {}).get("token") or "")
    if not expected or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="invalid wake token")
    drain_queue.spawn()
    return {"status": "started"}


@app.function(
    image=image, secrets=[secret], cpu=0.25, memory=256, timeout=120,
    schedule=modal.Period(minutes=SWEEP_EVERY_MINUTES),
)
def sweep() -> dict:
    """Start a worker for every job still waiting (bounded). Covers a wake call that never arrived."""
    queue, _ = _queue()
    waiting = len(queue)
    started = min(waiting, MAX_PARALLEL_JOBS)
    for _ in range(started):
        drain_queue.spawn()
    return {"waiting": waiting, "started": started}
