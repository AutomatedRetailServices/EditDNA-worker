"""Read-only: why the last export jobs failed (RQ failed registry of the `cutsell` queue, last 3 hours):
state, meta error code, and the last lines of the exception. Nothing is changed. Report encrypted."""
import json
import os

import modal

app = modal.App("cutsell-expfail-probe")
secret = modal.Secret.from_name("cutsell-worker")
image = modal.Image.debian_slim(python_version="3.11").pip_install("redis", "rq")


@app.function(image=image, secrets=[secret], timeout=300)
def run():
    from datetime import datetime, timezone, timedelta
    from redis import Redis
    from rq import Queue
    from rq.job import Job
    from rq.registry import FailedJobRegistry
    r = Redis.from_url(os.environ["REDIS_URL"])
    out = []
    since = datetime.now(timezone.utc) - timedelta(hours=3)
    for name in ("cutsell", "cutsell-export", "export"):
        q = Queue(name, connection=r)
        reg = FailedJobRegistry(queue=q)
        for jid in reg.get_job_ids()[-60:]:
            try:
                job = Job.fetch(jid, connection=r)
            except Exception:
                continue
            ended = job.ended_at
            if ended and ended.replace(tzinfo=timezone.utc) < since:
                continue
            exc = (job.exc_info or "").strip().splitlines()
            out.append({"queue": name, "func": job.func_name, "ended": str(ended), "meta": {k: v for k, v in (job.meta or {}).items()
                        if k in ("error_code", "stage", "progress_percent", "error", "error_detail")},
                        "exc": exc[-12:]})
    return {"failed": out}


@app.local_entrypoint()
def main():
    with open("engine_check.json", "w") as f:
        json.dump(run.remote(), f, ensure_ascii=False, default=str)
