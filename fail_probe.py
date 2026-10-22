"""Read-only: the creator's cut jobs of the last 2 hours (status, stage, error, traceback tail) and her projects.
Nothing is written. Report encrypted."""
import json
import os

import modal

app = modal.App("cutsell-fail-probe")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
USER = "usr_25bb7b9fc4a54767a475fda9a9ccb6c9"


@app.function(image=image, secrets=[secret], timeout=600)
def check():
    import datetime as dt
    import pickle
    import zlib
    from redis import Redis
    from rq.job import Job
    from cutsell_worker.project_store import list_projects
    r = Redis.from_url(os.environ["REDIS_URL"])
    since = dt.datetime(2026, 10, 22, 20, 15, tzinfo=dt.timezone.utc)
    jobs = []
    for key in r.scan_iter("rq:job:*", count=2000):
        k = key.decode()
        if k.count(":") != 2:
            continue
        raw = r.hget(key, "meta")
        if not raw:
            continue
        try:
            try:
                meta = pickle.loads(raw)
            except Exception:
                meta = pickle.loads(zlib.decompress(raw))
        except Exception:
            continue
        if meta.get("user_id") != USER:
            continue
        try:
            job = Job.fetch(k.split(":")[-1], connection=r)
        except Exception:
            continue
        enq = job.enqueued_at
        if enq and enq.tzinfo is None:
            enq = enq.replace(tzinfo=dt.timezone.utc)
        if not enq or enq < since:
            continue
        exc = ""
        try:
            lr = job.latest_result()
            exc = (lr.exc_string if lr is not None else "") or ""
        except Exception:
            pass
        exc = exc or job.exc_info or ""
        payload = job.args[0] if job.args else {}
        names = [s.get("original_name") for s in (payload or {}).get("sources", [])] if isinstance(payload, dict) else []
        jobs.append({"id": job.id, "func": job.func_name, "status": str(job.get_status()), "enqueued": str(enq),
                     "started": str(job.started_at), "ended": str(job.ended_at), "names": names,
                     "meta": {k2: str(v)[:400] for k2, v in meta.items() if "token" not in k2},
                     "exc_tail": exc[-1800:]})
    jobs.sort(key=lambda j: j["enqueued"])
    projects = list_projects(user_id=USER, client=r, limit=15)
    keep = ("project_id", "state", "title", "updated_at", "created_at")
    return {"jobs": jobs, "projects": [{k: p.get(k) for k in keep} for p in projects]}


@app.local_entrypoint()
def main():
    with open("engine_check.json", "w") as f:
        json.dump(check.remote(), f, indent=1, default=str)
