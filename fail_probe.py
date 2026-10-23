"""Read-only: why did the 23 Oct 17:51 UTC cuts fail (RQ job error + the last failed jobs). No writes."""
import json

import modal

app = modal.App("cutsell-fail-probe")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)


@app.function(image=image, secrets=[secret], timeout=300)
def audit():
    import os
    from redis import Redis
    from rq.job import Job
    from rq.registry import FailedJobRegistry
    from rq import Queue
    r = Redis.from_url(os.environ["REDIS_URL"], socket_timeout=15)
    out = {}
    def job_info(jid):
        try:
            j = Job.fetch(jid, connection=r)
            exc = (j.exc_info or "")
            return {"func": j.func_name, "status": str(j.get_status()), "enqueued": str(j.enqueued_at), "started": str(j.started_at),
                    "ended": str(j.ended_at), "meta": {k: v for k, v in (j.meta or {}).items() if k in ("stage", "error_code", "progress")},
                    "exc_tail": exc[-2500:], "result_error": (j.result or {}).get("error") if isinstance(j.result, dict) else None}
        except Exception as e:
            return {"fetch_error": e.__class__.__name__ + ": " + str(e)[:200]}
    out["ca083ad4"] = job_info("ca083ad4-03cc-49f3-afe1-4df1d580e4f5")
    q = Queue("cutsell", connection=r)
    reg = FailedJobRegistry(queue=q)
    ids = reg.get_job_ids()[-12:]
    out["recent_failed"] = {}
    for jid in ids:
        info = job_info(jid)
        out["recent_failed"][jid] = {k: (v[-600:] if k == "exc_tail" else v) for k, v in info.items()}
    return out


@app.local_entrypoint()
def main():
    with open("engine_check.json", "w") as f:
        json.dump({"label": "why cuts failed 23 Oct 17:51", "rows": [audit.remote()]}, f, indent=1, default=str)
