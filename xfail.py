"""Read-only: today's export jobs (failed and finished) with their error and timings. Report encrypted."""
import json
import os

import modal

app = modal.App("cutsell-export-failures")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)


@app.function(image=image, secrets=[secret], timeout=600)
def read():
    from redis import Redis
    from rq import Queue
    from rq.job import Job
    from rq.registry import FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry
    r = Redis.from_url(os.environ["REDIS_URL"])
    q = Queue("cutsell", connection=r)
    out = []
    for reg, label in ((FailedJobRegistry(queue=q), "failed"), (FinishedJobRegistry(queue=q), "finished"),
                       (StartedJobRegistry(queue=q), "started")):
        for jid in reg.get_job_ids():
            try:
                j = Job.fetch(jid, connection=r)
            except Exception:
                continue
            fn = str(j.func_name).rsplit(".", 1)[-1]
            if "export" not in fn and jid != "af0808a6-8a25-4fe9-97d4-6e96160a186f":
                continue
            res = None
            try:
                res = j.return_value() if label == "finished" else None
            except Exception:
                pass
            out.append({"id": jid, "func": fn, "state": label, "enqueued": str(j.enqueued_at), "started": str(j.started_at),
                        "ended": str(j.ended_at), "meta": {k: v for k, v in (j.meta or {}).items() if k in ("stage", "progress_percent", "timings", "error", "export_failed")},
                        "exc": "\n".join(str(j.exc_info or "").strip().splitlines()[-40:]),
                        "result": (json.dumps(res, default=str)[:3000] if res is not None else None)})
    return sorted(out, key=lambda x: x["enqueued"])


@app.local_entrypoint()
def main():
    rows = read.remote()
    with open("engine_check.json", "w") as f:
        json.dump({"jobs": rows}, f, ensure_ascii=False, default=str)
    print(f"::notice title=export jobs::{len(rows)}")
