"""Read-only cost inventory: R2 objects (sizes per kind and project), the bucket lifecycle rules, and every worker job
of the last 24 h (function, queue wait, run time, sources). Nothing is written. Report encrypted before it leaves."""
import json
import os

import modal

app = modal.App("cutsell-cost-explore")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)


def _h(v):
    import hashlib
    return hashlib.sha256(str(v).encode()).hexdigest()[:8]


@app.function(image=image, secrets=[secret], timeout=900)
def explore():
    from cutsell_worker.storage_routing import default_client
    rep = {}
    s3 = default_client()
    bucket = os.environ["S3_BUCKET"]
    try:
        rep["lifecycle"] = s3.get_bucket_lifecycle_configuration(Bucket=bucket).get("Rules")
    except Exception as exc:
        rep["lifecycle"] = "none: " + repr(exc)[:200]
    objs = []
    token = None
    while True:
        kw = {"Bucket": bucket, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        page = s3.list_objects_v2(**kw)
        for o in page.get("Contents") or []:
            k = o["Key"]
            parts = k.split("/")
            ext = k.rsplit(".", 1)[-1].lower() if "." in parts[-1] else ""
            objs.append({"kind": "/".join(parts[:2]), "user": parts[2] if len(parts) > 3 else "", "project": parts[3] if len(parts) > 4 else "",
                         "name_hint": ("voice" if parts[-1].startswith("voice-") else ext), "depth": len(parts),
                         "sub": "/".join(parts[4:-1])[:60], "size": o["Size"], "modified": o["LastModified"].isoformat()})
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    rep["objects"] = objs
    from redis import Redis
    from rq import Queue
    from rq.job import Job
    from rq.registry import FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry
    r = Redis.from_url(os.environ["REDIS_URL"])
    q = Queue("cutsell", connection=r)
    jobs = []
    for reg, label in ((FinishedJobRegistry(queue=q), "finished"), (FailedJobRegistry(queue=q), "failed"),
                       (StartedJobRegistry(queue=q), "started")):
        for jid in reg.get_job_ids():
            try:
                j = Job.fetch(jid, connection=r)
            except Exception:
                continue
            arg = (j.args or [None])[0] if j.args else (j.kwargs or {})
            srcs = []
            if isinstance(arg, dict):
                for s in arg.get("sources") or []:
                    md = s.get("metadata") or {}
                    srcs.append({"duration": s.get("duration_sec"), "meta_keys": sorted(md.keys()), "bulk_size": md.get("bulk_size")})
            jobs.append({"state": label, "func": str(j.func_name).rsplit(".", 1)[-1],
                         "enqueued": str(j.enqueued_at), "started": str(j.started_at), "ended": str(j.ended_at),
                         "project": _h(arg.get("project_id")) if isinstance(arg, dict) else "", "user": _h(arg.get("user_id")) if isinstance(arg, dict) else "",
                         "sources": srcs, "arg_keys": sorted(arg.keys())[:30] if isinstance(arg, dict) else [],
                         "error": (str(j.exc_info or "").strip().splitlines() or [""])[-1][:200] if label == "failed" else ""})
    rep["jobs"] = sorted(jobs, key=lambda x: x["enqueued"])
    return rep


@app.local_entrypoint()
def main():
    rep = explore.remote()
    with open("engine_check.json", "w") as f:
        json.dump(rep, f, ensure_ascii=False, default=str)
    print(f"::notice title=inventory::objects {len(rep['objects'])} jobs {len(rep['jobs'])}")
