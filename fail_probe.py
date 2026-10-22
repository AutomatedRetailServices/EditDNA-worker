"""Read-only: (1) her projects of 22 Oct (mode/sources), (2) storage inventory of the current bucket (R2) and the old
Amazon bucket: bytes by kind and by age. Lists only -- nothing is written or deleted. Report encrypted."""
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
def projects():
    from redis import Redis
    from cutsell_worker.project_store import list_projects
    r = Redis.from_url(os.environ["REDIS_URL"])
    out = []
    for p in list_projects(user_id=USER, client=r, limit=40):
        if str(p.get("created_at", ""))[:10] != "2026-10-22":
            continue
        row = {k: v for k, v in p.items() if not isinstance(v, (dict, list)) or k in ("sources", "metadata")}
        if "sources" in row:
            row["sources"] = [{k: s.get(k) for k in ("original_name", "duration_sec", "source_order", "metadata")} for s in row["sources"] or []]
        out.append(row)
    return out


def _kind(key):
    k = key.lower()
    for name in ("voice", "audio", "render", "export", "deliver", "overlay", "broll", "thumb", "proxy", "upload"):
        if name in k:
            return name
    return "/".join(key.split("/")[:2])


@app.function(image=image, secrets=[secret], timeout=1500)
def inventory():
    import datetime as dt
    from collections import defaultdict
    from cutsell_worker import storage_routing as sr
    now = dt.datetime.now(dt.timezone.utc)
    result = {}
    targets = [("current", sr.current_bucket(), sr.default_client())]
    fb = sr.fallback_storage()
    if fb:
        targets.append(("old_amazon", fb.bucket, sr._make_fallback_client(fb)))
    for label, bucket, client in targets:
        by_kind = defaultdict(lambda: [0, 0])
        by_age = defaultdict(lambda: [0, 0])
        by_prefix = defaultdict(lambda: [0, 0])
        total = [0, 0]
        try:
            for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
                for o in page.get("Contents", []):
                    size, key = int(o["Size"]), o["Key"]
                    age = (now - o["LastModified"]).days
                    band = "0-7d" if age < 7 else "7-30d" if age < 30 else "30-90d" if age < 90 else "90d+"
                    for d, k in ((by_kind, _kind(key)), (by_age, band), (by_prefix, "/".join(key.split("/")[:2]))):
                        d[k][0] += 1
                        d[k][1] += size
                    total[0] += 1
                    total[1] += size
            result[label] = {"bucket_hash": hash(bucket) % 10**6, "objects": total[0], "gb": round(total[1] / 1e9, 2),
                             "by_kind_gb": {k: [v[0], round(v[1] / 1e9, 2)] for k, v in by_kind.items()},
                             "by_age_gb": {k: [v[0], round(v[1] / 1e9, 2)] for k, v in by_age.items()},
                             "by_prefix_gb": {k: [v[0], round(v[1] / 1e9, 2)] for k, v in sorted(by_prefix.items(), key=lambda x: -x[1][1])[:15]}}
        except Exception as e:
            result[label] = f"{e.__class__.__name__}: {str(e)[:200]}"
    return result


@app.local_entrypoint()
def main():
    out = {}
    for name, fn in (("projects", projects), ("inventory", inventory)):
        try:
            out[name] = fn.remote()
        except Exception as e:
            out[name] = f"{e.__class__.__name__}: {str(e)[:300]}"
    with open("engine_check.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
