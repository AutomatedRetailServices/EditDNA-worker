"""Storage operations approved by the Product Owner on 22 Oct (20:19):
 a) copy the test inputs (Gold / benchmarks) from the old Amazon bucket to R2 (same keys), verify sizes;
 b) copy her uploads (cutsell/uploads, exports, overlays, voice-overs) to R2 (same keys), verify sizes; Amazon keeps them;
 c) delete the MEDIA files of old test-run outputs on Amazon (cutsell/benchmarks, cutsell/serverless, editdna/outputs, ...),
    keeping every .json / text file.
storage_plan.json says which step runs. Deletion only with "confirm": true. The report lists every count and size."""
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
TEST_INPUTS = ("Editdna longform validation/", "Editdna good videos/", "Editdna bloopers videos/", "Yaskira/", "Bloop",
               "IMG_0", "editdna/training/", "editdna/models/", "editdna/benchmarks/", "editdna/sentence", "editdna/semantic",
               "cutsell/benchmark-inputs/")
USER = ("cutsell/uploads/", "cutsell/exports/", "cutsell/overlay-assets/", "cutsell/voice-over-uploads/")
OUTPUTS = ("cutsell/benchmarks/", "cutsell/serverless/", "editdna/outputs/", "cutsell/august-comparison/",
           "cutsell/legacy-comparison/", "cutsell/smokes/", "pruebas/")
MEDIA = (".mp4", ".mov", ".m4v", ".wav", ".m4a", ".mp3", ".webm", ".mkv", ".avi", ".aac", ".flac", ".ts")


def _clients():
    from cutsell_worker import storage_routing as sr
    fb = sr.fallback_storage()
    return sr._make_fallback_client(fb), fb.bucket, sr.default_client(), sr.current_bucket()


def _list(client, bucket, prefixes=None):
    out = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for o in page.get("Contents", []):
            if prefixes is None or o["Key"].startswith(prefixes):
                out[o["Key"]] = int(o["Size"])
    return out


@app.function(image=image, secrets=[secret], timeout=1800, cpu=1.0, memory=2048, max_containers=24)
def copy_one(key: str, size: int):
    old, old_bucket, r2, r2_bucket = _clients()
    try:
        if int(r2.head_object(Bucket=r2_bucket, Key=key)["ContentLength"]) == size:
            return {"key": key, "status": "already"}
    except Exception:
        pass
    body = old.get_object(Bucket=old_bucket, Key=key)
    r2.upload_fileobj(body["Body"], r2_bucket, key, ExtraArgs={"ContentType": body.get("ContentType") or "application/octet-stream"})
    got = int(r2.head_object(Bucket=r2_bucket, Key=key)["ContentLength"])
    return {"key": key, "status": "copied" if got == size else "SIZE_MISMATCH", "size": size, "r2_size": got}


@app.function(image=image, secrets=[secret], timeout=1800)
def plan_lists():
    old, old_bucket, r2, r2_bucket = _clients()
    return {"old": _list(old, old_bucket), "r2": _list(r2, r2_bucket)}


@app.function(image=image, secrets=[secret], timeout=1800)
def delete_keys(keys: list):
    old, old_bucket, _, _ = _clients()
    deleted, errors = 0, []
    for i in range(0, len(keys), 1000):
        chunk = keys[i:i + 1000]
        res = old.delete_objects(Bucket=old_bucket, Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True})
        errors += [e.get("Key") for e in res.get("Errors", [])]
        deleted += len(chunk) - len(res.get("Errors", []))
    return {"deleted": deleted, "errors": errors[:20], "error_count": len(errors)}


def _gb(n):
    return round(n / 1e9, 3)


@app.local_entrypoint()
def main():
    PLAN = json.load(open("storage_plan.json"))   # read here only: the containers don't have the file
    lists = plan_lists.remote()
    old, r2 = lists["old"], lists["r2"]
    report = {"plan": PLAN, "before": {"old_files": len(old), "old_gb": _gb(sum(old.values())), "r2_files": len(r2),
                                       "r2_gb": _gb(sum(r2.values()))}}
    for step in PLAN.get("copy", []):
        prefixes = TEST_INPUTS if step == "a" else USER
        wanted = {k: v for k, v in old.items() if k.startswith(prefixes) and not k.endswith("/")}
        todo = [(k, v) for k, v in wanted.items() if r2.get(k) != v]
        results = list(copy_one.starmap(todo, return_exceptions=True)) if todo else []
        bad = [r if isinstance(r, dict) else {"error": str(r)[:200]} for r in results
               if not isinstance(r, dict) or r.get("status") not in ("copied", "already")]
        r2_after = plan_lists.remote()["r2"]
        missing = [k for k, v in wanted.items() if r2_after.get(k) != v]
        report[f"copy_{step}"] = {"files": len(wanted), "gb": _gb(sum(wanted.values())), "copied_now": len(todo),
                                  "problems": bad[:20], "missing_or_different_after": missing[:50],
                                  "complete": not missing}
        r2 = r2_after
    if "c" in PLAN.get("delete", []):
        media = sorted(k for k in old if k.startswith(OUTPUTS) and k.lower().endswith(MEDIA))
        kept = [k for k in old if k.startswith(OUTPUTS) and not k.lower().endswith(MEDIA)]
        report["delete_c"] = {"media_files": len(media), "media_gb": _gb(sum(old[k] for k in media)),
                              "kept_files": len(kept), "kept_gb": _gb(sum(old[k] for k in kept)),
                              "sample": media[:15]}
        if PLAN.get("confirm") is True:
            report["delete_c"]["result"] = delete_keys.remote(media)
    if "a" in PLAN.get("delete", []):
        wanted = {k: v for k, v in old.items() if k.startswith(TEST_INPUTS) and not k.endswith("/")}
        missing = [k for k, v in wanted.items() if r2.get(k) != v]
        report["delete_a"] = {"files": len(wanted), "gb": _gb(sum(wanted.values())), "not_safe_in_r2": missing[:20]}
        if PLAN.get("confirm") is True and not missing:
            report["delete_a"]["result"] = delete_keys.remote(sorted(wanted))
    after = plan_lists.remote()
    report["after"] = {"old_files": len(after["old"]), "old_gb": _gb(sum(after["old"].values())),
                       "r2_files": len(after["r2"]), "r2_gb": _gb(sum(after["r2"].values()))}
    with open("engine_check.json", "w") as f:
        json.dump(report, f, indent=1, default=str)
