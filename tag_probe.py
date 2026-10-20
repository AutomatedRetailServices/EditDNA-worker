"""Read-only: brand/product detection (Haiku 5.5) on the creator's own stored cuts: the stored transcript, and the
original video from storage only when a photo is needed. Nothing is written to Redis or storage. Report encrypted."""
import json
import os
import time

import modal

app = modal.App("cutsell-tag-probe")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
REAL_USER = "1625a6e9"
MAX_VIDEOS = 20


@app.function(image=image, secrets=[secret], timeout=300)
def pick():
    import hashlib
    from redis import Redis
    r = Redis.from_url(os.environ["REDIS_URL"])
    rows = []
    for key in r.scan_iter("cutsell:v1:draft:*", count=500):
        try:
            snap = json.loads(r.get(key))
        except Exception:
            continue
        if hashlib.sha256(str(snap.get("user_id")).encode()).hexdigest()[:8] != REAL_USER:
            continue
        srcs = snap.get("sources") or []
        if not srcs or not srcs[0].get("uri"):
            continue
        rows.append({"key": key.decode(), "uri": srcs[0]["uri"], "dur": srcs[0].get("duration_sec"),
                     "saved": snap.get("saved_at") or ""})
    seen, out = set(), []
    for row in sorted(rows, key=lambda x: x["saved"], reverse=True):
        if row["uri"] in seen:
            continue
        seen.add(row["uri"])
        out.append(row)
    return out[:MAX_VIDEOS]


@app.function(image=image, secrets=[secret], timeout=600, cpu=1.0, memory=2048, max_containers=4)
def one(row):
    import tempfile
    from pathlib import Path
    from redis import Redis
    from cutsell_worker import product_tagging as T
    from cutsell_worker.storage import download_source
    r = Redis.from_url(os.environ["REDIS_URL"])
    snap = json.loads(r.get(row["key"]))
    draft = (snap.get("undo") or [snap.get("draft")])[0] or snap.get("draft") or {}
    transcript, first = T.transcript_lines(draft)
    text_result, info = T.classify_text(transcript, [])
    out = {"dur": row["dur"], "transcript_head": transcript[:600], "text": text_result, "text_usage": info["usage"],
           "text_error": info["error"]}
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        video = None
        if T.needs_photo(text_result):
            try:
                video = download_source(row["uri"], str(Path(d) / "v.mp4"))
            except Exception as exc:
                out["download_error"] = exc.__class__.__name__
        result = T.tag(draft, [], video_path=video, duration=row["dur"])
        if video:
            frames = T.grab_frames(video, T.moment_for_photo(text_result, first, float(row["dur"] or 10)), d)
            import base64
            out["frames_b64"] = [base64.b64encode(f).decode() for f in frames]
    out["final"] = {k: result.get(k) for k in ("brand", "product", "category", "confidence", "evidence", "used_photo",
                                               "needs_photo")}
    out["log"] = result.get("log")
    out["seconds"] = round(time.time() - t0, 1)
    return out


@app.local_entrypoint()
def main():
    rows = pick.remote()
    results = []
    for x in one.map(rows, return_exceptions=True, order_outputs=True):
        results.append(x if isinstance(x, dict) else {"error": repr(x)[:300]})
        with open("engine_check.json", "w") as f:
            json.dump({"videos": results}, f, ensure_ascii=False)
    print(f"::notice title=tag probe::{len(results)} videos")
