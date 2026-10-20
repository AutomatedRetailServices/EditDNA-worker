"""Backfill: classify the cuts that already exist (stored transcripts; a photo only when needed) into Edits · Products.

    BACKFILL_WRITE=0 modal run scripts/backfill_product_tags.py   # dry run (default): prints what it would do
    BACKFILL_WRITE=1 modal run scripts/backfill_product_tags.py   # writes the catalog (after the server is deployed)

Never overrides a video the creator placed by hand; skips videos already classified. Stops at BACKFILL_MAX_USD.
"""
import json
import os

import modal

app = modal.App("cutsell-product-backfill")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
WRITE = os.environ.get("BACKFILL_WRITE", "0") == "1"
MAX_USD = float(os.environ.get("BACKFILL_MAX_USD", "2.0"))


@app.function(image=image, secrets=[secret], timeout=3600, cpu=1.0, memory=2048)
def backfill(write: bool, max_usd: float):
    import tempfile
    from pathlib import Path
    from redis import Redis
    from cutsell_worker import product_catalog as catalog, product_tagging as tagging
    from cutsell_worker.storage import download_source
    r = Redis.from_url(os.environ["REDIS_URL"])
    spent, rows = 0.0, []
    for key in r.scan_iter("cutsell:v1:draft:*", count=500):
        if spent >= max_usd:
            rows.append({"stopped": "budget"})
            break
        try:
            snap = json.loads(r.get(key))
        except Exception:
            continue
        user_id, project_id = snap.get("user_id"), snap.get("project_id")
        if not user_id or not project_id:
            continue
        entry = catalog.video_entry(user_id, project_id, client=r) or {}
        if entry.get("source") == "user" or entry.get("suggestion"):
            continue                                            # placed by hand, or already classified
        draft = (snap.get("undo") or [snap.get("draft")])[0] or snap.get("draft") or {}
        sources = snap.get("sources") or []
        known = catalog.known_names(catalog.load(user_id, client=r))
        with tempfile.TemporaryDirectory() as d:
            text, info = tagging.classify_text(tagging.transcript_lines(draft)[0], known, tagging.album_hint_of(sources))
            video = None
            if tagging.needs_photo(text) and sources and sources[0].get("uri"):
                try:
                    video = download_source(sources[0]["uri"], str(Path(d) / "v.mp4"))
                except Exception:
                    video = None
            result = tagging.tag(draft, known, video_path=video, duration=sources[0].get("duration_sec") if sources else None,
                                 album_hint=tagging.album_hint_of(sources))
        spent += (result.get("log") or {}).get("cost_usd") or 0
        if write:
            catalog.record_ai_result(user_id, project_id, result, client=r)
        rows.append({"brand": result.get("brand"), "product": result.get("product"), "confidence": result.get("confidence"),
                     "used_photo": result.get("used_photo")})
    return {"videos": len(rows), "spent_usd": round(spent, 4), "written": write, "rows": rows}


@app.local_entrypoint()
def main():
    out = backfill.remote(WRITE, MAX_USD)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
