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
        draft = (snap.get("undo") or [snap.get("draft")])[0] or snap.get("draft") or {}   # the cut as the AI made it
        sources = sorted(snap.get("sources") or [], key=lambda s: s.get("source_order") or 0)
        known = catalog.known_names(catalog.load(user_id, client=r))
        result = tagging.tag(draft, known, album_hint=tagging.album_hint_of(sources))  # text only, no video yet
        spent += (result.get("log") or {}).get("cost_usd") or 0
        row = {"brand": result.get("brand"), "product": result.get("product"), "confidence": result.get("confidence"),
               "needs_photo": result.get("needs_photo"), "used_photo": False}
        if write:
            catalog.record_ai_result(user_id, project_id, result, client=r)
            if result.get("needs_photo") and sources and sources[0].get("uri"):
                with tempfile.TemporaryDirectory() as d:
                    try:
                        video = download_source(sources[0]["uri"], str(Path(d) / "v.mp4"))
                        follow = tagging.photo_followup(user_id=user_id, project_id=project_id, video_path=video, client=r)
                        spent += ((follow.get("log") or {}).get("cost_usd") or 0)
                        row.update({"used_photo": True, "confidence": follow.get("confidence")})
                    except Exception as exc:
                        row["photo_error"] = exc.__class__.__name__
        rows.append(row)
    return {"videos": len(rows), "spent_usd": round(spent, 4), "written": write, "rows": rows}


@app.local_entrypoint()
def main():
    out = backfill.remote(WRITE, MAX_USD)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
