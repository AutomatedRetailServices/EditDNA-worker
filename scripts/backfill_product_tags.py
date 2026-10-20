"""Backfill: classify the cuts that already exist (stored transcripts; a photo only when needed) into Edits · Products.

    BACKFILL_WRITE=0 modal run scripts/backfill_product_tags.py   # dry run (default): prints what it would do
    BACKFILL_WRITE=1 modal run scripts/backfill_product_tags.py   # writes the catalog (after the server is deployed)

Never overrides a video the creator placed by hand; resumable (a video already classified only gets its pending photo step). 20 videos at a time; at most ~$0.001 per video.
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


@app.function(image=image, secrets=[secret], timeout=600)
def keys() -> list[str]:
    from redis import Redis
    r = Redis.from_url(os.environ["REDIS_URL"])
    return [k.decode() for k in r.scan_iter("cutsell:v1:draft:*", count=500)]


@app.function(image=image, secrets=[secret], timeout=600, cpu=1.0, memory=2048, max_containers=20)
def one(key: str, write: bool) -> dict:
    """One cut: the text step (if never classified), then the photo step (if it asked for one). Resumable."""
    import tempfile
    from pathlib import Path
    from redis import Redis
    from cutsell_worker import product_catalog as catalog, product_tagging as tagging
    from cutsell_worker.storage import download_source
    r = Redis.from_url(os.environ["REDIS_URL"])
    try:
        snap = json.loads(r.get(key))
    except Exception:
        return {"skip": "unreadable"}
    user_id, project_id = snap.get("user_id"), snap.get("project_id")
    if not user_id or not project_id:
        return {"skip": "no_ids"}
    entry = catalog.video_entry(user_id, project_id, client=r) or {}
    if entry.get("source") == "user":
        return {"skip": "placed_by_hand"}
    sources = sorted(snap.get("sources") or [], key=lambda s: s.get("source_order") or 0)
    spent, row = 0.0, {}
    suggestion = entry.get("suggestion")
    if not suggestion and not entry.get("product_id"):
        draft = (snap.get("undo") or [snap.get("draft")])[0] or snap.get("draft") or {}
        known = catalog.known_names(catalog.load(user_id, client=r))
        result = tagging.tag(draft, known, album_hint=tagging.album_hint_of(sources))
        spent += (result.get("log") or {}).get("cost_usd") or 0
        if write:
            catalog.record_ai_result(user_id, project_id, result, client=r)
        suggestion = result
    row.update({"brand": suggestion.get("brand"), "product": suggestion.get("product"),
                "confidence": suggestion.get("confidence"), "used_photo": bool(suggestion.get("used_photo"))})
    if write and suggestion.get("needs_photo") and sources and sources[0].get("uri"):
        with tempfile.TemporaryDirectory() as d:
            try:
                video = download_source(sources[0]["uri"], str(Path(d) / "v.mp4"))
                follow = tagging.photo_followup(user_id=user_id, project_id=project_id, video_path=video, client=r)
                spent += ((follow.get("log") or {}).get("cost_usd") or 0)
                after = (catalog.video_entry(user_id, project_id, client=r) or {}).get("suggestion") or {}
                row.update({"used_photo": follow.get("status") == "done", "brand": after.get("brand"),
                            "product": after.get("product"), "confidence": after.get("confidence")})
            except Exception as exc:
                row["photo_error"] = exc.__class__.__name__
    row["spent_usd"] = spent
    return row


@app.local_entrypoint()
def main():
    all_keys = keys.remote()
    rows = list(one.map(all_keys, kwargs={"write": WRITE}, return_exceptions=True))
    rows = [r if isinstance(r, dict) else {"error": str(r)[:200]} for r in rows]
    out = {"videos": len(rows), "spent_usd": round(sum(r.get("spent_usd") or 0 for r in rows), 4), "written": WRITE,
           "rows": rows}
    with open("engine_check.json", "w") as handle:
        json.dump(out, handle)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
