"""Plan my week on the creator's real Ready videos (read-only): the week the rules build, the rules checked, and the
posting caption Haiku writes for 12 of her videos (not stored). Report encrypted (no keys)."""
import json
import os

import modal

app = modal.App("cutsell-plan-probe")
secret = modal.Secret.from_name("cutsell-worker")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
         .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"]))
REAL_USER = "1625a6e9"


@app.function(image=image, secrets=[secret], timeout=900)
def probe():
    import hashlib
    from datetime import date
    from concurrent.futures import ThreadPoolExecutor
    from redis import Redis
    from cutsell_worker import post_captions, post_plan, product_catalog
    from cutsell_worker.draft_store import get_draft_snapshot
    from cutsell_worker.project_store import list_projects
    r = Redis.from_url(os.environ["REDIS_URL"])
    user = None
    for key in r.scan_iter("cutsell:v1:draft:*", count=500):
        snap = json.loads(r.get(key))
        if hashlib.sha256(str(snap.get("user_id")).encode()).hexdigest()[:8] == REAL_USER:
            user = snap["user_id"]
            break
    projects = list_projects(user_id=user, limit=500, client=r)
    videos = post_plan.candidates(user, projects, client=r)
    report = {"ready_videos": len(videos), "by_source": {}, "products": len({post_plan._product_key(v) for v in videos})}
    for v in videos:
        report["by_source"][v["source"]] = report["by_source"].get(v["source"], 0) + 1
    prefs = dict(post_plan.DEFAULT_PREFS)
    week = post_plan.build(videos, prefs, date(2026, 10, 19))
    by_id = {v["project_id"]: v for v in videos}
    report["week"] = [{"date": s["date"], "time": s["time"], "product": by_id[s["project_id"]].get("product"),
                       "brand": by_id[s["project_id"]].get("brand"), "source": by_id[s["project_id"]]["source"],
                       "mix_code": by_id[s["project_id"]].get("mix_code")} for s in week["slots"]]
    report["missing"] = week["missing"]
    report["rule_problems"] = post_plan.check(week["slots"], videos)
    catalog = product_catalog.load(user, client=r)
    picked = [s["project_id"] for s in week["slots"]][:12]

    def one(pid):
        draft = get_draft_snapshot(user_id=user, project_id=pid, client=r).get("draft") or {}
        v = by_id[pid]
        spoken = post_captions.spoken_text(draft)
        out = post_captions.write(spoken, brand=v.get("brand"), product=v.get("product"))
        return {"product": f"{v.get('brand')} · {v.get('product')}", "spoken": spoken[:300], **out}
    with ThreadPoolExecutor(max_workers=4) as pool:
        report["captions"] = list(pool.map(one, picked))
    costs = [c["log"]["cost_usd"] for c in report["captions"] if c["log"].get("cost_usd") is not None]
    report["caption_cost_avg"] = round(sum(costs) / max(1, len(costs)), 6)
    report["caption_cost_total"] = round(sum(costs), 6)
    return report


@app.local_entrypoint()
def main():
    out = probe.remote()
    with open("engine_check.json", "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("::notice title=plan probe::done")
