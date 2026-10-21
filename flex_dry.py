"""Read-only: (1) what production (deployed 72ecfa5) shows her today in Plan; (2) Goal v2 on her real library with the
new code (feat/plan-flexible): her usable videos, the live summary and the plan the rules build -- nothing stored.
The two app states (before / after Build) are returned for the simulator. Report encrypted."""
import json
import os

import modal

app = modal.App("cutsell-flex-dry")
secret = modal.Secret.from_name("cutsell-worker")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
         .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"]))
REAL_USER = "1625a6e9"


@app.function(image=image, secrets=[secret], timeout=300)
def run():
    import hashlib
    from datetime import datetime, timezone, timedelta
    from redis import Redis
    from cutsell_worker import post_plan, project_store
    r = Redis.from_url(os.environ["REDIS_URL"])
    user = None
    for key in r.scan_iter("cutsell:v1:draft:*", count=500):
        snap = json.loads(r.get(key))
        if hashlib.sha256(str(snap.get("user_id")).encode()).hexdigest()[:8] == REAL_USER:
            user = snap["user_id"]
            break
    projects = project_store.list_projects(user_id=user, limit=500, client=r)
    plan = post_plan.load(user, client=r)
    now = datetime.now(timezone.utc)
    current = post_plan.view(user, projects, client=r, now=now)        # what her app gets today (new code, read-only)
    report = {"now_summary": current["summary"], "now_week_slots": len((current.get("week") or {}).get("slots") or []),
              "now_prefs": current["prefs"], "personal": len(current.get("personal") or [])}
    videos = post_plan.candidates(user, projects, client=r, plan=plan)
    today = post_plan.today_for(plan["prefs"], now)
    prefs = {**plan["prefs"], "times": ["08:15", "12:30", "18:45"], "per_day": 3, "days": [0, 1, 2, 3, 4],
             "count": len([v for v in videos if not v.get("posted_at")]), "start": (today + timedelta(days=1)).isoformat(),
             "remind": True}
    week = post_plan.build_flex(videos, prefs, today, keep=[])
    by_id = {v["project_id"]: v for v in videos}
    slots = [{**s, **{k: by_id[s["project_id"]].get(k) for k in ("product", "brand", "product_id", "mix_code", "source",
                                                                  "posted_at", "caption", "caption_source", "duration_sec")}}
             for s in week["slots"]]
    report["rules_check"] = post_plan.check(week["slots"], videos)
    report["plan"] = {k: week[k] for k in ("start", "end", "asked", "placed", "ends_early")}
    report["slots"] = [{"date": s["date"], "time": s["time"], "product": s["product"], "brand": s["brand"], "source": s["source"]}
                       for s in slots]
    before = {"prefs": prefs, "today": today.isoformat(), "summary": current["summary"], "week": None,
              "personal": current.get("personal") or []}
    after = {**before, "week": {"start": week["start"], "slots": slots, "missing": None, "end": week["end"], "flex": True,
                                "asked": week["asked"], "placed": week["placed"], "ends_early": week["ends_early"]}}
    report["fixture"] = {"before": before, "after": after}
    return report


@app.local_entrypoint()
def main():
    with open("engine_check.json", "w") as f:
        json.dump(run.remote(), f, ensure_ascii=False, default=str)
