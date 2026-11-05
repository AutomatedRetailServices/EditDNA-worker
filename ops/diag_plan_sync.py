"""One-off diagnosis (PO 4 Nov: "cuando selecciono todos los videos no suben al día que indico").

Read-only: loads ONE account's plan and library (identified by its storage hash, never the id) and counts why
chosen videos may not show in the week. Prints NUMBERS ONLY (the repository and its Actions logs are public):
no titles, no ids, no captions. Nothing is written. Run: modal run ops/diag_plan_sync.py (ephemeral app).
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
app = modal.App("cutsell-diag-plan-sync")
SCOPE = "1625a6e9f075ad8ddf01"          # sha256(user)[:20], the same hash the storage keys use


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=300, cpu=0.5)
def diagnose():
    import collections, hashlib, json, os
    from redis import Redis
    from cutsell_worker import post_plan, project_store

    r = Redis.from_url(os.environ["REDIS_URL"])
    ids = [x.decode() for x in r.zrevrange(f"cutsell:v1:projects:{SCOPE}", 0, 499)]
    print(f"library: {len(ids)} projects")
    first = None
    for pid in ids:
        raw = r.get(f"cutsell:v1:project:{SCOPE}:{hashlib.sha256(pid.encode()).hexdigest()[:20]}")
        if raw:
            first = json.loads(raw); break
    if not first:
        print("no project readable"); return
    user = first["user_id"]
    projects = project_store.list_projects(user_id=user, client=r, limit=500)
    print("states:", dict(collections.Counter(p.get("state") for p in projects)))
    plan = post_plan.load(user, client=r)
    prefs = plan["prefs"]
    print("prefs: times=%d days=%s start=%s count=%s remind=%s" % (
        len(prefs.get("times") or []), prefs.get("days"), prefs.get("start"), prefs.get("count"), prefs.get("remind")))
    week = plan.get("week") or {}
    slots = week.get("slots") or []
    print("stored week: flex=%s start=%s end=%s asked=%s placed=%s slots=%d pinned=%d built_at=%s" % (
        week.get("flex"), week.get("start"), week.get("end"), week.get("asked"), week.get("placed"), len(slots),
        sum(1 for s in slots if s.get("pinned")), (week.get("built_at") or "")[:16]))
    print("stored slots per day:", dict(sorted(collections.Counter(s["date"] for s in slots).items())))
    ready_ids = {str(p["project_id"]) for p in projects if p.get("state") == "draft_ready"}
    personal = {pid for pid, v in plan["videos"].items() if v.get("personal")}
    cands = post_plan.candidates(user, projects, client=r, plan=plan)
    cand_ids = {v["project_id"] for v in cands}
    print(f"ready={len(ready_ids)} personal={len(personal & ready_ids)} candidates={len(cand_ids)} "
          f"posted_among_candidates={sum(1 for v in cands if v.get('posted_at'))} "
          f"hidden_as_same_recording={len(ready_ids - cand_ids - personal)}")
    stored_ids = {s["project_id"] for s in slots}
    shown = post_plan.view(user, projects, client=r)
    shown_slots = (shown.get("week") or {}).get("slots") or []
    print(f"view: today={shown.get('today')} slots_shown={len(shown_slots)} of stored={len(slots)}; "
          f"stored but not a candidate={len(stored_ids - cand_ids)}")
    print("shown slots per day:", dict(sorted(collections.Counter(s["date"] for s in shown_slots).items())))
    offered = ready_ids - personal - stored_ids
    print(f"app would offer {len(offered)}; server keeps {len(offered & cand_ids)}; "
          f"drops {len(offered - cand_ids)} as another cut of the same recording")
    meta = plan["videos"]
    lost = [pid for pid, v in meta.items() if v.get("scheduled_for") and pid not in stored_ids]
    print(f"scheduled_for set but not in the week: {len(lost)} (not candidates: {len([p for p in lost if p not in cand_ids])})")
    print("lost by scheduled day:", dict(sorted(collections.Counter(meta[p]['scheduled_for'][:10] for p in lost).items())))


@app.local_entrypoint()
def main():
    diagnose.remote()
