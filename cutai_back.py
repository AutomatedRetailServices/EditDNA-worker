"""Product Owner, 21 Oct (correction): "Cut AI" IS CutSell. The 2 videos go back to CutSell · CutSell.ai (as hers, so
the AI never moves them); the "Cut AI" product and brand are removed from her catalog if nothing is left in them.
Then how many videos Plan can use. Writes only her catalog. Report encrypted."""
import json
import os

import modal

app = modal.App("cutsell-cutai-back")
secret = modal.Secret.from_name("cutsell-worker")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
         .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"]))
REAL_USER = "1625a6e9"
VIDEOS = ["0ddb40", "2f7faf"]


@app.function(image=image, secrets=[secret], timeout=300)
def run():
    import hashlib
    from redis import Redis
    from cutsell_worker import product_catalog as catalog, project_store, post_plan
    r = Redis.from_url(os.environ["REDIS_URL"])
    user = None
    for key in r.scan_iter("cutsell:v1:draft:*", count=500):
        snap = json.loads(r.get(key))
        if hashlib.sha256(str(snap.get("user_id")).encode()).hexdigest()[:8] == REAL_USER:
            user = snap["user_id"]
            break
    projects = project_store.list_projects(user_id=user, limit=500, client=r)
    ids = [p["project_id"] for p in projects]
    report = {"errors": []}
    data = catalog.load(user, client=r)
    names = {b: v.get("name") for b, v in data["brands"].items()}
    target = next((pid for pid, p in data["products"].items()
                   if p.get("name") == "CutSell.ai" and names.get(p.get("brand_id")) == "CutSell"), None)
    move = [i for i in ids if any(i.endswith(e) for e in VIDEOS)]
    if not target or len(move) != 2:
        report["errors"].append(f"target {bool(target)} videos {len(move)}")
        return report
    catalog.assign(user, move, product_id=target, client=r)

    def drop_empty(d):
        used = {v.get("product_id") for v in d["videos"].values()}
        gone = []
        for pid, p in list(d["products"].items()):
            if p.get("name") == "Cut AI" and pid not in used:
                bid = p.get("brand_id")
                del d["products"][pid]
                gone.append("product Cut AI")
                if bid and not any(q.get("brand_id") == bid for q in d["products"].values()) \
                        and (d["brands"].get(bid) or {}).get("name") == "Cut AI":
                    del d["brands"][bid]
                    gone.append("brand Cut AI")
        return gone
    report["removed"] = catalog._update(user, drop_empty, client=r)
    data = catalog.load(user, client=r)
    names = {b: v.get("name") for b, v in data["brands"].items()}
    report["products"] = [{"brand": names.get(p.get("brand_id")), "product": p.get("name"),
                           "videos": sum(1 for v in data["videos"].values() if v.get("product_id") == pid),
                           "by_her": sum(1 for v in data["videos"].values() if v.get("product_id") == pid
                                         and v.get("source") == "user")}
                          for pid, p in data["products"].items()]
    report["brands"] = sorted(names.values())
    plan = post_plan.load(user, client=r)
    usable = post_plan.candidates(user, projects, client=r, plan=plan)
    report["plan"] = {"ready_videos": sum(p.get("state") == "draft_ready" for p in projects),
                      "personal": sum(1 for v in plan["videos"].values() if v.get("personal")),
                      "usable": len(usable), "unposted": sum(1 for v in usable if not v.get("posted_at")),
                      "list": [{"product": v.get("product"), "brand": v.get("brand"), "sec": v["duration_sec"],
                                "other_cuts": v.get("other_cuts", 0), "source": v["source"]} for v in usable]}
    return report


@app.local_entrypoint()
def main():
    out = run.remote()
    with open("engine_check.json", "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("::notice title=cutai back::errors", len(out.get("errors") or []))
