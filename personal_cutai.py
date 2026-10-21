"""Product Owner, 21 Oct: (3) the two "Cut AI" videos are another app -> out of CutSell.ai, into brand/product "Cut AI";
(5) her 10 videos without a product are personal -> flagged so Plan never schedules them. Writes only her catalog and
her plan entries. Report encrypted."""
import json
import os

import modal

app = modal.App("cutsell-personal-cutai")
secret = modal.Secret.from_name("cutsell-worker")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
         .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"]))
REAL_USER = "1625a6e9"
CUT_AI = ["0ddb40", "2f7faf"]


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
    cut = [i for i in ids if any(i.endswith(e) for e in CUT_AI)]
    if len(cut) == 2:
        out = catalog.assign(user, cut, brand="Cut AI", product="Cut AI", client=r)
        report["cut_ai"] = {"videos": len(cut), "product_id": out["product_id"][-6:]}
    else:
        report["errors"].append(f"cut ai matched {len(cut)}")
    data = catalog.load(user, client=r)
    personal = [p for p in projects if not (data["videos"].get(p["project_id"]) or {}).get("product_id")]
    report["personal"] = [{"id": p["project_id"][-6:], "title": p.get("title"), "state": p.get("state")} for p in personal]
    if len(personal) == 10:
        post_plan.set_personal(user, [p["project_id"] for p in personal], client=r)
    else:
        report["errors"].append(f"expected 10 videos without a product, found {len(personal)}; nothing flagged")
    names = {b: v.get("name") for b, v in data["brands"].items()}
    report["products"] = [{"brand": names.get(p.get("brand_id")), "product": p.get("name"),
                           "videos": sum(1 for v in data["videos"].values() if v.get("product_id") == pid)}
                          for pid, p in data["products"].items()]
    plan = post_plan.load(user, client=r)
    report["flagged"] = sum(1 for v in plan["videos"].values() if v.get("personal"))
    week = post_plan.candidates(user, projects, client=r, plan=plan)
    report["plannable"] = len(week)
    return report


@app.local_entrypoint()
def main():
    out = run.remote()
    with open("engine_check.json", "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("::notice title=personal::errors", len(out["errors"]))
