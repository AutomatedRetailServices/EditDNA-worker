"""Product Owner, 21 Oct (option B): create these products in the creator's catalog and put her videos in them.
Writes ONLY her catalog (product + video links, marked as hers so the AI never moves them). Report encrypted."""
import json
import os

import modal

app = modal.App("cutsell-assign-products")
secret = modal.Secret.from_name("cutsell-worker")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
         .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"]))
REAL_USER = "1625a6e9"
PLAN = [  # (brand, product, ends of the project ids)
    ("No brand", "Perfume", ["b6f9d6", "f1ae74", "b992d5", "b0acdd"]),
    ("No brand", "Glass of water ($29.99)", ["ed2f59", "83f2d2", "c33470"]),
    ("No brand", "Cotton knit pants", ["1070c8", "a0cfb8"]),
    ("No brand", "Baby shoes", ["a2569c", "79c635"]),
    ("CutSell", "CutSell.ai", ["c9b4ec", "0ddb40", "2f7faf"]),
]


@app.function(image=image, secrets=[secret], timeout=300)
def run():
    import hashlib
    from redis import Redis
    from cutsell_worker import product_catalog as catalog, project_store
    r = Redis.from_url(os.environ["REDIS_URL"])
    user = None
    for key in r.scan_iter("cutsell:v1:draft:*", count=500):
        snap = json.loads(r.get(key))
        if hashlib.sha256(str(snap.get("user_id")).encode()).hexdigest()[:8] == REAL_USER:
            user = snap["user_id"]
            break
    ids = [p["project_id"] for p in project_store.list_projects(user_id=user, limit=500, client=r)]
    before = catalog.load(user, client=r)
    report = {"before": {"brands": len(before["brands"]), "products": len(before["products"])}, "done": [], "errors": []}
    for brand, product, ends in PLAN:
        full = []
        for end in ends:
            match = [i for i in ids if i.endswith(end)]
            if len(match) != 1:
                report["errors"].append(f"{product}: {end} matched {len(match)}")
                continue
            if (before["videos"].get(match[0]) or {}).get("product_id"):
                report["errors"].append(f"{product}: {end} already has a product, left as is")
                continue
            full.append(match[0])
        if full:
            out = catalog.assign(user, full, brand=brand, product=product, client=r)
            report["done"].append({"brand": brand, "product": product, "videos": len(full), "product_id": out["product_id"][-6:]})
    after = catalog.load(user, client=r)
    names = {b: v.get("name") for b, v in after["brands"].items()}
    report["after"] = [{"brand": names.get(p.get("brand_id")), "product": p.get("name"),
                        "videos": sum(1 for v in after["videos"].values() if v.get("product_id") == pid)}
                       for pid, p in after["products"].items()]
    report["untagged_left"] = sum(1 for i in ids if not (after["videos"].get(i) or {}).get("product_id"))
    return report


@app.local_entrypoint()
def main():
    out = run.remote()
    with open("engine_check.json", "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("::notice title=assign::done", len(out["done"]), "errors", len(out["errors"]))
