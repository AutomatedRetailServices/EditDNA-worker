"""Production check after the 23 Oct deploy, with a test user (anonymous session): a talking video (Single), two in a
Bulk (Batch API), a quiet one (SIN VOZ), and the restore guard. Prints no token or link."""
import json, os, sys, time, uuid, requests
B = "https://cutsell-api.onrender.com"
s = requests.post(B + "/v1/auth/session", timeout=60).json()
H = {"authorization": "Bearer " + s["access_token"]}
U = s["user_id"]

def project(title):
    return requests.post(B + "/v1/projects", json={"user_id": U, "title": title}, headers=H, timeout=60).json()["project_id"]

def upload(pid, path, ctype="video/mp4"):
    size = os.path.getsize(path)
    p = requests.post(B + "/v1/uploads/presign", json={"project_id": pid, "user_id": U, "original_name": os.path.basename(path),
                                                       "content_type": ctype, "size_bytes": size}, headers=H, timeout=60).json()
    with open(path, "rb") as f:
        if p["method"].upper() == "PUT":
            r = requests.put(p["upload_url"], data=f, headers=p.get("headers") or {}, timeout=300)
        else:
            r = requests.post(p["upload_url"], data=p["fields"], files={"file": f}, timeout=300)
    assert r.status_code in (200, 204), r.status_code
    return p["source_uri"], size

def start(pid, uri, name, meta=None):
    body = {"project_id": pid, "user_id": U, "sources": [{"original_name": name, "uri": uri, "source_order": 0, "metadata": meta}]}
    return requests.post(B + "/v1/flow-b/jobs", json=body, headers=H, timeout=60).json()["job_id"]

def wait(job, limit=2400):
    t0 = time.time()
    while time.time() - t0 < limit:
        j = requests.get(B + f"/v1/jobs/{job}", headers=H, timeout=60).json()
        if j["state"] in ("finished", "failed", "canceled"):
            return j, round(time.time() - t0)
        time.sleep(10)
    return {"state": "timeout"}, limit

out = {}
jobs = {}

def run_phase(items):
    started = {}
    for name, path, meta in items:
        pid = project("Prod check " + name)
        uri, size = upload(pid, path)
        jobs[name] = {"pid": pid, "uri": uri, "meta": meta, "job": start(pid, uri, path, meta), "started": time.time()}
        started[name] = jobs[name]
    for name, j in started.items():
        res, secs = wait(j["job"])
        row = {"state": res.get("state"), "error": res.get("error"), "seconds": round(time.time() - j["started"])}
        if res.get("state") == "finished":
            d = requests.get(B + f"/v1/projects/{j['pid']}/draft", params={"user_id": U}, headers=H, timeout=60).json()
            src = (d.get("draft", {}).get("diagnostics", {}).get("simple_engine", {}).get("sources") or [{}])[0]
            pc = src.get("prompt_cache") or {}
            row.update({"cut_s": src.get("cut_duration_sec"), "kept_clips": len(d.get("draft", {}).get("selected") or []),
                        "available_flag": [x.get("available") for x in d.get("sources", [])],
                        "cache": {k: pc.get(k) for k in ("cache_on", "ttl", "cache_used", "usd", "usd_normal_api", "batch_calls",
                                                         "batch_fallbacks", "batch_wait_s", "read_tokens", "write_1h_tokens")}})
        out[name] = row

# Today's server allows 2 cuts at once per user: Single + quiet first, then a Bulk of 2.
run_phase((("single", "talk.mp4", None), ("quiet", "quiet.mp4", None)))
bulk = {"bulk_size": "2", "bulk_id": "prodcheck-" + uuid.uuid4().hex[:6]}
run_phase((("bulk_a", "talk.mp4", bulk), ("bulk_b", "talk.mp4", bulk)))
# restore guard: a copy that is still stored must be refused (409)
r = requests.post(B + "/v1/uploads/multipart/start", json={"project_id": jobs["single"]["pid"], "user_id": U, "original_name": "talk.mp4",
                  "content_type": "video/mp4", "size_bytes": 15258086, "restore_uri": jobs["single"]["uri"]}, headers=H, timeout=60)
out["restore_guard_still_stored"] = r.status_code
r = requests.post(B + "/v1/uploads/multipart/start", json={"project_id": jobs["single"]["pid"], "user_id": U, "original_name": "talk.mp4",
                  "content_type": "video/mp4", "size_bytes": 15258086, "restore_uri": jobs["quiet"]["uri"]}, headers=H, timeout=60)
out["restore_guard_other_project"] = r.status_code
json.dump(out, open("engine_check.json", "w"), indent=1)
