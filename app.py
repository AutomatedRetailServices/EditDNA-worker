# app.py — FastAPI + RQ (1.16) with safe worker funcs and admin purge tools
import os, json, time, uuid
from typing import List, Optional, Dict, Any

import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rq import Queue
from rq.job import Job
from rq.registry import (
    StartedJobRegistry,
    FailedJobRegistry,
    FinishedJobRegistry,
    ScheduledJobRegistry,
    DeferredJobRegistry,
)

# ---------- infra ----------
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL env var")

r = redis.from_url(REDIS_URL, decode_responses=True)
q = Queue("default", connection=r)

app = FastAPI(title="editdna-web", version="1.1.0")

# ---------- session helpers ----------
def _sid_key(session_id: str) -> str:
    return f"session:{session_id}"

def _new_session(initial: Dict[str, Any]) -> str:
    sid = uuid.uuid4().hex
    r.set(_sid_key(sid), json.dumps(initial))
    return sid

def _get_session(session_id: str) -> Dict[str, Any]:
    raw = r.get(_sid_key(session_id))
    if not raw:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return json.loads(raw)

def _save_session(session_id: str, data: Dict[str, Any]) -> None:
    r.set(_sid_key(session_id), json.dumps(data))

# ---------- models ----------
class ProcessUrlsIn(BaseModel):
    urls: List[str]
    tone: Optional[str] = None
    product_link: Optional[str] = None
    features_csv: Optional[str] = None

class AnalyzeIn(BaseModel):
    session_id: str

class ManifestIn(BaseModel):
    session_id: str
    preset_key: Optional[str] = Field(default=None)
    filename: Optional[str] = Field(default="final.mp4")
    fps: Optional[int] = Field(default=30)
    scale: Optional[int] = Field(default=720)
    manifest: Optional[Dict[str, Any]] = None

class StitchIn(BaseModel):
    session_id: str
    manifest: Dict[str, Any]

# ---------- worker functions (must be importable; no lambdas) ----------
def nop() -> dict:
    time.sleep(0.5)
    return {"ok": True}

def analyze_core_from_session(session_id: str) -> dict:
    data = _get_session(session_id)
    time.sleep(1.0)
    data["analysis"] = {"ok": True, "assets": data.get("files", []), "notes": "analysis complete (dummy)"}
    _save_session(session_id, data)
    return {"ok": True, "session_id": session_id}

def stitch_core(session_id: str) -> dict:
    data = _get_session(session_id)
    time.sleep(1.0)
    manifest = data.get("manifest")
    if not manifest:
        raise ValueError("No manifest set on session")
    data["stitch"] = {
        "status": "finished",
        "result": {"ok": True, "public_url": f"https://example.com/sessions/{session_id}/{manifest.get('filename','final.mp4')}"}
    }
    _save_session(session_id, data)
    return data["stitch"]

# ---------- health / job status ----------
@app.post("/enqueue_nop")
def enqueue_nop():
    job = q.enqueue(nop)
    return {"ok": True, "job_id": job.id}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=r)
    except Exception:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    status = job.get_status()
    payload: Dict[str, Any] = {"job_id": job_id, "status": status}
    if status == "finished":
        payload["result"] = job.result
    elif status == "failed":
        payload["result"] = None
        payload["error"] = str(job.exc_info or job.description or "Job failed")
    return payload

# ---------- admin: inspect & purge queues/registries ----------
def _registry_counts() -> Dict[str, int]:
    return {
        "queued": len(q),
        "started": len(StartedJobRegistry(queue=q)),
        "scheduled": len(ScheduledJobRegistry(queue=q)),
        "deferred": len(DeferredJobRegistry(queue=q)),
        "failed": len(FailedJobRegistry(queue=q)),
        "finished": len(FinishedJobRegistry(queue=q)),
    }

@app.get("/admin/health")
def admin_health():
    return {"ok": True, "registries": _registry_counts()}

@app.post("/admin/purge_all")
def admin_purge_all():
    total_deleted = 0

    # Empty the main queue
    total_deleted += q.empty()

    # Clear registries
    for Reg in [StartedJobRegistry, ScheduledJobRegistry, DeferredJobRegistry, FailedJobRegistry, FinishedJobRegistry]:
        reg = Reg(queue=q)
        ids = reg.get_job_ids()
        for jid in ids:
            try:
                job = Job.fetch(jid, connection=r)
                job.delete()
            except Exception:
                pass
        try:
            reg.clean()
        except Exception:
            pass
        total_deleted += len(ids)

    return {"ok": True, "deleted": total_deleted, "registries_after": _registry_counts()}

# ---------- real API ----------
@app.post("/process_urls")
def process_urls(inp: ProcessUrlsIn):
    files = [{"file_id": uuid.uuid4().hex[:8], "source": "url", "url": u} for u in inp.urls]
    session = {
        "meta": {"tone": inp.tone, "product_link": inp.product_link, "features_csv": inp.features_csv},
        "files": files,
        "analysis": None,
        "manifest": None,
        "stitch": None
    }
    sid = _new_session(session)
    return {"ok": True, "session_id": sid, "files": files}

@app.post("/analyze")
def analyze(inp: AnalyzeIn):
    job = q.enqueue(analyze_core_from_session, inp.session_id)
    return {"ok": True, "job_id": job.id}

@app.post("/analyze_sync")
def analyze_sync(inp: AnalyzeIn):
    job = q.enqueue(analyze_core_from_session, inp.session_id)
    for _ in range(120):  # ~60s
        status = job.get_status()
        if status == "finished":
            return {"ok": True, "result": job.result}
        if status == "failed":
            return {"ok": False, "error": str(job.exc_info or 'failed')}
        time.sleep(0.5)
    return {"ok": False, "error": "analyze timed out, check /jobs/{job_id}", "job_id": job.id}

@app.post("/manifest")
def set_manifest(inp: ManifestIn):
    data = _get_session(inp.session_id)
    manifest = inp.manifest or {
        "preset_key": inp.preset_key,
        "filename": inp.filename or "final.mp4",
        "fps": inp.fps or 30,
        "scale": inp.scale or 720,
        "segments": [
            {"file_id": data["files"][0]["file_id"], "start": 0.0, "end": 2.0}
        ]
    }
    data["manifest"] = manifest
    _save_session(inp.session_id, data)
    return {"ok": True, "session_id": inp.session_id, "manifest": manifest}

@app.post("/stitch")
def stitch(inp: StitchIn):
    data = _get_session(inp.session_id)
    data["manifest"] = inp.manifest
    _save_session(inp.session_id, data)
    job = q.enqueue(stitch_core, inp.session_id)
    return {"ok": True, "job_id": job.id, "session_id": inp.session_id}

@app.get("/debug/session/{session_id}")
def debug_session(session_id: str):
    return _get_session(session_id)
