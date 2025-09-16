# app.py  — FastAPI + RQ (v1.16) endpoints with safe, named worker functions
import os, json, time, uuid
from typing import List, Optional, Dict, Any

import redis
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

from rq import Queue
from rq.job import Job  # NOTE: job import must be from rq.job in RQ 1.16

# ---------- infra ----------
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL env var")

r = redis.from_url(REDIS_URL, decode_responses=True)
q = Queue("default", connection=r)

app = FastAPI(title="editdna-web", version="1.0.0")

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
    # Minimal “segments” schema; you can extend later
    manifest: Optional[Dict[str, Any]] = None

class StitchIn(BaseModel):
    session_id: str
    manifest: Dict[str, Any]

# ---------- worker functions (MUST be top-level, importable) ----------
def nop() -> dict:
    """Tiny job to prove the queue works."""
    time.sleep(0.5)
    return {"ok": True}

def analyze_core_from_session(session_id: str) -> dict:
    """Pretend analysis; writes analysis result back to the session."""
    data = _get_session(session_id)
    # fake analysis work
    time.sleep(1.0)
    data["analysis"] = {
        "ok": True,
        "assets": data.get("files", []),
        "notes": "analysis complete (dummy)"
    }
    _save_session(session_id, data)
    return {"ok": True, "session_id": session_id}

def stitch_core(session_id: str) -> dict:
    """Pretend stitching; writes a dummy public_url result."""
    data = _get_session(session_id)
    # fake stitching work
    time.sleep(1.0)
    manifest = data.get("manifest")
    if not manifest:
        raise ValueError("No manifest set on session")

    data["stitch"] = {
        "status": "finished",
        "result": {
            "ok": True,
            # Replace with your real S3 output if you have it
            "public_url": f"https://example.com/sessions/{session_id}/{manifest.get('filename','final.mp4')}"
        }
    }
    _save_session(session_id, data)
    return data["stitch"]

# ---------- health/diag ----------
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

# ---------- real API ----------
@app.post("/process_urls")
def process_urls(inp: ProcessUrlsIn):
    # store basic session with file ids
    files = [{"file_id": uuid.uuid4().hex[:8], "source": "url", "url": u} for u in inp.urls]
    session = {
        "meta": {
            "tone": inp.tone,
            "product_link": inp.product_link,
            "features_csv": inp.features_csv
        },
        "files": files,
        "analysis": None,
        "manifest": None,
        "stitch": None
    }
    sid = _new_session(session)
    return {"ok": True, "session_id": sid, "files": files}

@app.post("/analyze")
def analyze(inp: AnalyzeIn):
    # enqueue named function (no lambdas!)
    job = q.enqueue(analyze_core_from_session, inp.session_id)
    return {"ok": True, "job_id": job.id}

@app.post("/analyze_sync")
def analyze_sync(inp: AnalyzeIn):
    # Convenience: block until done (for small jobs).
    job = q.enqueue(analyze_core_from_session, inp.session_id)
    # Poll briefly (avoid long timeouts on Render)
    for _ in range(120):  # ~60s
        if job.get_status() == "finished":
            return {"ok": True, "result": job.result}
        if job.get_status() == "failed":
            return {"ok": False, "error": str(job.exc_info or 'failed')}
        time.sleep(0.5)
    return {"ok": False, "error": "analyze timed out, check /jobs/{job_id}", "job_id": job.id}

@app.post("/manifest")
def set_manifest(inp: ManifestIn):
    data = _get_session(inp.session_id)
    # If caller didn’t send manifest field, build minimal from first file
    manifest = inp.manifest or {
        "preset_key": inp.preset_key,
        "filename": inp.filename or "final.mp4",
        "fps": inp.fps or 30,
        "scale": inp.scale or 720,
        "segments": [
            # Example: play the first asset 0–2s; you can post a richer manifest later
            {"file_id": data["files"][0]["file_id"], "start": 0.0, "end": 2.0}
        ]
    }
    data["manifest"] = manifest
    _save_session(inp.session_id, data)
    return {"ok": True, "session_id": inp.session_id, "manifest": manifest}

@app.post("/stitch")
def stitch(inp: StitchIn):
    # Save/overwrite manifest then enqueue stitch job
    data = _get_session(inp.session_id)
    data["manifest"] = inp.manifest
    _save_session(inp.session_id, data)

    job = q.enqueue(stitch_core, inp.session_id)
    return {"ok": True, "job_id": job.id, "session_id": inp.session_id}

@app.get("/debug/session/{session_id}")
def debug_session(session_id: str):
    return _get_session(session_id)
