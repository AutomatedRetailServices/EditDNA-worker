# app.py
# FastAPI + RQ (v1.16.x) web app with Redis-backed session storage.
# Endpoints:
#  - POST /process_urls
#  - POST /analyze
#  - POST /manifest
#  - POST /stitch
#  - GET  /jobs/{job_id}
#  - GET  /debug/session/{session_id}
#  - GET  /admin/health
#  - POST /enqueue_nop   (smoke test for queue)

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rq import Queue
from rq.job import Job
from rq.registry import FailedJobRegistry, FinishedJobRegistry, ScheduledJobRegistry, StartedJobRegistry

# -----------------------------------------------------------------------------
# Redis & RQ setup
# -----------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

r = redis.from_url(REDIS_URL, decode_responses=True)  # store JSON strings
q = Queue("default", connection=r)

# -----------------------------------------------------------------------------
# Pydantic models (request bodies)
# -----------------------------------------------------------------------------
class ProcessUrlsIn(BaseModel):
    urls: List[str]
    tone: Optional[str] = None
    product_link: Optional[str] = None
    features_csv: Optional[str] = None

class AnalyzeIn(BaseModel):
    session_id: str

class ManifestIn(BaseModel):
    session_id: str
    preset_key: Optional[str] = None
    filename: str = "final.mp4"
    fps: int = 30
    scale: int = 720
    manifest: Optional[Dict[str, List[Dict[str, Any]]]] = None  # e.g. { "CTA":[{file_id,start,end},...] }

class StitchIn(BaseModel):
    session_id: str
    manifest: Dict[str, List[Dict[str, Any]]]
    filename: str = "final.mp4"
    fps: int = 30
    scale: int = 720

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="EditDNA Web API")

# -----------------------------------------------------------------------------
# Helper: Redis keys
# -----------------------------------------------------------------------------
def key_session(session_id: str) -> str:
    return f"session:{session_id}"

def key_analysis(session_id: str) -> str:
    return f"analysis:{session_id}"

def key_manifest(session_id: str) -> str:
    return f"manifest:{session_id}"

# -----------------------------------------------------------------------------
# Queueable functions (MUST be top-level named functions for RQ to import)
# -----------------------------------------------------------------------------
def nop() -> str:
    """Simple no-op used to verify worker/queue."""
    return "ok"

def analyze_core_from_session(session_id: str, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pretend analysis: reads session, writes an analysis result for later steps."""
    sess_raw = r.get(key_session(session_id))
    if not sess_raw:
        raise ValueError(f"Unknown session_id: {session_id}")

    session = json.loads(sess_raw)

    # Simulate some work
    time.sleep(1.0)

    # Create a tiny, deterministic "analysis"
    result = {
        "ok": True,
        "session_id": session_id,
        "files": session["files"],  # echo back
        "detected_scenes": [{"file_id": f["file_id"], "start": 0.0, "end": 2.0} for f in session["files"]],
    }
    r.set(key_analysis(session_id), json.dumps(result))
    return result

def stitch_core(session_id: str, filename: str, manifest: Dict[str, List[Dict[str, Any]]], fps: int, scale: int) -> Dict[str, Any]:
    """Pretend stitch: consumes manifest + analysis and returns a fake public URL."""
    # Ensure analysis exists (optional sanity check)
    _ = r.get(key_analysis(session_id))  # not strictly required for this mocked stitch

    # Simulate work
    time.sleep(1.0)

    # In a real implementation you'd render and upload; here we just fabricate a URL.
    public_url = f"https://script2clipshop-video-automatedetailservices.s3.us-east-1.amazonaws.com/sessions/{session_id}/{filename}"
    return {"ok": True, "public_url": public_url, "fps": fps, "scale": scale, "manifest_keys": list(manifest.keys())}

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.post("/process_urls")
def process_urls(body: ProcessUrlsIn):
    if not body.urls:
        raise HTTPException(status_code=400, detail="urls must not be empty")

    session_id = uuid.uuid4().hex
    files = []
    for u in body.urls:
        files.append({"file_id": uuid.uuid4().hex[:8], "source": "url", "url": u})

    session = {
        "session_id": session_id,
        "urls": body.urls,
        "tone": body.tone,
        "product_link": body.product_link,
        "features_csv": body.features_csv,
        "files": files,
    }
    r.set(key_session(session_id), json.dumps(session))
    return {"ok": True, "session_id": session_id, "files": files}

@app.post("/analyze")
def analyze(body: AnalyzeIn):
    # Enqueue the named function so RQ can import it: "app.analyze_core_from_session"
    job = q.enqueue(analyze_core_from_session, body.session_id, None)
    return {"ok": True, "job_id": job.get_id()}

@app.post("/manifest")
def save_manifest(body: ManifestIn):
    # Allow the client to send the manifest now or build one from defaults later.
    manifest = body.manifest or {}
    r.set(key_manifest(body.session_id), json.dumps({
        "filename": body.filename,
        "fps": body.fps,
        "scale": body.scale,
        "manifest": manifest
    }))
    return {"ok": True, "session_id": body.session_id, "manifest": manifest}

@app.post("/stitch")
def stitch(body: StitchIn):
    # Enqueue the named function so RQ can import it: "app.stitch_core"
    job = q.enqueue(stitch_core, body.session_id, body.filename, body.manifest, body.fps, body.scale)
    return {"ok": True, "job_id": job.get_id(), "session_id": body.session_id}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = Job.fetch(job_id, connection=r)
    except Exception:
        raise HTTPException(status_code=404, detail="unknown_job_id")

    payload: Dict[str, Any] = {"job_id": job_id, "status": job.get_status()}

    if job.is_failed:
        payload["error"] = str(job._exc_info) if job._exc_info else "failed"
    elif job.is_finished:
        payload["result"] = job.result

    return payload

@app.get("/debug/session/{session_id}")
def debug_session(session_id: str):
    data = {
        "session": r.get(key_session(session_id)),
        "analysis": r.get(key_analysis(session_id)),
        "manifest": r.get(key_manifest(session_id)),
    }
    # Return parsed JSON where possible
    out = {}
    for k, v in data.items():
        out[k] = json.loads(v) if v else None
    return out

@app.get("/admin/health")
def admin_health():
    regs = {
        "queued": len(q.jobs),
        "started": len(StartedJobRegistry(queue=q)),
        "scheduled": len(ScheduledJobRegistry(queue=q)),
        "deferred": 0,  # not used in this minimal setup
        "failed": len(FailedJobRegistry(queue=q)),
        "finished": len(FinishedJobRegistry(queue=q)),
    }
    return {"ok": True, "registries": regs}

# ---- Smoke test endpoint to verify queue/worker wiring -----------------------
@app.post("/enqueue_nop")
def enqueue_nop():
    job = q.enqueue(nop)  # IMPORTANT: named function, not a lambda
    return {"ok": True, "job_id": job.get_id()}

# Root for friendliness
@app.get("/")
def root():
    return {"ok": True, "service": "editdna-web", "queue": q.name}
