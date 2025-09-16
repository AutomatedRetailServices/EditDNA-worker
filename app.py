# app.py
import os, json, time, uuid, redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rq import Queue, Job
from rq.registry import (StartedJobRegistry, FinishedJobRegistry,
                         FailedJobRegistry, ScheduledJobRegistry, DeferredJobRegistry)

# ---- Redis + RQ -------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

# IMPORTANT: decode_responses=False to avoid unicode/bytes issues with RQ internals
r = redis.from_url(REDIS_URL, decode_responses=False)
q = Queue("default", connection=r)

# ---- FastAPI ---------------------------------------------------------------
app = FastAPI(title="editdna-web")

# Small helper to make Redis keys
def key_session(session_id: str) -> str:
    return f"session:{session_id}".encode()

# ---------------------------- MODELS ----------------------------------------
class ProcessUrlsIn(BaseModel):
    urls: list[str]
    tone: str | None = None
    product_link: str | None = None
    features_csv: str | None = None

class ManifestIn(BaseModel):
    session_id: str
    preset_key: str | None = None
    filename: str | None = None
    fps: int | None = None
    scale: int | None = None

class AnalyzeIn(BaseModel):
    session_id: str

# --------------------------- WORKER FUNCS -----------------------------------
def analyze_core_from_session(session_id: str, _unused=None):
    """
    This is a placeholder worker task. It just reads the session payload
    and writes a tiny 'analysis' result back to Redis, then returns it.
    Replace with your real analysis later.
    """
    k = key_session(session_id)
    raw = r.get(k)
    if not raw:
        raise ValueError(f"session {session_id} not found")

    payload = json.loads(raw.decode("utf-8"))
    # pretend “analysis”
    analysis = {
        "num_assets": len(payload.get("files", [])),
        "tone": payload.get("tone"),
        "ts": int(time.time()),
    }

    # store side-effect
    r.hset(k + b":analysis", mapping={
        b"json": json.dumps(analysis).encode("utf-8")
    })
    return {"session_id": session_id, "analysis": analysis}

# keep a simple nop we used to prove the pipeline
def nop_echo(data: dict):
    return {"echo": data}

# --------------------------- ROUTES -----------------------------------------
@app.post("/enqueue_nop")
def enqueue_nop():
    job = q.enqueue(nop_echo, {"hello": "world"}, job_timeout=300)
    return {"ok": True, "job_id": job.id}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = Job.fetch(job_id, connection=r)
    except Exception:
        raise HTTPException(status_code=404, detail="unknown_job_id")

    status = job.get_status()
    result = job.result if status == "finished" else None
    err = None
    if status == "failed" and job.exc_info:
        # return only the error line to keep it short
        err = job.exc_info.splitlines()[-1]

    enq = job.enqueued_at.isoformat() if job.enqueued_at else None
    end = job.ended_at.isoformat() if job.ended_at else None

    return {
        "job_id": job.id,
        "status": status,
        "result": result,
        "error": err,
        "enqueued_at": enq,
        "ended_at": end,
    }

@app.get("/admin/health")
def health():
    regs = {
        "queued": q.count,
        "started": lambda: StartedJobRegistry("default", connection=r).count,
        "scheduled": lambda: ScheduledJobRegistry("default", connection=r).count,
        "deferred": lambda: DeferredJobRegistry("default", connection=r).count,
        "failed": lambda: FailedJobRegistry("default", connection=r).count,
        "finished": lambda: FinishedJobRegistry("default", connection=r).count,
    }
    # evaluate the callables
    out = {k: (v() if callable(v) else v) for k, v in regs.items()}
    return {"ok": True, "registries": out}

# ---------- REAL FLOW: /process_urls -> enqueue analyze -> later /manifest --
@app.post("/process_urls")
def process_urls(body: ProcessUrlsIn):
    if not body.urls:
        raise HTTPException(400, "urls required")

    session_id = uuid.uuid4().hex
    files = [{"file_id": uuid.uuid4().hex[:8], "source": "url", "url": u}
             for u in body.urls]

    payload = {
        "session_id": session_id,
        "files": files,
        "tone": body.tone,
        "product_link": body.product_link,
        "features_csv": body.features_csv,
    }

    # store entire payload as bytes
    r.set(key_session(session_id), json.dumps(payload).encode("utf-8"))

    # enqueue analysis in the background
    job = q.enqueue("app.analyze_core_from_session", session_id, None, job_timeout=1800)

    return {"ok": True, "session_id": session_id, "job_id": job.id, "files": files}

@app.post("/analyze")
def analyze(body: AnalyzeIn):
    # manual enqueue of analysis (optional; /process_urls already enqueues)
    job = q.enqueue("app.analyze_core_from_session", body.session_id, None, job_timeout=1800)
    return {"ok": True, "job_id": job.id}

@app.post("/manifest")
def manifest(body: ManifestIn):
    # placeholder — just echo current session + analysis (if any)
    raw = r.get(key_session(body.session_id))
    if not raw:
        raise HTTPException(404, "unknown session_id")

    payload = json.loads(raw.decode("utf-8"))
    analysis_raw = r.hget(key_session(body.session_id) + b":analysis", b"json")
    analysis = json.loads(analysis_raw.decode("utf-8")) if analysis_raw else None

    return {
        "ok": True,
        "session_id": body.session_id,
        "manifest": {
            "segments": [{"file_id": f["file_id"], "start": 0.0, "end": 2.0}
                         for f in payload["files"][:1]]
        },
        "analysis": analysis,
    }

