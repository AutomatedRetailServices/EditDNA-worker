# app.py — FastAPI + RQ (web enqueues by string; worker imports the code)
import os
import time
import redis
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
from rq import Queue
from rq.job import Job

# -------------------------
# Redis / RQ
# -------------------------
REDIS_URL = os.getenv("REDIS_URL", "")
if not REDIS_URL:
    raise RuntimeError("Missing REDIS_URL")

# IMPORTANT: keep decode_responses=False (RQ stores bytes)
conn = redis.from_url(REDIS_URL, decode_responses=False)
q = Queue("default", connection=conn)

# ---- Job function paths (these live in worker.py) ----
TASK_NOP            = "worker.task_nop"
TASK_CHECK_URLS     = "worker.check_urls"
TASK_ANALYZE        = "worker.analyze_session"
TASK_DIAG_OPENAI    = "worker.diag_openai"
TASK_NET_PROBE      = "worker.net_probe"
TASK_RENDER         = "worker.job_render"
TASK_RENDER_CHUNKED = "worker.job_render_chunked"

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="editdna API", version="1.5.0")
application = app  # alias some hosts expect

@app.get("/")
def root():
    return {"ok": True, "service": "editdna", "time": int(time.time())}

@app.get("/admin/health")
def health():
    try:
        conn.ping()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# 0) Tiny test job
# -------------------------
@app.post("/enqueue_nop")
def enqueue_nop(payload: dict = Body(default={"echo": {"hello": "world"}})):
    job = q.enqueue(TASK_NOP, job_timeout=300)
    return {"job_id": job.get_id()}

# -------------------------
# 1) Check URLs
# Body: {"urls": ["https://...","s3://bucket/key", ...], "session_id": "optional"}
# -------------------------
@app.post("/process_urls")
def process_urls(payload: dict = Body(...)):
    urls = payload.get("urls")
    if not urls or not isinstance(urls, list):
        raise HTTPException(status_code=400, detail="Provide 'urls': [ ... ]")
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    job = q.enqueue(TASK_CHECK_URLS, {"urls": urls, "session_id": session_id}, job_timeout=600)
    return {"job_id": job.get_id(), "session_id": session_id}

# -------------------------
# 2) Analyze (OpenAI or stub)
# Body: { "session_id": "...", "product_link": "...", "features_csv": "a,b,c", "tone": "casual" }
# -------------------------
@app.post("/analyze")
def analyze(payload: dict = Body(...)):
    if not payload.get("session_id"):
        raise HTTPException(status_code=400, detail="Missing 'session_id'")
    job = q.enqueue(TASK_ANALYZE, payload, job_timeout=1800)
    return {"job_id": job.get_id(), "session_id": payload["session_id"]}

# -------------------------
# 3) Render (all-at-once)
# Body: { "session_id": "...", "files": ["https://...","s3://..."], "output_prefix": "editdna/outputs" }
# -------------------------
@app.post("/render")
def render(payload: dict = Body(...)):
    session_id = payload.get("session_id")
    files = payload.get("files")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing 'session_id'")
    if not files or not isinstance(files, list):
        raise HTTPException(status_code=400, detail="Provide 'files': ['s3://...','https://...']")
    output_prefix = payload.get("output_prefix") or "editdna/outputs"
    job = q.enqueue(
        TASK_RENDER,
        {"session_id": session_id, "files": files, "output_prefix": output_prefix},
        job_timeout=60 * 60,
    )
    return {"job_id": job.get_id(), "session_id": session_id}

# -------------------------
# 4) Render (chunked — low memory, recommended)
# Body: { "session_id": "...", "files": [...], "output_prefix": "editdna/outputs" }
# -------------------------
@app.post("/render_chunked")
def render_chunked(payload: dict = Body(...)):
    session_id = payload.get("session_id")
    files = payload.get("files")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing 'session_id'")
    if not files or not isinstance(files, list):
        raise HTTPException(status_code=400, detail="Provide 'files': ['s3://...','https://...']")
    output_prefix = payload.get("output_prefix") or "editdna/outputs"
    job = q.enqueue(
        TASK_RENDER_CHUNKED,
        {"session_id": session_id, "files": files, "output_prefix": output_prefix},
        job_timeout=60 * 60,
    )
    return {"job_id": job.get_id(), "session_id": session_id}

# -------------------------
# 5) Diagnostics
# -------------------------
@app.post("/diag/openai")
def diag_openai():
    job = q.enqueue(TASK_DIAG_OPENAI, job_timeout=180)
    return {"job_id": job.get_id()}

@app.post("/diag/net")
def diag_net():
    job = q.enqueue(TASK_NET_PROBE, job_timeout=120)
    return {"job_id": job.get_id()}

# -------------------------
# 6) Poll a job
# -------------------------
@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = Job.fetch(job_id, connection=conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    data = {
        "job_id": job.get_id(),
        "status": job.get_status(),
        "result": None,
        "error": None,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
    }

    if job.is_failed:
        data["error"] = str(job.exc_info or (job.meta or {}).get("error") or "Job failed")
    elif job.is_finished:
        data["result"] = job.result

    return JSONResponse(data)
