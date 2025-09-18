# app.py — FastAPI + RQ (with /render_chunked)
import os, time
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

# ---- Match functions in worker.py ----
TASK_NOP           = "worker.task_nop"
TASK_CHECK_URLS    = "worker.check_urls"
TASK_ANALYZE       = "worker.analyze_session"
TASK_DIAG_OPENAI   = "worker.diag_openai"
TASK_NET_PROBE     = "worker.net_probe"
TASK_RENDER        = "worker.job_render"
TASK_RENDER_CHUNK  = "worker.job_render_chunked"

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="editdna API", version="1.4.0")
application = app  # alias for some ASGI hosts

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
# -------------------------
@app.post("/analyze")
def analyze(payload: dict = Body(...)):
    if not payload.get("session_id"):
        raise HTTPException(status_code=400, detail="Missing 'session_id'")
    job = q.enqueue(TASK_ANALYZE, payload, job_timeout=1800)
    return {"job_id": job.get_id(), "session_id": payload["session_id"]}

# -------------------------
# 3) Render (original all-at-once)
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
# 4) Render (chunked low-memory)
# -------------------------
@app.post("/render

          
