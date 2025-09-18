# app.py — FastAPI + RQ (Redis Queue)
import os
import time
from typing import Dict, Any, List, Optional

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
# We keep your existing queue name to match render.yaml
q = Queue("default", connection=conn)

# ---- Match the functions that exist in worker.py ----
TASK_NOP           = "worker.task_nop"
TASK_CHECK_URLS    = "worker.check_urls"        # ✅ matches worker.py
TASK_ANALYZE       = "worker.analyze_session"   # ✅ matches worker.py
TASK_DIAG_OPENAI   = "worker.diag_openai"
TASK_NET_PROBE     = "worker.net_probe"
TASK_RENDER        = "worker.job_render"        # ✅ NEW: added in worker.py step 3

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="editdna API", version="1.3.0")

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
# 1) Check S3/public URLs — returns job_id AND session_id
# Body: { "urls": ["https://.../IMG_0001.mov", ...] , optional "session_id": "sess-..." }
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
# 2) Analyze (OpenAI if available, else stub)
# Body: { "session_id": "...", "tone": "...", "product_link": "...", "features_csv": "a,b,c" }
# -------------------------
@app.post("/analyze")
def analyze(payload: dict = Body(...)):
    if not payload.get("session_id"):
        raise HTTPException(status_code=400, detail="Missing 'session_id'")
    job = q.enqueue(TASK_ANALYZE, payload, job_timeout=1800)
    return {"job_id": job.get_id(), "session_id": payload["session_id"]}

# -------------------------
# 3) NEW — Render job (downloads from S3 → ffmpeg stitch → uploads MP4)
# Body:
# {
#   "session_id": "sess-...",
#   "files": ["s3://bucket/raw/a.mov", "s3://bucket/raw/b.mov"],
#   "output_prefix": "editdna/outputs"   # optional
# }
# -------------------------
@app.post("/render")
def render(payload: dict = Body(...)):
    session_id = payload.get("session_id
