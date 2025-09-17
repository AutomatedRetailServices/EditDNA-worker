# app.py — FastAPI + RQ 1.16, returns session_id right away
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

# IMPORTANT: keep decode_responses=False
conn = redis.from_url(REDIS_URL, decode_responses=False)
q = Queue("default", connection=conn)

# RQ resolves these by import string in the worker process
TASK_NOP = "worker.task_nop"
TASK_CHECK_URLS = "worker.check_urls"
TASK_ANALYZE_SESSION = "worker.analyze_session"

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="editdna API", version="1.0.2")

@app.get("/")
def root():
    return {"ok": True, "service": "editdna", "time": int(time.time())}

@app.get("/admin/health")
def health():
    try:
        conn.ping()
        return {"status": "ok"}
    except Exception as e
