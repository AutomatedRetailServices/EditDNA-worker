import os, json, time
import redis
import requests

# ENV required:
# REDIS_URL  = redis connection string (e.g., rediss://:password@host:port/0)
# API_BASE   = public URL of your API service, e.g. https://script2clipshop-worker.onrender.com

REDIS_URL = os.getenv("REDIS_URL", "")
API_BASE  = os.getenv("API_BASE", "").rstrip("/")

if not REDIS_URL:
    raise SystemExit("ERROR: REDIS_URL is not set")
if not API_BASE:
    raise SystemExit("ERROR: API_BASE is not set (e.g. https://script2clipshop-worker.onrender.com)")

r = redis.from_url(REDIS_URL, decode_responses=True)

def _job_key(job_id: str) -> str:
    return f"job:{job_id}"

def _update(job: dict, **fields):
    job.update(fields)
    r.set(_job_key(job["job_id"]), json.dumps(job))

def _handle_stitch(job: dict):
    _update(job, status="running", started_at=int(time.time()))
    try:
        # Call your API's synchronous /stitch (same payload), but from this worker.
        payload = {
            "session_id": job["session_id"],
            "filename": job["filename"],
            "manifest": job["manifest"]
        }
        resp = requests.post(f"{API
