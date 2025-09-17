# app.py
import os, json, uuid, time
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import redis as redis_lib
from rq import Queue
from rq.job import Job, NoSuchJobError

REDIS_URL = os.getenv("REDIS_URL", "")
r = redis_lib.from_url(REDIS_URL, decode_responses=True)
q = Queue("default", connection=r)

app = FastAPI()

class ProcessUrlsIn(BaseModel):
    urls: list[str]
    session_id: str | None = None
    tone: str | None = None
    product_link: str | None = None
    features_csv: str | None = None

class AnalyzeIn(BaseModel):
    session_id: str

def nop_task():
    return {"echo": {"hello": "world"}}

def check_urls_task(payload: dict):
    out = []
    for u in payload.get("urls", []):
        try:
            import httpx
            with httpx.Client(follow_redirects=True, timeout=30) as c:
                r = c.head(u)
            size = int(r.headers.get("content-length", "0")) if r.status_code == 200 else None
            out.append({"url": u, "status": "OK" if r.status_code == 200 else "BAD",
                        "http": r.status_code, "size": size})
        except Exception as e:
            out.append({"url": u, "status": "ERROR", "http": None, "size": None, "error": str(e)})
    return {"checked": out}

def analyze_task(session_id: str):
    # placeholder – returns when worker runs something for the session
    return {"echo": {"sess": session_id}}

@app.post("/enqueue_nop")
def enqueue_nop():
    job = q.enqueue(nop_task)
    return {"ok": True, "job_id": job.get_id()}

@app.post("/process_urls")
def process_urls(body: ProcessUrlsIn):
    sess = body.session_id or str(uuid.uuid4())
    job = q.enqueue(check_urls_task, {"urls": body.urls})
    return {"ok": True, "job_id": job.get_id(), "session_id": sess}

@app.post("/analyze")
def analyze(body: AnalyzeIn):
    job = q.enqueue(analyze_task, body.session_id)
    return {"ok": True, "job_id": job.get_id()}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = Job.fetch(job_id, connection=r)
    except NoSuchJobError:
        raise HTTPException(status_code=404, detail="unknown_job_id")
    payload = {
        "job_id": job_id,
        "status": job.get_status(refresh=True),
        "result": job.result,
        "error": job.meta.get("error") if job.meta else None,
        "enqueued_at": getattr(job, "enqueued_at", None),
        "ended_at": getattr(job, "ended_at", None),
    }
    return JSONResponse(payload)

@app.get("/admin/health")
def health():
    # Basic rq registry counts
    from rq.registry import FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry, ScheduledJobRegistry, DeferredJobRegistry
    return {
        "ok": True,
        "registries": {
            "queued": q.count,
            "started": StartedJobRegistry(queue=q).count,
            "scheduled": ScheduledJobRegistry(queue=q).count,
            "deferred": DeferredJobRegistry(queue=q).count,
            "failed": FailedJobRegistry(queue=q).count,
            "finished": FinishedJobRegistry(queue=q).count,
        },
    }

@app.get("/admin/debug")
def debug():
    # Help confirm both services share exactly the same Redis view
    info = r.info()
    key = "debug:echo"
    token = str(uuid.uuid4())
    r.set(key, token, ex=60)
    readback = r.get(key)
    return {
        "redis_url_tail": REDIS_URL[-60:],   # last 60 chars so you can compare
        "redis_db": info.get("db0", info.get("db", "?")),
        "redis_mode": info.get("redis_mode"),
        "server": {"version": info.get("redis_version"), "proto": info.get("resp")},
        "echo_roundtrip_ok": readback == token,
        "queue_name": q.name,
    }
