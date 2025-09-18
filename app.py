from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis import Redis
from rq import Queue
import os
import uuid

# Import your job functions
from jobs import analyze_session, render_from_files

# Initialize FastAPI app
app = FastAPI()

# Connect to Redis
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(redis_url)
q = Queue("default", connection=redis_conn)

# Health check
@app.get("/")
async def root():
    return {"status": "ok", "message": "EditDNA.ai API is running"}

# Echo test
@app.post("/enqueue_nop")
async def enqueue_nop(request: Request):
    job = q.enqueue(lambda: {"echo": {"hello": "world"}})
    return {"job_id": job.id}

# Check uploaded URLs
@app.post("/process_urls")
async def process_urls(request: Request):
    payload = await request.json()
    files = payload.get("files", [])
    if not files:
        return JSONResponse(content={"error": "No files provided"}, status_code=400)

    session_id = f"sess-{uuid.uuid4().hex[:8]}"
    return {"ok": True, "session_id": session_id, "files": files}

# Analyze session with OpenAI
@app.post("/analyze")
async def analyze(request: Request):
    payload = await request.json()
    session_id = payload.get("session_id")
    product_link = payload.get("product_link")
    features_csv = payload.get("features_csv", "")
    tone = payload.get("tone", "casual")

    if not session_id or not product_link:
        return JSONResponse(
            content={"error": "Missing session_id or product_link"}, status_code=400
        )

    job = q.enqueue(analyze_session, session_id, product_link, features_csv, tone)
    return {"job_id": job.id, "session_id": session_id}

# Render final video from uploaded files
@app.post("/render")
async def render(request: Request):
    payload = await request.json()
    session_id = payload.get("session_id")
    files = payload.get("files", [])

    if not session_id or not files:
        return JSONResponse(
            content={"error": "Missing session_id or files"}, status_code=400
        )

    job = q.enqueue(render_from_files, session_id, files)
    return {"job_id": job.id, "session_id": session_id}

# Check job status
@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    from rq.job import Job

    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        return JSONResponse(content={"error": "Job not found"}, status_code=404)

    return {
        "job_id": job.id,
        "status": job.get_status(),
        "result": job.result,
        "error": job.exc_info,
        "enqueued_at": job.enqueued_at,
        "ended_at": job.ended_at,
    }
