# app.py — FastAPI + RQ web service (aligned with your s3_utils.py)

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from redis import Redis
from rq import Queue
from rq.job import Job

from s3_utils import parse_s3_url, AWS_REGION, S3_BUCKET

# -----------------------------------------------------------------------------
# Redis / RQ setup
# -----------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis = Redis.from_url(REDIS_URL)
queue = Queue("default", connection=redis, default_timeout=60 * 60)  # 60 min

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="editdna", version="1.0.0")

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class RenderRequest(BaseModel):
    session_id: Optional[str] = "session"
    files: List[str | HttpUrl]
    output_prefix: Optional[str] = "editdna/outputs"
    portrait: Optional[bool] = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _presign_if_needed(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    If the job result contains s3:// URIs, add an HTTPS presigned URL for easy viewing.
    - Supports result["output_s3"]
    - Supports result["outputs"] = [{ "output_s3": ... }, ...]
    """
    if not isinstance(result, dict):
        return result

    s3 = boto3.client("s3", region_name=AWS_REGION)

    def presign_one(s3_uri: str) -> Optional[str]:
        try:
            bucket, key = parse_s3_url(s3_uri)
            return s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket or S3_BUCKET, "Key": key},
                ExpiresIn=3600,
            )
        except ClientError:
            return None

    # Single output
    if "output_s3" in result and "output_url" not in result:
        url = presign_one(result["output_s3"])
        if url:
            result["output_url"] = url

    # Batch outputs
    if isinstance(result.get("outputs"), list):
        for item in result["outputs"]:
            if isinstance(item, dict) and "output_s3" in item and "output_url" not in item:
                url = presign_one(item["output_s3"])
                if url:
                    item["output_url"] = url

    return result


def _job_payload(job: Job) -> Dict[str, Any]:
    status = job.get_status(refresh=True)
    result = None
    error = None

    if status == "finished":
        result = job.result
        result = _presign_if_needed(result)  # add HTTPS if we have s3://
    elif status == "failed":
        error = job.exc_info

    return {
        "job_id": job.id,
        "status": status,
        "result": result,
        "error": error,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
    }


def _get_job_or_404(job_id: str) -> Job:
    try:
        return Job.fetch(job_id, connection=redis)
    except Exception:
        raise HTTPException(status_code=404, detail="Not Found")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "service": "editdna",
            "time": int(datetime.utcnow().timestamp()),
        }
    )


@app.post("/enqueue_nop")
def enqueue_nop() -> JSONResponse:
    job = queue.enqueue("worker.task_nop")
    return JSONResponse({"job_id": job.id})


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = _get_job_or_404(job_id)
    return JSONResponse(_job_payload(job))


@app.post("/render")
def render(req: RenderRequest) -> JSONResponse:
    """
    Enqueue a render job; accepts HTTPS S3 URLs. Worker uploads final MP4 to S3.
    """
    payload = {
        "session_id": req.session_id or "session",
        "files": [str(x) for x in req.files],
        "output_prefix": req.output_prefix or "editdna/outputs",
        "portrait": bool(req.portrait),
    }

    # Pass a single dict payload so both signatures are supported
    job = queue.enqueue("worker.job_render", payload)

    return JSONResponse(
        {
            "job_id": job.id,
            "session_id": payload["session_id"],
        }
    )
