"""Internal endpoints for private-S3 historical benchmarks."""

from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from rq.exceptions import NoSuchJobError
from rq.job import Job

import benchmark_s3
from jobs import enqueue_benchmark, get_queue

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


class BenchmarkRequest(BaseModel):
    source_prefixes: list[str] = Field(min_length=1, max_length=4)
    dataset_key: str
    mode: Literal["old_vs_new", "inventory_only"] = "old_vs_new"
    limit: int | None = Field(default=None, ge=1, le=1000)
    render_outputs: bool = False
    use_semantic_v2: bool = False
    use_take_judge_v2: bool = False

    @field_validator("source_prefixes")
    @classmethod
    def prefixes_are_allowed(cls, values):
        return [benchmark_s3.validate_input_prefix(value) for value in values]

    @field_validator("dataset_key")
    @classmethod
    def dataset_is_allowed(cls, value):
        return benchmark_s3.validate_dataset_key(value)


def fetch(job_id: str) -> Job:
    try: return Job.fetch(job_id, connection=get_queue().connection)
    except NoSuchJobError as exc: raise HTTPException(404, "Benchmark job not found") from exc


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
def run(request: BenchmarkRequest):
    job_id = f"benchmark-{uuid4().hex}"
    enqueue_benchmark(job_id, request.model_dump())
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = fetch(job_id); meta = job.meta or {}; current = job.get_status(refresh=True)
    total, processed = int(meta.get("total_sessions", 0)), int(meta.get("processed_sessions", 0))
    return {"status": getattr(current, "value", str(current)), "total_sessions": total, "processed_sessions": processed,
            "successful_sessions": int(meta.get("successful_sessions", max(0, processed-int(meta.get("failed_sessions", 0))))),
            "failed_sessions": int(meta.get("failed_sessions", 0)), "current_session": meta.get("current_session"),
            "progress_percent": round(processed * 100 / total, 1) if total else 0,
            "output_prefix": f"editdna/benchmarks/{job_id}/", "errors_count": int(meta.get("errors_count", 0))}


@router.get("/jobs/{job_id}/results")
def results(job_id: str):
    job = fetch(job_id); current = job.get_status(refresh=True)
    if getattr(current, "value", str(current)) != "finished": raise HTTPException(409, "Benchmark is not finished")
    result = job.result or {}; s3 = benchmark_s3.client()
    # Signed URLs are deliberately created only in this handler and never logged.
    return {"summary": result.get("summary", {}), "result_keys": result.get("result_keys", []),
            "urls": {key: benchmark_s3.presign_output(s3, key, job_id) for key in result.get("result_keys", [])}}
