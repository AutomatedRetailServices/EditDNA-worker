"""HTTP endpoints for submitting and inspecting render jobs."""

from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Body, HTTPException, status
from pydantic import AnyHttpUrl, BaseModel, Field, TypeAdapter, model_validator
from rq.exceptions import NoSuchJobError
from rq.job import Job

from jobs import enqueue_render, get_queue

router = APIRouter()


class RenderRequest(BaseModel):
    """Canonical API request; ``files`` and ``file_urls`` are aliases."""

    session_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    input_url: Optional[str] = None
    files: Optional[List[AnyHttpUrl]] = None
    file_urls: Optional[List[AnyHttpUrl]] = None
    mode: Literal["clean", "human", "blooper"] = "human"

    @model_validator(mode="after")
    def require_input_url(self) -> "RenderRequest":
        if self.input_url:
            # Deprecated single-input compatibility alias. Validation still
            # happens at the API boundary, before a job can be enqueued.
            TypeAdapter(AnyHttpUrl).validate_python(self.input_url)
        if not self.input_url and not self.files and not self.file_urls:
            raise ValueError("at least one URL must be provided in input_url, files, or file_urls")
        return self

    def canonical_urls(self) -> List[str]:
        """Combine all compatibility aliases, de-duplicating in input order."""
        values = ([self.input_url] if self.input_url else []) + [
            str(url) for group in (self.files, self.file_urls) if group for url in group
        ]
        return list(dict.fromkeys(str(TypeAdapter(AnyHttpUrl).validate_python(url)) for url in values))


@router.post("/render", status_code=status.HTTP_202_ACCEPTED)
async def render(payload: RenderRequest = Body(...)) -> Dict[str, Any]:
    """Validate and enqueue a render without executing the media pipeline inline."""
    session_id = payload.session_id or f"render-{uuid4().hex}"
    job = enqueue_render(
        session_id=session_id,
        files=None,
        file_urls=payload.canonical_urls(),
        mode=payload.mode,
    )
    return {
        "job_id": job.id,
        "session_id": session_id,
        "status": "queued",
        "mode": payload.mode,
    }


def _status_value(job: Job) -> str:
    job_status = job.get_status(refresh=True)
    return getattr(job_status, "value", str(job_status))


def _public_progress(job: Job, current_status: str) -> Dict[str, Any]:
    meta = job.meta if isinstance(getattr(job, "meta", None), dict) else {}
    stage = meta.get("stage")
    progress = meta.get("progress", 0)
    messages = {
        "queued": "Render queued",
        "downloading": "Downloading sources",
        "analyzing": "Analyzing sources",
        "selecting": "Selecting clips",
        "rendering": "Rendering video",
        "uploading": "Uploading rendered video",
        "finished": "Render finished",
        "failed": "Render failed",
        "canceled": "Render canceled",
    }
    if current_status == "finished":
        stage, progress = "finished", 100
    elif current_status == "failed":
        stage = "failed"
    elif current_status in ("canceled", "stopped") or stage == "canceled":
        stage = "canceled"
    elif stage not in {"queued", "downloading", "analyzing", "selecting", "rendering", "uploading"}:
        stage, progress = "queued", 0
    try:
        progress = min(100, max(0, int(progress)))
    except (TypeError, ValueError):
        progress = 0
    return {
        "stage": stage,
        "progress": progress,
        "message": messages[stage],
        "cancel_requested": bool(meta.get("cancel_requested", False)),
    }


@router.get("/jobs/{job_id}")
async def job_status(job_id: str) -> Dict[str, Any]:
    """Return a JSON-safe, deliberately limited view of an RQ job."""
    try:
        job = Job.fetch(job_id, connection=get_queue().connection)
    except NoSuchJobError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    current_status = _status_value(job)
    public_progress = _public_progress(job, current_status)
    return {
        "job_id": job.id,
        "status": current_status,
        **public_progress,
        "result": job.result if current_status == "finished" else None,
        "error": "Render job failed" if current_status == "failed" else None,
    }


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    """Cancel waiting work or request cancellation at the next safe boundary."""
    try:
        job = Job.fetch(job_id, connection=get_queue().connection)
    except NoSuchJobError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    current_status = _status_value(job)
    meta = job.meta if isinstance(getattr(job, "meta", None), dict) else {}
    if current_status in ("finished", "failed"):
        raise HTTPException(status_code=409, detail=f"Cannot cancel {current_status} job")
    if current_status in ("canceled", "stopped") or meta.get("stage") == "canceled":
        return {"job_id": job.id, "status": "canceled", "cancel_requested": True}

    if current_status in ("queued", "deferred", "scheduled"):
        job.cancel()
        job.meta.update(
            cancel_requested=True,
            stage="canceled",
            message="Render canceled",
        )
        job.save_meta()
        return {"job_id": job.id, "status": "canceled", "cancel_requested": True}
    if current_status == "started":
        job.meta["cancel_requested"] = True
        job.save_meta()
        return {"job_id": job.id, "status": "started", "cancel_requested": True}

    raise HTTPException(status_code=409, detail=f"Cannot cancel {current_status} job")
