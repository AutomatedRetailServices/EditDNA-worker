"""RQ job inspection/cancellation for the clean CutSell service."""
from __future__ import annotations

from dataclasses import dataclass

from .queueing import get_queue


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    state: str
    progress: int | None = None
    result: object | None = None
    error: str | None = None


def _map_status(status: str) -> str:
    mapping = {
        "queued": "uploaded",
        "deferred": "uploaded",
        "scheduled": "uploaded",
        "started": "analyzing",
        "finished": "finished",
        "failed": "failed",
        "stopped": "canceled",
        "canceled": "canceled",
    }
    return mapping.get(status, status or "unknown")


def fetch_job_snapshot(job_id: str, *, queue=None) -> JobSnapshot:
    target = queue or get_queue()
    connection = target.connection
    from rq.job import Job
    from rq.exceptions import NoSuchJobError

    try:
        job = Job.fetch(job_id, connection=connection)
    except NoSuchJobError:
        raise KeyError(job_id) from None

    status = str(job.get_status(refresh=True))
    meta = dict(getattr(job, "meta", {}) or {})
    progress = meta.get("progress_percent")
    if progress is not None:
        try:
            progress = max(0, min(100, int(progress)))
        except (TypeError, ValueError):
            progress = None
    error = None
    if status == "failed":
        error = str(meta.get("error_code") or "processing_failed")
    return JobSnapshot(
        job_id=str(job.id),
        state=_map_status(status),
        progress=progress,
        result=(job.result if status == "finished" else None),
        error=error,
    )


def cancel_job(job_id: str, *, queue=None) -> JobSnapshot:
    target = queue or get_queue()
    connection = target.connection
    from rq.job import Job
    from rq.exceptions import NoSuchJobError

    try:
        job = Job.fetch(job_id, connection=connection)
    except NoSuchJobError:
        raise KeyError(job_id) from None

    status = str(job.get_status(refresh=True))
    if status == "finished":
        return JobSnapshot(str(job.id), "finished", result=job.result)
    if status in {"failed", "stopped", "canceled"}:
        return JobSnapshot(str(job.id), _map_status(status))
    job.cancel()
    return JobSnapshot(str(job.id), "canceled")
