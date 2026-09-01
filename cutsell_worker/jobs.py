"""RQ job inspection/cancellation/retry for the clean CutSell service."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .queueing import enqueue_export, enqueue_flow_b, get_queue
from .usage_limits import release_processing_slot


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    state: str
    progress: int | None = None
    result: object | None = None
    error: str | None = None


def _normalize_status(status: object) -> str:
    if isinstance(status, Enum):
        raw = status.value
    else:
        raw = status
    text = str(raw or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


def _map_status(status: object) -> str:
    normalized = _normalize_status(status)
    mapping = {
        "queued": "uploaded",
        "deferred": "uploaded",
        "scheduled": "uploaded",
        "started": "analyzing",
        "finished": "finished",
        "failed": "failed",
        "stopped": "canceled",
        "canceled": "canceled",
        "cancelled": "canceled",
    }
    return mapping.get(normalized, normalized or "unknown")


def _fetch_job(job_id: str, connection):
    from rq.job import Job
    from rq.exceptions import NoSuchJobError
    try:
        return Job.fetch(job_id, connection=connection)
    except NoSuchJobError:
        raise KeyError(job_id) from None


def _job_payload(job) -> dict:
    args = tuple(getattr(job, "args", ()) or ())
    if len(args) == 1 and isinstance(args[0], dict):
        return dict(args[0])
    return {}


def _assert_job_owner(job, user_id: str | None) -> None:
    if user_id is None:
        return
    payload = _job_payload(job)
    owner = str(payload.get("user_id") or "")
    if not owner or owner != str(user_id):
        raise PermissionError("job does not belong to this user")


def fetch_job_snapshot(job_id: str, *, user_id: str | None = None, queue=None) -> JobSnapshot:
    target = queue or get_queue()
    job = _fetch_job(job_id, target.connection)
    _assert_job_owner(job, user_id)
    status = _normalize_status(job.get_status(refresh=True))
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


def cancel_job(job_id: str, *, user_id: str | None = None, queue=None) -> JobSnapshot:
    target = queue or get_queue()
    job = _fetch_job(job_id, target.connection)
    _assert_job_owner(job, user_id)
    status = _normalize_status(job.get_status(refresh=True))
    if status == "finished":
        return JobSnapshot(str(job.id), "finished", result=job.result)
    if status in {"failed", "stopped", "canceled", "cancelled"}:
        return JobSnapshot(str(job.id), _map_status(status))

    meta = dict(getattr(job, "meta", {}) or {})
    should_release = (
        status in {"queued", "deferred", "scheduled"}
        and bool(meta.get("cutsell_slot_reserved"))
        and bool(meta.get("user_id"))
    )
    job.cancel()
    if should_release:
        release_processing_slot(user_id=str(meta["user_id"]))
        try:
            job.meta["cutsell_slot_reserved"] = False
            job.save_meta()
        except Exception:
            pass
    return JobSnapshot(str(job.id), "canceled")


def retry_job(job_id: str, *, user_id: str, queue=None):
    """Create a fresh job from a failed/canceled CutSell job without mutating the original."""
    target = queue or get_queue()
    job = _fetch_job(job_id, target.connection)
    _assert_job_owner(job, user_id)
    status = _normalize_status(job.get_status(refresh=True))
    if status not in {"failed", "stopped", "canceled", "cancelled"}:
        raise ValueError("only failed or canceled jobs can be retried")
    payload = _job_payload(job)
    if not payload:
        raise ValueError("job payload is unavailable for safe retry")
    func_name = str(getattr(job, "func_name", "") or "")
    if func_name == "cutsell_worker.worker_job.run_flow_b_job":
        return enqueue_flow_b(payload, queue=target)
    if func_name == "cutsell_worker.export_job.run_export_job":
        return enqueue_export(payload, queue=target)
    raise ValueError("job type is not retryable by CutSell")
