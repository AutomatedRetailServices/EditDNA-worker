"""Redis/RQ queue wiring for CutSell jobs."""
from __future__ import annotations

from dataclasses import dataclass

from .config import load_runtime_config
from .usage_limits import release_processing_slot, reserve_processing_slot


@dataclass(frozen=True)
class QueueSubmission:
    job_id: str
    queue_name: str


def get_queue(name: str = "cutsell"):
    config = load_runtime_config()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL is not configured")
    from redis import Redis
    from rq import Queue
    connection = Redis.from_url(config.redis_url)
    return Queue(name, connection=connection)


def enqueue_flow_b(payload: dict, *, queue=None, timeout: int = 3600) -> QueueSubmission:
    user_id = str(payload.get("user_id") or "")
    if not user_id:
        raise ValueError("Flow B payload requires user_id")
    slot = reserve_processing_slot(user_id=user_id)
    if not slot.get("allowed"):
        raise RuntimeError("processing_concurrency_limit")
    try:
        target = queue or get_queue()
        job = target.enqueue(
            "cutsell_worker.worker_job.run_flow_b_job",
            payload,
            job_timeout=timeout,
            result_ttl=86400,
            failure_ttl=86400,
            meta={
                "user_id": user_id,
                "project_id": str(payload.get("project_id") or ""),
                "cutsell_slot_reserved": True,
            },
        )
        return QueueSubmission(job_id=str(job.id), queue_name=str(getattr(target, "name", "cutsell")))
    except Exception:
        release_processing_slot(user_id=user_id)
        raise


def enqueue_export(payload: dict, *, queue=None, timeout: int = 3600) -> QueueSubmission:
    target = queue or get_queue()
    job = target.enqueue(
        "cutsell_worker.export_job.run_export_job",
        payload,
        job_timeout=timeout,
        result_ttl=86400,
        failure_ttl=86400,
        meta={
            "user_id": str(payload.get("user_id") or ""),
            "project_id": str(payload.get("project_id") or ""),
        },
    )
    return QueueSubmission(job_id=str(job.id), queue_name=str(getattr(target, "name", "cutsell")))


def enqueue_batch_item(
    *, batch_id: str, user_id: str, index: int, queue=None, timeout: int = 3600
) -> QueueSubmission:
    target = queue or get_queue()
    job = target.enqueue(
        "cutsell_worker.batch_job.run_batch_item",
        batch_id,
        user_id,
        int(index),
        job_timeout=timeout,
        result_ttl=86400,
        failure_ttl=86400,
        meta={"user_id": user_id, "batch_id": batch_id, "batch_index": int(index)},
    )
    return QueueSubmission(job_id=str(job.id), queue_name=str(getattr(target, "name", "cutsell")))
