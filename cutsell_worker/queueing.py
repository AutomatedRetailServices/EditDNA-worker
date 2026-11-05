"""Redis/RQ queue wiring for CutSell jobs."""
from __future__ import annotations

import threading

from dataclasses import dataclass

from .config import load_runtime_config
from .usage_limits import bulk_counter, release_processing_slot, reserve_processing_slot, work_gate
from .worker_wake import wake_worker


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


BULK_MAX_VIDEOS = 10


class ProcessingLimitError(RuntimeError):
    """The user already has as many cuts processing as allowed (API answers 429, the app retries later)."""


def bulk_extra_slots(payload: dict, *, client=None) -> int:
    """A Bulk send's videos run side by side (Batch API: up to 30 min each), so the send gets one slot per video,
    at most BULK_MAX_VIDEOS. Single / Combine: 0.

    PO 5 Nov (security review): the phone's "bulk_size" is no longer believed on its own. The server counts the
    videos it has received for that Bulk (its bulk_id): the Nth video of a send may use N slots, never more than
    the size the phone announced, never more than BULK_MAX_VIDEOS. One request claiming "Bulk of 10" gets 1."""
    size, bulk_id = 0, ""
    for src in payload.get("sources") or []:
        meta = (src or {}).get("metadata") or {}
        try:
            size = max(size, int(str(meta.get("bulk_size") or "0")))
        except ValueError:
            continue
        bulk_id = bulk_id or str(meta.get("bulk_id") or "")
    if size <= 1:
        return 0
    received = bulk_counter(user_id=str(payload.get("user_id") or ""), bulk_id=bulk_id, client=client) \
        if payload.get("user_id") and bulk_id else None
    if received is None:
        received = size              # no counter reachable: no slot limit is enforced without Redis either
    return max(0, min(received, size, BULK_MAX_VIDEOS) - 1)


def guard(user_id: str, kind: str) -> None:
    """The one gate for paid work (usage_limits.work_gate): raises ProcessingLimitError (API: 429)."""
    if not kind:
        return
    reason = work_gate(user_id=str(user_id or ""), kind=kind)
    if reason:
        raise ProcessingLimitError(reason)


def enqueue_flow_b(payload: dict, *, queue=None, timeout: int = 3600) -> QueueSubmission:
    user_id = str(payload.get("user_id") or "")
    reserved = False
    if user_id:
        slot = reserve_processing_slot(user_id=user_id, extra=bulk_extra_slots(payload))
        if not slot.get("allowed"):
            raise ProcessingLimitError("processing_concurrency_limit")
        reserved = True
    try:
        target = queue or get_queue()
        meta = {
            "user_id": user_id,
            "project_id": str(payload.get("project_id") or ""),
            "cutsell_slot_reserved": reserved,
        }
        job = target.enqueue(
            "cutsell_worker.worker_job.run_flow_b_job",
            payload,
            job_timeout=timeout,
            result_ttl=86400,
            failure_ttl=86400,
            meta=meta,
        )
        # PO 28 Oct (speed): waking the worker runs alongside, like every other job here; the cut's submit
        # used to wait for it (measured 2.0 s warm, 5.4 s when the wake endpoint itself was cold).
        threading.Thread(target=wake_worker, name="wake-worker", daemon=True).start()
        return QueueSubmission(job_id=str(job.id), queue_name=str(getattr(target, "name", "cutsell")))
    except Exception:
        if reserved:
            release_processing_slot(user_id=user_id)
        raise


def enqueue_video_followup(payload: dict, *, queue=None, kind: str | None = "video_ready", timeout: int = 1200) -> QueueSubmission:
    """Audio first (iOS build 23): the video of an audio-first source finished uploading."""
    guard(str(payload.get("user_id") or ""), kind)
    target = queue or get_queue()
    job = target.enqueue(
        "cutsell_worker.worker_job.run_video_followup_job",
        payload,
        job_timeout=timeout,
        result_ttl=86400,
        failure_ttl=86400,
        meta={"user_id": str(payload.get("user_id") or ""), "project_id": str(payload.get("project_id") or "")},
    )
    threading.Thread(target=wake_worker, name="wake-worker", daemon=True).start()
    return QueueSubmission(job_id=str(job.id), queue_name=str(getattr(target, "name", "cutsell")))


def enqueue_mix(payload: dict, *, queue=None, kind: str | None = "mix", timeout: int = 3600) -> QueueSubmission:
    """One job for a whole Mix (cutsell_worker.mix_job). Started again by video-ready / retry: it is resumable."""
    guard(str(payload.get("user_id") or ""), kind)
    target = queue or get_queue()
    job = target.enqueue(
        "cutsell_worker.mix_job.run_mix_job",
        payload,
        job_timeout=timeout,
        result_ttl=86400,
        failure_ttl=86400,
        meta={"user_id": str(payload.get("user_id") or ""), "project_id": str(payload.get("mix_id") or ""), "mix": True},
    )
    threading.Thread(target=wake_worker, name="wake-worker", daemon=True).start()
    return QueueSubmission(job_id=str(job.id), queue_name=str(getattr(target, "name", "cutsell")))


def enqueue_captions(payload: dict, *, queue=None, timeout: int = 900) -> QueueSubmission:
    """Plan my week: posting captions for planned videos that do not have one yet (written once)."""
    target = queue or get_queue()
    job = target.enqueue("cutsell_worker.post_captions.run_caption_job", payload, job_timeout=timeout,
                         result_ttl=86400, failure_ttl=86400,
                         meta={"user_id": str(payload.get("user_id") or ""), "captions": True})
    threading.Thread(target=wake_worker, name="wake-worker", daemon=True).start()
    return QueueSubmission(job_id=str(job.id), queue_name=str(getattr(target, "name", "cutsell")))


def enqueue_export(payload: dict, *, queue=None, kind: str | None = "export", timeout: int = 3600) -> QueueSubmission:
    guard(str(payload.get("user_id") or ""), kind)
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
    # The export is queued; waking the worker runs alongside so the iPhone gets its job id at once
    # (it used to wait up to ~2 s for the wake call before Export could show anything).
    threading.Thread(target=wake_worker, name="wake-worker", daemon=True).start()
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
    wake_worker()
    return QueueSubmission(job_id=str(job.id), queue_name=str(getattr(target, "name", "cutsell")))


def enqueue_unseen_clean_cut_benchmark(
    payload: dict,
    *,
    queue=None,
    timeout: int = 10800,
) -> QueueSubmission:
    """Enqueue the paid-compute benchmark on the same RunPod GPU worker as product jobs."""
    target = queue or get_queue()
    benchmark_id = str(payload.get("benchmark_id") or "")
    job = target.enqueue(
        "cutsell_worker.validation_job.run_unseen_clean_cut_benchmark",
        payload,
        job_timeout=timeout,
        result_ttl=86400,
        failure_ttl=86400,
        meta={
            "benchmark_id": benchmark_id,
            "brain_backend": "runpod_local",
            "external_brain_calls_enabled": False,
        },
    )
    return QueueSubmission(job_id=str(job.id), queue_name=str(getattr(target, "name", "cutsell")))


def enqueue_clean_audio(payload: dict, *, queue=None, kind: str | None = "clean_audio", timeout: int = 1800) -> QueueSubmission:
    """Editor v5 Clean up voice: make the cleaned copy of one sound on the worker (audio_clean)."""
    guard(str(payload.get("user_id") or ""), kind)
    target = queue or get_queue()
    job = target.enqueue(
        "cutsell_worker.audio_clean.run_clean_audio_job",
        payload,
        job_timeout=timeout,
        result_ttl=86400,
        failure_ttl=86400,
        meta={"user_id": str(payload.get("user_id") or ""), "project_id": str(payload.get("project_id") or "")},
    )
    threading.Thread(target=wake_worker, name="wake-worker", daemon=True).start()
    return QueueSubmission(job_id=str(job.id), queue_name=str(getattr(target, "name", "cutsell")))
