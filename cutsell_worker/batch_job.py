"""RQ wrapper that processes CutSell batch items strictly one after another."""
from __future__ import annotations

from .batch import get_batch, update_batch_item
from .queueing import enqueue_batch_item, get_queue
from .worker_job import run_flow_b_job


def run_batch_item(batch_id: str, user_id: str, index: int) -> dict:
    from rq import get_current_job

    current_job = get_current_job()
    job_id = str(getattr(current_job, "id", "")) or None
    record = get_batch(user_id=user_id, batch_id=batch_id, include_payloads=True)
    payloads = list(record.get("payloads") or ())
    if index < 0 or index >= len(payloads):
        raise IndexError("batch item index is invalid")

    update_batch_item(
        user_id=user_id,
        batch_id=batch_id,
        index=index,
        state="processing",
        job_id=job_id,
    )

    state = "finished"
    result = None
    error = None
    try:
        result = run_flow_b_job(dict(payloads[index]))
    except Exception as exc:
        state = "failed"
        error = exc.__class__.__name__
    finally:
        update_batch_item(
            user_id=user_id,
            batch_id=batch_id,
            index=index,
            state=state,
            job_id=job_id,
            result=result,
            error=error,
        )
        next_index = index + 1
        if next_index < len(payloads):
            queue = get_queue()
            submission = enqueue_batch_item(
                batch_id=batch_id,
                user_id=user_id,
                index=next_index,
                queue=queue,
            )
            update_batch_item(
                user_id=user_id,
                batch_id=batch_id,
                index=next_index,
                state="queued",
                job_id=submission.job_id,
            )

    return {
        "batch_id": batch_id,
        "index": index,
        "state": state,
        "error": error,
        "result": result,
    }
