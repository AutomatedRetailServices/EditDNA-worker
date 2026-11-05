"""Safe retry endpoint for failed CutSell processing/render jobs."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cutsell_worker.jobs import retry_job
from cutsell_worker.queueing import ProcessingLimitError, guard

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


class RetryRequest(BaseModel):
    user_id: str


@router.post("/{job_id}/retry")
def retry_failed_job(job_id: str, payload: RetryRequest):
    try:
        guard(payload.user_id, "retry")          # PO 5 Nov: the one gate for paid work
        submission = retry_job(job_id, user_id=payload.user_id)
        return {
            "original_job_id": job_id,
            "job_id": submission.job_id,
            "queue": submission.queue_name,
            "state": "uploaded",
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found") from None
    except PermissionError:
        raise HTTPException(status_code=403, detail="job does not belong to this user") from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ProcessingLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "60"}) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
