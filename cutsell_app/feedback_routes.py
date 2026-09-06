"""Creator feedback endpoint for CutSell production evaluation."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cutsell_worker.feedback import build_feedback_event, store_feedback_event

router = APIRouter(prefix="/v1/projects", tags=["feedback"])


class FeedbackRequest(BaseModel):
    user_id: str
    rating: str
    draft: dict[str, Any]
    reason: str | None = None
    clip_id: str | None = None
    time_sec: float | None = None
    processing_metrics: dict[str, Any] | None = None


@router.post("/{project_id}/feedback")
def submit_feedback(project_id: str, payload: FeedbackRequest):
    try:
        event = build_feedback_event(
            user_id=payload.user_id,
            project_id=project_id,
            rating=payload.rating,
            draft=payload.draft,
            reason=payload.reason,
            clip_id=payload.clip_id,
            time_sec=payload.time_sec,
            processing_metrics=payload.processing_metrics,
        )
        return store_feedback_event(
            event,
            user_id=payload.user_id,
            project_id=project_id,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
