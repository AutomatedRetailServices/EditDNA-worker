"""Mobile notification outbox routes for CutSell."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from cutsell_worker.notifications import list_notifications

router = APIRouter(prefix="/v1/notifications", tags=["notifications"])


@router.get("")
def get_notifications(user_id: str, limit: int = Query(default=30, ge=1, le=100)):
    try:
        return {"notifications": list_notifications(user_id=user_id, limit=limit)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
