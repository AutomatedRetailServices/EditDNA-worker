"""23 Oct: the iPhone registers its push token; the worker sends "your videos are ready" (also with CutSell closed)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cutsell_worker import push

router = APIRouter(prefix="/v1/push", tags=["push"])


class PushRegisterRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=200)
    token: str = Field(min_length=32, max_length=200)
    language: str = "en"
    environment: str = "production"


@router.post("/register")
def register(payload: PushRegisterRequest):
    """Saves this iPhone's token. push_active: the server sends the "ready" notifications (the app then posts none
    of its own for them)."""
    try:
        push.register(user_id=payload.user_id, token=payload.token, language=payload.language,
                      environment=payload.environment)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return {"ok": True, "push_active": push.active()}
