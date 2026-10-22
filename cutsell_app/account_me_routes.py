"""You · Account card: the signed-in account's email (from Sign in with Apple, stored at sign-in) and the name the
creator chose (Apple gives the name only once, so the app sends it; she can change it). Guests have neither."""
from __future__ import annotations

import hashlib

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/account", tags=["account"])


def _name_key(user_id: str) -> str:
    return "cutsell:v1:account_name:" + hashlib.sha256(user_id.encode()).hexdigest()[:24]


def _email(user_id: str) -> str | None:
    from cutsell_worker.config import load_runtime_config
    config = load_runtime_config()
    if not config.database_url:
        return None
    from sqlalchemy import text
    from cutsell_worker.commercial_store import _engine
    with _engine(config.database_url).connect() as conn:
        row = conn.execute(text("SELECT email FROM cutsell_users WHERE user_id=:u"), {"u": user_id}).first()
    return str(row[0]) if row and row[0] else None


def _redis():
    from cutsell_worker.project_store import _redis_client
    return _redis_client(None)


@router.get("/me")
def get_me(user_id: str):
    try:
        raw = _redis().get(_name_key(user_id))
        name = raw.decode() if isinstance(raw, bytes) else raw
    except Exception:
        name = None
    try:
        email = _email(user_id)
    except Exception:
        email = None
    return {"user_id": user_id, "email": email, "name": name or None}


class NameRequest(BaseModel):
    user_id: str
    name: str = Field(min_length=1, max_length=60)


@router.put("/me")
def set_name(payload: NameRequest):
    name = " ".join(payload.name.split())
    if not name:
        raise HTTPException(status_code=422, detail="name is empty")
    try:
        _redis().set(_name_key(payload.user_id), name)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return get_me(payload.user_id)
