"""Account lifecycle routes for persistent CutSell users."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cutsell_worker.account_lifecycle import delete_account_data

router = APIRouter(prefix="/v1/account", tags=["account"])


class AccountDeleteRequest(BaseModel):
    user_id: str
    confirmation: str


@router.delete("")
def delete_account(payload: AccountDeleteRequest):
    if payload.confirmation != "DELETE MY ACCOUNT":
        raise HTTPException(status_code=409, detail="account deletion requires confirmation DELETE MY ACCOUNT")
    try:
        return delete_account_data(user_id=payload.user_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
