"""Closed-beta auth bootstrap for the CutSell mobile client."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from cutsell_worker.auth import create_session

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/session")
def create_mobile_session():
    try:
        return create_session()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
