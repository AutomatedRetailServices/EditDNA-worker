"""Closed-beta bootstrap plus persistent Sign in with Apple authentication."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cutsell_worker.apple_auth import verify_apple_identity_token
from cutsell_worker.auth import create_session, stable_apple_user_id
from cutsell_worker.commercial_observability import capture_operational_event
from cutsell_worker.config import load_runtime_config

router = APIRouter(prefix="/v1/auth", tags=["auth"])


class AppleSessionRequest(BaseModel):
    identity_token: str
    nonce: str | None = None


@router.post("/session")
def create_mobile_session():
    """Legacy anonymous closed-beta bootstrap. Do not use for commercial accounts."""
    try:
        return create_session()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.post("/apple")
def create_apple_session(payload: AppleSessionRequest):
    try:
        identity = verify_apple_identity_token(payload.identity_token, nonce=payload.nonce)
        user_id = stable_apple_user_id(identity.subject)
        config = load_runtime_config()
        durable_status = "not_configured"
        if config.database_url:
            from cutsell_worker.commercial_store import upsert_user
            upsert_user(
                config.database_url,
                user_id=user_id,
                apple_subject=identity.subject,
                email=identity.email,
            )
            durable_status = "written"
        session = create_session(user_id=user_id)
        capture_operational_event("auth.apple.success", user_id=user_id, durable_status=durable_status)
        return {**session, "account_provider": "apple", "durable_status": durable_status}
    except PermissionError as exc:
        capture_operational_event("auth.apple.rejected", reason=str(exc)[:120])
        raise HTTPException(status_code=401, detail="Apple identity verification failed") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except Exception as exc:
        capture_operational_event("auth.apple.error", reason=exc.__class__.__name__)
        raise HTTPException(status_code=503, detail="Apple authentication unavailable") from None
