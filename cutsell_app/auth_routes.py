"""Closed-beta bootstrap plus persistent Sign in with Apple authentication."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from cutsell_worker.account_lifecycle import delete_account_data
from cutsell_worker.apple_auth import verify_apple_identity_token
from cutsell_worker.auth import check_anonymous_session_allowance, create_session, stable_apple_user_id
from cutsell_worker.commercial_observability import capture_operational_event
from cutsell_worker.config import load_runtime_config

router = APIRouter(prefix="/v1/auth", tags=["auth"])


class AppleSessionRequest(BaseModel):
    identity_token: str
    nonce: str | None = None


class AccountDeleteRequest(BaseModel):
    user_id: str
    confirmation: str


# Headers set by the edge proxy (Cloudflare in front of Render) that a client cannot forge.
# X-Forwarded-For is NOT used: its leftmost entry is whatever the client sent.
_TRUSTED_CLIENT_IP_HEADERS = ("cf-connecting-ip", "true-client-ip")


def _trusted_client_ip(request: Request) -> tuple[str | None, str]:
    for name in _TRUSTED_CLIENT_IP_HEADERS:
        value = (request.headers.get(name) or "").strip()
        if value:
            return value, name
    return None, "none"


@router.post("/session")
def create_mobile_session(request: Request):
    """Legacy anonymous closed-beta bootstrap. Do not use for commercial accounts.

    Rate-limited: per network when the proxy gives the real client IP, and overall."""
    client_ip, ip_source = _trusted_client_ip(request)
    try:
        allowed, reason = check_anonymous_session_allowance(client_ip=client_ip)
        if not allowed:
            capture_operational_event("auth.anonymous.rate_limited", reason=reason, ip_source=ip_source)
            raise HTTPException(status_code=429, detail="Too many new accounts right now. Please try again later.")
        session = create_session()
        capture_operational_event("auth.anonymous.created", ip_source=ip_source)
        return session
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


@router.delete("/account")
def delete_authenticated_account(payload: AccountDeleteRequest):
    # Auth middleware binds payload.user_id to the bearer session before this route runs.
    if payload.confirmation != "DELETE MY ACCOUNT":
        raise HTTPException(status_code=409, detail="account deletion requires confirmation DELETE MY ACCOUNT")
    try:
        result = delete_account_data(user_id=payload.user_id)
        capture_operational_event("account.deleted", user_id=payload.user_id)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
