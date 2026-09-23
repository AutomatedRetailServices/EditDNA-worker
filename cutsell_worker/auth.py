"""Opaque bearer sessions for CutSell beta and persistent accounts.

Tokens are returned once to the mobile client; Redis stores only their SHA-256 hash.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import secrets
from typing import Any
from uuid import uuid4

from .config import load_runtime_config

SESSION_TTL_SEC = 60 * 60 * 24 * 90


def _redis_client(client=None):
    if client is not None:
        return client
    config = load_runtime_config()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL is required for auth sessions")
    from redis import Redis
    return Redis.from_url(config.redis_url)


def _token_hash(token: str) -> str:
    return hashlib.sha256(str(token).encode()).hexdigest()


def _user_scope(user_id: str) -> str:
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:24]


def stable_apple_user_id(subject: str) -> str:
    value = str(subject or "").strip()
    if not value:
        raise ValueError("Apple subject is required")
    return "usr_apple_" + hashlib.sha256(value.encode()).hexdigest()[:32]


def session_key(token: str) -> str:
    return f"cutsell:v1:session:{_token_hash(token)}"


def session_index_key(user_id: str) -> str:
    return f"cutsell:v1:sessions:{_user_scope(user_id)}"


def create_session(*, user_id: str | None = None, client=None) -> dict[str, Any]:
    target = _redis_client(client)
    resolved_user_id = str(user_id or f"usr_{uuid4().hex}")
    if not resolved_user_id.startswith("usr_"):
        raise ValueError("user_id must be a CutSell user identifier")
    token = secrets.token_urlsafe(48)
    key = session_key(token)
    record = {
        "schema_version": "cutsell.session.v1",
        "user_id": resolved_user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    encoded = json.dumps(record, separators=(",", ":"))
    # Real Redis clients use a pipeline so the session and revocation index are
    # written together. Minimal test doubles/legacy clients may only implement
    # setex/get; keep that compatibility without affecting production behavior.
    if hasattr(target, "pipeline"):
        pipe = target.pipeline()
        pipe.setex(key, SESSION_TTL_SEC, encoded)
        pipe.sadd(session_index_key(resolved_user_id), key)
        pipe.expire(session_index_key(resolved_user_id), SESSION_TTL_SEC)
        pipe.execute()
    else:
        target.setex(key, SESSION_TTL_SEC, encoded)
    return {
        "user_id": resolved_user_id,
        "access_token": token,
        "token_type": "bearer",
        "expires_in": SESSION_TTL_SEC,
    }


def resolve_session(token: str, *, client=None) -> dict[str, Any]:
    value = str(token or "").strip()
    if not value:
        raise PermissionError("missing bearer token")
    target = _redis_client(client)
    raw = target.get(session_key(value))
    if raw is None:
        raise PermissionError("invalid or expired bearer token")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    record = json.loads(str(raw))
    if not isinstance(record, dict) or not str(record.get("user_id") or "").startswith("usr_"):
        raise PermissionError("invalid auth session")
    return record


def revoke_all_sessions(*, user_id: str, client=None) -> dict[str, Any]:
    """Revoke all newly indexed sessions for one user.

    Legacy sessions created before the index existed expire naturally; commercial
    rollout should rotate beta sessions when persistent auth is enabled.
    """
    target = _redis_client(client)
    if not hasattr(target, "smembers") or not hasattr(target, "pipeline"):
        return {"status": "legacy_unindexed", "session_count": 0}
    index = session_index_key(user_id)
    raw_keys = target.smembers(index) or set()
    keys = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in raw_keys]
    pipe = target.pipeline()
    if keys:
        pipe.delete(*keys)
    pipe.delete(index)
    pipe.execute()
    return {"status": "revoked", "session_count": len(keys)}
