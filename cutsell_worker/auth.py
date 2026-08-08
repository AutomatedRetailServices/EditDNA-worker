"""Opaque bearer sessions for CutSell closed beta.

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


def session_key(token: str) -> str:
    return f"cutsell:v1:session:{_token_hash(token)}"


def create_session(*, client=None) -> dict[str, Any]:
    target = _redis_client(client)
    user_id = f"usr_{uuid4().hex}"
    token = secrets.token_urlsafe(48)
    record = {
        "schema_version": "cutsell.session.v1",
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    target.setex(session_key(token), SESSION_TTL_SEC, json.dumps(record, separators=(",", ":")))
    return {
        "user_id": user_id,
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
