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


# Anonymous accounts are free to create and each one carries its own processing quota, so
# their creation is rate-limited. Per network (hashed IP, only when a trusted proxy header
# gives the real client address) and overall. Limits are environment-tunable.
ANON_PER_IP_HOUR_DEFAULT = 5
ANON_PER_IP_DAY_DEFAULT = 20
ANON_GLOBAL_HOUR_DEFAULT = 300


def _env_limit(name: str, default: int) -> int:
    import os
    try:
        return max(0, int(str(os.environ.get(name, "")).strip() or default))
    except ValueError:
        return default


def check_anonymous_session_allowance(*, client_ip: str | None, client=None, now: datetime | None = None) -> tuple[bool, str]:
    """Counts one anonymous-session request and says whether it is allowed.

    Returns (allowed, reason). A limit of 0 disables that limit. The IP is never stored:
    only a salted SHA-256 prefix is used as the counter key."""
    target = _redis_client(client)
    moment = now or datetime.now(timezone.utc)
    hour, day = moment.strftime("%Y%m%d%H"), moment.strftime("%Y%m%d")
    checks: list[tuple[str, int, int, str]] = []
    global_limit = _env_limit("CUTSELL_ANON_SESSIONS_GLOBAL_PER_HOUR", ANON_GLOBAL_HOUR_DEFAULT)
    if global_limit:
        checks.append((f"cutsell:v1:anon:global:{hour}", global_limit, 3600, "anonymous_signup_global_limit"))
    if client_ip:
        scope = hashlib.sha256(("cutsell-anon-ip:" + client_ip).encode()).hexdigest()[:24]
        per_hour = _env_limit("CUTSELL_ANON_SESSIONS_PER_IP_PER_HOUR", ANON_PER_IP_HOUR_DEFAULT)
        per_day = _env_limit("CUTSELL_ANON_SESSIONS_PER_IP_PER_DAY", ANON_PER_IP_DAY_DEFAULT)
        if per_hour:
            checks.append((f"cutsell:v1:anon:ip:{scope}:{hour}", per_hour, 3600, "anonymous_signup_network_limit"))
        if per_day:
            checks.append((f"cutsell:v1:anon:ip:{scope}:d{day}", per_day, 86400, "anonymous_signup_network_limit"))
    for key, limit, ttl, reason in checks:
        count = int(target.incr(key))
        if count == 1:
            target.expire(key, ttl + 60)
        if count > limit:
            return False, reason
    return True, "allowed"


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
    # Sliding expiry: a session in use never runs out, so an active guest does not lose
    # their projects 90 days after first opening the app.
    if hasattr(target, "expire"):
        try:
            target.expire(session_key(value), SESSION_TTL_SEC)
            target.expire(session_index_key(str(record["user_id"])), SESSION_TTL_SEC)
        except Exception:
            pass
    return record


def revoke_session(token: str, *, client=None) -> dict[str, Any]:
    """Log out: revoke just this one session token."""
    value = str(token or "").strip()
    if not value:
        raise PermissionError("missing bearer token")
    target = _redis_client(client)
    key = session_key(value)
    raw = target.get(key)
    if raw is None:
        return {"status": "already_signed_out"}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        user_id = str(json.loads(str(raw)).get("user_id") or "")
    except Exception:
        user_id = ""
    target.delete(key)
    if user_id and hasattr(target, "srem"):
        target.srem(session_index_key(user_id), key)
    return {"status": "signed_out"}


# ---- Sign in with Apple <-> CutSell account binding ----------------------------------
# An Apple identity points to ONE CutSell user_id. All of a user's data (projects, drafts,
# uploads, exports) is keyed by that user_id, so binding an Apple identity to the guest
# account the person already used keeps every project without moving any data.

def is_guest_user_id(user_id: str) -> bool:
    value = str(user_id or "")
    return value.startswith("usr_") and not value.startswith("usr_apple_")


def _apple_scope(subject: str) -> str:
    return hashlib.sha256(("cutsell-apple:" + str(subject)).encode()).hexdigest()[:32]


def apple_binding_key(subject: str) -> str:
    return f"cutsell:v1:apple:{_apple_scope(subject)}"


def user_apple_key(user_id: str) -> str:
    return f"cutsell:v1:user-apple:{_user_scope(user_id)}"


def _decoded(raw) -> str | None:
    if raw is None:
        return None
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def resolve_apple_account(subject: str, *, guest_user_id: str | None = None, client=None) -> tuple[str, str]:
    """Returns (user_id, outcome) for an Apple identity.

    outcome:
    - "existing_account": this Apple ID already has a CutSell account (signing in again,
      new phone, reinstall). Same user_id -> same projects.
    - "guest_upgraded": first Apple sign-in on a phone that was using a guest account; the
      Apple ID now points to that guest account, so its projects become the Apple account's.
    - "new_account": first Apple sign-in with no guest account to keep.
    - "existing_account_guest_not_merged": this Apple ID already had an account AND the phone
      had a different guest account. The guest's projects stay on the guest account
      (moving stored videos between accounts is not supported yet)."""
    if not str(subject or "").strip():
        raise ValueError("Apple subject is required")
    target = _redis_client(client)
    key = apple_binding_key(subject)
    existing = _decoded(target.get(key))
    if existing:
        if guest_user_id and guest_user_id != existing and is_guest_user_id(guest_user_id):
            return existing, "existing_account_guest_not_merged"
        return existing, "existing_account"
    candidate, outcome = stable_apple_user_id(subject), "new_account"
    if guest_user_id and is_guest_user_id(guest_user_id) and target.get(user_apple_key(guest_user_id)) is None:
        candidate, outcome = guest_user_id, "guest_upgraded"
    # SET NX: if two sign-ins race, both end on the same account.
    if not target.set(key, candidate, nx=True):
        winner = _decoded(target.get(key)) or candidate
        return winner, "existing_account"
    target.set(user_apple_key(candidate), _apple_scope(subject))
    return candidate, outcome


def delete_apple_binding(*, user_id: str, client=None) -> bool:
    """Called on account deletion: the Apple ID may sign up again as a fresh account."""
    target = _redis_client(client)
    scope = _decoded(target.get(user_apple_key(user_id)))
    if not scope:
        return False
    target.delete(f"cutsell:v1:apple:{scope}")
    target.delete(user_apple_key(user_id))
    return True


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
