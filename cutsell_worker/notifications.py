"""User-scoped notification outbox for CutSell mobile background completion events."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from .config import load_runtime_config
from .redis_atomic_list import append_if_absent

MAX_NOTIFICATIONS = 100
ALLOWED_KINDS = {"draft_ready", "render_finished", "processing_failed", "render_failed"}


def _scope_hash(value: str) -> str:
    if not value or len(value) > 200:
        raise ValueError("notification user ID must contain 1 to 200 characters")
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def notification_key(user_id: str) -> str:
    return f"cutsell:v1:notifications:{_scope_hash(user_id)}"


def _redis_client(client=None):
    if client is not None:
        return client
    config = load_runtime_config()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL is required for notifications")
    from redis import Redis
    return Redis.from_url(config.redis_url)


def publish_notification(
    *, user_id: str, project_id: str, kind: str, payload: dict | None = None, client=None,
    idempotency_key: str | None = None,
) -> dict:
    """D-288.3: `idempotency_key`, when given, makes a repeat call for the
    same `(kind, idempotency_key)` pair a no-op that returns the ORIGINAL
    notification instead of firing a duplicate -- a caller recovering
    from an interruption (e.g. `export_job.resume_delivery_after_
    approval` retried after a crash right after a successful finalize)
    can safely call this again. `idempotency_key=None` (every existing
    caller) is byte-for-byte the prior always-append behavior."""
    normalized = str(kind or "")
    if normalized not in ALLOWED_KINDS:
        raise ValueError("unsupported notification kind")
    target = _redis_client(client)
    key = notification_key(user_id)
    record = {
        "notification_id": f"ntf_{uuid4().hex}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_id": str(project_id),
        "kind": normalized,
        "payload": dict(payload or {}),
        "idempotency_key": idempotency_key,
    }
    # D-288.4/D-288.4.1: EVERY insert is one atomic check-and-append.
    # With an `idempotency_key`, two concurrent callers for the same
    # `(kind, idempotency_key)` receive the single stored notification
    # (and its one id). Without one, the freshly minted, unique
    # `notification_id` is the match field -- it can never already exist,
    # so this is a plain atomic append: a keyless write can no longer read
    # the list, lose to a concurrent atomic insert, and overwrite it with
    # its own stale copy (the D-288.4 keyless path still did exactly that).
    match = (
        {"kind": normalized, "idempotency_key": idempotency_key}
        if idempotency_key else {"notification_id": record["notification_id"]}
    )
    return append_if_absent(target, key, match=match, record=record, max_len=MAX_NOTIFICATIONS)


def list_notifications(*, user_id: str, limit: int = 30, client=None) -> list[dict]:
    if not 1 <= int(limit) <= 100:
        raise ValueError("notification limit must be between 1 and 100")
    target = _redis_client(client)
    raw = target.get(notification_key(user_id))
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    items = json.loads(raw) if raw else []
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items[: int(limit)] if isinstance(item, dict)]
