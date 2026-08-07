"""User-scoped notification outbox for CutSell mobile background completion events."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from .config import load_runtime_config

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


def publish_notification(*, user_id: str, project_id: str, kind: str, payload: dict | None = None, client=None) -> dict:
    normalized = str(kind or "")
    if normalized not in ALLOWED_KINDS:
        raise ValueError("unsupported notification kind")
    record = {
        "notification_id": f"ntf_{uuid4().hex}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_id": str(project_id),
        "kind": normalized,
        "payload": dict(payload or {}),
    }
    target = _redis_client(client)
    key = notification_key(user_id)
    raw = target.get(key)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    items = json.loads(raw) if raw else []
    if not isinstance(items, list):
        items = []
    items.insert(0, record)
    target.set(key, json.dumps(items[:MAX_NOTIFICATIONS], ensure_ascii=False))
    return record


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
