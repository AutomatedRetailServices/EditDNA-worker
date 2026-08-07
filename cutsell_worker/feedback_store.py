"""Bounded production feedback store for CutSell human evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any
from uuid import uuid4

from .config import load_runtime_config
from .draft_store import get_draft_snapshot

MAX_FEEDBACK_PER_PROJECT = 200


def _scope(value: str) -> str:
    value = str(value or "").strip()
    if not value or len(value) > 200:
        raise ValueError("feedback scope identifiers must contain 1 to 200 characters")
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def feedback_key(*, user_id: str, project_id: str) -> str:
    return f"cutsell:v1:feedback:{_scope(user_id)}:{_scope(project_id)}"


def _redis_client(client=None):
    if client is not None:
        return client
    config = load_runtime_config()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL is required for feedback persistence")
    from redis import Redis
    return Redis.from_url(config.redis_url)


def _decode(raw) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    value = json.loads(str(raw))
    if not isinstance(value, list):
        raise ValueError("stored feedback is invalid")
    return [dict(item) for item in value if isinstance(item, dict)]


def save_edit_feedback(
    *,
    user_id: str,
    project_id: str,
    rating: str,
    category: str | None = None,
    note: str | None = None,
    clip_id: str | None = None,
    time_sec: float | None = None,
    client=None,
    draft_client=None,
) -> dict[str, Any]:
    normalized = str(rating or "").lower().strip()
    if normalized not in {"good", "bad"}:
        raise ValueError("feedback rating must be good or bad")
    if time_sec is not None and float(time_sec) < 0:
        raise ValueError("feedback time cannot be negative")
    snapshot = get_draft_snapshot(user_id=user_id, project_id=project_id, client=draft_client or client)
    draft = dict(snapshot.get("draft") or {})
    diagnostics = dict(draft.get("diagnostics") or {})
    config = load_runtime_config()
    record = {
        "feedback_id": f"fb_{uuid4().hex}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_id": str(user_id),
        "project_id": str(project_id),
        "rating": normalized,
        "category": (str(category).strip()[:80] if category else None),
        "note": (str(note).strip()[:500] if note else None),
        "clip_id": (str(clip_id)[:120] if clip_id else None),
        "time_sec": (round(float(time_sec), 3) if time_sec is not None else None),
        "strategy": draft.get("strategy"),
        "selected_clip_ids": [str(item.get("clip_id") or "") for item in draft.get("selected") or () if item.get("clip_id")],
        "alternate_clip_ids": [str(item.get("clip_id") or "") for item in draft.get("alternates") or () if item.get("clip_id")],
        "take_judge_status_counts": diagnostics.get("take_judge_status_counts") or {},
        "semantic_model": config.semantic_model,
        "visual_model": config.visual_model,
        "take_judge_model": config.take_judge_model,
        "draft_revision": snapshot.get("revision"),
    }
    target = _redis_client(client)
    key = feedback_key(user_id=user_id, project_id=project_id)
    history = _decode(target.get(key))
    history.insert(0, record)
    target.set(key, json.dumps(history[:MAX_FEEDBACK_PER_PROJECT], ensure_ascii=False))
    return record


def list_edit_feedback(*, user_id: str, project_id: str, client=None) -> list[dict[str, Any]]:
    target = _redis_client(client)
    return _decode(target.get(feedback_key(user_id=user_id, project_id=project_id)))
