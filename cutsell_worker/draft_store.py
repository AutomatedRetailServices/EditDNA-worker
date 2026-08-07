"""Revisioned Redis persistence for recoverable CutSell mobile drafts.

The clean editor remains stateless; this module is the persistence boundary. Drafts
are scoped by hashed user/project IDs and use optimistic revisions so a stale mobile
client cannot overwrite a newer autosave.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from .config import load_runtime_config


class DraftConflictError(RuntimeError):
    pass


def _scope_hash(value: str) -> str:
    if not value or len(value) > 200:
        raise ValueError("draft scope identifiers must contain 1 to 200 characters")
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def draft_key(*, user_id: str, project_id: str) -> str:
    return f"cutsell:v1:draft:{_scope_hash(user_id)}:{_scope_hash(project_id)}"


def _redis_client(client=None):
    if client is not None:
        return client
    config = load_runtime_config()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL is required for draft persistence")
    from redis import Redis
    return Redis.from_url(config.redis_url)


def _decode(raw) -> dict[str, Any]:
    if raw is None:
        raise KeyError("draft not found")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(str(raw))
    if not isinstance(data, dict) or not isinstance(data.get("draft"), dict):
        raise ValueError("stored draft snapshot is invalid")
    return data


def _validate_draft(draft: dict[str, Any], *, project_id: str) -> None:
    if not isinstance(draft, dict):
        raise ValueError("draft must be an object")
    if str(draft.get("project_id") or "") != project_id:
        raise ValueError("draft project does not match persistence project")
    if not isinstance(draft.get("selected"), list):
        raise ValueError("draft selected clips must be a list")


def _snapshot(
    *,
    project_id: str,
    user_id: str,
    draft: dict[str, Any],
    sources: list[dict[str, Any]],
    revision: int,
) -> dict[str, Any]:
    return {
        "schema_version": "cutsell.draft.v1",
        "project_id": project_id,
        "user_id": user_id,
        "revision": int(revision),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "draft": draft,
        "sources": sources,
    }


def get_draft_snapshot(*, user_id: str, project_id: str, client=None) -> dict[str, Any]:
    target = _redis_client(client)
    return _decode(target.get(draft_key(user_id=user_id, project_id=project_id)))


def create_initial_draft(
    *,
    user_id: str,
    project_id: str,
    draft: dict[str, Any],
    sources: list[dict[str, Any]],
    client=None,
) -> tuple[dict[str, Any], bool]:
    """Persist the AI draft only when no recoverable draft exists yet."""
    _validate_draft(draft, project_id=project_id)
    if not sources:
        raise ValueError("initial draft persistence requires source metadata")
    target = _redis_client(client)
    key = draft_key(user_id=user_id, project_id=project_id)
    record = _snapshot(
        project_id=project_id,
        user_id=user_id,
        draft=draft,
        sources=list(sources),
        revision=1,
    )
    created = bool(target.set(key, json.dumps(record, ensure_ascii=False), nx=True))
    if created:
        return record, True
    return _decode(target.get(key)), False


def save_draft_snapshot(
    *,
    user_id: str,
    project_id: str,
    draft: dict[str, Any],
    expected_revision: int,
    client=None,
    max_watch_retries: int = 3,
) -> dict[str, Any]:
    """Autosave one edited draft using optimistic revision protection."""
    _validate_draft(draft, project_id=project_id)
    if int(expected_revision) < 1:
        raise ValueError("expected_revision must be at least 1")
    target = _redis_client(client)
    key = draft_key(user_id=user_id, project_id=project_id)

    for _attempt in range(max(1, int(max_watch_retries))):
        pipe = target.pipeline()
        try:
            pipe.watch(key)
            current = _decode(pipe.get(key))
            current_revision = int(current.get("revision") or 0)
            if current_revision != int(expected_revision):
                raise DraftConflictError(
                    f"draft revision conflict: expected {expected_revision}, current {current_revision}"
                )
            record = _snapshot(
                project_id=project_id,
                user_id=user_id,
                draft=draft,
                sources=list(current.get("sources") or ()),
                revision=current_revision + 1,
            )
            pipe.multi()
            pipe.set(key, json.dumps(record, ensure_ascii=False))
            pipe.execute()
            return record
        except DraftConflictError:
            raise
        except KeyError:
            raise
        except Exception as exc:
            # Avoid importing redis at module import time so pure CI/unit tests do not
            # require the runtime package. Retry only Redis optimistic-lock races.
            if exc.__class__.__name__ != "WatchError":
                raise
        finally:
            try:
                pipe.reset()
            except Exception:
                pass
    raise DraftConflictError("draft changed concurrently; retry autosave")
