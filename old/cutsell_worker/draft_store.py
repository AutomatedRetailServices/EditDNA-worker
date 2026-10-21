"""Revisioned Redis persistence for recoverable CutSell mobile drafts.

The clean editor remains stateless; this module is the persistence boundary. Drafts
are scoped by hashed user/project IDs and use optimistic revisions so a stale mobile
client cannot overwrite a newer autosave. A bounded undo/redo history lives in the
same snapshot so edit recovery survives app termination.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from .config import load_runtime_config


class DraftConflictError(RuntimeError):
    pass


class DraftHistoryError(RuntimeError):
    pass


MAX_HISTORY = 20


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
    data.setdefault("undo", [])
    data.setdefault("redo", [])
    return data


def _validate_draft(draft: dict[str, Any], *, project_id: str) -> None:
    if not isinstance(draft, dict):
        raise ValueError("draft must be an object")
    if str(draft.get("project_id") or "") != project_id:
        raise ValueError("draft project does not match persistence project")
    if not isinstance(draft.get("selected"), list):
        raise ValueError("draft selected clips must be a list")


def _bounded(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(items[-MAX_HISTORY:])


def _snapshot(
    *,
    project_id: str,
    user_id: str,
    draft: dict[str, Any],
    sources: list[dict[str, Any]],
    revision: int,
    undo: list[dict[str, Any]] | None = None,
    redo: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "cutsell.draft.v1",
        "project_id": project_id,
        "user_id": user_id,
        "revision": int(revision),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "draft": draft,
        "sources": sources,
        "undo": _bounded(list(undo or [])),
        "redo": _bounded(list(redo or [])),
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
    """Persist the AI draft. When the project already has a draft (it is being processed again),
    the new cut is merged with the creator's edits (`draft_merge.merge_reprocessed_draft`) as a
    new revision, the previous draft kept for Undo. Returns (snapshot, created)."""
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
    # Editor v5: the project was processed again. The creator's edits are merged into the new cut
    # (draft_merge.merge_reprocessed_draft) instead of being wiped -- or, before v5, instead of the
    # new cut being ignored. The previous draft stays recoverable through Undo.
    return _merge_into_existing(
        target=target, user_id=user_id, project_id=project_id, draft=draft, sources=list(sources),
    ), False


def _merge_into_existing(*, target, user_id: str, project_id: str, draft: dict[str, Any],
                         sources: list[dict[str, Any]], attempts: int = 3) -> dict[str, Any]:
    from .draft_merge import changes_anything, merge_reprocessed_draft
    from .serde import draft_from_dict

    key = draft_key(user_id=user_id, project_id=project_id)
    for _attempt in range(max(1, attempts)):
        current = _decode(target.get(key))
        # Edited by the creator (any autosave/undo history): their cut is kept, only the AI's own
        # parts are refreshed. Untouched (revision 1, no history): the new cut is taken.
        edited = int(current.get("revision") or 1) > 1 or bool(current.get("undo"))
        merged = merge_reprocessed_draft(current["draft"], draft, creator_edited=edited)
        if edited and not changes_anything(current["draft"], merged):
            return current  # nothing of the AI's changed: no new revision
        try:
            draft_from_dict(merged)
        except (TypeError, ValueError, KeyError):
            # A merge that would not render is never stored: the creator's draft stays as it was.
            return current
        keep_sources = list(current.get("sources") or ()) if edited else sources

        def transition(existing, revision, merged=merged, keep_sources=keep_sources):
            return _snapshot(
                project_id=project_id, user_id=user_id, draft=merged, sources=keep_sources,
                revision=revision + 1,
                undo=_bounded([*list(existing.get("undo") or ()), existing["draft"]]), redo=[],
            )

        try:
            return _write_transition(
                user_id=user_id, project_id=project_id, expected_revision=int(current.get("revision") or 1),
                transition=transition, client=target,
            )
        except DraftConflictError:
            continue  # an autosave landed in between: merge again with the newest draft
    return _decode(target.get(key))


def _write_transition(
    *,
    user_id: str,
    project_id: str,
    expected_revision: int,
    transition,
    client=None,
    max_watch_retries: int = 3,
) -> dict[str, Any]:
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
            record = transition(current, current_revision)
            pipe.multi()
            pipe.set(key, json.dumps(record, ensure_ascii=False))
            pipe.execute()
            return record
        except (DraftConflictError, DraftHistoryError, KeyError):
            raise
        except Exception as exc:
            if exc.__class__.__name__ != "WatchError":
                raise
        finally:
            try:
                pipe.reset()
            except Exception:
                pass
    raise DraftConflictError("draft changed concurrently; retry edit")


def save_draft_snapshot(
    *,
    user_id: str,
    project_id: str,
    draft: dict[str, Any],
    expected_revision: int,
    client=None,
    max_watch_retries: int = 3,
) -> dict[str, Any]:
    """Autosave one edited draft and preserve the prior state for Undo."""
    _validate_draft(draft, project_id=project_id)

    def transition(current, current_revision):
        undo = _bounded([*list(current.get("undo") or ()), current["draft"]])
        return _snapshot(
            project_id=project_id,
            user_id=user_id,
            draft=draft,
            sources=list(current.get("sources") or ()),
            revision=current_revision + 1,
            undo=undo,
            redo=[],
        )

    return _write_transition(
        user_id=user_id,
        project_id=project_id,
        expected_revision=expected_revision,
        transition=transition,
        client=client,
        max_watch_retries=max_watch_retries,
    )


def undo_draft_snapshot(
    *,
    user_id: str,
    project_id: str,
    expected_revision: int,
    client=None,
) -> dict[str, Any]:
    """Restore the most recent autosaved state and preserve the current state for Redo."""
    def transition(current, current_revision):
        undo = list(current.get("undo") or ())
        if not undo:
            raise DraftHistoryError("nothing to undo")
        previous = undo.pop()
        redo = _bounded([*list(current.get("redo") or ()), current["draft"]])
        return _snapshot(
            project_id=project_id,
            user_id=user_id,
            draft=previous,
            sources=list(current.get("sources") or ()),
            revision=current_revision + 1,
            undo=undo,
            redo=redo,
        )

    return _write_transition(
        user_id=user_id,
        project_id=project_id,
        expected_revision=expected_revision,
        transition=transition,
        client=client,
    )


def redo_draft_snapshot(
    *,
    user_id: str,
    project_id: str,
    expected_revision: int,
    client=None,
) -> dict[str, Any]:
    """Reapply the most recently undone state."""
    def transition(current, current_revision):
        redo = list(current.get("redo") or ())
        if not redo:
            raise DraftHistoryError("nothing to redo")
        next_draft = redo.pop()
        undo = _bounded([*list(current.get("undo") or ()), current["draft"]])
        return _snapshot(
            project_id=project_id,
            user_id=user_id,
            draft=next_draft,
            sources=list(current.get("sources") or ()),
            revision=current_revision + 1,
            undo=undo,
            redo=redo,
        )

    return _write_transition(
        user_id=user_id,
        project_id=project_id,
        expected_revision=expected_revision,
        transition=transition,
        client=client,
    )
