"""Redis-backed CutSell project library for mobile recovery and history."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any
from uuid import uuid4

from .config import load_runtime_config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scope(value: str) -> str:
    value = str(value or "").strip()
    if not value or len(value) > 200:
        raise ValueError("identifier must contain 1 to 200 characters")
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def project_key(*, user_id: str, project_id: str) -> str:
    return f"cutsell:v1:project:{_scope(user_id)}:{_scope(project_id)}"


def project_index_key(*, user_id: str) -> str:
    return f"cutsell:v1:projects:{_scope(user_id)}"


def _redis_client(client=None):
    if client is not None:
        return client
    config = load_runtime_config()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL is required for project persistence")
    from redis import Redis
    return Redis.from_url(config.redis_url)


def _decode(raw) -> dict[str, Any]:
    if raw is None:
        raise KeyError("project not found")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(str(raw))
    if not isinstance(data, dict):
        raise ValueError("stored project is invalid")
    return data


def create_project(*, user_id: str, title: str | None = None, client=None) -> dict[str, Any]:
    target = _redis_client(client)
    project_id = f"prj_{uuid4().hex}"
    now = _now()
    record = {
        "schema_version": "cutsell.project.v1",
        "project_id": project_id,
        "user_id": str(user_id),
        "title": str(title or "Untitled Cut").strip()[:120] or "Untitled Cut",
        "state": "created",
        "created_at": now,
        "updated_at": now,
        "sources": [],
        "latest_job_id": None,
        "render_versions": [],
    }
    key = project_key(user_id=user_id, project_id=project_id)
    index = project_index_key(user_id=user_id)
    pipe = target.pipeline()
    pipe.set(key, json.dumps(record, ensure_ascii=False))
    pipe.zadd(index, {project_id: datetime.now(timezone.utc).timestamp()})
    pipe.execute()
    return record


def get_project(*, user_id: str, project_id: str, client=None) -> dict[str, Any]:
    target = _redis_client(client)
    record = _decode(target.get(project_key(user_id=user_id, project_id=project_id)))
    if str(record.get("user_id") or "") != str(user_id) or str(record.get("project_id") or "") != str(project_id):
        raise KeyError("project not found")
    return record


def list_projects(*, user_id: str, client=None, limit: int = 50) -> list[dict[str, Any]]:
    target = _redis_client(client)
    count = max(1, min(int(limit), 100))
    ids = target.zrevrange(project_index_key(user_id=user_id), 0, count - 1)
    output = []
    for raw_id in ids:
        project_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
        try:
            output.append(get_project(user_id=user_id, project_id=project_id, client=target))
        except KeyError:
            continue
    return output


def update_project(
    *,
    user_id: str,
    project_id: str,
    state: str | None = None,
    sources: list[dict[str, Any]] | None = None,
    latest_job_id: str | None = None,
    title: str | None = None,
    render_version: dict[str, Any] | None = None,
    client=None,
) -> dict[str, Any]:
    target = _redis_client(client)
    key = project_key(user_id=user_id, project_id=project_id)
    current = get_project(user_id=user_id, project_id=project_id, client=target)
    if state is not None:
        current["state"] = str(state)
    if sources is not None:
        current["sources"] = list(sources)
    if latest_job_id is not None:
        current["latest_job_id"] = str(latest_job_id)
    if title is not None:
        current["title"] = str(title).strip()[:120] or current.get("title") or "Untitled Cut"
    if render_version is not None:
        versions = list(current.get("render_versions") or ())
        versions.append(dict(render_version))
        current["render_versions"] = versions[-20:]
    current["updated_at"] = _now()
    target.set(key, json.dumps(current, ensure_ascii=False))
    target.zadd(project_index_key(user_id=user_id), {project_id: datetime.now(timezone.utc).timestamp()})
    return current
