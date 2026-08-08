"""Durable commercial persistence for CutSell accounts, projects and usage.

Redis remains the processing/cache layer. This store is intentionally optional until
DATABASE_URL is configured, allowing migration-safe dual write before production.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _engine(database_url: str):
    from sqlalchemy import create_engine
    return create_engine(database_url, future=True, pool_pre_ping=True)


def initialize_schema(database_url: str) -> None:
    from sqlalchemy import text
    engine = _engine(database_url)
    ddl = (
        """
        CREATE TABLE IF NOT EXISTS cutsell_users (
            user_id VARCHAR(120) PRIMARY KEY,
            apple_subject VARCHAR(255) UNIQUE,
            email VARCHAR(320),
            status VARCHAR(32) NOT NULL DEFAULT 'active',
            created_at VARCHAR(64) NOT NULL,
            updated_at VARCHAR(64) NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cutsell_projects (
            project_id VARCHAR(120) PRIMARY KEY,
            user_id VARCHAR(120) NOT NULL,
            title VARCHAR(160) NOT NULL,
            state VARCHAR(64) NOT NULL,
            project_json TEXT NOT NULL,
            created_at VARCHAR(64) NOT NULL,
            updated_at VARCHAR(64) NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cutsell_projects_user ON cutsell_projects(user_id, updated_at)",
        """
        CREATE TABLE IF NOT EXISTS cutsell_usage_events (
            event_id VARCHAR(160) PRIMARY KEY,
            user_id VARCHAR(120) NOT NULL,
            project_id VARCHAR(120),
            event_type VARCHAR(80) NOT NULL,
            quantity DOUBLE PRECISION NOT NULL,
            unit VARCHAR(40) NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at VARCHAR(64) NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cutsell_usage_user ON cutsell_usage_events(user_id, created_at)",
    )
    with engine.begin() as conn:
        for statement in ddl:
            conn.execute(text(statement))


def upsert_user(
    database_url: str,
    *,
    user_id: str,
    apple_subject: str | None = None,
    email: str | None = None,
) -> dict[str, Any]:
    from sqlalchemy import text
    initialize_schema(database_url)
    now = _now()
    engine = _engine(database_url)
    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT user_id, apple_subject, email, status, created_at, updated_at FROM cutsell_users WHERE user_id=:user_id"),
            {"user_id": user_id},
        ).mappings().first()
        if existing:
            conn.execute(text(
                "UPDATE cutsell_users SET apple_subject=COALESCE(:apple_subject, apple_subject), "
                "email=COALESCE(:email, email), updated_at=:updated_at WHERE user_id=:user_id"
            ), {"user_id": user_id, "apple_subject": apple_subject, "email": email, "updated_at": now})
        else:
            conn.execute(text(
                "INSERT INTO cutsell_users(user_id, apple_subject, email, status, created_at, updated_at) "
                "VALUES(:user_id,:apple_subject,:email,'active',:created_at,:updated_at)"
            ), {"user_id": user_id, "apple_subject": apple_subject, "email": email, "created_at": now, "updated_at": now})
        row = conn.execute(
            text("SELECT user_id, apple_subject, email, status, created_at, updated_at FROM cutsell_users WHERE user_id=:user_id"),
            {"user_id": user_id},
        ).mappings().one()
    return dict(row)


def durable_upsert_project(database_url: str, record: dict[str, Any]) -> None:
    from sqlalchemy import text
    initialize_schema(database_url)
    now = _now()
    project_id = str(record.get("project_id") or "")
    user_id = str(record.get("user_id") or "")
    if not project_id or not user_id:
        raise ValueError("durable project requires project_id and user_id")
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    created_at = str(record.get("created_at") or now)
    engine = _engine(database_url)
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT project_id FROM cutsell_projects WHERE project_id=:project_id"), {"project_id": project_id}).first()
        if exists:
            conn.execute(text(
                "UPDATE cutsell_projects SET user_id=:user_id,title=:title,state=:state,project_json=:project_json,updated_at=:updated_at "
                "WHERE project_id=:project_id"
            ), {
                "project_id": project_id, "user_id": user_id,
                "title": str(record.get("title") or "Untitled Cut")[:160],
                "state": str(record.get("state") or "created")[:64],
                "project_json": payload, "updated_at": now,
            })
        else:
            conn.execute(text(
                "INSERT INTO cutsell_projects(project_id,user_id,title,state,project_json,created_at,updated_at) "
                "VALUES(:project_id,:user_id,:title,:state,:project_json,:created_at,:updated_at)"
            ), {
                "project_id": project_id, "user_id": user_id,
                "title": str(record.get("title") or "Untitled Cut")[:160],
                "state": str(record.get("state") or "created")[:64],
                "project_json": payload, "created_at": created_at, "updated_at": now,
            })


def record_usage(
    database_url: str,
    *,
    event_id: str,
    user_id: str,
    event_type: str,
    quantity: float,
    unit: str,
    project_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    from sqlalchemy import text
    initialize_schema(database_url)
    if quantity < 0:
        raise ValueError("usage quantity must be non-negative")
    engine = _engine(database_url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO cutsell_usage_events(event_id,user_id,project_id,event_type,quantity,unit,metadata_json,created_at) "
            "VALUES(:event_id,:user_id,:project_id,:event_type,:quantity,:unit,:metadata_json,:created_at)"
        ), {
            "event_id": event_id, "user_id": user_id, "project_id": project_id,
            "event_type": event_type, "quantity": float(quantity), "unit": unit,
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":")),
            "created_at": _now(),
        })


def usage_total(database_url: str, *, user_id: str, event_type: str) -> float:
    from sqlalchemy import text
    initialize_schema(database_url)
    engine = _engine(database_url)
    with engine.begin() as conn:
        value = conn.execute(text(
            "SELECT COALESCE(SUM(quantity),0) FROM cutsell_usage_events WHERE user_id=:user_id AND event_type=:event_type"
        ), {"user_id": user_id, "event_type": event_type}).scalar_one()
    return float(value or 0.0)


def durable_delete_project(database_url: str, *, user_id: str, project_id: str) -> int:
    from sqlalchemy import text
    initialize_schema(database_url)
    engine = _engine(database_url)
    with engine.begin() as conn:
        result = conn.execute(
            text("DELETE FROM cutsell_projects WHERE project_id=:project_id AND user_id=:user_id"),
            {"project_id": project_id, "user_id": user_id},
        )
    return int(result.rowcount or 0)


def durable_delete_user(database_url: str, *, user_id: str) -> dict[str, int]:
    """Delete durable account/project metadata while retaining aggregate-free integrity."""
    from sqlalchemy import text
    initialize_schema(database_url)
    engine = _engine(database_url)
    with engine.begin() as conn:
        usage = conn.execute(text("DELETE FROM cutsell_usage_events WHERE user_id=:user_id"), {"user_id": user_id})
        projects = conn.execute(text("DELETE FROM cutsell_projects WHERE user_id=:user_id"), {"user_id": user_id})
        users = conn.execute(text("DELETE FROM cutsell_users WHERE user_id=:user_id"), {"user_id": user_id})
    return {
        "usage_events": int(usage.rowcount or 0),
        "projects": int(projects.rowcount or 0),
        "users": int(users.rowcount or 0),
    }
