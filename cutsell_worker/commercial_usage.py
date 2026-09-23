"""Calendar-month durable usage queries for CutSell plans."""
from __future__ import annotations

from datetime import datetime, timezone


def month_start_utc(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    return current.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()


def monthly_usage_total(database_url: str, *, user_id: str, event_type: str, now: datetime | None = None) -> float:
    from sqlalchemy import create_engine, text
    from .commercial_store import initialize_schema

    initialize_schema(database_url)
    engine = create_engine(database_url, future=True, pool_pre_ping=True)
    with engine.begin() as conn:
        value = conn.execute(text(
            "SELECT COALESCE(SUM(quantity),0) FROM cutsell_usage_events "
            "WHERE user_id=:user_id AND event_type=:event_type AND created_at>=:month_start"
        ), {
            "user_id": user_id,
            "event_type": event_type,
            "month_start": month_start_utc(now),
        }).scalar_one()
    return float(value or 0.0)
