"""Commercial usage guardrails for CutSell processing."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from .config import load_runtime_config


@dataclass(frozen=True)
class UsageDecision:
    allowed: bool
    reason: str
    requested_minutes: float
    monthly_used_minutes: float | None = None
    monthly_limit_minutes: float | None = None


def check_processing_allowance(*, user_id: str, durations_sec: list[float]) -> UsageDecision:
    config = load_runtime_config()
    requested_sec = sum(max(0.0, float(item or 0.0)) for item in durations_sec)
    requested_minutes = requested_sec / 60.0
    if requested_minutes > float(config.max_source_minutes):
        return UsageDecision(
            False,
            "source_duration_limit",
            requested_minutes,
            monthly_limit_minutes=float(config.monthly_processing_minutes),
        )

    if config.database_url:
        try:
            from .commercial_store import usage_total
            used = usage_total(config.database_url, user_id=user_id, event_type="processing_minutes")
            limit = float(config.monthly_processing_minutes)
            if used + requested_minutes > limit:
                return UsageDecision(False, "monthly_processing_limit", requested_minutes, used, limit)
            return UsageDecision(True, "allowed", requested_minutes, used, limit)
        except Exception:
            # DB migration/telemetry outages do not break the closed beta, but are observable elsewhere.
            pass

    return UsageDecision(
        True,
        "allowed_without_durable_meter",
        requested_minutes,
        None,
        float(config.monthly_processing_minutes),
    )


def record_processing_minutes(*, user_id: str, project_id: str, minutes: float, metadata: dict | None = None) -> dict:
    config = load_runtime_config()
    if minutes < 0:
        raise ValueError("processing minutes must be non-negative")
    if not config.database_url:
        return {"status": "not_configured"}
    try:
        from .commercial_store import record_usage
        event_id = f"usage_{uuid4().hex}"
        record_usage(
            config.database_url,
            event_id=event_id,
            user_id=user_id,
            project_id=project_id,
            event_type="processing_minutes",
            quantity=float(minutes),
            unit="minutes",
            metadata=metadata,
        )
        return {"status": "recorded", "event_id": event_id}
    except Exception as exc:
        return {"status": "degraded", "reason": exc.__class__.__name__}
