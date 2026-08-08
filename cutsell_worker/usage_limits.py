"""Commercial usage guardrails for CutSell processing."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
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

    # Product default: imported gallery footage has no arbitrary visible duration cap.
    # A positive CUTSELL_MAX_SOURCE_MINUTES can still be enabled later for a specific
    # plan/infrastructure safety policy.
    if config.max_source_minutes > 0 and requested_minutes > float(config.max_source_minutes):
        return UsageDecision(
            False,
            "source_duration_limit",
            requested_minutes,
            monthly_limit_minutes=float(config.monthly_processing_minutes),
        )

    if config.database_url:
        try:
            from .commercial_usage import monthly_usage_total
            used = monthly_usage_total(config.database_url, user_id=user_id, event_type="processing_minutes")
            limit = float(config.monthly_processing_minutes)
            if used + requested_minutes > limit:
                return UsageDecision(False, "monthly_processing_limit", requested_minutes, used, limit)
            return UsageDecision(True, "allowed", requested_minutes, used, limit)
        except Exception:
            # Migration/telemetry outages do not break closed beta processing.
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


def _concurrency_key(user_id: str) -> str:
    digest = hashlib.sha256(str(user_id).encode()).hexdigest()[:24]
    return f"cutsell:v1:concurrency:{digest}"


def reserve_processing_slot(*, user_id: str, client=None, ttl_sec: int = 7200) -> dict:
    """Atomically reserve one processing slot for a user before enqueueing work."""
    config = load_runtime_config()
    if client is None:
        if not config.redis_url:
            return {"allowed": True, "status": "not_configured", "active": None}
        from redis import Redis
        client = Redis.from_url(config.redis_url)
    limit = max(1, int(config.max_concurrent_jobs_per_user))
    script = """
    local current = tonumber(redis.call('GET', KEYS[1]) or '0')
    local limit = tonumber(ARGV[1])
    if current >= limit then
      return {0, current}
    end
    local next = redis.call('INCR', KEYS[1])
    redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
    return {1, next}
    """
    allowed, active = client.eval(script, 1, _concurrency_key(user_id), limit, int(ttl_sec))
    return {
        "allowed": bool(int(allowed)),
        "status": "reserved" if int(allowed) else "concurrency_limit",
        "active": int(active),
        "limit": limit,
    }


def release_processing_slot(*, user_id: str, client=None) -> dict:
    """Release one reserved processing slot; idempotently floors at zero."""
    config = load_runtime_config()
    if client is None:
        if not config.redis_url:
            return {"status": "not_configured", "active": None}
        from redis import Redis
        client = Redis.from_url(config.redis_url)
    script = """
    local current = tonumber(redis.call('GET', KEYS[1]) or '0')
    if current <= 1 then
      redis.call('DEL', KEYS[1])
      return 0
    end
    return redis.call('DECR', KEYS[1])
    """
    active = int(client.eval(script, 1, _concurrency_key(user_id)))
    return {"status": "released", "active": active}
