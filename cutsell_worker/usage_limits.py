"""Commercial usage guardrails for CutSell processing."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from uuid import uuid4

from .config import load_runtime_config


MAX_SOURCE_SECONDS = 600          # one source / clip: 10 minutes


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

    # Cap each source independently. A project may contain multiple videos, while
    # an optional plan limit may impose a tighter cap on their total duration.
    if any(float(item or 0.0) > MAX_SOURCE_SECONDS for item in durations_sec) or (
        config.max_source_minutes > 0 and requested_minutes > float(config.max_source_minutes)
    ):
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


def reserve_processing_slot(*, user_id: str, client=None, ttl_sec: int = 7200, extra: int = 0) -> dict:
    """Atomically reserve one processing slot for a user before enqueueing work.

    `extra` (23 Oct): more slots for one Bulk send -- each of its videos may wait up to 30 min in the Batch API,
    so a Bulk of N needs N slots at once (capped by the caller)."""
    config = load_runtime_config()
    client = _redis(client)
    if client is None:
        return {"allowed": True, "status": "not_configured", "active": None}
    limit = max(1, int(config.max_concurrent_jobs_per_user)) + max(0, int(extra))
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
    client = _redis(client)
    if client is None:
        return {"status": "not_configured", "active": None}
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


# ---------------------------------------------------------------- one gate for every kind of paid work (PO 5 Nov)
# Security review 4 Nov: only "make a cut" checked a limit; Bulk, Mix (and its retry), server exports, Clean up
# voice, video-ready and job retry did not. Every route that queues paid work now passes here: the month's minutes
# must not be used up, and each kind has an hourly ceiling far above real use (a person never exports 60 videos in
# one hour) so a script cannot queue work without end.
MONTHLY_KINDS = {"mix", "batch", "retry"}
HOURLY_LIMITS = {
    "export": 60,
    "mix": 30,
    "clean_audio": 60,
    "batch": 10,
    "video_ready": 200,
    "retry": 30,
}


def _redis(client=None):
    if client is not None:
        return client
    config = load_runtime_config()
    if not config.redis_url:
        return None
    from redis import Redis
    return Redis.from_url(config.redis_url)


def monthly_used_up(*, user_id: str) -> bool:
    """True when this month's processing minutes are already used up (the meter must be configured)."""
    decision = check_processing_allowance(user_id=user_id, durations_sec=[])
    used, limit = decision.monthly_used_minutes, decision.monthly_limit_minutes
    return used is not None and limit is not None and used >= limit


def count_work(*, user_id: str, kind: str, client=None, now: float | None = None) -> dict:
    """One more `kind` job this hour for this user; over HOURLY_LIMITS[kind] -> not allowed."""
    import time
    limit = HOURLY_LIMITS.get(kind)
    target = _redis(client)
    if not limit or target is None:
        return {"allowed": True, "status": "not_limited"}
    hour = int((now if now is not None else time.time()) // 3600)
    digest = hashlib.sha256(str(user_id).encode()).hexdigest()[:24]
    key = f"cutsell:v1:work:{kind}:{digest}:{hour}"
    try:
        count = int(target.incr(key))
        if count == 1:
            target.expire(key, 3700)
    except Exception:
        return {"allowed": True, "status": "counter_unavailable"}
    return {"allowed": count <= limit, "status": "allowed" if count <= limit else "hourly_limit", "count": count,
            "limit": limit}


def work_gate(*, user_id: str, kind: str, client=None) -> str | None:
    """None when this paid work may be queued, else the reason (the API answers 429 with it)."""
    if not user_id:
        return None
    # New cutting work spends her month's minutes; exports, Clean up voice and the video of a cut already made do
    # not (they finish work she already has), so a used-up month never locks her out of her finished videos.
    if kind in MONTHLY_KINDS and monthly_used_up(user_id=user_id):
        return "monthly_processing_limit"
    if not count_work(user_id=user_id, kind=kind, client=client)["allowed"]:
        return f"{kind}_hourly_limit"
    return None


def bulk_counter(*, user_id: str, bulk_id: str, client=None) -> int | None:
    """How many videos of this Bulk send the server has received (counted here, not told by the phone)."""
    target = _redis(client)
    if target is None or not bulk_id:
        return None
    digest = hashlib.sha256(f"{user_id}|{bulk_id}".encode()).hexdigest()[:24]
    key = f"cutsell:v1:bulk:{digest}"
    try:
        count = int(target.incr(key))
        if count == 1:
            target.expire(key, 7200)
    except Exception:
        return None
    return count


class MeasuredLimitExceeded(ValueError):
    """The video's REAL length (measured on the worker before any paid step) is over a limit."""


def check_measured(*, user_id: str, durations_sec: list[float]) -> None:
    """PO 5 Nov (security review): the phone's declared length is not trusted. Before speech-to-text or the AI
    hear anything, the worker checks the lengths it measured itself: no source over 10 minutes, and the month's
    minutes not passed. Raises MeasuredLimitExceeded (the cut fails cleanly; nothing paid has run)."""
    decision = check_processing_allowance(user_id=user_id, durations_sec=durations_sec)
    if not decision.allowed:
        raise MeasuredLimitExceeded(decision.reason)
