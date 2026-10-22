"""Which prompt-cache lifetime the next video uses (Product Owner, 2026-10-22).

Every video (Single, Combine, Bulk, Mix; every user) caches the fixed instructions of each step. While few videos are
being cut (fewer than CUTSELL_ENGINE_CACHE_BUSY_PER_HOUR, default 12, in the last hour) the cache lives 1 hour, so a
video 20 minutes after the previous one still reads it. With many videos it lives 5 minutes (cheaper to write) and every
video renews it. The count is shared by all workers (Redis); without Redis the 5-minute cache is used.
"""
from __future__ import annotations

import logging
import os
import time
import uuid

KEY = "cutsell:engine:video_starts"
WINDOW_SEC = 3600
_LOG = logging.getLogger("cutsell.simple_engine.cache_policy")


def busy_per_hour() -> int:
    try:
        return max(1, int(os.environ.get("CUTSELL_ENGINE_CACHE_BUSY_PER_HOUR", "12")))
    except ValueError:
        return 12


def ttl_for(videos_last_hour: int) -> str:
    return "1h" if videos_last_hour < busy_per_hour() else "5m"


def note_video(client=None, *, now: float | None = None) -> tuple[str, int | None]:
    """Counts this video and returns (ttl, videos in the last hour before it). Never raises."""
    now = time.time() if now is None else now
    try:
        if client is None:
            from redis import Redis
            url = os.environ.get("REDIS_URL", "").strip()
            if not url:
                return "5m", None
            client = Redis.from_url(url, socket_timeout=3)
        client.zremrangebyscore(KEY, 0, now - WINDOW_SEC)
        before = int(client.zcard(KEY))
        client.zadd(KEY, {f"{now:.3f}-{uuid.uuid4().hex[:6]}": now})
        client.expire(KEY, WINDOW_SEC * 2)
        return ttl_for(before), before
    except Exception as exc:  # the cache lifetime must never stop a video
        _LOG.warning("cache_policy_unavailable %s", exc.__class__.__name__)
        return "5m", None
