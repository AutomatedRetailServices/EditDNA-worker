"""What happens when a subscription ends (Product Owner, 8 Nov 2026).

Decision: when a creator cancels, she keeps CutSell until the end of the period she already paid for (Apple's rule:
cancelling only turns auto-renew off). The day Apple says the subscription has really ENDED (or a free trial ends
without paying), EVERYTHING of hers is deleted -- the same full deletion as "Delete my account" -- and nothing can
be recovered. Before that she is warned in the app with the exact date, so she can export what she wants to keep.
If she pays again before that date, nothing is deleted.

Built before payments exist: today no account has a subscription state, so nothing here ever deletes anything. When
StoreKit 2 + App Store Server Notifications V2 are added, the notification handler only has to call
`record(user_id, state=..., ends_at=...)`; the warning and the deletion then work on their own.

States: "active" (paying or in the trial), "canceled" (auto-renew off; still usable until `ends_at`), "expired"
(ended: deleted at the next daily run). No state = no payments yet = never deleted by this module.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

STATES = ("active", "canceled", "expired")
ENDING_INDEX = "cutsell:v1:subscription-ending"      # sorted set: user_id -> the time her data is deleted
GRACE_SEC = 3600                                      # one hour after the end, so a late renewal can still win
_LOG = logging.getLogger("cutsell.subscription_lifecycle")

WARNING_EN = ("You canceled your subscription. On {date} all your videos, cuts and CutSell data will be deleted "
              "forever. They can't be recovered. Export to Photos anything you want to keep before that date.")
WARNING_ES = ("Cancelaste tu suscripción. El {date} se borrarán para siempre todos tus videos, cortes y datos de "
              "CutSell. No se pueden recuperar. Exporta a tus Fotos lo que quieras guardar antes de esa fecha.")


def _key(user_id: str) -> str:
    return "cutsell:v1:subscription:" + hashlib.sha256(str(user_id).encode()).hexdigest()[:20]


def _redis(client=None):
    if client is not None:
        return client
    from .project_store import _redis_client
    return _redis_client(None)


def record(user_id: str, *, state: str, ends_at: float | None = None, client=None, now: float | None = None) -> dict:
    """Called by the payments code when Apple reports a change. `ends_at`: when the paid period (or trial) ends."""
    if state not in STATES:
        raise ValueError(f"unknown subscription state {state!r}")
    if state in ("canceled", "expired") and ends_at is None:
        raise ValueError("a canceled or expired subscription needs its end date")
    now = time.time() if now is None else now
    target = _redis(client)
    entry = {"state": state, "ends_at": float(ends_at) if ends_at is not None else None, "updated_at": now}
    pipe = target.pipeline()
    pipe.set(_key(user_id), json.dumps(entry))
    if state == "active":
        pipe.zrem(ENDING_INDEX, user_id)                 # paying again: nothing is deleted
    else:
        pipe.zadd(ENDING_INDEX, {user_id: float(ends_at)})
    pipe.execute()
    return entry


def status(user_id: str, *, client=None) -> dict[str, Any]:
    """For the app: the state, the deletion date and the warning (only while canceled)."""
    try:
        raw = _redis(client).get(_key(user_id))
    except Exception:
        raw = None
    if not raw:
        return {"state": "none"}
    entry = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    out: dict[str, Any] = {"state": entry.get("state"), "ends_at": entry.get("ends_at")}
    if entry.get("state") in ("canceled", "expired") and entry.get("ends_at"):
        date = datetime.fromtimestamp(float(entry["ends_at"]), tz=timezone.utc).strftime("%Y-%m-%d")
        out.update({"deletes_on": date, "warning": WARNING_EN.format(date=date),
                    "warning_es": WARNING_ES.format(date=date)})
    return out


def purge_ended(*, client=None, now: float | None = None, delete=None) -> dict[str, Any]:
    """The daily run: full deletion of every account whose subscription ended (state still canceled/expired and the
    end + 1 hour has passed). Re-reads the state first, so a renewal recorded in between always wins."""
    now = time.time() if now is None else now
    target = _redis(client)
    if delete is None:
        from .account_lifecycle import delete_account_data

        def delete(user_id):
            return delete_account_data(user_id=user_id, redis_client=target)
    due = target.zrangebyscore(ENDING_INDEX, 0, now - GRACE_SEC)
    deleted, kept, failed = 0, 0, 0
    for raw in due:
        user_id = raw.decode() if isinstance(raw, bytes) else str(raw)
        current = status(user_id, client=target)
        if current.get("state") not in ("canceled", "expired") or float(current.get("ends_at") or now) > now - GRACE_SEC:
            target.zrem(ENDING_INDEX, user_id)
            kept += 1
            continue
        try:
            result = delete(user_id)
        except Exception as exc:
            _LOG.warning("subscription_purge_failed %s", exc.__class__.__name__)
            failed += 1
            continue
        if (result or {}).get("status") == "partial":
            failed += 1                                    # stays in the index: finished on the next run
            continue
        target.delete(_key(user_id))
        target.zrem(ENDING_INDEX, user_id)
        deleted += 1
    report = {"due": len(due), "deleted": deleted, "kept": kept, "failed": failed}
    _LOG.info("subscription_purge %s", report)
    return report
