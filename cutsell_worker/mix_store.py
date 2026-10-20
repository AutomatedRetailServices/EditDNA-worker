"""Mix records: one JSON per Mix in Redis (the Mix id is the id of the project its clips are uploaded to).

    {"mix_id", "user_id", "product_id", "brand", "product", "sections": [{"name", "code", "clips": [{"clip_id", "uri",
     "audio_uri", "original_name", "duration_sec"}]}], "combos": [{"n", "picks", "code", "project_id", "state",
     "duration_sec"}], "clips": {clip_id: {"state", ...}}, "state", "timing", "usage", ...}

Written with WATCH/MULTI (the job and the API may both touch a Mix)."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable

from .project_store import _redis_client

LOCK_TTL_SEC = 3600


def _scope(value: str) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()[:20]


def mix_key(user_id: str, mix_id: str) -> str:
    return f"cutsell:v1:mix:{_scope(user_id)}:{_scope(mix_id)}"


def index_key(user_id: str) -> str:
    return f"cutsell:v1:mix-index:{_scope(user_id)}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_new(record: dict[str, Any], *, client=None) -> dict[str, Any]:
    r = _redis_client(client)
    pipe = r.pipeline()
    pipe.set(mix_key(record["user_id"], record["mix_id"]), json.dumps(record, ensure_ascii=False))
    pipe.zadd(index_key(record["user_id"]), {record["mix_id"]: time.time()})
    pipe.execute()
    return record


def load(user_id: str, mix_id: str, *, client=None) -> dict[str, Any]:
    raw = _redis_client(client).get(mix_key(user_id, mix_id))
    if raw is None:
        raise KeyError("mix not found")
    return json.loads(raw)


def exists(user_id: str, mix_id: str, *, client=None) -> bool:
    return bool(_redis_client(client).exists(mix_key(user_id, mix_id)))


def update(user_id: str, mix_id: str, change: Callable[[dict], Any], *, client=None) -> dict[str, Any]:
    r = _redis_client(client)
    key = mix_key(user_id, mix_id)
    from redis.exceptions import WatchError
    for _ in range(20):
        with r.pipeline() as pipe:
            try:
                pipe.watch(key)
                raw = pipe.get(key)
                if raw is None:
                    raise KeyError("mix not found")
                data = json.loads(raw)
                change(data)
                data["updated_at"] = now()
                pipe.multi()
                pipe.set(key, json.dumps(data, ensure_ascii=False))
                pipe.execute()
                return data
            except WatchError:
                continue
    raise RuntimeError("mix update conflict")


def recent(user_id: str, *, limit: int = 20, client=None) -> list[dict[str, Any]]:
    r = _redis_client(client)
    ids = [i.decode() if isinstance(i, bytes) else i for i in r.zrevrange(index_key(user_id), 0, limit - 1)]
    out = []
    for mix_id in ids:
        raw = r.get(mix_key(user_id, mix_id))
        if raw is not None:
            out.append(json.loads(raw))
    return out


def lock(user_id: str, mix_id: str, *, client=None) -> bool:
    return bool(_redis_client(client).set(mix_key(user_id, mix_id) + ":lock", "1", nx=True, ex=LOCK_TTL_SEC))


def unlock(user_id: str, mix_id: str, *, client=None) -> None:
    _redis_client(client).delete(mix_key(user_id, mix_id) + ":lock")


def summary(record: dict[str, Any]) -> dict[str, Any]:
    """What the app shows: the group in Processing, each video's code and state, failed clips."""
    combos = record.get("combos") or []
    clips = record.get("clips") or {}
    failed = [{"clip_id": cid, "section": c.get("section"), "label": c.get("label"), "error": c.get("error")}
              for cid, c in clips.items() if c.get("state") == "failed"]
    return {
        "mix_id": record["mix_id"], "state": record.get("state"), "product_id": record.get("product_id"),
        "brand": record.get("brand"), "product": record.get("product"),
        "sections": [{"name": s["name"], "code": s["code"], "clips": len(s["clips"])} for s in record.get("sections") or []],
        "videos": [{"n": c["n"], "code": c["code"], "project_id": c["project_id"], "state": c.get("state"),
                    "duration_sec": c.get("duration_sec")} for c in combos],
        "ready": sum(1 for c in combos if c.get("state") == "ready"), "total": len(combos),
        "clip_count": sum(len(s["clips"]) for s in record.get("sections") or []),
        "failed_clips": failed, "created_at": record.get("created_at"), "timing": record.get("timing") or {},
    }
