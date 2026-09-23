"""Sequential high-output batch orchestration for up to ten CutSell projects."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from .config import load_runtime_config

MAX_BATCH_ITEMS = 10


def _scope_hash(value: str) -> str:
    if not value or len(value) > 200:
        raise ValueError("batch scope identifiers must contain 1 to 200 characters")
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def batch_key(*, user_id: str, batch_id: str) -> str:
    return f"cutsell:v1:batch:{_scope_hash(user_id)}:{_scope_hash(batch_id)}"


def _redis_client(client=None):
    if client is not None:
        return client
    config = load_runtime_config()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL is required for batch processing")
    from redis import Redis
    return Redis.from_url(config.redis_url)


def _decode(raw):
    if raw is None:
        raise KeyError("batch not found")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    value = json.loads(str(raw))
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise ValueError("stored batch is invalid")
    return value


def create_batch(*, user_id: str, payloads: list[dict], client=None) -> dict:
    if not 1 <= len(payloads) <= MAX_BATCH_ITEMS:
        raise ValueError("batch requires 1 to 10 projects")
    for payload in payloads:
        if str(payload.get("user_id") or "") != user_id:
            raise PermissionError("all batch items must belong to the same user")
        if not payload.get("project_id") or not payload.get("sources"):
            raise ValueError("each batch item requires project_id and sources")
    batch_id = f"batch_{uuid4().hex}"
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "schema_version": "cutsell.batch.v1",
        "batch_id": batch_id,
        "user_id": user_id,
        "created_at": now,
        "updated_at": now,
        "state": "queued",
        "current_index": 0,
        "items": [
            {
                "index": index,
                "project_id": payload["project_id"],
                "state": "queued" if index == 0 else "waiting",
                "job_id": None,
                "result": None,
                "error": None,
            }
            for index, payload in enumerate(payloads)
        ],
        "payloads": payloads,
    }
    target = _redis_client(client)
    target.set(batch_key(user_id=user_id, batch_id=batch_id), json.dumps(record, ensure_ascii=False))
    return record


def get_batch(*, user_id: str, batch_id: str, client=None, include_payloads: bool = False) -> dict:
    target = _redis_client(client)
    record = _decode(target.get(batch_key(user_id=user_id, batch_id=batch_id)))
    if str(record.get("user_id") or "") != user_id:
        raise PermissionError("batch does not belong to this user")
    output = dict(record)
    if not include_payloads:
        output.pop("payloads", None)
    return output


def update_batch_item(
    *, user_id: str, batch_id: str, index: int, state: str,
    job_id: str | None = None, result=None, error: str | None = None, client=None,
) -> dict:
    target = _redis_client(client)
    key = batch_key(user_id=user_id, batch_id=batch_id)
    record = _decode(target.get(key))
    items = list(record["items"])
    if index < 0 or index >= len(items):
        raise IndexError("batch item index is invalid")
    item = dict(items[index])
    item.update({"state": state, "job_id": job_id or item.get("job_id"), "result": result, "error": error})
    items[index] = item
    record["items"] = items
    record["current_index"] = index
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    terminal = {"finished", "failed", "canceled"}
    if all(str(entry.get("state")) in terminal for entry in items):
        record["state"] = "finished"
    elif state == "processing":
        record["state"] = "processing"
    target.set(key, json.dumps(record, ensure_ascii=False))
    return record
