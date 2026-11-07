"""Server storage kept to the minimum (Product Owner, 22 Oct).

- Originals and finished exports are not kept on the server (exports are made on the iPhone; the server export is a
  fallback whose file is temporary).
- The light 1080p copy and the voice file of a cut are removed 3 days after the cut was LAST USED (opened, exported,
  cut again). The cut itself (decisions, captions, keyframes: text in Redis) is kept forever.
- When a cut whose copy was removed is opened, the iPhone makes the copy again from the original in Photos and uploads it
  to the same place ("Restoring video…"): `restore_target` below.

Nothing here deletes by itself: `candidates()` only reports. `remove()` deletes only with
CUTSELL_STORAGE_LIFECYCLE_DELETE=1 AND confirm=True, and only under the app's upload prefix -- never QA material,
benchmarks or anything outside `cutsell/uploads/`. Running it is the Product Owner's decision.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import time
from typing import Any, Iterable

from .uploads import scoped_upload_prefix, upload_prefix

LAST_USED_KEY = "cutsell:storage:last_used"
REPORT_KEY = "cutsell:storage:last_report"
KEEP_DAYS = 3


def keep_hours() -> float:
    """PO 7 Nov 2026: 24 hours after the last use (CUTSELL_STORAGE_KEEP_HOURS)."""
    try:
        return max(1.0, float(os.environ.get("CUTSELL_STORAGE_KEEP_HOURS", "24")))
    except ValueError:
        return 24.0


_VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".qt")


def is_source_copy(key: str) -> bool:
    """Only the light video copy the iPhone uploaded ({32 hex}-{name}.mov/.mp4 directly in the project folder):
    the one thing the iPhone can upload again from Photos. Never the voice file, clean audio (clean/), timeline
    media (timeline/), Mix videos (mix-..., mix/) or anything else: those cannot be restored from the phone."""
    import re
    prefix = upload_prefix()
    if not key.startswith(prefix):
        return False
    parts = key[len(prefix):].split("/")
    if len(parts) != 3:                 # user16 / project16 / file -- no subfolders
        return False
    name = parts[2].lower()
    return bool(re.match(r"^[0-9a-f]{32}-", name)) and name.endswith(_VIDEO_SUFFIXES)
_LOG = logging.getLogger("cutsell.storage_lifecycle")


def _redis(client=None):
    if client is not None:
        return client
    url = os.environ.get("REDIS_URL", "").strip()
    if not url:
        return None
    from redis import Redis
    return Redis.from_url(url, socket_timeout=3)


def media_uris(source: dict) -> list[str]:
    """The stored files of one source: the video copy and its voice (audio first)."""
    out = [str(source.get("uri") or "")]
    meta = source.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("audio_uri"):
        out.append(str(meta["audio_uri"]))
    return [u for u in out if u.startswith("s3://")]


def touch(uris: Iterable[str], client=None, *, now: float | None = None) -> None:
    """Marks the files as used now (opening a cut, exporting it, cutting it again). Never raises."""
    uris = [u for u in uris if u]
    if not uris:
        return
    try:
        target = _redis(client)
        if target is None:
            return
        stamp = time.time() if now is None else now
        target.zadd(LAST_USED_KEY, {u: stamp for u in uris})
    except Exception as exc:
        _LOG.warning("storage_touch_failed %s", exc.__class__.__name__)


def candidates(s3, bucket: str, client=None, *, now: float | None = None, keep_days: float | None = None,
               only_source_copies: bool = False) -> dict[str, Any]:
    """Report only: the app's uploaded files not used for `keep_days` (last use, else the upload date).
    only_source_copies: the daily cleanup looks only at the iPhone's video copies (`is_source_copy`)."""
    now = time.time() if now is None else now
    keep_days = KEEP_DAYS if keep_days is None else keep_days
    limit = now - keep_days * 86400
    target = _redis(client)
    prefix = upload_prefix()
    out = {"bucket_files": 0, "bucket_bytes": 0, "old_files": 0, "old_bytes": 0, "keys": []}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key, size = obj["Key"], int(obj["Size"])
            if only_source_copies and not is_source_copy(key):
                continue
            out["bucket_files"] += 1
            out["bucket_bytes"] += size
            used = None
            if target is not None:
                try:
                    used = target.zscore(LAST_USED_KEY, f"s3://{bucket}/{key}")
                except Exception:
                    used = None
            last = float(used) if used is not None else obj["LastModified"].timestamp()
            if last < limit:
                out["old_files"] += 1
                out["old_bytes"] += size
                out["keys"].append(key)
    return out


def remove(s3, bucket: str, keys: list[str], *, confirm: bool = False) -> int:
    """Deletes the reported files. Off unless CUTSELL_STORAGE_LIFECYCLE_DELETE=1 and confirm=True."""
    if not confirm or os.environ.get("CUTSELL_STORAGE_LIFECYCLE_DELETE", "0").strip() != "1":
        return 0
    prefix = upload_prefix()
    removed = 0
    for key in keys:
        if not key.startswith(prefix):
            continue   # never anything outside the app's uploads
        s3.delete_object(Bucket=bucket, Key=key)
        removed += 1
    return removed


def restore_target(*, uri: str, user_id: str, project_id: str, bucket: str) -> str:
    """The object key a restored copy is uploaded to: the SAME key, only inside this user's own project prefix."""
    head = f"s3://{bucket}/"
    if not uri.startswith(head):
        raise ValueError("source is not in the current storage")
    key = uri[len(head):]
    if not key.startswith(scoped_upload_prefix(user_id=user_id, project_id=project_id)):
        raise PermissionError("source does not belong to this project")
    return key


def daily(s3, bucket: str, client=None, *, now: float | None = None) -> dict[str, Any]:
    """The daily cleanup (PO 7 Nov 2026). Reports the iPhone video copies unused for `keep_hours()`, and deletes them
    only when CUTSELL_STORAGE_LIFECYCLE_DELETE=1 (off at first: report only). The report is kept in Redis (numbers
    only, no names) so the Product Owner can see what it would remove."""
    import json
    now = time.time() if now is None else now
    found = candidates(s3, bucket, client, now=now, keep_days=keep_hours() / 24.0, only_source_copies=True)
    deleting = os.environ.get("CUTSELL_STORAGE_LIFECYCLE_DELETE", "0").strip() == "1"
    removed = remove(s3, bucket, [k for k in found["keys"] if is_source_copy(k)], confirm=deleting)
    report = {"at": dt.datetime.fromtimestamp(now, tz=dt.timezone.utc).isoformat(), "keep_hours": keep_hours(),
              "copies": found["bucket_files"], "copies_bytes": found["bucket_bytes"],
              "unused": found["old_files"], "unused_bytes": found["old_bytes"],
              "mode": "delete" if deleting else "report_only", "removed": removed}
    try:
        target = _redis(client)
        if target is not None:
            target.set(REPORT_KEY, json.dumps(report))
    except Exception as exc:
        _LOG.warning("storage_report_store_failed %s", exc.__class__.__name__)
    _LOG.info("storage_daily %s", report)
    return report
