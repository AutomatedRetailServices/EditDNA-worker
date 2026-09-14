"""D-282A -- opaque, server-issued export delivery reference.

D-282's own `cutsell_app/timeline_routes.py` originally returned the
renderer's real local `output_path` directly in `TimelineExportResponse`
-- a raw server filesystem path leaked to the client. This module mints
an opaque `export_id` mapping (server-side only, short-lived, ownership-
checked) to that real path, mirroring the SAME Redis-backed-record
pattern this gate's own sibling module (`timeline_upload_registration.py`)
and the pre-existing `multipart_uploads.py` both already use. The route
returns only `export_id`; the client fetches bytes through a NEW, owned
download route that resolves the reference server-side -- the real path
never appears in any client-facing JSON.

## Scope discipline

This is a LOCAL-artifact delivery reference, matching this whole D-276..
D-282 lineage's own "fake/local storage only, no real S3 mutation"
discipline -- it is not a replacement for, and does not modify, the
EXISTING real remote tenant-safe delivery path (`exports.py`/`tenant_
safe_delivery.py`/`export_job.py`), which stays completely untouched. A
future gate that wires timeline exports through that same real S3/
presigned-GET delivery layer can swap this reference's resolution target
without changing the client-facing contract this module establishes
(an opaque id, never a path).
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any
from uuid import uuid4

from .config import load_runtime_config

EXPORT_REFERENCE_TTL_SEC = 24 * 60 * 60

EXPORT_REFERENCE_NOT_FOUND = "EXPORT_REFERENCE_NOT_FOUND"
EXPORT_REFERENCE_NOT_OWNED = "EXPORT_REFERENCE_NOT_OWNED"


class ExportReferenceResolutionError(Exception):
    """Carries a bounded outcome code -- never a raw exception/stack trace
    reaches the API layer."""

    def __init__(self, outcome: str):
        super().__init__(outcome)
        self.outcome = outcome


def _redis_client(client=None):
    if client is not None:
        return client
    config = load_runtime_config()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL is required for timeline export reference registration")
    from redis import Redis
    return Redis.from_url(config.redis_url)


def _reference_key(export_id: str) -> str:
    if not export_id:
        raise ValueError("export_id is required")
    digest = hashlib.sha256(export_id.encode()).hexdigest()
    return f"cutsell:v1:timeline_export_ref:{digest}"


def register_export_reference(
    *,
    user_id: str,
    project_id: str,
    local_output_path: str,
    format_qc_status: str | None,
    redis_client=None,
) -> str:
    """Mints an opaque `export_id`. The real local artifact path is
    stored ONLY in this Redis record -- it is never returned to a caller
    of this function's own return value, which is the id alone."""
    if not str(local_output_path or "").strip():
        raise ValueError("local_output_path must be non-empty")
    export_id = f"exp_{uuid4().hex}"
    record = {
        "schema_version": "cutsell.timeline_export_reference.v1",
        "export_id": export_id,
        "user_id": user_id,
        "project_id": project_id,
        "local_output_path": local_output_path,
        "format_qc_status": format_qc_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    client = _redis_client(redis_client)
    client.set(_reference_key(export_id), json.dumps(record, ensure_ascii=False), ex=EXPORT_REFERENCE_TTL_SEC)
    return export_id


def resolve_export_reference(
    *,
    export_id: str,
    user_id: str,
    project_id: str,
    redis_client=None,
) -> dict[str, Any]:
    """The ONLY way a real local artifact path is ever recovered for
    delivery -- ownership + project scope re-checked here, exactly the
    same discipline as every other D-279/D-281/D-282 resolver."""
    client = _redis_client(redis_client)
    raw = client.get(_reference_key(export_id))
    if raw is None:
        raise ExportReferenceResolutionError(EXPORT_REFERENCE_NOT_FOUND)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    record = json.loads(str(raw))
    if record.get("user_id") != user_id or record.get("project_id") != project_id:
        raise ExportReferenceResolutionError(EXPORT_REFERENCE_NOT_OWNED)
    return record
