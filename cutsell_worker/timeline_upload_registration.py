"""D-282A -- opaque, server-issued upload identity for timeline asset ingest.

D-282's own `cutsell_app/timeline_routes.py` originally accepted a raw,
client-supplied `source_uri` directly on `POST /timeline-assets` -- a real
storage-authority leak: nothing stopped a client from naming ANY bucket/
object key (or, in a real production wiring, an object it never actually
uploaded, or one belonging to a different user/project). This module closes
that seam with the SAME pattern this codebase already uses for resumable
multipart uploads (`multipart_uploads.py`'s own Redis-backed, ownership-
checked upload session): the client never gets to invent a storage
reference -- it only ever holds an opaque `upload_id` this module minted,
bound server-side to the exact user/project/media-class that requested it.

## Flow (Stage 3)

1. client calls `register_timeline_upload` (via the route's own
   `POST /{project_id}/timeline-uploads`) for a specific `media_class`
2. this module allocates a real, scoped storage target via the EXISTING,
   UNMODIFIED `uploads.create_presigned_upload` / `uploads.create_
   presigned_voice_over_upload` (never a new storage-scoping mechanism)
3. this module mints `upload_id = f"tup_{uuid4().hex}"` and stores a
   short-lived Redis record binding it to `{user_id, project_id,
   media_class, source_uri}` -- the real storage reference NEVER leaves
   this record
4. client PUTs bytes directly to the returned presigned URL (unchanged)
5. client calls timeline asset ingest with `upload_id` ONLY
6. `resolve_and_consume_timeline_upload` re-checks ownership, project
   scope, and media class, consumes the record exactly once (no replay),
   and hands the real `source_uri` back to the caller INTERNALLY -- it is
   never serialized into any client-facing response

## Fake/local-storage testing (Stage 9/31/36, unchanged discipline)

`register_timeline_upload` takes an injectable `presign_upload` callable
(the SAME `client=None`-style injection seam this codebase already uses
for `persist_media`/`fetch_media`/`materialize`) so tests never need real
S3/boto3: the default dispatches to `uploads.py`'s own real presign
functions; tests inject a fake that returns a `local://` fake-storage
reference, the exact scheme `timeline_asset_upload_bridge.py`'s own
`_default_fetch_uploaded_media` already establishes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Callable
from uuid import uuid4

from .config import load_runtime_config
from .uploads import create_presigned_upload, create_presigned_voice_over_upload

UPLOAD_SESSION_TTL_SEC = 24 * 60 * 60

MEDIA_CLASS_VIDEO = "video"
MEDIA_CLASS_VOICE_OVER = "voice_over"
_VALID_MEDIA_CLASSES = frozenset({MEDIA_CLASS_VIDEO, MEDIA_CLASS_VOICE_OVER})

UPLOAD_NOT_FOUND = "UPLOAD_NOT_FOUND"
UPLOAD_NOT_OWNED = "UPLOAD_NOT_OWNED"
UPLOAD_MEDIA_CLASS_MISMATCH = "UPLOAD_MEDIA_CLASS_MISMATCH"
UPLOAD_ALREADY_CONSUMED = "UPLOAD_ALREADY_CONSUMED"


class TimelineUploadResolutionError(Exception):
    """Carries a bounded outcome code -- never a raw exception/stack trace
    reaches the API layer (same discipline as D-279's own outcome errors)."""

    def __init__(self, outcome: str):
        super().__init__(outcome)
        self.outcome = outcome


def _valid_media_class(media_class: str) -> str:
    if media_class not in _VALID_MEDIA_CLASSES:
        raise ValueError(f"unknown media_class: {media_class!r}")
    return media_class


def _redis_client(client=None):
    if client is not None:
        return client
    config = load_runtime_config()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL is required for timeline upload registration")
    from redis import Redis
    return Redis.from_url(config.redis_url)


def _session_key(upload_id: str) -> str:
    if not upload_id:
        raise ValueError("upload_id is required")
    digest = hashlib.sha256(upload_id.encode()).hexdigest()
    return f"cutsell:v1:timeline_upload:{digest}"


def _default_presign_upload(
    *,
    media_class: str,
    project_id: str,
    user_id: str,
    original_name: str,
    content_type: str | None,
    size_bytes: int,
    expires_in: int,
) -> dict[str, Any]:
    """Real production dispatch -- reuses `uploads.py`'s own, completely
    unmodified presign functions. Never a new storage-scoping mechanism."""
    if media_class == MEDIA_CLASS_VOICE_OVER:
        return create_presigned_voice_over_upload(
            project_id=project_id, user_id=user_id, original_name=original_name,
            content_type=content_type, size_bytes=size_bytes, expires_in=expires_in,
        )
    return create_presigned_upload(
        project_id=project_id, user_id=user_id, original_name=original_name,
        content_type=content_type, size_bytes=size_bytes, expires_in=expires_in,
    )


def register_timeline_upload(
    *,
    project_id: str,
    user_id: str,
    media_class: str,
    original_name: str,
    content_type: str | None,
    size_bytes: int,
    expires_in: int = 900,
    presign_upload: Callable[..., dict[str, Any]] | None = None,
    redis_client=None,
) -> dict[str, Any]:
    """Stage 1-4/5: mints ONE opaque, single-use `upload_id` bound to this
    exact user/project/media_class. Returns only what the client needs to
    perform the PUT -- never anything the ingest route later needs back
    from the client (the real `source_uri`/`object_key` stay server-side)."""
    _valid_media_class(media_class)
    presign_fn = presign_upload or _default_presign_upload
    presign = presign_fn(
        media_class=media_class, project_id=project_id, user_id=user_id,
        original_name=original_name, content_type=content_type,
        size_bytes=size_bytes, expires_in=expires_in,
    )
    upload_id = f"tup_{uuid4().hex}"
    session = {
        "schema_version": "cutsell.timeline_upload.v1",
        "upload_id": upload_id,
        "user_id": user_id,
        "project_id": project_id,
        "media_class": media_class,
        "source_uri": presign["source_uri"],
        "object_key": presign.get("object_key"),
        "content_type": presign.get("content_type"),
        "size_bytes": presign.get("max_bytes"),
        "consumed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    client = _redis_client(redis_client)
    client.set(_session_key(upload_id), json.dumps(session, ensure_ascii=False), ex=UPLOAD_SESSION_TTL_SEC)
    return {
        "upload_id": upload_id,
        "method": presign.get("method", "POST"),
        "upload_url": presign["upload_url"],
        "fields": presign.get("fields", {}),
        "content_type": presign.get("content_type"),
        "max_bytes": presign.get("max_bytes"),
        "expires_in": presign.get("expires_in", expires_in),
    }


def resolve_and_consume_timeline_upload(
    *,
    upload_id: str,
    user_id: str,
    project_id: str,
    expected_media_class: str,
    redis_client=None,
) -> str:
    """Stage 5/6: the ONLY way a real `source_uri` is ever recovered for
    ingest -- never accepted directly from a client request body.
    Ownership, project scope, and media class are all re-checked here
    (Stage 4/5's own binding); a session is consumed exactly once so a
    captured/replayed `upload_id` can never be reused (Stage 11 item 10's
    own "resolves storage internally" -- and never twice)."""
    client = _redis_client(redis_client)
    key = _session_key(upload_id)
    raw = client.get(key)
    if raw is None:
        raise TimelineUploadResolutionError(UPLOAD_NOT_FOUND)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    session = json.loads(str(raw))
    if session.get("user_id") != user_id or session.get("project_id") != project_id:
        # Stage: cross-user AND cross-project reuse both resolve the SAME
        # bounded outcome -- no leak distinguishing "exists for someone
        # else" from "does not exist" (same discipline as D-279's own
        # ASSET_NOT_OWNED / ASSET_NOT_FOUND separation is deliberately
        # NOT mirrored here: an upload session is never independently
        # discoverable by a non-owner, so there is nothing to distinguish).
        raise TimelineUploadResolutionError(UPLOAD_NOT_OWNED)
    if session.get("media_class") != expected_media_class:
        raise TimelineUploadResolutionError(UPLOAD_MEDIA_CLASS_MISMATCH)
    if session.get("consumed"):
        raise TimelineUploadResolutionError(UPLOAD_ALREADY_CONSUMED)
    session["consumed"] = True
    client.set(key, json.dumps(session, ensure_ascii=False), ex=UPLOAD_SESSION_TTL_SEC)
    return str(session["source_uri"])
