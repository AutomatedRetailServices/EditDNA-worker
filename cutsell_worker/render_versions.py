"""Recoverable render-version history for CutSell projects.

Render files live in S3. Redis stores only bounded metadata so projects can reopen
previous exports without keeping stale presigned URLs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any
from uuid import uuid4

from .config import load_runtime_config

MAX_RENDER_VERSIONS = 20


def _scope_hash(value: str) -> str:
    if not value or len(value) > 200:
        raise ValueError("render version scope identifiers must contain 1 to 200 characters")
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def render_versions_key(*, user_id: str, project_id: str) -> str:
    return f"cutsell:v1:renders:{_scope_hash(user_id)}:{_scope_hash(project_id)}"


def _redis_client(client=None):
    if client is not None:
        return client
    config = load_runtime_config()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL is required for render history")
    from redis import Redis
    return Redis.from_url(config.redis_url)


def _decode(raw) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    value = json.loads(str(raw))
    if not isinstance(value, list):
        raise ValueError("stored render version history is invalid")
    return [dict(item) for item in value if isinstance(item, dict)]


def add_render_version(
    *,
    user_id: str,
    project_id: str,
    export_uri: str,
    size_bytes: int,
    selected_count: int,
    text_overlay_count: int = 0,
    media_overlay_count: int = 0,
    client=None,
) -> dict[str, Any]:
    if not str(export_uri).startswith("s3://"):
        raise ValueError("render version requires S3 export URI")
    target = _redis_client(client)
    key = render_versions_key(user_id=user_id, project_id=project_id)
    record = {
        "render_version_id": f"rv_{uuid4().hex}",
        "project_id": project_id,
        "user_id": user_id,
        "export_uri": str(export_uri),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": max(0, int(size_bytes)),
        "selected_count": max(0, int(selected_count)),
        "text_overlay_count": max(0, int(text_overlay_count)),
        "media_overlay_count": max(0, int(media_overlay_count)),
    }
    history = _decode(target.get(key))
    history.insert(0, record)
    target.set(key, json.dumps(history[:MAX_RENDER_VERSIONS], ensure_ascii=False))
    return record


def list_render_versions(*, user_id: str, project_id: str, client=None) -> list[dict[str, Any]]:
    target = _redis_client(client)
    return _decode(target.get(render_versions_key(user_id=user_id, project_id=project_id)))


def sign_export_uri(export_uri: str, *, expires_in: int = 3600, client=None) -> str:
    if not 60 <= int(expires_in) <= 86400:
        raise ValueError("render version expiry must be between 60 and 86400 seconds")
    value = str(export_uri)
    if not value.startswith("s3://"):
        raise ValueError("render version export URI must be S3")
    bucket_key = value[5:]
    bucket, sep, key = bucket_key.partition("/")
    if not bucket or not sep or not key:
        raise ValueError("render version export URI is invalid")
    config = load_runtime_config()
    if config.s3_bucket and bucket != config.s3_bucket:
        raise ValueError("render version bucket is not allowed")
    if not key.startswith("cutsell/exports/"):
        raise ValueError("render version key is outside CutSell exports")
    if client is None:
        import boto3
        client = boto3.client("s3", region_name=config.aws_region or "us-east-1")
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=int(expires_in),
    )


def hydrated_render_versions(*, user_id: str, project_id: str, client=None, s3_client=None) -> list[dict[str, Any]]:
    output = []
    for item in list_render_versions(user_id=user_id, project_id=project_id, client=client):
        current = dict(item)
        try:
            current["download_url"] = sign_export_uri(current["export_uri"], client=s3_client)
            current["download_url_status"] = "ready"
        except Exception as exc:
            current["download_url_status"] = "degraded"
            current["download_url_reason"] = exc.__class__.__name__
        output.append(current)
    return output
