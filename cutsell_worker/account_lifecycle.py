"""Deletion lifecycle for CutSell projects and accounts.

Deletes project-scoped S3 media/artifacts, recoverable Redis state and optional
durable SQL records. Account deletion revokes indexed sessions and removes every
known project before durable user metadata is deleted.
"""
from __future__ import annotations

import hashlib
from typing import Any

from .auth import revoke_all_sessions, session_index_key
from .batch import batch_key
from .config import load_runtime_config
from .draft_store import draft_key
from .exports import EXPORT_PREFIX
from .notifications import notification_key
from .overlay_uploads import OVERLAY_PREFIX
from .project_store import get_project, list_projects, project_index_key, project_key
from .timeline_asset_storage import ASSET_PREFIX
from .uploads import upload_prefix


def _scope16(value: str) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()[:16]


def _scope20(value: str) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()[:20]


def _project_prefixes(*, user_id: str, project_id: str) -> tuple[str, ...]:
    user16 = _scope16(user_id)
    project16 = _scope16(project_id)
    return (
        f"{upload_prefix()}{user16}/{project16}/",
        f"{OVERLAY_PREFIX}{user16}/{project16}/",
        f"{ASSET_PREFIX}{user16}/{project16}/",
        f"{EXPORT_PREFIX}{user16}/{project16}/",
    )


def _delete_s3_prefix(client, *, bucket: str, prefix: str) -> int:
    deleted = 0
    token = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        keys = [item["Key"] for item in page.get("Contents") or () if item.get("Key")]
        for start in range(0, len(keys), 1000):
            chunk = keys[start:start + 1000]
            if chunk:
                client.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": [{"Key": key} for key in chunk], "Quiet": True},
                )
                deleted += len(chunk)
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return deleted


def delete_project_data(*, user_id: str, project_id: str, redis_client=None, s3_client=None) -> dict[str, Any]:
    """Delete one owned project and all known project-scoped media/artifacts."""
    # Ownership check must happen before destructive operations.
    get_project(user_id=user_id, project_id=project_id, client=redis_client)
    config = load_runtime_config()

    if redis_client is None:
        if not config.redis_url:
            raise RuntimeError("REDIS_URL is required for project deletion")
        from redis import Redis
        redis_client = Redis.from_url(config.redis_url)

    deleted_objects = 0
    if config.s3_bucket:
        if s3_client is None:
            import boto3
            s3_client = boto3.client("s3", region_name=config.aws_region or "us-east-1")
        for prefix in _project_prefixes(user_id=user_id, project_id=project_id):
            deleted_objects += _delete_s3_prefix(s3_client, bucket=config.s3_bucket, prefix=prefix)

    pipe = redis_client.pipeline()
    pipe.delete(project_key(user_id=user_id, project_id=project_id))
    pipe.delete(draft_key(user_id=user_id, project_id=project_id))
    pipe.zrem(project_index_key(user_id=user_id), project_id)
    pipe.execute()

    durable = {"status": "not_configured"}
    if config.database_url:
        try:
            from .commercial_store import durable_delete_project
            durable = {
                "status": "deleted",
                "rows": durable_delete_project(config.database_url, user_id=user_id, project_id=project_id),
            }
        except Exception as exc:
            durable = {"status": "degraded", "reason": exc.__class__.__name__}

    return {
        "status": "deleted",
        "project_id": project_id,
        "s3_objects_deleted": deleted_objects,
        "durable": durable,
    }


def delete_account_data(*, user_id: str, redis_client=None, s3_client=None) -> dict[str, Any]:
    """Delete known CutSell account data and revoke indexed sessions."""
    config = load_runtime_config()
    if redis_client is None:
        if not config.redis_url:
            raise RuntimeError("REDIS_URL is required for account deletion")
        from redis import Redis
        redis_client = Redis.from_url(config.redis_url)

    projects = list_projects(user_id=user_id, client=redis_client, limit=100)
    deleted_projects = []
    deleted_objects = 0
    for project in projects:
        project_id = str(project.get("project_id") or "")
        if not project_id:
            continue
        result = delete_project_data(
            user_id=user_id,
            project_id=project_id,
            redis_client=redis_client,
            s3_client=s3_client,
        )
        deleted_projects.append(project_id)
        deleted_objects += int(result.get("s3_objects_deleted") or 0)

    sessions = revoke_all_sessions(user_id=user_id, client=redis_client)
    pipe = redis_client.pipeline()
    pipe.delete(project_index_key(user_id=user_id))
    pipe.delete(notification_key(user_id))
    pipe.delete(session_index_key(user_id))
    # Remove any batch records under the same user hash without touching other users.
    for key in redis_client.scan_iter(match=f"cutsell:v1:batch:{_scope20(user_id)}:*"):
        pipe.delete(key)
    pipe.execute()

    durable = {"status": "not_configured"}
    if config.database_url:
        try:
            from .commercial_store import durable_delete_user
            durable = {"status": "deleted", **durable_delete_user(config.database_url, user_id=user_id)}
        except Exception as exc:
            durable = {"status": "degraded", "reason": exc.__class__.__name__}

    return {
        "status": "deleted",
        "user_id": user_id,
        "projects_deleted": deleted_projects,
        "s3_objects_deleted": deleted_objects,
        "sessions": sessions,
        "durable": durable,
    }
