"""Deletion lifecycle for CutSell projects and accounts.

Deletes project-scoped S3 media/artifacts, recoverable Redis state and optional
durable SQL records. Account deletion revokes indexed sessions and removes every
known project before durable user metadata is deleted.

D-281 Stage 42: additively wires the D-279/D-281 manual-timeline asset
registry's own live Redis records (`timeline_asset_registry_store.py` --
NOT `timeline_asset_storage.py`'s unrelated filmstrip/waveform presentation
assets, already covered by `ASSET_PREFIX` below) into project deletion, via
`delete_project_timeline_assets`. Timeline media bytes stored in S3
(`timeline_asset_s3`: B-roll, primary video) live inside the project's upload
prefix and voice-over recordings under the voice-over prefix; both prefixes
are deleted below.
"""
from __future__ import annotations

import hashlib
from typing import Any

from .auth import delete_apple_binding, revoke_all_sessions, revoke_apple_token, session_index_key
from .config import load_runtime_config
from .draft_store import draft_key
from .exports import EXPORT_PREFIX
from .feedback import FEEDBACK_PREFIX
from .notifications import notification_key
from .overlay_uploads import OVERLAY_PREFIX
from .project_store import get_project, project_index_key, project_key
from .timeline_asset_registry_store import delete_project_timeline_assets
from .timeline_asset_storage import ASSET_PREFIX
from .uploads import upload_prefix, voice_over_upload_prefix


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
        f"{FEEDBACK_PREFIX}{user16}/{project16}/",
        # Voice-over recordings (D-282) live under their own prefix; timeline media
        # (B-roll, primary video) lives inside the upload prefix (first entry).
        f"{voice_over_upload_prefix()}{user16}/{project16}/",
    )


def _user_prefixes(*, user_id: str) -> tuple[str, ...]:
    """Every storage "drawer" of one user (all her projects together). Account deletion empties these after the
    project-by-project pass, so a project missing from the index, a Mix output or a pending-review render is not
    left behind (PO 6 Nov 2026: when she deletes her account, everything goes and nothing can be recovered)."""
    from .pending_watch_listen_review import PENDING_REVIEW_PREFIX
    user16 = _scope16(user_id)
    return tuple(f"{prefix}{user16}/" for prefix in (
        upload_prefix(), OVERLAY_PREFIX, ASSET_PREFIX, EXPORT_PREFIX, FEEDBACK_PREFIX,
        voice_over_upload_prefix(), PENDING_REVIEW_PREFIX))


def _project_key_patterns(*, user_id: str, project_id: str) -> tuple[str, ...]:
    return (
        f"cutsell:v1:pending_watch_listen:{_scope20(user_id)}:{_scope20(project_id)}:*",
        f"cutsell:v1:clean:{_scope20(user_id)}:{_scope16(project_id)}:*",
    )


def _user_keys(user_id: str) -> list[str]:
    """Per-account Redis records outside the projects (PO 6 Nov: these were left behind)."""
    from .mix_store import index_key as mix_index_key
    from .post_plan import plan_key, prints_key
    from .product_catalog import catalog_key
    return [
        project_index_key(user_id=user_id),
        notification_key(user_id),
        session_index_key(user_id),
        prints_key(user_id),
        plan_key(user_id),
        catalog_key(user_id),
        mix_index_key(user_id),
        "cutsell:v1:account_name:" + hashlib.sha256(user_id.encode()).hexdigest()[:24],
    ]


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


def delete_project_data(*, user_id: str, project_id: str, redis_client=None, s3_client=None,
                        include_old_storage: bool = False, old_storage_client=None,
                        require_record: bool = True) -> dict[str, Any]:
    """Delete one owned project and all known project-scoped media/artifacts.

    `include_old_storage` is set ONLY by account deletion (Product Owner, 15 Oct 2026): after the
    move to Cloudflare R2 the user's files from before the move are also deleted from the old
    Amazon bucket (FALLBACK_*), and only under this user's own prefixes. Nothing else ever writes
    to or deletes from the old bucket."""
    config = load_runtime_config()

    if redis_client is None:
        if not config.redis_url:
            raise RuntimeError("REDIS_URL is required for project deletion")
        from redis import Redis
        redis_client = Redis.from_url(config.redis_url)
    if require_record:
        get_project(user_id=user_id, project_id=project_id, client=redis_client)
    # Account deletion (require_record=False) cleans the project's prefixes and keys even when its record is
    # already gone, so no file is left behind under her prefixes.

    deleted_objects = 0
    if config.s3_bucket:
        if s3_client is None:
            import boto3
            s3_client = boto3.client("s3", region_name=config.aws_region or "us-east-1")
        for prefix in _project_prefixes(user_id=user_id, project_id=project_id):
            deleted_objects += _delete_s3_prefix(s3_client, bucket=config.s3_bucket, prefix=prefix)

    old_storage_deleted = 0
    if include_old_storage:
        from . import storage_routing

        fallback = storage_routing.fallback_storage()
        if fallback is not None and fallback.bucket != config.s3_bucket:
            old_client = old_storage_client or storage_routing.fallback_client_factory(fallback)
            for prefix in _project_prefixes(user_id=user_id, project_id=project_id):
                old_storage_deleted += _delete_s3_prefix(old_client, bucket=fallback.bucket, prefix=prefix)

    pipe = redis_client.pipeline()
    pipe.delete(project_key(user_id=user_id, project_id=project_id))
    pipe.delete(draft_key(user_id=user_id, project_id=project_id))
    pipe.zrem(project_index_key(user_id=user_id), project_id)
    # PO 6 Nov: the project's render list, pending reviews and clean-audio states were left behind.
    from .render_versions import render_versions_key
    pipe.delete(render_versions_key(user_id=user_id, project_id=project_id))
    for pattern in _project_key_patterns(user_id=user_id, project_id=project_id):
        for key in redis_client.scan_iter(match=pattern):
            pipe.delete(key)
    pipe.execute()

    # D-281 Stage 42: additive -- the manual-timeline asset registry's own
    # live records participate in project deletion.
    timeline_cleanup = delete_project_timeline_assets(user_id=user_id, project_id=project_id, client=redis_client)

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
        "old_storage_objects_deleted": old_storage_deleted,
        "timeline_assets_deleted": timeline_cleanup["timeline_assets_deleted"],
        "durable": durable,
    }


def _members(client, key: str) -> list:
    """Members of an index that may be a sorted set, a set or a list."""
    kind = client.type(key)
    kind = kind.decode() if isinstance(kind, bytes) else str(kind)
    if kind == "zset":
        return client.zrange(key, 0, -1)
    if kind == "set":
        return list(client.smembers(key))
    if kind == "list":
        return client.lrange(key, 0, -1)
    return []


def delete_account_data(*, user_id: str, redis_client=None, s3_client=None,
                        old_storage_client=None) -> dict[str, Any]:
    """Delete known CutSell account data and revoke indexed sessions."""
    config = load_runtime_config()
    if redis_client is None:
        if not config.redis_url:
            raise RuntimeError("REDIS_URL is required for account deletion")
        from redis import Redis
        redis_client = Redis.from_url(config.redis_url)

    # Every project in the index (no 100 cap). Each one is retried on the next request if it fails.
    raw_ids = redis_client.zrange(project_index_key(user_id=user_id), 0, -1)
    project_ids = [raw.decode("utf-8") if isinstance(raw, bytes) else str(raw) for raw in raw_ids]
    deleted_projects, failed_projects = [], []
    deleted_objects = 0
    old_deleted = 0
    for project_id in project_ids:
        if not project_id:
            continue
        try:
            result = delete_project_data(
                user_id=user_id,
                project_id=project_id,
                redis_client=redis_client,
                s3_client=s3_client,
                include_old_storage=True,
                old_storage_client=old_storage_client,
                require_record=False,
            )
        except Exception as exc:
            failed_projects.append({"project_id": project_id, "error": exc.__class__.__name__})
            continue
        deleted_projects.append(project_id)
        deleted_objects += int(result.get("s3_objects_deleted") or 0)
        old_deleted += int(result.get("old_storage_objects_deleted") or 0)

    # The whole drawer of each storage area, in the current and the old bucket (anything the pass above missed).
    if not failed_projects:
        try:
            if config.s3_bucket:
                if s3_client is None:
                    import boto3
                    s3_client = boto3.client("s3", region_name=config.aws_region or "us-east-1")
                for prefix in _user_prefixes(user_id=user_id):
                    deleted_objects += _delete_s3_prefix(s3_client, bucket=config.s3_bucket, prefix=prefix)
            from . import storage_routing
            fallback = storage_routing.fallback_storage()
            if fallback is not None and fallback.bucket != config.s3_bucket:
                old_client = old_storage_client or storage_routing.fallback_client_factory(fallback)
                for prefix in _user_prefixes(user_id=user_id):
                    old_deleted += _delete_s3_prefix(old_client, bucket=fallback.bucket, prefix=prefix)
        except Exception as exc:
            failed_projects.append({"project_id": "*storage*", "error": exc.__class__.__name__})

    if failed_projects:
        # Sessions stay alive so the app can ask again; nothing is reported as deleted when it is not.
        return {
            "status": "partial",
            "user_id": user_id,
            "projects_deleted": deleted_projects,
            "projects_failed": failed_projects,
            "s3_objects_deleted": deleted_objects,
            "old_storage_objects_deleted": old_deleted,
        }

    from .mix_store import mix_key
    from .mix_store import index_key as mix_index_key
    mix_ids = [raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
               for raw in _members(redis_client, mix_index_key(user_id))]

    apple_revoke = revoke_apple_token(user_id=user_id, client=redis_client)
    sessions = revoke_all_sessions(user_id=user_id, client=redis_client)
    apple_binding_removed = delete_apple_binding(user_id=user_id, client=redis_client)
    pipe = redis_client.pipeline()
    for key in _user_keys(user_id):
        pipe.delete(key)
    for mix_id in mix_ids:
        pipe.delete(mix_key(user_id, mix_id))
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
        "mixes_deleted": len(mix_ids),
        "s3_objects_deleted": deleted_objects,
        "old_storage_objects_deleted": old_deleted,
        "sessions": sessions,
        "apple_binding_removed": apple_binding_removed,
        "apple_revoke": apple_revoke,
        "durable": durable,
    }
