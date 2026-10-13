"""Real S3 storage for timeline assets (B-roll, voice-over, primary video).

D-281/D-282 built the timeline asset registry with injectable storage seams
(`persist_media`, `fetch_media`, `materialize`) and only a local-directory
implementation. Production (Render, no persistent disk) had none wired, so the
timeline routes could not store anything. This module is the real backend for
those same seams; it does not change qualification or composition rules.

Where the bytes live
--------------------
Qualified timeline media is stored under the project's own upload scope:

    {upload_prefix()}{user16}/{project16}/timeline/{asset_id}{suffix}

so project and account deletion (`account_lifecycle._project_prefixes`) remove
it with the rest of the project. Only server-generated names are used.

Primary video
-------------
`register_primary_video` runs at the end of a cut (worker). It registers the
cut's first source video as the project's `PRIMARY_SOURCE` timeline asset, the
base that voice-over and B-roll placements are saved against. When the source
was accepted as uploaded, the asset points at the uploaded object itself (no
copy); when the cut had to normalize it, the normalized file is stored once.
It never raises: a failure is reported as `status=degraded` and the cut still
finishes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from . import timeline_asset_registry as reg
from . import timeline_asset_registry_store as store
from .config import load_runtime_config
from .storage import parse_s3_uri
from .uploads import scoped_upload_prefix, scoped_voice_over_upload_prefix

TIMELINE_MEDIA_FOLDER = "timeline/"


def _bucket() -> str:
    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required for timeline asset storage")
    return config.s3_bucket


def s3_configured() -> bool:
    return bool(load_runtime_config().s3_bucket)


def _client(client=None):
    if client is not None:
        return client
    import boto3

    config = load_runtime_config()
    return boto3.client("s3", region_name=config.aws_region or "us-east-1")


def timeline_media_prefix(*, user_id: str, project_id: str) -> str:
    return f"{scoped_upload_prefix(user_id=user_id, project_id=project_id)}{TIMELINE_MEDIA_FOLDER}"


def _allowed_prefixes(*, user_id: str, project_id: str) -> tuple[str, ...]:
    """Objects one project may read: its own video uploads (which include its
    timeline media) and its own voice-over uploads. Nothing else in the bucket."""
    return (
        scoped_upload_prefix(user_id=user_id, project_id=project_id),
        scoped_voice_over_upload_prefix(user_id=user_id, project_id=project_id),
    )


def _checked_key(uri: str, *, user_id: str, project_id: str) -> tuple[str, str]:
    bucket, key = parse_s3_uri(uri)
    if bucket != _bucket():
        raise RuntimeError("timeline media is outside the CutSell bucket")
    if ".." in key.split("/") or not any(key.startswith(p) for p in _allowed_prefixes(user_id=user_id, project_id=project_id)):
        raise RuntimeError("timeline media is outside this project's storage scope")
    return bucket, key


def s3_persister(
    *,
    user_id: str,
    project_id: str,
    reuse: dict[str, str] | None = None,
    client=None,
) -> Callable[[str, str], str]:
    """`persist_media(local_path, storage_key)` for the registry store.

    `storage_key` is the store's own server-generated key; only its file name
    (`{asset_id}{suffix}`) is kept. `reuse` maps a local path to the s3:// URI
    it was downloaded from, so an unchanged upload is referenced, not copied."""
    prefix = timeline_media_prefix(user_id=user_id, project_id=project_id)
    reuse = dict(reuse or {})

    def _persist(local_path: str, storage_key: str) -> str:
        source = Path(local_path)
        if not source.exists() or source.stat().st_size <= 0:
            raise ValueError("asset media is missing or empty")
        reused = reuse.get(str(local_path))
        if reused:
            _checked_key(reused, user_id=user_id, project_id=project_id)
            return reused
        name = Path(storage_key).name
        if not name or name in (".", "..") or "/" in name:
            raise ValueError("unsafe storage key")
        key = f"{prefix}{name}"
        bucket = _bucket()
        _client(client).upload_file(str(source), bucket, key)
        return f"s3://{bucket}/{key}"

    return _persist


def s3_fetcher(
    *,
    user_id: str,
    project_id: str,
    scratch_dir: str,
    client=None,
) -> Callable[[str], str]:
    """`fetch_media`/`materialize` for the registry: downloads an s3:// object
    of THIS project into `scratch_dir` and returns the local path. A `local://`
    reference (tests, old local-mode records) is passed through unchanged."""

    def _fetch(reference: str) -> str:
        if reference.startswith("local://"):
            return reference[len("local://"):]
        bucket, key = _checked_key(reference, user_id=user_id, project_id=project_id)
        suffix = Path(key).suffix.lower()[:8]
        destination = Path(scratch_dir) / f"{uuid4().hex}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        _client(client).download_file(bucket, key, str(destination))
        return str(destination)

    return _fetch


def _ready_primary(*, user_id: str, project_id: str, client=None) -> reg.TimelineMediaAsset | None:
    for asset in store.list_timeline_assets(user_id=user_id, project_id=project_id, client=client):
        if (
            asset.role == reg.TimelineAssetRole.PRIMARY_SOURCE
            and asset.media_kind == reg.TimelineMediaKind.VIDEO
            and asset.qualification_status == reg.TimelineAssetQualificationStatus.READY
        ):
            return asset
    return None


def register_primary_video(
    *,
    user_id: str,
    project_id: str,
    source_uri: str,
    original_local_path: str,
    cut_local_path: str,
    scratch_dir: str,
    redis_client=None,
    s3_client=None,
) -> dict[str, Any]:
    """Registers the cut's primary video as the project's PRIMARY_SOURCE
    timeline asset. Idempotent (a retried cut keeps the existing one). Never
    raises; returns a small public-safe status dict (no paths, no URIs)."""
    try:
        existing = _ready_primary(user_id=user_id, project_id=project_id, client=redis_client)
        if existing is not None:
            return {"status": "already_registered", "asset_id": existing.asset_id}
        persist = s3_persister(
            user_id=user_id, project_id=project_id,
            reuse={str(original_local_path): source_uri}, client=s3_client,
        )
        normalization_dir = os.path.join(scratch_dir, "primary-video")
        os.makedirs(normalization_dir, exist_ok=True)
        asset = store.create_video_timeline_asset(
            user_id=user_id, project_id=project_id, role=reg.TimelineAssetRole.PRIMARY_SOURCE,
            local_source_path=str(cut_local_path), persist_media=persist,
            normalization_output_dir=normalization_dir, client=redis_client,
        )
        ready = asset.qualification_status == reg.TimelineAssetQualificationStatus.READY
        return {
            "status": "registered" if ready else "not_ready",
            "asset_id": asset.asset_id,
            "qualification_status": asset.qualification_status.value,
        }
    except Exception as exc:  # the cut must finish even if this fails
        return {"status": "degraded", "reason": exc.__class__.__name__}
