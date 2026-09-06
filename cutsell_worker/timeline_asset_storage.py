"""S3 persistence and fresh signed URLs for mobile timeline presentation assets.

Stable S3 URIs are persisted with the draft/source metadata. Expiring download URLs
are generated on recovery so reopening a project never depends on stale URLs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from .config import load_runtime_config

ASSET_PREFIX = "cutsell/timeline-assets/"


def _scope_hash(value: str) -> str:
    if not value or len(value) > 200:
        raise ValueError("timeline asset scope identifiers must contain 1 to 200 characters")
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _source_token(source_asset_id: str) -> str:
    if not source_asset_id or len(source_asset_id) > 200:
        raise ValueError("invalid source_asset_id")
    return hashlib.sha256(source_asset_id.encode()).hexdigest()[:20]


def _prefix(*, user_id: str, project_id: str, source_asset_id: str) -> str:
    return (
        f"{ASSET_PREFIX}{_scope_hash(user_id)}/{_scope_hash(project_id)}/"
        f"{_source_token(source_asset_id)}/"
    )


def store_timeline_assets(
    *,
    user_id: str,
    project_id: str,
    source_asset_id: str,
    filmstrip: tuple[dict, ...],
    waveform: tuple[float, ...],
    client=None,
) -> dict:
    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")
    if client is None:
        import boto3
        client = boto3.client("s3", region_name=config.aws_region or "us-east-1")
    prefix = _prefix(user_id=user_id, project_id=project_id, source_asset_id=source_asset_id)

    stored_frames = []
    for index, item in enumerate(filmstrip):
        path = Path(str(item["path"]))
        if not path.exists() or path.stat().st_size <= 0:
            raise ValueError("timeline frame is missing or empty")
        key = f"{prefix}frame-{index:03d}.jpg"
        client.upload_file(str(path), config.s3_bucket, key, ExtraArgs={"ContentType": "image/jpeg"})
        stored_frames.append({
            "time": float(item["time"]),
            "uri": f"s3://{config.s3_bucket}/{key}",
        })

    waveform_key = f"{prefix}waveform.json"
    waveform_body = json.dumps({"peaks": [float(value) for value in waveform]}, separators=(",", ":")).encode()
    client.put_object(
        Bucket=config.s3_bucket,
        Key=waveform_key,
        Body=waveform_body,
        ContentType="application/json",
    )
    return {
        "status": "ready",
        "filmstrip": stored_frames,
        "waveform_uri": f"s3://{config.s3_bucket}/{waveform_key}",
        "waveform_bucket_count": len(waveform),
    }


def _parse_allowed_asset_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    config = load_runtime_config()
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError("invalid timeline asset uri")
    key = parsed.path.lstrip("/")
    if not config.s3_bucket or parsed.netloc != config.s3_bucket or not key.startswith(ASSET_PREFIX):
        raise ValueError("timeline asset uri is outside allowed scope")
    return parsed.netloc, key


def sign_timeline_assets(metadata: dict, *, expires_in: int = 3600, client=None) -> dict:
    """Return a copy with fresh HTTPS URLs while preserving stable S3 URIs."""
    if not 60 <= int(expires_in) <= 86400:
        raise ValueError("timeline asset expiry must be between 60 and 86400 seconds")
    assets = dict(metadata or {})
    if assets.get("status") != "ready":
        return assets
    config = load_runtime_config()
    if client is None:
        import boto3
        client = boto3.client("s3", region_name=config.aws_region or "us-east-1")

    signed_frames = []
    for item in assets.get("filmstrip") or ():
        bucket, key = _parse_allowed_asset_uri(str(item["uri"]))
        signed_frames.append({
            **dict(item),
            "download_url": client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=int(expires_in),
            ),
        })
    waveform_uri = str(assets.get("waveform_uri") or "")
    waveform_url = None
    if waveform_uri:
        bucket, key = _parse_allowed_asset_uri(waveform_uri)
        waveform_url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=int(expires_in),
        )
    return {
        **assets,
        "filmstrip": signed_frames,
        "waveform_download_url": waveform_url,
        "signed_url_expires_in": int(expires_in),
    }
