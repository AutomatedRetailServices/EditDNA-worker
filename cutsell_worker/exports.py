"""S3 export storage for rendered CutSell drafts."""
from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from .config import load_runtime_config

EXPORT_PREFIX = "cutsell/exports/"


def _scope_hash(value: str) -> str:
    if not value or len(value) > 200:
        raise ValueError("export scope identifiers must contain 1 to 200 characters")
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def store_export(
    path: str,
    *,
    project_id: str,
    user_id: str,
    expires_in: int = 3600,
    client=None,
) -> dict:
    source = Path(path)
    if not source.exists() or source.stat().st_size <= 0:
        raise ValueError("rendered export is missing or empty")
    if not 60 <= int(expires_in) <= 86400:
        raise ValueError("export expiry must be between 60 and 86400 seconds")
    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")
    key = (
        f"{EXPORT_PREFIX}{_scope_hash(user_id)}/{_scope_hash(project_id)}/"
        f"{uuid4().hex}.mp4"
    )
    if client is None:
        import boto3
        client = boto3.client("s3", region_name=config.aws_region or "us-east-1")
    client.upload_file(
        str(source),
        config.s3_bucket,
        key,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": config.s3_bucket, "Key": key},
        ExpiresIn=int(expires_in),
    )
    return {
        "export_uri": f"s3://{config.s3_bucket}/{key}",
        "download_url": url,
        "expires_in": int(expires_in),
        "size_bytes": source.stat().st_size,
    }
