"""Direct-to-S3 mobile upload contracts for CutSell.

The API signs bounded uploads; video bytes never transit the FastAPI service. Product
jobs only accept objects inside the configured CutSell upload prefix/bucket.
"""
from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path
import re
from urllib.parse import quote
from uuid import uuid4

from .config import load_runtime_config
from .storage import parse_s3_uri

DEFAULT_UPLOAD_PREFIX = "cutsell/uploads/"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-m4v",
    "video/webm",
    "application/octet-stream",
}


def upload_prefix() -> str:
    prefix = os.getenv("CUTSELL_UPLOAD_PREFIX", DEFAULT_UPLOAD_PREFIX)
    if not prefix or prefix.startswith("/") or ".." in prefix.split("/") or not prefix.endswith("/"):
        raise ValueError("CUTSELL_UPLOAD_PREFIX must be a safe relative prefix ending in '/'")
    return prefix


def _safe_name(original_name: str) -> str:
    base = Path(original_name or "").name
    suffix = Path(base).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError("unsupported video extension")
    stem = Path(base).stem
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")[:72] or "video"
    return clean + suffix


def _scope_hash(value: str) -> str:
    if not value or len(value) > 200:
        raise ValueError("upload scope identifiers must contain 1 to 200 characters")
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def create_presigned_upload(
    *,
    project_id: str,
    user_id: str,
    original_name: str,
    content_type: str | None,
    size_bytes: int,
    expires_in: int = 900,
    client=None,
) -> dict:
    if not 1 <= int(size_bytes) <= MAX_UPLOAD_BYTES:
        raise ValueError("upload size is outside allowed range")
    if not 60 <= int(expires_in) <= 3600:
        raise ValueError("upload expiry must be between 60 and 3600 seconds")
    safe_name = _safe_name(original_name)
    detected = mimetypes.guess_type(safe_name)[0]
    resolved_type = (content_type or detected or "application/octet-stream").lower()
    if resolved_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError("unsupported video content type")

    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")
    prefix = upload_prefix()
    key = (
        f"{prefix}{_scope_hash(user_id)}/{_scope_hash(project_id)}/"
        f"{uuid4().hex}-{safe_name}"
    )
    if client is None:
        import boto3
        client = boto3.client("s3", region_name=config.aws_region or "us-east-1")
    post = client.generate_presigned_post(
        Bucket=config.s3_bucket,
        Key=key,
        Fields={"Content-Type": resolved_type},
        Conditions=[
            {"Content-Type": resolved_type},
            ["content-length-range", 1, int(size_bytes)],
        ],
        ExpiresIn=int(expires_in),
    )
    return {
        "method": "POST",
        "upload_url": post["url"],
        "fields": post.get("fields", {}),
        "source_uri": f"s3://{config.s3_bucket}/{quote(key, safe='/._-')}",
        "object_key": key,
        "content_type": resolved_type,
        "max_bytes": int(size_bytes),
        "expires_in": int(expires_in),
    }


def validate_product_source_uri(uri: str) -> tuple[str, str]:
    bucket, key = parse_s3_uri(uri)
    config = load_runtime_config()
    if not config.s3_bucket or bucket != config.s3_bucket:
        raise ValueError("source uri bucket is not allowed")
    prefix = upload_prefix()
    if not key.startswith(prefix):
        raise ValueError("source uri is outside CutSell upload prefix")
    if Path(key).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError("source uri does not reference a supported video")
    return bucket, key
