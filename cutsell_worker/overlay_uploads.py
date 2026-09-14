"""Direct-to-S3 upload contracts for CutSell photo/video overlay assets."""
from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
import re
from uuid import uuid4

from .config import load_runtime_config
from .storage import parse_s3_uri

OVERLAY_PREFIX = "cutsell/overlay-assets/"
MAX_OVERLAY_BYTES = 500 * 1024 * 1024
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
CONTENT_TYPES = {
    "image/jpeg", "image/png",
    "video/mp4", "video/quicktime", "video/x-m4v", "video/webm",
    "application/octet-stream",
}


def _scope_hash(value: str) -> str:
    if not value or len(value) > 200:
        raise ValueError("overlay scope identifiers must contain 1 to 200 characters")
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def overlay_prefix(*, user_id: str, project_id: str) -> str:
    return f"{OVERLAY_PREFIX}{_scope_hash(user_id)}/{_scope_hash(project_id)}/"


def _safe_name(name: str) -> tuple[str, str]:
    base = Path(name or "").name
    suffix = Path(base).suffix.lower()
    if suffix in PHOTO_EXTENSIONS:
        kind = "photo"
    elif suffix in VIDEO_EXTENSIONS:
        kind = "video"
    else:
        raise ValueError("unsupported overlay extension")
    clean_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(base).stem).strip("-._")[:72] or "overlay"
    return clean_stem + suffix, kind


def create_overlay_presigned_upload(
    *, project_id: str, user_id: str, original_name: str,
    content_type: str | None, size_bytes: int, expires_in: int = 900, client=None,
) -> dict:
    if not 1 <= int(size_bytes) <= MAX_OVERLAY_BYTES:
        raise ValueError("overlay upload size is outside allowed range")
    if not 60 <= int(expires_in) <= 3600:
        raise ValueError("overlay upload expiry must be between 60 and 3600 seconds")
    safe_name, kind = _safe_name(original_name)
    detected = mimetypes.guess_type(safe_name)[0]
    resolved_type = (content_type or detected or "application/octet-stream").lower()
    if resolved_type not in CONTENT_TYPES:
        raise ValueError("unsupported overlay content type")
    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")
    key = f"{overlay_prefix(user_id=user_id, project_id=project_id)}{uuid4().hex}-{safe_name}"
    if client is None:
        import boto3
        client = boto3.client("s3", region_name=config.aws_region or "us-east-1")
    post = client.generate_presigned_post(
        Bucket=config.s3_bucket, Key=key,
        Fields={"Content-Type": resolved_type},
        Conditions=[{"Content-Type": resolved_type}, ["content-length-range", 1, int(size_bytes)]],
        ExpiresIn=int(expires_in),
    )
    return {
        "method": "POST", "upload_url": post["url"], "fields": post.get("fields", {}),
        "uri": f"s3://{config.s3_bucket}/{key}", "object_key": key,
        "kind": kind, "content_type": resolved_type, "max_bytes": int(size_bytes), "expires_in": int(expires_in),
    }


def validate_overlay_uri(uri: str, *, user_id: str, project_id: str) -> tuple[str, str, str]:
    bucket, key = parse_s3_uri(uri)
    config = load_runtime_config()
    if not config.s3_bucket or bucket != config.s3_bucket:
        raise ValueError("overlay uri bucket is not allowed")
    prefix = overlay_prefix(user_id=user_id, project_id=project_id)
    if not key.startswith(prefix):
        raise ValueError("overlay uri is outside allowed CutSell project scope")
    suffix = Path(key).suffix.lower()
    if suffix in PHOTO_EXTENSIONS:
        kind = "photo"
    elif suffix in VIDEO_EXTENSIONS:
        kind = "video"
    else:
        raise ValueError("overlay uri has unsupported extension")
    return bucket, key, kind
