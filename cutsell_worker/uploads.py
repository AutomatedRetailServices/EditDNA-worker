"""Direct-to-S3 mobile upload contracts for CutSell.

The API signs bounded uploads; video bytes never transit the FastAPI service. Product
jobs only accept objects inside the configured CutSell upload prefix/bucket and the
same hashed user/project scope that requested the upload.

## D-282 Stage 6/7/11 -- voice-over audio upload allowlist (additive)

D-281 left the VO-recording direct-upload contract as an explicit, named
gap: this module's own allowlist (`ALLOWED_VIDEO_EXTENSIONS`/`ALLOWED_
CONTENT_TYPES`) covers VIDEO containers only. This gate closes it with
the SMALLEST safe V1 addition -- `.m4a` (AAC-in-MP4) only -- grounded in
actual evidence, not invented: (1) D-281's own real test suite already
qualified a `.m4a` VO fixture end-to-end through `source_media_profile.
probe_source_media_profile` (real ffprobe evidence this exact container/
codec is already fully supported by the qualification stack); (2) `.m4a`
(AAC-in-MP4) is iOS's own `AVAudioRecorder` default output container --
a well-established platform fact, not a guess. A separate prefix
(`DEFAULT_VOICE_OVER_UPLOAD_PREFIX`), a separate, tighter byte ceiling,
and a separate validator function keep this from silently broadening the
existing VIDEO upload allowlist's own attack surface (Stage 7's own
explicit instruction) -- `prepare_upload_target`/`create_presigned_
upload`/`ALLOWED_VIDEO_EXTENSIONS` are untouched.
"""
from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path
import re
from uuid import uuid4

from .config import load_runtime_config
from .storage import parse_s3_uri

DEFAULT_UPLOAD_PREFIX = "cutsell/uploads/"
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-m4v",
    "video/webm",
    "application/octet-stream",
}

# D-282 Stage 6/7/11: the smallest safe V1 voice-over audio upload
# allowlist -- a distinct prefix/limit/extension set from the VIDEO
# allowlist above, never merged with it.
DEFAULT_VOICE_OVER_UPLOAD_PREFIX = "cutsell/voice-over-uploads/"
# A spoken voice-over track is never full-length video -- reuses the
# SAME bytes-range-limiting mechanism as MAX_UPLOAD_BYTES (Stage 8's own
# "bounded file size using existing limits if applicable"), just a
# tighter ceiling appropriate to a short recorded audio clip.
MAX_VOICE_OVER_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_VOICE_OVER_AUDIO_EXTENSIONS = {".m4a"}
ALLOWED_VOICE_OVER_CONTENT_TYPES = {
    "audio/mp4",
    "audio/x-m4a",
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


def scoped_upload_prefix(*, user_id: str, project_id: str) -> str:
    return f"{upload_prefix()}{_scope_hash(user_id)}/{_scope_hash(project_id)}/"


def prepare_upload_target(
    *,
    project_id: str,
    user_id: str,
    original_name: str,
    content_type: str | None,
    size_bytes: int,
) -> dict:
    """Validate mobile upload metadata and allocate one scoped S3 object key."""
    if not 1 <= int(size_bytes) <= MAX_UPLOAD_BYTES:
        raise ValueError("upload size is outside allowed range")
    safe_name = _safe_name(original_name)
    detected = mimetypes.guess_type(safe_name)[0]
    resolved_type = (content_type or detected or "application/octet-stream").lower()
    if resolved_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError("unsupported video content type")
    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")
    key = f"{scoped_upload_prefix(user_id=user_id, project_id=project_id)}{uuid4().hex}-{safe_name}"
    return {
        "bucket": config.s3_bucket,
        "region": config.aws_region or "us-east-1",
        "object_key": key,
        "source_uri": f"s3://{config.s3_bucket}/{key}",
        "content_type": resolved_type,
        "size_bytes": int(size_bytes),
    }


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
    if not 60 <= int(expires_in) <= 3600:
        raise ValueError("upload expiry must be between 60 and 3600 seconds")
    target = prepare_upload_target(
        project_id=project_id,
        user_id=user_id,
        original_name=original_name,
        content_type=content_type,
        size_bytes=size_bytes,
    )
    if client is None:
        import boto3
        client = boto3.client("s3", region_name=target["region"])
    post = client.generate_presigned_post(
        Bucket=target["bucket"],
        Key=target["object_key"],
        Fields={"Content-Type": target["content_type"]},
        Conditions=[
            {"Content-Type": target["content_type"]},
            ["content-length-range", 1, target["size_bytes"]],
        ],
        ExpiresIn=int(expires_in),
    )
    return {
        "method": "POST",
        "upload_url": post["url"],
        "fields": post.get("fields", {}),
        "source_uri": target["source_uri"],
        "object_key": target["object_key"],
        "content_type": target["content_type"],
        "max_bytes": target["size_bytes"],
        "expires_in": int(expires_in),
    }


def validate_product_source_uri(
    uri: str,
    *,
    project_id: str | None = None,
    user_id: str | None = None,
) -> tuple[str, str]:
    bucket, key = parse_s3_uri(uri)
    config = load_runtime_config()
    if not config.s3_bucket or bucket != config.s3_bucket:
        raise ValueError("source uri bucket is not allowed")
    required_prefix = upload_prefix()
    if project_id is not None or user_id is not None:
        if not project_id or not user_id:
            raise ValueError("both project_id and user_id are required for scoped source validation")
        required_prefix = scoped_upload_prefix(user_id=user_id, project_id=project_id)
    if not key.startswith(required_prefix):
        raise ValueError("source uri is outside allowed CutSell upload scope")
    if Path(key).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError("source uri does not reference a supported video")
    return bucket, key


def create_presigned_source_download(
    uri: str,
    *,
    project_id: str,
    user_id: str,
    expires_in: int = 1800,
    client=None,
) -> dict:
    """Create a short-lived GET URL only for a source owned by this user/project."""
    if not 60 <= int(expires_in) <= 3600:
        raise ValueError("source download expiry must be between 60 and 3600 seconds")
    bucket, key = validate_product_source_uri(
        uri,
        project_id=project_id,
        user_id=user_id,
    )
    config = load_runtime_config()
    if client is None:
        import boto3
        client = boto3.client("s3", region_name=config.aws_region or "us-east-1")
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=int(expires_in),
    )
    return {"url": url, "expires_in": int(expires_in)}


# =============================================================================
# D-282 Stage 6/7/10/37 -- voice-over audio upload contract (additive,
# structurally distinct from the VIDEO functions above; never merged).
# =============================================================================

def voice_over_upload_prefix() -> str:
    prefix = os.getenv("CUTSELL_VOICE_OVER_UPLOAD_PREFIX", DEFAULT_VOICE_OVER_UPLOAD_PREFIX)
    if not prefix or prefix.startswith("/") or ".." in prefix.split("/") or not prefix.endswith("/"):
        raise ValueError("CUTSELL_VOICE_OVER_UPLOAD_PREFIX must be a safe relative prefix ending in '/'")
    return prefix


def _safe_voice_over_name(original_name: str) -> str:
    base = Path(original_name or "").name
    suffix = Path(base).suffix.lower()
    if suffix not in ALLOWED_VOICE_OVER_AUDIO_EXTENSIONS:
        raise ValueError("unsupported voice-over audio extension")
    stem = Path(base).stem
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")[:72] or "voice-over"
    return clean + suffix


def scoped_voice_over_upload_prefix(*, user_id: str, project_id: str) -> str:
    return f"{voice_over_upload_prefix()}{_scope_hash(user_id)}/{_scope_hash(project_id)}/"


def prepare_voice_over_upload_target(
    *,
    project_id: str,
    user_id: str,
    original_name: str,
    content_type: str | None,
    size_bytes: int,
) -> dict:
    """The voice-over sibling of `prepare_upload_target` -- validates
    against the audio allowlist only, allocates a key under the
    SEPARATE `voice_over_upload_prefix()`, never the video prefix."""
    if not 1 <= int(size_bytes) <= MAX_VOICE_OVER_UPLOAD_BYTES:
        raise ValueError("voice-over upload size is outside allowed range")
    safe_name = _safe_voice_over_name(original_name)
    detected = mimetypes.guess_type(safe_name)[0]
    resolved_type = (content_type or detected or "application/octet-stream").lower()
    if resolved_type not in ALLOWED_VOICE_OVER_CONTENT_TYPES:
        raise ValueError("unsupported voice-over audio content type")
    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")
    key = f"{scoped_voice_over_upload_prefix(user_id=user_id, project_id=project_id)}{uuid4().hex}-{safe_name}"
    return {
        "bucket": config.s3_bucket,
        "region": config.aws_region or "us-east-1",
        "object_key": key,
        "source_uri": f"s3://{config.s3_bucket}/{key}",
        "content_type": resolved_type,
        "size_bytes": int(size_bytes),
    }


def create_presigned_voice_over_upload(
    *,
    project_id: str,
    user_id: str,
    original_name: str,
    content_type: str | None,
    size_bytes: int,
    expires_in: int = 900,
    client=None,
) -> dict:
    if not 60 <= int(expires_in) <= 3600:
        raise ValueError("voice-over upload expiry must be between 60 and 3600 seconds")
    target = prepare_voice_over_upload_target(
        project_id=project_id, user_id=user_id, original_name=original_name,
        content_type=content_type, size_bytes=size_bytes,
    )
    if client is None:
        import boto3
        client = boto3.client("s3", region_name=target["region"])
    post = client.generate_presigned_post(
        Bucket=target["bucket"],
        Key=target["object_key"],
        Fields={"Content-Type": target["content_type"]},
        Conditions=[
            {"Content-Type": target["content_type"]},
            ["content-length-range", 1, target["size_bytes"]],
        ],
        ExpiresIn=int(expires_in),
    )
    return {
        "method": "POST",
        "upload_url": post["url"],
        "fields": post.get("fields", {}),
        "source_uri": target["source_uri"],
        "object_key": target["object_key"],
        "content_type": target["content_type"],
        "max_bytes": target["size_bytes"],
        "expires_in": int(expires_in),
    }


def validate_voice_over_source_uri(
    uri: str,
    *,
    project_id: str,
    user_id: str,
) -> tuple[str, str]:
    """The voice-over sibling of `validate_product_source_uri` -- requires
    the SEPARATE voice-over prefix and the audio extension allowlist; a
    video-uploaded object can never pass this check, and vice versa."""
    bucket, key = parse_s3_uri(uri)
    config = load_runtime_config()
    if not config.s3_bucket or bucket != config.s3_bucket:
        raise ValueError("source uri bucket is not allowed")
    required_prefix = scoped_voice_over_upload_prefix(user_id=user_id, project_id=project_id)
    if not key.startswith(required_prefix):
        raise ValueError("source uri is outside allowed CutSell voice-over upload scope")
    if Path(key).suffix.lower() not in ALLOWED_VOICE_OVER_AUDIO_EXTENSIONS:
        raise ValueError("source uri does not reference a supported voice-over audio file")
    return bucket, key
