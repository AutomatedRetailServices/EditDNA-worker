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
    object_key: str | None = None,
    object_metadata: dict[str, str] | None = None,
    client=None,
) -> dict:
    """D-269A Stage 4/6: `object_key` lets a caller (the live export path)
    supply the D-269 tenant-safe key instead of this module's own legacy
    `uuid4()`-based key; the legacy key remains the default so any other
    existing caller is byte-for-byte unaffected. `object_metadata`, when
    given, is attached to the uploaded object (D-268 Stage 32 item 2 --
    binding `job_id`/`render_identity`/`output_sha256` into the object's
    own metadata, a real, already-available boto3 mechanism). After
    upload, this now performs one real `head_object` call (Stage 6: "the
    live export path must consume remote metadata") and returns it as
    `remote_head` so the caller can run D-269's own `verify_remote_
    delivery` against real (or, in tests, faked) remote evidence rather
    than trusting `upload_file`'s bare success alone."""
    source = Path(path)
    if not source.exists() or source.stat().st_size <= 0:
        raise ValueError("rendered export is missing or empty")
    if not 60 <= int(expires_in) <= 86400:
        raise ValueError("export expiry must be between 60 and 86400 seconds")
    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")
    key = object_key or (
        f"{EXPORT_PREFIX}{_scope_hash(user_id)}/{_scope_hash(project_id)}/"
        f"{uuid4().hex}.mp4"
    )
    if client is None:
        import boto3
        client = boto3.client("s3", region_name=config.aws_region or "us-east-1")
    extra_args = {"ContentType": "video/mp4"}
    if object_metadata:
        extra_args["Metadata"] = dict(object_metadata)
    client.upload_file(
        str(source),
        config.s3_bucket,
        key,
        ExtraArgs=extra_args,
    )
    remote_head: dict = {"exists": False}
    try:
        head = client.head_object(Bucket=config.s3_bucket, Key=key)
        remote_head = {
            "exists": True,
            "key": key,
            "size_bytes": int(head.get("ContentLength") or 0),
            "metadata": dict(head.get("Metadata") or {}),
        }
    except Exception as exc:
        # Never destroy an otherwise-successful upload because the
        # verification HEAD itself failed/errored -- the caller's own
        # `verify_remote_delivery` treats `exists=False` as a real,
        # honestly-reported verification failure (Stage 6/8), never a
        # silent pass.
        remote_head = {"exists": False, "reason": exc.__class__.__name__}
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
        "bucket": config.s3_bucket,
        "object_key": key,
        "remote_head": remote_head,
    }
