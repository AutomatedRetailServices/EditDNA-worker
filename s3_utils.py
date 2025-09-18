# s3_utils.py — S3 helpers for EditDNA.ai
import os
import uuid
import mimetypes
from typing import Tuple
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")

if not S3_BUCKET:
    raise RuntimeError("Missing S3_BUCKET env var")

s3 = boto3.client("s3", region_name=AWS_REGION)


def parse_s3_url(s3_url: str) -> Tuple[str, str]:
    """
    Supports:
      - s3://bucket/key
      - https://bucket.s3.amazonaws.com/key
      - https://bucket.s3.<region>.amazonaws.com/key
      - https://s3.<region>.amazonaws.com/bucket/key  (path-style)
    If a plain key is passed, defaults to env S3_BUCKET.
    """
    if s3_url.startswith("s3://"):
        # s3://bucket/key
        p = urlparse(s3_url)
        bucket = p.netloc
        key = p.path.lstrip("/")
        return bucket, key

    if s3_url.startswith("http://") or s3_url.startswith("https://"):
        p = urlparse(s3_url)
        host = p.netloc
        path = p.path.lstrip("/")

        # virtual-hosted-style: bucket.s3.<region>.amazonaws.com/key
        if ".s3." in host and ".amazonaws.com" in host:
            bucket = host.split(".s3.", 1)[0]
            key = path
            return bucket, key

        # path-style: s3.<region>.amazonaws.com/bucket/key
        if host.startswith("s3.") and ".amazonaws.com" in host:
            # first segment of path is bucket
            parts = path.split("/", 1)
            if len(parts) == 2:
                bucket, key = parts[0], parts[1]
                return bucket, key

        # fallback: treat as key under default bucket
        return S3_BUCKET, path

    # fallback: treat as a bare key under default bucket
    return S3_BUCKET, s3_url.lstrip("/")


def head_exists(s3_url: str) -> bool:
    """True if the object exists in S3."""
    bucket, key = parse_s3_url(s3_url)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def download_to_tmp(s3_url: str, dest_dir: str) -> str:
    """Download an S3 object to a local path inside dest_dir; returns the local path."""
    import os
    os.makedirs(dest_dir, exist_ok=True)
    bucket, key = parse_s3_url(s3_url)
    filename = os.path.basename(key) or f"file-{uuid.uuid4().hex}"
    local_path = os.path.join(dest_dir, filename)
    s3.download_file(bucket, key, local_path)
    return local_path


def upload_file(local_path: str, key_prefix: str, content_type: str | None = None) -> str:
    """Upload local_path to s3://S3_BUCKET/<key_prefix>/<basename> and return the s3 URI."""
    import os
    basename = os.path.basename(local_path)
    key_prefix = key_prefix.strip("/")

    # Guess content-type if not provided
    ctype = content_type
    if not ctype:
        ctype, _ = mimetypes.guess_type(local_path)
    if not ctype:
        ctype = "application/octet-stream"

    key = f"{key_prefix}/{basename}"
    s3.upload_file(
        local_path,
        S3_BUCKET,
        key,
        ExtraArgs={"ACL": "private", "ContentType": ctype},
    )
    return f"s3://{S3_BUCKET}/{key}"


def new_session_id() -> str:
    return f"sess-{uuid.uuid4()}"
