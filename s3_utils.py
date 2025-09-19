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


def parse_s3_url(s3_url: str) -> Tuple[str | None, str]:
    """
    Returns (bucket, key). If bucket can't be derived from the URL, returns (None, key).
    Supports:
      - s3://bucket/key
      - https://bucket.s3.amazonaws.com/key
      - https://bucket.s3.<region>.amazonaws.com/key
      - https://s3.<region>.amazonaws.com/bucket/key
    """
    if s3_url.startswith("s3://"):
        p = urlparse(s3_url)
        return p.netloc, p.path.lstrip("/")

    if s3_url.startswith(("http://", "https://")):
        p = urlparse(s3_url)
        host = p.netloc
        path = p.path.lstrip("/")

        if ".s3." in host and ".amazonaws.com" in host:
            bucket = host.split(".s3.", 1)[0]
            return bucket, path

        if host.startswith("s3.") and ".amazonaws.com" in host:
            parts = path.split("/", 1)
            if len(parts) == 2:
                return parts[0], parts[1]

        return None, path

    return None, s3_url.lstrip("/")


def head_exists(s3_url: str) -> bool:
    bucket, key = parse_s3_url(s3_url)
    b = bucket or S3_BUCKET
    try:
        s3.head_object(Bucket=b, Key=key)
        return True
    except ClientError:
        return False


def download_to_tmp(s3_url: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    bucket, key = parse_s3_url(s3_url)
    b = bucket or S3_BUCKET
    filename = os.path.basename(key) or f"file-{uuid.uuid4().hex}"
    local_path = os.path.join(dest_dir, filename)
    s3.download_file(b, key, local_path)
    return local_path


def upload_file(local_path: str, key_prefix: str, content_type: str | None = None) -> str:
    basename = os.path.basename(local_path)
    key_prefix = key_prefix.strip("/")

    ctype = content_type or mimetypes.guess_type(local_path)[0] or "application/octet-stream"
    key = f"{key_prefix}/{basename}"

    s3.upload_file(
        local_path,
        S3_BUCKET,
        key,
        ExtraArgs={"ACL": "private", "ContentType": ctype},
    )
    return f"s3://{S3_BUCKET}/{key}"


def presigned_url(bucket: str, key: str, expires: int = 3600) -> str:
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


def new_session_id() -> str:
    return f"sess-{uuid.uuid4()}"
