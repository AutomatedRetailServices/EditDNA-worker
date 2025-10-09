# s3_utils.py
import os
import boto3
from botocore.config import Config
import mimetypes
import urllib.parse
import pathlib
import requests

S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", os.environ.get("AWS_REGION", "us-east-1"))

# lazy client so env can be overridden before first use
_s3_client = None
def _s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            region_name=AWS_REGION,
            config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"})
        )
    return _s3_client

def _clean_prefix(p: str) -> str:
    p = (p or "").strip().strip("/")
    return p

def upload_file(local_path: str, key_prefix: str, content_type: str | None = None, public: bool = False) -> str:
    """
    Upload local_path to s3://{S3_BUCKET}/{key_prefix}/{basename}.
    Returns the s3:// URI.
    """
    assert S3_BUCKET, "S3_BUCKET not configured"
    base = pathlib.Path(local_path).name
    key = f"{_clean_prefix(key_prefix)}/{base}" if key_prefix else base

    if content_type is None:
        guessed, _ = mimetypes.guess_type(base)
        content_type = guessed or "application/octet-stream"

    extra = {"ContentType": content_type}
    if public:
        extra["ACL"] = "public-read"

    _s3().upload_file(local_path, S3_BUCKET, key, ExtraArgs=extra)
    return f"s3://{S3_BUCKET}/{key}"

def presigned_url(bucket: str | None = None, key: str | None = None, *, expires: int = 3600) -> str:
    """
    If bucket/key omitted, assumes default bucket and literal key.
    """
    b = bucket or S3_BUCKET
    assert b and key, "bucket and key are required (or set S3_BUCKET)"
    return _s3().generate_presigned_url(
        "get_object",
        Params={"Bucket": b, "Key": key},
        ExpiresIn=expires,
    )

def download_to_tmp(url: str, tmpdir: str) -> str:
    """
    Accepts s3://bucket/key or https://… (streamed).
    Saves into tmpdir and returns local path.
    """
    os.makedirs(tmpdir, exist_ok=True)

    if url.startswith("s3://"):
        # s3://bucket/key
        _, rest = url.split("s3://", 1)
        bucket, key = rest.split("/", 1)
        local = os.path.join(tmpdir, pathlib.Path(key).name)
        _s3().download_file(bucket, key, local)
        return local

    # Treat everything else as HTTP(S)
    pu = urllib.parse.urlparse(url)
    # try filename from path; fall back to last segment
    filename = pathlib.Path(pu.path or "").name or "download"
    # remove dangerous pieces
    filename = filename.replace("/", "_")
    local = os.path.join(tmpdir, filename)

    with requests.get(url, stream=True, timeout=(10, 60)) as r:
        r.raise_for_status()
        with open(local, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return local
