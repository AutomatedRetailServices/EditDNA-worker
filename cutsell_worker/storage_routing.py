"""Media storage routing for the move from Amazon S3 to Cloudflare R2.

The current storage is whatever the environment configures (`S3_BUCKET`, the default boto3
credentials, and `AWS_ENDPOINT_URL_S3` + `AWS_REGION=auto` for R2). Every WRITE goes there, always.

Projects made before the move still point at the old Amazon bucket. Nothing is ever deleted from
it, and it stays readable through four OPTIONAL variables holding the old Amazon settings:

    FALLBACK_S3_BUCKET, FALLBACK_AWS_REGION, FALLBACK_AWS_ACCESS_KEY_ID, FALLBACK_AWS_SECRET_ACCESS_KEY

Reads and read links (`download`, `presigned_get`, `head`) are routed here:
  - a URI in the fallback bucket (an old cut) is read from Amazon, with a client built on the
    explicit regional Amazon endpoint so a global `AWS_ENDPOINT_URL_S3` (R2) cannot redirect it;
  - a URI in the current bucket is read from the current storage; if that key is missing there
    (404 / NoSuchKey) the same key is read from the fallback bucket;
  - `is_readable_bucket` lets URI validation accept the fallback bucket FOR READS ONLY -- tenant
    and prefix checks stay with the caller, unchanged.

With no FALLBACK_* variables set, every function here makes exactly the call the code made before
(same client, same bucket, no extra request).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from .config import load_runtime_config

FALLBACK_ENV = ("FALLBACK_S3_BUCKET", "FALLBACK_AWS_REGION", "FALLBACK_AWS_ACCESS_KEY_ID",
                "FALLBACK_AWS_SECRET_ACCESS_KEY")
_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound", "NoSuchObject"}


@dataclass(frozen=True)
class FallbackStorage:
    bucket: str
    region: str
    access_key_id: str
    secret_access_key: str

    @property
    def endpoint_url(self) -> str:
        return f"https://s3.{self.region}.amazonaws.com"


def fallback_storage() -> FallbackStorage | None:
    """The old Amazon storage, or None when it is not (fully) configured. Credentials are
    required: without them the default (R2) credentials would be sent to Amazon."""
    bucket = os.getenv("FALLBACK_S3_BUCKET", "").strip()
    key_id = os.getenv("FALLBACK_AWS_ACCESS_KEY_ID", "").strip()
    secret = os.getenv("FALLBACK_AWS_SECRET_ACCESS_KEY", "").strip()
    if not bucket or not key_id or not secret:
        return None
    region = os.getenv("FALLBACK_AWS_REGION", "").strip() or "us-east-1"
    return FallbackStorage(bucket=bucket, region=region, access_key_id=key_id, secret_access_key=secret)


def current_bucket() -> str | None:
    return load_runtime_config().s3_bucket


def is_fallback_bucket(bucket: str) -> bool:
    fallback = fallback_storage()
    return fallback is not None and bool(bucket) and bucket == fallback.bucket


def is_readable_bucket(bucket: str, current: str | None = None) -> bool:
    """For READ validation only: the current bucket (`current`, else the configured one), or the
    old Amazon bucket when configured."""
    current = current if current is not None else current_bucket()
    return (bool(current) and bucket == current) or is_fallback_bucket(bucket)


def default_client(*, region_name: str | None = None):
    """The current storage's client (boto3 honours AWS_ENDPOINT_URL_S3 for R2).

    Region from the runtime config (AWS_REGION, e.g. "auto" on R2) when not given, and always
    Signature V4: without a region boto3 falls back to us-east-1 and signs PRESIGNED links with
    the legacy V2 scheme, which Cloudflare R2 refuses (401) -- found live on the Clean up link."""
    import boto3
    from botocore.config import Config

    region = region_name or load_runtime_config().aws_region or None
    return boto3.client("s3", region_name=region, config=Config(signature_version="s3v4"))


def _make_fallback_client(fallback: FallbackStorage):
    import boto3

    return boto3.client(
        "s3", region_name=fallback.region, endpoint_url=fallback.endpoint_url,
        aws_access_key_id=fallback.access_key_id, aws_secret_access_key=fallback.secret_access_key,
    )


#: Builds the Amazon client for the old bucket (replaced in tests by a fake).
fallback_client_factory: Callable[[FallbackStorage], object] = _make_fallback_client


def is_not_found(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error") or {}
        code = str(error.get("Code") or "")
        status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        return code in _NOT_FOUND_CODES or status == 404
    return False


def _fallback_client(fallback: FallbackStorage, client):
    return client if client is not None else fallback_client_factory(fallback)


def download(bucket: str, key: str, destination: str, *, client=None, fallback_client=None) -> str:
    """Download one object to `destination` with the routing described in the module docstring."""
    fallback = fallback_storage()
    if fallback is not None and bucket == fallback.bucket:
        _fallback_client(fallback, fallback_client).download_file(bucket, key, str(destination))
        return str(destination)
    primary = client if client is not None else default_client()
    try:
        primary.download_file(bucket, key, str(destination))
    except Exception as exc:
        if fallback is None or bucket != current_bucket() or not is_not_found(exc):
            raise
        _fallback_client(fallback, fallback_client).download_file(fallback.bucket, key, str(destination))
    return str(destination)


def head(bucket: str, key: str, *, client=None, fallback_client=None) -> dict:
    """head_object with the same routing (a current-bucket miss is looked up on the fallback)."""
    fallback = fallback_storage()
    if fallback is not None and bucket == fallback.bucket:
        return _fallback_client(fallback, fallback_client).head_object(Bucket=bucket, Key=key)
    primary = client if client is not None else default_client()
    try:
        return primary.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if fallback is None or bucket != current_bucket() or not is_not_found(exc):
            raise
        return _fallback_client(fallback, fallback_client).head_object(Bucket=fallback.bucket, Key=key)


def presigned_get(bucket: str, key: str, *, expires_in: int, client=None, fallback_client=None,
                  params: dict | None = None) -> str:
    """A GET link for reading. An old-bucket object is signed by Amazon; a current-bucket key
    that the current storage does not have (checked with one HEAD, only when a fallback is
    configured) is signed on the fallback bucket. Without a fallback: exactly the old call."""
    extra = dict(params or {})
    fallback = fallback_storage()
    if fallback is not None and bucket == fallback.bucket:
        return _fallback_client(fallback, fallback_client).generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key, **extra}, ExpiresIn=int(expires_in))
    primary = client if client is not None else default_client()
    if fallback is not None and bucket == current_bucket():
        try:
            primary.head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            if is_not_found(exc):
                return _fallback_client(fallback, fallback_client).generate_presigned_url(
                    "get_object", Params={"Bucket": fallback.bucket, "Key": key, **extra},
                    ExpiresIn=int(expires_in))
            # any other answer: sign on the current storage as before
    return primary.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key, **extra}, ExpiresIn=int(expires_in))
