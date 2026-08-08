"""S3 source adapter for CutSell. Secret values are read by boto3 from env/runtime."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError("source uri must be a valid s3://bucket/key URI")
    return parsed.netloc, parsed.path.lstrip("/")


def download_source(uri: str, destination: str, *, client=None) -> str:
    bucket, key = parse_s3_uri(uri)
    if client is None:
        import boto3
        client = boto3.client("s3")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(path))
    return str(path)
