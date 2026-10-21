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
    """Download one stored object. Routed through `storage_routing` (R2 move): an old-bucket URI
    is read from the Amazon fallback, a current-bucket miss is retried there. Without FALLBACK_*
    settings this is the same single `download_file` call as before."""
    from .storage_routing import download

    bucket, key = parse_s3_uri(uri)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    return download(bucket, key, str(path), client=client)
