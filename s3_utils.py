import os, tempfile, boto3, urllib.parse, requests

S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

_s3 = boto3.client("s3", region_name=AWS_REGION)

def upload_file(local_path: str, key_prefix: str, content_type: str = "application/octet-stream") -> str:
    assert S3_BUCKET, "S3_BUCKET not configured"
    base = os.path.basename(local_path)
    key = f"{key_prefix.strip('/')}/{base}"
    _s3.upload_file(
        local_path, S3_BUCKET, key,
        ExtraArgs={"ContentType": content_type, "ACL": "private"}  # keep object private; we hand back a presigned URL
    )
    return f"s3://{S3_BUCKET}/{key}"

def presigned_url(bucket: str, key: str, expires: int = 3600) -> str:
    return _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires
    )

def download_to_tmp(url: str, tmpdir: str) -> str:
    """
    Accepts s3://bucket/key or https://…amazonaws.com/…
    Saves into tmpdir and returns local path.
    """
    os.makedirs(tmpdir, exist_ok=True)
    if url.startswith("s3://"):
        _, rest = url.split("s3://", 1)
        bucket, key = rest.split("/", 1)
        local = os.path.join(tmpdir, os.path.basename(key))
        _s3.download_file(bucket, key, local)
        return local
    # assume https
    local = os.path.join(tmpdir, os.path.basename(urllib.parse.urlparse(url).path))
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return local
