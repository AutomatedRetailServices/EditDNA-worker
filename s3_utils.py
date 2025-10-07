# s3_utils.py — S3 helpers used by jobs.py
import os, tempfile, boto3, urllib.parse, requests

S3_BUCKET = os.environ.get("S3_BUCKET", "").strip()
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

_session = boto3.session.Session(region_name=AWS_REGION)
_s3 = _session.client("s3")

def upload_file(local_path: str, prefix: str, content_type: str = "application/octet-stream") -> str:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not configured")
    key = f"{prefix.strip('/')}/" + os.path.basename(local_path)
    extra = {"ContentType": content_type}
    _s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs=extra)
    return f"s3://{S3_BUCKET}/{key}"

def presigned_url(bucket: str, key: str, expires: int = 3600) -> str:
    return _s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )

def _is_s3_https(url: str) -> bool:
    # e.g. https://bucket.s3.us-east-1.amazonaws.com/path/file.mp4
    return url.startswith("http") and ".s3." in url

def download_to_tmp(s3_or_https: str, tmpdir: str) -> str:
    """
    Accepts:
      - s3://bucket/key
      - https S3 URLs (virtual host or path style)
    Downloads to tmpdir and returns local path.
    """
    if s3_or_https.startswith("s3://"):
        _, rest = s3_or_https.split("s3://", 1)
        bucket, key = rest.split("/", 1)
        fd, path = tempfile.mkstemp(dir=tmpdir, suffix=os.path.splitext(key)[1] or ".bin")
        os.close(fd)
        _s3.download_file(bucket, key, path)
        return path

    if _is_s3_https(s3_or_https):
        # direct HTTP download
        r = requests.get(s3_or_https, stream=True, timeout=60)
        r.raise_for_status()
        ext = os.path.splitext(urllib.parse.urlparse(s3_or_https).path)[1] or ".bin"
        fd, path = tempfile.mkstemp(dir=tmpdir, suffix=ext)
        os.close(fd)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return path


    
    # Otherwise, assume it's already local path
    return s3_or_https
