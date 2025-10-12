import os, boto3

def _resolve_region(region: str|None):
    return region or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"

def _resolve_bucket(bucket: str|None, bucket_name: str|None):
    b = bucket or bucket_name or os.environ.get("S3_BUCKET")
    if not b:
        raise RuntimeError("S3 bucket not provided. Set S3_BUCKET env or pass bucket/bucket_name.")
    return b

def upload_file(local_path: str,
                key: str,
                bucket: str|None=None,
                bucket_name: str|None=None,
                region: str|None=None,
                content_type: str|None=None,
                acl: str="private"):
    """
    Flexible uploader. Accepts either bucket= or bucket_name=.
    Uses AWS_DEFAULT_REGION/AWS_REGION if region not provided.
    """
    b = _resolve_bucket(bucket, bucket_name)
    r = _resolve_region(region)
    s3 = boto3.client("s3", region_name=r)
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    if acl:
        extra["ACL"] = acl
    with open(local_path, "rb") as f:
        s3.upload_fileobj(f, b, key, ExtraArgs=extra if extra else None)
    return {
        "bucket": b,
        "key": key,
        "s3_url": f"s3://{b}/{key}",
        "https_url": f"https://{b}.s3.amazonaws.com/{key}",
        "region": r,
    }

def presigned_url(key: str,
                  bucket: str|None=None,
                  bucket_name: str|None=None,
                  region: str|None=None,
                  expires: int=3600):
    """
    Create a time-limited HTTPS URL for a given S3 key.
    """
    b = _resolve_bucket(bucket, bucket_name)
    r = _resolve_region(region)
    s3 = boto3.client("s3", region_name=r)
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": b, "Key": key},
        ExpiresIn=expires
    )
    return url
