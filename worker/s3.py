# /workspace/EditDNA-worker/worker/s3.py
import os
import boto3

# env-configurable
S3_BUCKET = os.environ.get("S3_BUCKET", "script2clipshop-video-automatedretailservices")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def upload_file(local_path: str, key: str):
    """
    Upload local_path to S3 at object 'key'.

    Returns a tuple:
        (s3_url, https_url)

    so the pipeline can fill both fields.
    """
    session = boto3.session.Session()
    s3_client = session.client("s3", region_name=AWS_REGION)

    # do the upload
    s3_client.upload_file(local_path, S3_BUCKET, key)

    # s3-style
    s3_url = f"s3://{S3_BUCKET}/{key}"

    # https-style (your bucket is in us-east-1)
    if AWS_REGION == "us-east-1":
        https_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
    else:
        https_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"

    return s3_url, https_url
