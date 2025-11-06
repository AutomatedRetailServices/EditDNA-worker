# /workspace/EditDNA-worker/worker/s3.py
import os
import boto3

# you can override these in the pod env
S3_BUCKET = os.environ.get("S3_BUCKET", "script2clipshop-video-automatedretailservices")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def upload_file(local_path: str, key: str) -> str:
    """
    Uploads local_path to S3_BUCKET at object key.
    Returns an https URL so the API can hand it to the frontend.
    """
    session = boto3.session.Session()
    s3_client = session.client("s3", region_name=AWS_REGION)

    s3_client.upload_file(local_path, S3_BUCKET, key)

    # URL style
    if AWS_REGION == "us-east-1":
        url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
    else:
        url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"

    return url
