cat > ~/editdna/jobs_smoke.py <<'PY'
import os, time
import boto3

def job_smoke_render():
    # prefer AWS_DEFAULT_REGION if set, else AWS_REGION else us-east-1
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
    bucket = os.environ["S3_BUCKET"]
    ts = int(time.time())
    key = f"smoke/out-{ts}.txt"

    s3 = boto3.client("s3", region_name=region)
    body = f"EditDNA smoke ok @ {ts}\n".encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/plain")

    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    return {"ok": True, "bucket": bucket, "key": key, "url": url, "s3_url": f"s3://{bucket}/{key}"}
PY
