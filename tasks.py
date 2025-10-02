# tasks.py
import os, json, time, uuid
from typing import Dict, Any

# minimal S3 text helper
try:
    from s3_utils import s3_put_text
except Exception:
    import boto3
    def s3_put_text(bucket: str, key: str, text: str):
        boto3.client("s3").put_object(
            Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType="text/plain"
        )

# 👇 real render (defined in jobs.py below)
from jobs import render_main

def job_render(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main render entry. On error, write a short breadcrumb to S3 and re-raise so /jobs shows failed.
    """
    bucket = os.environ.get("S3_BUCKET")
    prefix = params.get("output_prefix", "editdna/outputs")
    started = time.time()

    try:
        # 🔥 call the real pipeline
        return render_main(params, bucket=bucket, prefix=prefix)

    except Exception as e:
        err_key = f"{prefix}/errors/{uuid.uuid4().hex}.txt"
        try:
            s3_put_text(bucket, err_key, f"{type(e).__name__}: {e}")
        finally:
            raise
