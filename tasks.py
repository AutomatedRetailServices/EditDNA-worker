import os, json, time, uuid
from typing import Dict, Any

# If you already have a helper, we'll use it; otherwise fall back to boto3 here.
try:
    from s3_utils import s3_put_text  # your existing helper (if present)
except Exception:
    import boto3
    def s3_put_text(bucket: str, key: str, text: str):
        boto3.client("s3").put_object(
            Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType="text/plain"
        )

def job_render(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main render entry for the worker.
    For now, writes a manifest to S3 so the E2E pipeline is verifiable.
    Replace the 'manifest' section with your real render pipeline call when ready.
    """
    bucket = os.environ.get("S3_BUCKET")
    prefix = params.get("output_prefix", "editdna/outputs")
    started = time.time()

    try:
        # >>> When the full pipeline is ready, call it here and return its dict:
        # return render_main(params)

        # Temporary: write a manifest proving we received the request.
        key = f"{prefix}/manifest-{uuid.uuid4().hex}.json"
        s3_put_text(bucket, key, json.dumps({"received": params, "ts": started}))
        return {
            "ok": True,
            "bucket": bucket,
            "key": key,
            "url": f"https://{bucket}.s3.amazonaws.com/{key}",
        }
    except Exception as e:
        # Breadcrumb in S3 on error, then re-raise so /jobs shows 'failed'
        err_key = f"{prefix}/errors/{uuid.uuid4().hex}.txt"
        try:
            s3_put_text(bucket, err_key, f"{type(e).__name__}: {e}")
        finally:
            raise
