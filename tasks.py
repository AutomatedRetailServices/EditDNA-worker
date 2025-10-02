import os, json, time, uuid
from typing import Dict, Any

# Minimal S3 text writer
try:
    from s3_utils import s3_put_text
except Exception:
    import boto3
    def s3_put_text(bucket: str, key: str, text: str):
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType="text/plain"
        )

def job_render(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main render entrypoint.
    - On success → writes a manifest.json proving the request arrived.
    - On error → writes a short error.txt to S3 so you always see breadcrumbs.
    """
    bucket = os.environ.get("S3_BUCKET")
    prefix = params.get("output_prefix", "editdna/outputs")
    started = time.time()

    try:
        # >>> Later, replace with your real render pipeline call (ffmpeg, ASR, scoring, etc.)
        # result = render_main(params)
        # return result

        # For now: lightweight manifest for E2E testing
        key = f"{prefix}/manifest-{uuid.uuid4().hex}.json"
        payload = {
            "ok": True,
            "ts": started,
            "received": params
        }
        s3_put_text(bucket, key, json.dumps(payload))
        return {
            "ok": True,
            "bucket": bucket,
            "key": key,
            "url": f"https://{bucket}.s3.amazonaws.com/{key}"
        }

    except Exception as e:
        err_key = f"{prefix}/errors/{uuid.uuid4().hex}.txt"
        try:
            s3_put_text(bucket, err_key, f"{type(e).__name__}: {e}")
        finally:
            # Re-raise so RQ marks as failed (traceback visible in /jobs)
            raise
