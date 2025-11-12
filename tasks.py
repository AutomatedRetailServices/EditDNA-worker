import os
import json
import tempfile
import requests
from worker import pipeline

S3_PREFIX = os.getenv("S3_PREFIX", "editdna/outputs/").strip()

def _download(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return path

def job_render(payload: dict) -> dict:
    """
    RQ entrypoint.
    Expected payload:
      {
        "session_id": "funnel-test-1",
        "files": ["https://.../IMG_02.mov"],
        "portrait": true,
        "funnel_counts": {"HOOK":1, "FEATURE":5, "CTA":1},   # optional
        "max_duration_sec": null,                             # optional
        "s3_prefix": "editdna/outputs/"                       # optional
      }
    """
    session_id = payload.get("session_id") or "session"
    file_urls = payload.get("files") or []
    portrait = bool(payload.get("portrait", True))
    funnel_counts = payload.get("funnel_counts") or {}
    max_duration_sec = payload.get("max_duration_sec")
    s3_prefix = payload.get("s3_prefix") or S3_PREFIX

    if not file_urls:
        return {"ok": False, "error": "no files provided"}

    # We pass the URLs directly; pipeline will download/ASR internally.
    out = pipeline.run_pipeline(
        session_id=session_id,
        file_urls=file_urls,
        portrait=portrait,
        funnel_counts=funnel_counts,
        max_duration=max_duration_sec,
        s3_prefix=s3_prefix,
    )
    return out

# Optional local test
if __name__ == "__main__":
    test = {
        "session_id": "local-test",
        "files": [os.getenv("TEST_URL", "")],
        "portrait": True,
        "funnel_counts": {"HOOK":1, "FEATURE":4, "CTA":1},
        "max_duration_sec": None
    }
    print(json.dumps(job_render(test), indent=2))
