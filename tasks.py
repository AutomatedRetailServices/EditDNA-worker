# /workspace/EditDNA-worker/tasks.py
from __future__ import annotations
import os
import tempfile
import rq
from typing import List

from worker import video
import pipeline  # our pipeline.py in the same repo

# this is the queue name the worker is listening on
QUEUE_NAME = os.getenv("RQ_QUEUE", "default")


def _download_all(urls: List[str]) -> str:
    """
    We only process 1 video right now, so just download the first URL
    into /tmp and return the local path.
    """
    if not urls:
        raise ValueError("no input URLs")

    first = urls[0]
    local_path = video.download_to_local(first)
    return local_path


def job_render(payload: dict) -> dict:
    """
    RQ entry point.

    Expected payload shape (like your logs):
    {
      "session_id": "funnel-test-1",
      "files": ["https://.../IMG_03.mov"],
      "output_prefix": "editdna/outputs/"
    }
    """
    print("[worker.tasks] job_render() start")
    session_id = payload.get("session_id", f"session-{os.getpid()}")
    urls = payload.get("files") or payload.get("urls") or []
    s3_prefix = payload.get("output_prefix") or payload.get("s3_prefix") or "editdna/outputs/"

    print(f"  session_id={session_id}")
    print(f"  urls={urls}")
    print(f"  s3_prefix={s3_prefix}")

    # 1) download
    local_video_path = _download_all(urls)
    print(f"[worker.tasks] downloaded to {local_video_path}")

    try:
        # 2) run pipeline
        out = pipeline.run_pipeline(
            local_video_path=local_video_path,
            session_id=session_id,
            s3_prefix=s3_prefix,
        )
        print("[worker.tasks] job_render() OK")
        return out
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("[worker.tasks] ERROR in job_render()")
        print(tb)
        return {
            "ok": False,
            "error": str(e),
            "traceback": tb,
        }
