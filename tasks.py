# /workspace/EditDNA-worker/tasks.py
from __future__ import annotations
import os
from typing import List, Dict, Any

import pipeline  # your pipeline.py

# this is the queue name the worker is listening on
QUEUE_NAME = os.getenv("RQ_QUEUE", "default")


def job_render(payload: dict) -> dict:
    """
    RQ entry point.

    Expected payload shape (like your logs):
    {
      "session_id": "funnel-test-1",
      "files": ["https://.../IMG_02.mov"],
      "output_prefix": "editdna/outputs/"
    }
    """
    print("[worker.tasks] job_render() start")

    session_id = payload.get("session_id", "session-unknown")
    # in your logs it's sometimes "files", sometimes "urls"
    file_urls: List[str] = payload.get("files") or payload.get("urls") or []
    s3_prefix = payload.get("output_prefix") or payload.get("s3_prefix") or "editdna/outputs/"

    print(f"  session_id={session_id}")
    print(f"  urls={file_urls}")
    print(f"  s3_prefix={s3_prefix}")

    if not file_urls:
        err = "no input URLs in payload"
        print("[worker.tasks] ERROR:", err)
        return {"ok": False, "error": err}

    try:
        # NEW: call pipeline with file_urls (pipeline will download)
        out: Dict[str, Any] = pipeline.run_pipeline(
            session_id=session_id,
            file_urls=file_urls,
            portrait=False,          # or True if you want
            funnel_counts=None,
            max_duration=60.0,
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
