# /workspace/EditDNA-worker/tasks.py
from __future__ import annotations
import os
from typing import List, Dict, Any

from worker import video
import pipeline  # the file above

QUEUE_NAME = os.getenv("RQ_QUEUE", "default")


def _download_all(urls: List[str]) -> str:
    if not urls:
        raise ValueError("no input URLs")
    first = urls[0]
    local_path = video.download_to_local(first)
    return local_path


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    print("[worker.tasks] job_render() start")
    session_id = payload.get("session_id", f"session-{os.getpid()}")
    urls = payload.get("files") or payload.get("urls") or []
    s3_prefix = payload.get("output_prefix") or payload.get("s3_prefix") or "editdna/outputs/"

    print(f"  session_id={session_id}")
    print(f"  urls={urls}")
    print(f"  s3_prefix={s3_prefix}")

    local_video_path = _download_all(urls)
    print(f"[worker.tasks] downloaded to {local_video_path}")

    try:
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
