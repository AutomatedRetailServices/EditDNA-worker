# /workspace/EditDNA-worker/tasks.py
"""
tasks.py
RQ worker entrypoints for EditDNA Worker.

This version matches your current pipeline signature:
    run_pipeline(local_video_path, session_id, s3_prefix)
and downloads the remote video first.
"""

from __future__ import annotations

from typing import Any, Dict
import os
import json
import time
import uuid
import traceback
import urllib.request

import pipeline  # /workspace/EditDNA-worker/pipeline.py


def _download_to_tmp(url: str) -> str:
    """
    Download remote video to /tmp and return its local path.
    """
    # try to keep the original name if present
    base = os.path.basename(url.split("?")[0]) or "input.mp4"
    local_path = f"/tmp/{base}"
    urllib.request.urlretrieve(url, local_path)
    return local_path


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    session_id = payload.get("session_id", f"session-{uuid.uuid4().hex[:8]}")
    urls = payload.get("files", [])  # web is sending this
    s3_prefix = payload.get("output_prefix", "editdna/outputs/")

    print("[worker.tasks] job_render() start", flush=True)
    print(f"  session_id={session_id}", flush=True)
    print(f"  urls={urls}", flush=True)
    print(f"  s3_prefix={s3_prefix}", flush=True)

    if not urls:
        return {
            "ok": False,
            "error": "No input files provided in payload['files']"
        }

    first_url = urls[0]
    try:
        local_path = _download_to_tmp(first_url)
        print(f"[worker.tasks] downloaded to {local_path}", flush=True)
    except Exception as e:
        traceback.print_exc()
        return {
            "ok": False,
            "error": f"Failed to download input video: {e}"
        }

    try:
        # IMPORTANT: your pipeline only wants these 3
        out = pipeline.run_pipeline(
            local_video_path=local_path,
            session_id=session_id,
            s3_prefix=s3_prefix,
        )
        dt = time.time() - t0
        print(f"[worker.tasks] job_render() OK in {dt:.2f}s", flush=True)
        return out
    except Exception as e:
        print("[worker.tasks] ERROR in job_render()", flush=True)
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
