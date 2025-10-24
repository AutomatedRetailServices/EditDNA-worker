# tasks.py — adapter: web payload -> local file -> jobs.job_render(local_path)
import os
import tempfile
import requests
from typing import Any, Dict, Optional

# Import your pipeline entry (rename to avoid name clash)
from jobs import job_render as run_pipeline


def _download_to_tmp(url: str) -> str:
    """
    Downloads an http(s) video URL to a local temp file (mp4) and returns the path.
    """
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as f:
        for chunk in resp.iter_content(1024 * 512):
            if chunk:
                f.write(chunk)
    print(f"[tasks] downloaded -> {path}", flush=True)
    return path


def _resolve_input_path(payload: Dict[str, Any]) -> str:
    """
    Accepts several payload shapes and resolves to a LOCAL file path:
      - { "files": ["https://...mp4", ...] }
      - { "video_url": "https://...mp4" }
      - { "s3_url": "https://bucket.s3....mp4" }
      - { "local_path": "/abs/path/file.mp4" }  (advanced)
    """
    # Prefer `files[0]` if present
    files = payload.get("files") or []
    url: Optional[str] = None

    if isinstance(files, list) and files:
        url = files[0]
    else:
        # backward-compatible keys
        url = (
            payload.get("video_url")
            or payload.get("s3_url")
            or payload.get("s3_key")  # sometimes users pass a full https s3 key
            or payload.get("input")   # anything else the web might send
        )

    if not url:
        # allow advanced usage with a direct local path
        local_path = payload.get("local_path")
        if local_path and os.path.exists(local_path):
            print(f"[tasks] using local_path={local_path}", flush=True)
            return local_path
        raise ValueError("No input video provided. Expected 'files[0]' or 'video_url' or 's3_url'.")

    # If it already looks like a local path, use it
    if isinstance(url, str) and url.startswith("/"):
        print(f"[tasks] using local file: {url}", flush=True)
        return url

    # Else treat it as a downloadable URL (https S3 or public http)
    return _download_to_tmp(str(url))


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    RQ entrypoint: called as 'tasks.job_render' with the JSON payload from /render.
    - Resolves the first input video to a local path.
    - Applies optional env overrides (e.g., max_duration).
    - Invokes your pipeline run_pipeline(local_path).
    - Returns the pipeline's result dict.
    """
    print(f"[tasks] job_render payload keys={list(payload.keys())}", flush=True)

    # Optional cap override
    max_duration = payload.get("max_duration")
    if max_duration:
        os.environ["MAX_DURATION_SEC"] = str(max_duration)

    # Resolve input to local file
    local_path = _resolve_input_path(payload)

    # Call your existing pipeline (expects a local file path)
    result = run_pipeline(local_path)

    print(f"[tasks] done. result keys={list(result.keys())}", flush=True)
    return result
