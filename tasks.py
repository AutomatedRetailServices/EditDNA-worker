# tasks.py — RQ entrypoints for EditDNA worker (download → pipeline → return)
import os, tempfile, requests
from typing import Dict, Any, List
from rq import get_current_job

from jobs import job_render  # our pipeline (defined in jobs.py)

def _download_first(files: List[str]) -> str:
    """Download the first URL to a tmp .mp4 and return local path."""
    if not files:
        raise ValueError("files[] is empty")
    url = files[0]
    # honor short timeouts so bad URLs don't hang the worker
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        fd, path = tempfile.mkstemp(suffix=".mp4")
        with os.fdopen(fd, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return path

def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    RQ job entry: expects payload like:
    { "session_id": "...", "files": ["https://.../raw.mp4"], "max_duration": 120, ... }
    """
    j = get_current_job()
    session_id = payload.get("session_id", "session")
    files = payload.get("files") or []
    local_path = _download_first(files)
    try:
        result = run_pipeline(local_path=local_path, payload=payload)
        return result
    finally:
        try:
            os.remove(local_path)
        except Exception:
            pass

# --- thin wrapper to your pipeline ---
def run_pipeline(local_path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Calls the main pipeline in jobs.py with a local file path."""
    # pass through any options you may want later
    return job_render(local_path)
