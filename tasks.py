# tasks.py — queue adapter: downloads first URL, calls jobs.job_render(local_path)
import os, json, tempfile, uuid, logging, traceback
from typing import Dict, Any, Optional, List
import requests

from jobs import job_render as _run_pipeline  # your heavy pipeline

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("editdna.tasks")

def _download_to_tmp(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    log.info(f"[download] {url} -> {path}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return path

def run_pipeline(local_path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Delegates to jobs.job_render(local_path). Keeps wiggle room to pass future options via payload.
    """
    return _run_pipeline(local_path)

def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected payload (from web or Postman):
    {
      "session_id": "string",
      "files": ["https://...mp4", ...],
      ... (ignored or future options)
    }
    """
    try:
        # Backward-compat if someone passes a string:
        if isinstance(payload, str):
            payload = {"files": [payload]}

        files = payload.get("files") or []
        assert files and isinstance(files, list), "payload.files[] URL(s) required"
        src_url = files[0]

        local_path = _download_to_tmp(src_url)
        result = run_pipeline(local_path=local_path, payload=payload)
        log.info(f"[tasks] done. result keys={list(result.keys())}")
        return result
    except Exception as e:
        tb = traceback.format_exc(limit=8)
        log.error(f"[tasks] ERROR: {e}\n{tb}")
        raise
