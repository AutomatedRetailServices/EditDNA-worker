# tasks.py — adapter between the web payload and jobs.py
import os, tempfile, requests
from typing import Dict, Any
from jobs import job_render  # your existing full pipeline

def _download_first_file(payload: Dict[str, Any]) -> str:
    files = payload.get("files") or []
    if not files:
        raise RuntimeError("payload.files is empty")
    url = files[0]
    r = requests.get(url, stream=True, timeout=90)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as f:
        for chunk in r.iter_content(1024 * 512):
            f.write(chunk)
    print(f"[tasks] downloaded to {path}", flush=True)
    return path

def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    print(f"[tasks] job_render payload keys={list(payload.keys())}", flush=True)
    if "max_duration" in payload and payload["max_duration"]:
        os.environ["MAX_DURATION_SEC"] = str(payload["max_duration"])
    local_path = _download_first_file(payload)
    result = job_render(local_path)
    print(f"[tasks] job_render done; result keys={list(result.keys())}", flush=True)
    return result
