# tasks.py — bridge: web payload -> local jobs.job_render
import os, tempfile, requests
from typing import Dict, Any
from jobs import job_render as _job_render  # your full pipeline (expects local path)

def _download_first_file(payload: Dict[str, Any]) -> str:
    files = payload.get("files") or []
    if not files:
        raise RuntimeError("payload.files is empty")
    url = files[0]
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as f:
        for chunk in r.iter_content(1024 * 512):
            f.write(chunk)
    print(f"[tasks] downloaded -> {path}", flush=True)
    return path

# MUST match the name your web enqueues: "tasks.job_render"
def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    print(f"[tasks] payload keys={list(payload.keys())}", flush=True)

    # pass options via env knobs your jobs.py already reads
    if payload.get("max_duration"):
        os.environ["MAX_DURATION_SEC"] = str(payload["max_duration"])
    if "portrait" in payload:
        os.environ["PORTRAIT"] = "1" if payload["portrait"] else "0"
    if payload.get("audio"):
        os.environ["AUDIO_MODE"] = str(payload["audio"])

    local_path = _download_first_file(payload)
    res = _job_render(local_path)  # calls your existing pipeline
    print(f"[tasks] done; result keys={list(res.keys())}", flush=True)
    return res
