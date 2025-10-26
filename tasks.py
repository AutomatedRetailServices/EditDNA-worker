# tasks.py — adapter that downloads input and calls the heavy pipeline
import os, json, tempfile, requests
from typing import Any, Dict, List, Union

# Your heavy pipeline lives in jobs.py and expects a LOCAL PATH argument
from jobs import job_render as pipeline_job_render


def _to_dict(payload: Union[str, bytes, Dict[str, Any], None]) -> Dict[str, Any]:
    """Accept dict / JSON string / bytes / None and return a dict."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (bytes, bytearray)):
        try:
            return json.loads(payload.decode("utf-8"))
        except Exception:
            return {"files": [payload.decode("utf-8")]}
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return {"files": [payload]}
    # Fallback
    return {}


def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [str(x)]


def _download_to_tmp(url: str) -> str:
    """Download remote media to a local temp .mp4 and return its path."""
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
    return path


def job_render(payload: Union[str, bytes, Dict[str, Any], None]) -> Dict[str, Any]:
    """
    RQ entrypoint: normalize payload, download first file, call pipeline, return result.
    Expected payload shape (from your web layer):
      {
        "session_id": "...",
        "mode": "funnel",
        "files": ["https://.../raw.mp4"],
        "options": { ...optional env-style overrides... }
      }
    """
    data = _to_dict(payload)

    files = _as_list(data.get("files") or data.get("file"))
    if not files:
        raise ValueError("files[] required in payload")

    # optional per-job overrides (safe: strings/numbers only)
    opts = data.get("options") or {}
    for k, v in opts.items():
        if isinstance(v, (str, int, float, bool)):
            os.environ[str(k)] = str(v)

    # Download first URL and run the heavy pipeline
    src_url = files[0]
    local_path = _download_to_tmp(src_url)
    try:
        result = pipeline_job_render(local_path)  # jobs.py version that requires a LOCAL PATH
    finally:
        try:
            os.remove(local_path)
        except Exception:
            pass

    # add some provenance
    result["source_url"] = src_url
    result["session_id"] = data.get("session_id", "")
    return result
