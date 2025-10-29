# /workspace/editdna/tasks.py
# Safe RQ entrypoint shim that imports your job implementation
# from either `jobs` (same folder) or `worker.jobs` (package-style),
# and adapts legacy payloads. No relative imports, no circular refs.

from __future__ import annotations
import os, json, tempfile, urllib.request, traceback
from typing import Any, Dict, Optional

def _import_jobs():
    import importlib
    errors = []
    for modname in ("jobs", "worker.jobs"):
        try:
            m = importlib.import_module(modname)
            if hasattr(m, "job_render") or hasattr(m, "run_pipeline"):
                return m
        except Exception as e:
            errors.append(f"{modname}: {repr(e)}")
    raise ImportError(
        "Could not import job implementation from jobs or worker.jobs. "
        "Ensure one of them defines job_render() or run_pipeline().\n" + "\n".join(errors)
    )

def _download_to_tmp(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix=os.path.splitext(url.split("?")[0])[-1] or ".bin")
    os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path

def _norm_payload(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        # legacy: payload was a local path
        return {"local_path": payload}
    if isinstance(payload, (bytes, bytearray)):
        try:
            return json.loads(payload.decode("utf-8"))
        except Exception:
            return {"raw": payload}
    return {"raw": payload}

def job_render(payload: Any = None, **kwargs) -> Dict[str, Any]:
    try:
        jobs = _import_jobs()
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

    data = _norm_payload(payload)
    data.update(kwargs or {})

    # Ensure local_path if user only provided URLs
    local_path: Optional[str] = data.get("local_path")
    files = data.get("files") or data.get("file") or []
    if not local_path:
        if isinstance(files, list) and files and isinstance(files[0], str) and files[0].startswith(("http://","https://","s3://")):
            local_path = _download_to_tmp(files[0])
            data["local_path"] = local_path
        elif isinstance(files, str) and files.startswith(("http://","https://","s3://")):
            local_path = _download_to_tmp(files)
            data["local_path"] = local_path

    # Prefer user's job_render; fallback to run_pipeline
    if hasattr(jobs, "job_render"):
        return jobs.job_render(data)
    return jobs.run_pipeline(local_path=local_path, payload=data)
