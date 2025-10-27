# tasks.py
import os
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
from rq import get_current_job

# our pipeline
from jobs import job_render as _render_file
from jobs import ensure_local_file

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _coerce_payload(payload: Optional[dict]) -> dict:
    """
    Accepts either a dict (normal) or None.
    Web enqueuer always calls 'tasks.job_render' with kwargs.
    """
    return payload or {}

def _coerce_files(files: Optional[Union[str, List[str]]]) -> List[str]:
    if not files:
        return []
    if isinstance(files, str):
        return [files]
    return list(files)

def run_pipeline(local_path: str, payload: Optional[dict] = None) -> Dict[str, Any]:
    """
    Main file->funnel pipeline.
    """
    return _render_file(local_path, payload or {})

def job_render(payload: Optional[dict] = None,
               session_id: Optional[str] = None,
               files: Optional[Union[str, List[str]]] = None,
               mode: Optional[str] = None,
               options: Optional[dict] = None) -> Dict[str, Any]:
    """
    This exact symbol name MUST exist so RQ can import 'tasks.job_render'.
    Accepts kwargs from the web app. Supports direct 'files' OR payload['files'].
    """
    job = get_current_job()
    payload = _coerce_payload(payload)

    # Allow top-level kwargs to override payload (web sends both sometimes)
    if session_id: payload["session_id"] = session_id
    if mode:       payload["mode"] = mode
    if options:    payload.setdefault("options", {}).update(options)

    # Merge files
    merged_files = _coerce_files(files) or _coerce_files(payload.get("files"))
    if not merged_files:
        raise ValueError("No files provided. Expected 'files': [URL or s3://...]")

    # For now we process only the first file (extend later for multi)
    src = merged_files[0]
    local_path = ensure_local_file(src)

    result = run_pipeline(local_path=local_path, payload=payload)
    return {
        "ok": True,
        "job_id": job.id if job else None,
        "enqueued_at": job.enqueued_at.isoformat() + "Z" if job and job.enqueued_at else None,
        "started_at": _now_iso(),
        "ended_at": _now_iso(),
        **result
    }
