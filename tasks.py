#!/usr/bin/env python3
# tasks.py — RQ task wrappers that call jobs.py safely

from typing import Any, Dict
from jobs import job_render as _job_render, job_render_chunked as _job_render_chunked

def task_nop() -> Dict[str, Any]:
    return {"ok": True, "msg": "nop"}

def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return _job_render(payload)
    except Exception as e:
        return {"ok": False, "error": str(e), "payload": payload}

def job_render_chunked(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return _job_render_chunked(payload)
    except Exception as e:
        return {"ok": False, "error": str(e), "payload": payload}
