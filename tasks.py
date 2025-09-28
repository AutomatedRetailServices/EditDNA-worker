# tasks.py — RQ task wrappers

from typing import Any, Dict
from jobs import job_render, job_render_chunked

def task_nop() -> Dict[str, Any]:
    return {"echo": {"hello": "world"}}

def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    return job_render(payload)

def job_render_chunked(payload: Dict[str, Any]) -> Dict[str, Any]:
    return job_render_chunked(payload)
