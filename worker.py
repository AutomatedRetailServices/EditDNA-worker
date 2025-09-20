# worker.py — thin shim so RQ can import functions reliably
from __future__ import annotations
from typing import Any, Dict

from jobs import job_render as _job_render_impl, job_render_chunked as _job_render_chunked_impl

def task_nop() -> Dict[str, Any]:
    return {"echo": {"hello": "world"}}

def job_render(*args, **kwargs) -> Dict[str, Any]:
    return _job_render_impl(*args, **kwargs)

def job_render_chunked(*args, **kwargs) -> Dict[str, Any]:
    return _job_render_chunked_impl(*args, **kwargs)
