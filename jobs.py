# jobs.py — thin wrappers to worker.* for compatibility
from __future__ import annotations

from typing import Any, Dict, List

from worker import task_nop as task_nop  # re-export


def job_render(*args, **kwargs) -> Dict[str, Any]:
    from worker import job_render as _jr
    return _jr(*args, **kwargs)


def job_render_chunked(*args, **kwargs) -> Dict[str, Any]:
    from worker import job_render_chunked as _jrc
    return _jrc(*args, **kwargs)
