"""
Shim module so RQ can import `tasks.job_render`.

RQ loads by import string (e.g., "tasks.job_render"). This file lives at
PYTHONPATH root (/workspace/editdna) and simply forwards to your real
implementation in worker/tasks.py.
"""

from __future__ import annotations

# forwarders from your real worker implementation
from worker.tasks import job_render, run_pipeline  # noqa: F401

__all__ = ["job_render", "run_pipeline"]
