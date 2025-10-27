"""
Real worker entrypoints for EditDNA Worker.

This module connects RQ jobs → shim → real logic in jobs.py (root).
"""

from __future__ import annotations
from typing import Any, Dict

try:
    # Import your real logic from root jobs.py
    from ..jobs import job_render as _job_render_impl
    from ..jobs import run_pipeline as _run_pipeline_impl
except Exception as e:
    raise ImportError(f"[worker.tasks] Failed to import editdna.jobs: {e!r}")


def run_pipeline(*args, **kwargs) -> Dict[str, Any]:
    """Forwarder for run_pipeline — used internally by job_render."""
    return _run_pipeline_impl(*args, **kwargs)


def job_render(*args, **kwargs) -> Dict[str, Any]:
    """
    The main entrypoint RQ executes.
    This wrapper ensures backward compatibility and logs structured output.
    """
    try:
        print("[worker.tasks] job_render: received args/kwargs, forwarding to editdna.jobs.job_render...", flush=True)
        result = _job_render_impl(*args, **kwargs)
        print("[worker.tasks] job_render: completed successfully.", flush=True)
        return result
    except Exception as e:
        print(f"[worker.tasks] job_render: ERROR -> {e!r}", flush=True)
        raise
