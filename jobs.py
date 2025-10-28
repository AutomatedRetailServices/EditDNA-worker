# /workspace/editdna/tasks.py
# Minimal, robust shim that exposes tasks.job_render to RQ
# and imports your real implementation from worker.jobs or jobs.

import importlib
import os
import sys
from typing import Any

# Ensure /workspace/editdna is at the front of sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

def _import_jobs():
    """
    Try the two canonical locations:
      1) worker.jobs  (if you later move code under a worker/ folder)
      2) jobs         (current layout — your jobs.py in repo root)
    """
    errors = []
    for mod_name in ("worker.jobs", "jobs"):
        try:
            return importlib.import_module(mod_name)
        except Exception as e:
            errors.append(f"{mod_name}: {repr(e)}")
    raise ImportError(
        "Could not import job implementation from worker.jobs or jobs. "
        "Make sure one of these files defines job_render() or run_pipeline().\n"
        + "\n".join(errors)
    )

# Import once at import-time so RQ can resolve functions immediately.
_jobs = _import_jobs()

def job_render(payload: Any):
    """
    RQ entrypoint. Delegates to your real implementation.
    Prefer jobs.job_render(payload), else fall back to jobs.run_pipeline.
    """
    if hasattr(_jobs, "job_render"):
        return _jobs.job_render(payload)

    if hasattr(_jobs, "run_pipeline"):
        # Best-effort fallback: pass through payload; your run_pipeline
        # can decide how to interpret it (string local_path or dict).
        return _jobs.run_pipeline(payload)

    raise RuntimeError(
        "Neither job_render nor run_pipeline found in jobs implementation."
    )

# Optional convenience export if your web app imports it
def run_pipeline(*args, **kwargs):
    if hasattr(_jobs, "run_pipeline"):
        return _jobs.run_pipeline(*args, **kwargs)
    raise RuntimeError("run_pipeline not available in jobs implementation.")
