"""
RQ entrypoint shim for EditDNA worker.

- Imports the real implementation from one of:
    1) editdna.worker.jobs   (future-proof path)
    2) editdna.jobs          (your current layout)
    3) jobs                  (bare module on PYTHONPATH)

- Exposes: job_render(...), run_pipeline(...)
"""

from __future__ import annotations
import importlib
import typing as T

def _import_jobs():
    candidates = [
        "editdna.worker.jobs",
        "editdna.jobs",
        "jobs",
    ]
    last_err = None
    for modname in candidates:
        try:
            return importlib.import_module(modname)
        except Exception as e:
            last_err = e
    raise ImportError(
        "Could not import job implementation from worker.jobs or jobs. "
        "Make sure one of these files defines job_render() or run_pipeline()."
    ) from last_err

_jobs = _import_jobs()
_run_pipeline = getattr(_jobs, "run_pipeline", None)
_job_render   = getattr(_jobs, "job_render", None)

if _run_pipeline is None and _job_render is None:
    raise ImportError("Neither run_pipeline nor job_render found in the jobs module.")

def job_render(payload: T.Union[dict, str, None] = None, **kwargs):
    """
    Accepts either:
      - dict payload with keys like {'session_id','files','mode','options',...}
      - str local_path (legacy)
      - kwargs (legacy)
    """
    if _job_render is not None:
        return _job_render(payload if payload is not None else kwargs)

    # fall back to run_pipeline
    if isinstance(payload, str):
        return _run_pipeline(payload, kwargs or None)
    if isinstance(payload, dict):
        local_path = payload.get("local_path")
        return _run_pipeline(local_path, payload)
    local_path = kwargs.get("local_path")
    return _run_pipeline(local_path, kwargs or None)

def run_pipeline(local_path: str | None = None, payload: dict | None = None):
    if _run_pipeline is not None:
        return _run_pipeline(local_path, payload)
    pl = payload or {}
    if local_path and "local_path" not in pl:
        pl = dict(pl, local_path=local_path)
    return _job_render(pl)
