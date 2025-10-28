"""
RQ entrypoint shim for EditDNA worker.

- Always import the real implementation from either:
    1) editdna.worker.jobs   (preferred if you later reorganize)
    2) editdna.jobs          (your current layout)
    3) jobs                  (absolute on PYTHONPATH)

- Exposes: job_render(payload|local_path=...), run_pipeline(local_path, payload?)
"""

from __future__ import annotations
import importlib
import os
import typing as T

# --- import the real jobs module (first hit wins) ----------------------------
def _import_jobs():
    candidates = [
        "editdna.worker.jobs",   # if you later add worker/ subpackage
        "editdna.jobs",          # your current repo layout under package
        "jobs",                  # bare module on PYTHONPATH
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

# --- normalize callable handles ---------------------------------------------
_run_pipeline = getattr(_jobs, "run_pipeline", None)
_job_render   = getattr(_jobs, "job_render", None)

if _run_pipeline is None and _job_render is None:
    raise ImportError(
        "Neither run_pipeline nor job_render found in the jobs module."
    )

# --- public API used by RQ ---------------------------------------------------
def job_render(payload: T.Union[dict, str, None] = None, **kwargs):
    """
    Accepts either:
      - dict payload with keys like { 'session_id', 'files', 'mode', 'options', ... }, OR
      - str local_path to a media file (legacy), OR
      - None + kwargs (legacy).
    """
    # prefer real job_render if available
    if _job_render is not None:
        return _job_render(payload if payload is not None else kwargs)

    # else, adapt to run_pipeline signature
    if isinstance(payload, str):
        local_path = payload
        return _run_pipeline(local_path, kwargs or None)

    if isinstance(payload, dict):
        files = payload.get("files") or []
        local_path = payload.get("local_path")
        if not local_path:
            # in your real code you probably download first file to tmp; keep behavior consistent
            # we just forward to run_pipeline and let jobs.py handle download if it already does
            pass
        return _run_pipeline(local_path, payload)

    # legacy path: allow local_path via kwargs
    local_path = kwargs.get("local_path")
    return _run_pipeline(local_path, kwargs or None)


# Convenient alias so you can enqueue either name from the web app
def run_pipeline(local_path: str | None = None, payload: dict | None = None):
    if _run_pipeline is not None:
        return _run_pipeline(local_path, payload)
    # fall back through job_render if only that exists
    pl = payload or {}
    if local_path and "local_path" not in pl:
        pl = dict(pl, local_path=local_path)
    return _job_render(pl)
