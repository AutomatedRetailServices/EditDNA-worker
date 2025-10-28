# /workspace/editdna/tasks.py
from __future__ import annotations
import os, sys, importlib, traceback
from typing import Any, Dict

# Ensure /workspace/editdna is on sys.path so "jobs" is importable
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def _import_jobs():
    # Try both locations so it works on Render & RunPod
    for name in ("worker.jobs", "jobs"):
        try:
            return importlib.import_module(name)
        except Exception:
            pass
    raise ImportError(
        "Could not import job implementation from worker.jobs or jobs. "
        "Make sure one of these files defines job_render() or run_pipeline()."
    )

def job_render(payload: Dict[str, Any] | str):
    """
    RQ calls this name: tasks.job_render
    Accepts dict payload (preferred) or a string local_path (legacy).
    Delegates to run_pipeline() if present, else job_render() in jobs.py.
    """
    try:
        jobs = _import_jobs()
        # Normalize payload
        if isinstance(payload, str):
            local_path = payload
            payload_dict = None
        elif isinstance(payload, dict):
            local_path = payload.get("local_path") or payload.get("input_local") or payload.get("file")
            payload_dict = payload
        else:
            raise ValueError(f"Unexpected payload type: {type(payload)}")

        if hasattr(jobs, "run_pipeline"):
            return jobs.run_pipeline(local_path=local_path, payload=payload_dict)
        if hasattr(jobs, "job_render"):
            return jobs.job_render(local_path)
        raise RuntimeError("No run_pipeline() or job_render() found in jobs.py")

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
