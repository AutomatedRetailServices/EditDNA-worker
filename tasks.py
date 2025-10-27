# /workspace/editdna/tasks.py
from __future__ import annotations
import os, sys, traceback, importlib
from typing import Any, Dict

# --- Ensure proper sys.path for /workspace/editdna ---
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def _import_jobs_module():
    """
    Try importing the real job logic from either worker.jobs or jobs.
    This ensures compatibility on both Render and RunPod.
    """
    candidates = ["worker.jobs", "jobs"]
    for name in candidates:
        try:
            module = importlib.import_module(name)
            return module
        except Exception:
            continue
    raise ImportError(
        "Could not import job implementation from worker.jobs or jobs. "
        "Make sure one of these files defines job_render (and optionally run_pipeline)."
    )

def job_render(payload: Dict[str, Any] | str):
    """
    Main RQ entrypoint. Accepts either a dict payload or a local_path string.
    Calls run_pipeline() or job_render() from jobs.py depending on what’s available.
    """
    try:
        jobs = _import_jobs_module()
        # Normalize payload
        if isinstance(payload, str):
            local_path = payload
            payload_dict = None
        elif isinstance(payload, dict):
            local_path = payload.get("local_path") or payload.get("input_local") or payload.get("file")
            payload_dict = payload
        else:
            raise ValueError(f"Unexpected payload type: {type(payload)}")

        # Prefer run_pipeline() if defined, fallback to job_render()
        if hasattr(jobs, "run_pipeline"):
            return jobs.run_pipeline(local_path=local_path, payload=payload_dict)
        elif hasattr(jobs, "job_render"):
            return jobs.job_render(local_path)
        else:
            raise RuntimeError("No run_pipeline() or job_render() found in jobs.py")

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
