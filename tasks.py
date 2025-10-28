"""
tasks.py — Safe RQ task shim for EditDNA worker.
Ensures the worker can find jobs.py even without full package context.
"""

import importlib
import traceback

def _import_jobs():
    """Try to import jobs implementation safely."""
    try:
        return importlib.import_module("jobs")
    except Exception as e1:
        try:
            return importlib.import_module("worker.jobs")
        except Exception as e2:
            raise ImportError(
                "Could not import job implementation from worker.jobs or jobs. "
                "Make sure one of these files defines job_render() or run_pipeline()."
            ) from e2

def job_render(payload=None, **kwargs):
    """RQ entrypoint for rendering jobs."""
    try:
        jobs = _import_jobs()
        if hasattr(jobs, "job_render"):
            return jobs.job_render(payload=payload, **kwargs)
        elif hasattr(jobs, "run_pipeline"):
            return jobs.run_pipeline(payload=payload, **kwargs)
        else:
            raise RuntimeError("jobs.py found but missing job_render() or run_pipeline().")
    except Exception as e:
        trace = traceback.format_exc()
        print("💥 job_render() failed:", trace)
        return {"ok": False, "error": str(e), "trace": trace}
