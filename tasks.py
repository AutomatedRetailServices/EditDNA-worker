"""
tasks.py
This is the ONLY thing RQ calls (tasks.job_render).
It imports jobs.py safely and forwards payload.
It never crashes the worker on import.
"""

from __future__ import annotations
import traceback
import time
from typing import Any, Dict, Optional

def _import_jobs_module():
    """
    Try to import the real pipeline logic from jobs.py.
    We keep this tiny on purpose so that even if jobs.py explodes,
    the worker process itself (rq worker) stays alive.
    """
    try:
        import jobs  # editdna/jobs.py because PYTHONPATH=/workspace/editdna
        return jobs, None
    except Exception as e:
        return None, e

def _call_pipeline(jobs_mod, local_path: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    We support either jobs.run_pipeline(...) OR jobs.job_render(...).
    We standardize the return dict.
    """
    # Prefer run_pipeline if it exists (more flexible)
    if hasattr(jobs_mod, "run_pipeline"):
        return jobs_mod.run_pipeline(local_path=local_path, payload=payload)

    # Fallback: call job_render inside jobs.py
    if hasattr(jobs_mod, "job_render"):
        return jobs_mod.job_render(payload)

    return {
        "ok": False,
        "error": "jobs.py does not define run_pipeline() or job_render()"
    }

def job_render(payload: Any = None) -> Dict[str, Any]:
    """
    This is what Redis queue calls: tasks.job_render(payload)

    payload is what /render enqueues. Example:
    {
        "session_id": "...",
        "files": ["https://....mov"],
        "mode": "funnel",
        "options": { ... overrides ... }
    }
    """
    started = time.time()

    # normalize payload
    if not isinstance(payload, dict):
        payload = {}
    files = payload.get("files") or []
    local_path = payload.get("local_path")

    # basic debug info (this print shows in pod logs so you can see it's listening)
    print("[tasks.job_render] INCOMING PAYLOAD KEYS:", list(payload.keys()), flush=True)

    # import jobs
    jobs_mod, err = _import_jobs_module()
    if err is not None or jobs_mod is None:
        trace = traceback.format_exc()
        print("[tasks.job_render] IMPORT ERROR\n", trace, flush=True)
        return {
            "ok": False,
            "error": (
                "Could not import job implementation from jobs.py. "
                "Make sure editdna/jobs.py exists and has run_pipeline() or job_render()."
            ),
            "trace": trace,
        }

    # if no explicit local_path, maybe we will download it in jobs.run_pipeline anyway
    if not local_path and files:
        # we pass files through untouched. jobs.py will handle downloading.
        pass

    try:
        result = _call_pipeline(jobs_mod, local_path, payload)

        took = time.time() - started
        print(f"[tasks.job_render] DONE in {took:.2f}s ok={result.get('ok')}", flush=True)

        return result
    except Exception:
        trace = traceback.format_exc()
        print("[tasks.job_render] RUNTIME ERROR\n", trace, flush=True)
        return {
            "ok": False,
            "error": "Exception while running pipeline.",
            "trace": trace,
        }
