"""
tasks.py
RQ worker entrypoints for EditDNA Worker.

RQ will call tasks.job_render(payload)
and we forward into our real pipeline logic below.
"""

from __future__ import annotations

from typing import Any, Dict, List
import os
import sys
import json
import time
import uuid
import traceback

# make sure Python can see the repo
# /workspace is usually there, but we add it anyway
sys.path.insert(0, "/workspace")
# your repo lives here in the container
sys.path.insert(0, "/workspace/EditDNA-worker")

try:
    import pipeline  # /workspace/EditDNA-worker/pipeline.py
except Exception as e:
    pipeline = None
    print(f"[worker.tasks] ERROR: could not import pipeline: {e}", flush=True)


def _get_funnel_counts() -> Dict[str, int]:
    """
    Read FUNNEL_COUNTS from env as JSON, or fall back to safe defaults.
    Example env value:
      FUNNEL_COUNTS='{"HOOK":1,"PROBLEM":1,"FEATURE":1,"PROOF":1,"CTA":1}'
    """
    raw = os.getenv("FUNNEL_COUNTS")
    defaults: Dict[str, int] = {
        "HOOK": 1,
        "PROBLEM": 1,
        "FEATURE": 1,
        "PROOF": 1,
        "CTA": 1,
    }
    if not raw:
        return defaults

    try:
        data = json.loads(raw)
        for k, v in data.items():
            try:
                defaults[k] = int(v)
            except Exception:
                # bad value → keep default
                pass
        return defaults
    except Exception:
        return defaults


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    This is the job RQ will run: tasks.job_render
    """
    t0 = time.time()

    session_id = payload.get("session_id", f"session-{uuid.uuid4().hex[:8]}")

    # your webhook sometimes sends "files": ["..."], sometimes people call it "file_urls"
    file_urls: List[str] = payload.get("files") or payload.get("file_urls") or []
    portrait = bool(payload.get("portrait", True))
    max_duration = float(payload.get("max_duration", 120.0))

    # accept both names, prefer the explicit s3_prefix
    s3_prefix = (
        payload.get("s3_prefix")
        or payload.get("output_prefix")
        or "editdna/outputs/"
    )

    print("[worker.tasks] job_render() start", flush=True)
    print(f"  session_id={session_id}", flush=True)
    print(f"  file_urls={file_urls}", flush=True)
    print(f"  portrait={portrait}", flush=True)
    print(f"  max_duration={max_duration}", flush=True)
    print(f"  s3_prefix={s3_prefix}", flush=True)

    if not pipeline:
        # pipeline couldn’t be imported at all
        err = "pipeline module not importable (check PYTHONPATH or repo path)"
        print(f"[worker.tasks] ERROR: {err}", flush=True)
        return {
            "ok": False,
            "error": err,
        }

    try:
        result = pipeline.run_pipeline(
            session_id=session_id,
            file_urls=file_urls,
            portrait=portrait,
            max_duration=max_duration,
            s3_prefix=s3_prefix,
            funnel_counts=_get_funnel_counts(),
        )

        dt = time.time() - t0
        print(f"[worker.tasks] job_render() OK in {dt:.2f}s", flush=True)
        return result

    except Exception as e:
        print("[worker.tasks] ERROR in job_render()", flush=True)
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
