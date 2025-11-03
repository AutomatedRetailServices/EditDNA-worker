"""
tasks.py
RQ worker entrypoints for EditDNA Worker.

RQ will call tasks.job_render(payload)
and we forward into our real pipeline logic below.
"""

from __future__ import annotations
from typing import Any, Dict
import traceback
import uuid
import time
import os
import json

import pipeline  # imports /workspace/pipeline.py


def _get_funnel_counts() -> Dict[str, int]:
    """
    Read FUNNEL_COUNTS from env as JSON, or return safe defaults.
    Example:
      FUNNEL_COUNTS='{"HOOK":1,"PROBLEM":1,"FEATURE":1,"PROOF":1,"CTA":1}'
    """
    raw = os.getenv("FUNNEL_COUNTS")
    default_counts = {
        "HOOK": 1,
        "PROBLEM": 1,
        "FEATURE": 1,
        "PROOF": 1,
        "CTA": 1,
    }
    if not raw:
        return default_counts
    try:
        data = json.loads(raw)
        default_counts.update({k: int(v) for k, v in data.items()})
        return default_counts
    except Exception:
        return default_counts


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    This is what RQ actually runs.

    Flow:
    1. Read request (session_id, files, etc) from payload sent by the web.
    2. Call pipeline.run_pipeline() to generate clip(s).
    3. Return the pipeline's JSON so /jobs can show it.
    """
    t0 = time.time()

    # pull inputs from the payload coming from Render/web
    session_id = payload.get("session_id", f"session-{uuid.uuid4().hex[:8]}")
    file_urls = payload.get("files", [])
    portrait = bool(payload.get("portrait", True))
    max_duration = float(payload.get("max_duration", 220.0))
    s3_prefix = payload.get("output_prefix", "editdna/outputs/")

    # extra debug info:
    print("[worker.tasks] job_render() start", flush=True)
    print(f"  session_id={session_id}", flush=True)
    print(f"  file_urls={file_urls}", flush=True)
    print(f"  portrait={portrait}", flush=True)
    print(f"  max_duration={max_duration}", flush=True)
    print(f"  s3_prefix={s3_prefix}", flush=True)

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
