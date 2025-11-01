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

import pipeline  # <- this imports pipeline.py sitting next to this file


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
    s3_prefix = payload.get("output_prefix", "editdna/outputs")

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
            funnel_count="1",            # stub / future funnel logic
            max_duration=max_duration,
            s3_prefix=s3_prefix,
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
