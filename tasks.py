"""
tasks.py
RQ worker entrypoints for EditDNA Worker.

RQ will call tasks.job_render(payload)
and we forward into our real pipeline logic below.
"""

from __future__ import annotations

from typing import Any, Dict
import os
import json
import time
import uuid
import traceback

import pipeline  # /workspace/pipeline.py


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
        # merge over defaults so missing keys don’t crash the pipeline
        for k, v in data.items():
            try:
                defaults[k] = int(v)
            except Exception:
                # ignore bad value, keep default
                pass
        return defaults
    except Exception:
        # bad JSON in env → keep defaults
        return defaults


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    This is the job RQ will run: tasks.job_render
    It:
      - pulls args from the web payload
      - calls pipeline.run_pipeline(...)
      - returns whatever the pipeline returned
    """
    t0 = time.time()

    # 1) pull inputs from payload
    session_id = payload.get("session_id", f"session-{uuid.uuid4().hex[:8]}")
    file_urls = payload.get("files", [])
    portrait = bool(payload.get("portrait", True))
    max_duration = float(payload.get("max_duration", 120.0))
    # you used output_prefix in your earlier payloads
    s3_prefix = payload.get("output_prefix", "editdna/outputs/")

    print("[worker.tasks] job_render() start", flush=True)
    print(f"  session_id={session_id}", flush=True)
    print(f"  file_urls={file_urls}", flush=True)
    print(f"  portrait={portrait}", flush=True)
    print(f"  max_duration={max_duration}", flush=True)
    print(f"  s3_prefix={s3_prefix}", flush=True)

    try:
        # 2) run actual pipeline
        result = pipeline.run_pipeline(
            session_id=session_id,
            file_urls=file_urls,
            portrait=portrait,
            max_duration=max_duration,
            s3_prefix=s3_prefix,
            funnel_counts=_get_funnel_counts(),  # ← this was missing in your logs
        )

        dt = time.time() - t0
        print(f"[worker.tasks] job_render() OK in {dt:.2f}s", flush=True)
        # 3) return pipeline JSON
        return result

    except Exception as e:
        # on any error, give the API a JSON payload
        print("[worker.tasks] ERROR in job_render()", flush=True)
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
