"""
worker/tasks.py
This is the real job logic RQ should run.

RQ calls tasks.job_render   (module 'tasks' at repo root),
and that file just imports job_render from here.

So THIS file must:
- parse the payload from the web
- call pipeline.run_pipeline(...)
- return the JSON dict
"""

from __future__ import annotations
from typing import Any, Dict, List
import traceback
import uuid
import pipeline  # this imports /workspace/pipeline.py


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entrypoint that actually executes in the worker.

    Expected payload from web:
    {
        "session_id": "funnel-test-1",
        "files": ["https://.../input_video.mp4"],
        "portrait": true,
        "max_duration": 220,
        "output_prefix": "editdna/outputs/"
    }

    We sanitize, then forward to pipeline.run_pipeline().
    """

    try:
        # --- read/clean incoming payload ---
        session_id    = str(payload.get("session_id", uuid.uuid4().hex[:8]))
        file_urls_raw = payload.get("files", [])
        if not isinstance(file_urls_raw, list):
            file_urls_raw = [file_urls_raw]

        portrait      = bool(payload.get("portrait", True))
        max_duration  = float(payload.get("max_duration", 220.0))

        # not yet used deeply but keep wiring it forward:
        funnel_counts = str(payload.get("funnel_counts", "HOOK,PROBLEM,FEATURE,PROOF,CTA"))

        # --- call pipeline ---
        result = pipeline.run_pipeline(
            session_id=session_id,
            file_urls=file_urls_raw,
            portrait=portrait,
            funnel_counts=funnel_counts,
            max_duration=max_duration,
        )

        return result

    except Exception as e:
        traceback.print_exc()
        return {
            "ok": False,
            "error": f"job_render crashed: {e}",
        }
