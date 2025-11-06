"""
tasks.py
RQ worker entrypoints for EditDNA Worker.

We make this adaptive so it works even if pipeline.run_pipeline
uses a slightly different arg name (files, file_urls, urls, etc).
"""

from __future__ import annotations

from typing import Any, Dict
import os
import json
import time
import uuid
import traceback
import inspect

import pipeline  # your /workspace/EditDNA-worker/pipeline.py


def _get_funnel_counts() -> Dict[str, int]:
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
                pass
        return defaults
    except Exception:
        return defaults


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    session_id = payload.get("session_id", f"session-{uuid.uuid4().hex[:8]}")
    # what your web is sending:
    payload_files = payload.get("files", [])  # <- we start from here
    portrait = bool(payload.get("portrait", True))
    max_duration = float(payload.get("max_duration", 120.0))
    s3_prefix = payload.get("output_prefix", "editdna/outputs/")

    print("[worker.tasks] job_render() start", flush=True)
    print(f"  session_id={session_id}", flush=True)
    print(f"  incoming files={payload_files}", flush=True)

    # 1) find what the pipeline actually accepts
    sig = inspect.signature(pipeline.run_pipeline)
    params = sig.parameters.keys()

    # common names we might need to map to
    candidate_names = [
        "files",
        "file_urls",
        "urls",
        "inputs",
        "input_urls",
        "video_urls",
    ]

    chosen_name = None
    for name in candidate_names:
        if name in params:
            chosen_name = name
            break

    # build kwargs with the names the pipeline really has
    kwargs: Dict[str, Any] = {
        "session_id": session_id,
        "portrait": portrait,
        "max_duration": max_duration,
        "s3_prefix": s3_prefix,
        "funnel_counts": _get_funnel_counts(),
    }

    if chosen_name:
        kwargs[chosen_name] = payload_files
    else:
        # we couldn’t find any of the common names → tell the user exactly what we saw
        return {
            "ok": False,
            "error": (
                "pipeline.run_pipeline does not accept any of: "
                "files, file_urls, urls, inputs, input_urls, video_urls. "
                f"It has: {list(params)}"
            ),
        }

    try:
        result = pipeline.run_pipeline(**kwargs)
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
