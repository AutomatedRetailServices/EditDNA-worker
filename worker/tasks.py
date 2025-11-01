import os
import json
import time
import uuid
import traceback
from typing import Any, Dict

# IMPORTANT: no leading dots here.
# We are treating this folder as PYTHONPATH root, so modules import by simple name.
from s3_utils import upload_file_presign  # OK if unused, safe to leave
from captions import burn_captions_onto_video  # OK if unused, safe to leave
import pipeline  # <-- this is pipeline.py below

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return default
    cleaned = raw.split("#")[0].strip().split()[0]
    try:
        return float(cleaned)
    except Exception:
        return default

def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    This is what RQ actually runs.

    Flow:
    1. Read request (session_id, files, etc)
    2. Call pipeline.run_pipeline()
    3. Return the pipeline's JSON so /jobs can show it
    """

    t0 = time.time()

    # basic request fields
    session_id = payload.get("session_id", f"session-{uuid.uuid4().hex[:8]}")
    files = payload.get("files", [])
    portrait = bool(payload.get("portrait", True))

    if not isinstance(files, list) or len(files) == 0:
        return {
            "ok": False,
            "error": "No input files provided",
            "trace": None,
        }

    # pick max_duration (either from request or env MAX_DURATION_SEC)
    req_max = payload.get("max_duration", None)
    if req_max is not None:
        try:
            max_duration_final = float(str(req_max).strip().split()[0])
        except Exception:
            max_duration_final = _env_float("MAX_DURATION_SEC", 220.0)
    else:
        max_duration_final = _env_float("MAX_DURATION_SEC", 220.0)

    # funnel_counts (future marketing slots) - default shape
    funnel_counts = "1,3,3,3,1"
    if "options" in payload and isinstance(payload["options"], dict):
        fc = payload["options"].get("FUNNEL_COUNTS")
        if isinstance(fc, str) and fc.strip():
            funnel_counts = fc.strip()

    # run the pipeline
    try:
        result = pipeline.run_pipeline(
            session_id=session_id,
            file_urls=files,
            portrait=portrait,
            funnel_counts=funnel_counts,
            max_duration=max_duration_final,
        )
    except Exception as pipeline_err:
        return {
            "ok": False,
            "error": f"Pipeline crashed: {repr(pipeline_err)}",
            "trace": traceback.format_exc(),
        }

    if not result or not result.get("ok"):
        # pipeline said not ok
        return {
            "ok": False,
            "error": "Pipeline returned not-ok or empty result",
            "trace": json.dumps(result, indent=2, default=str),
        }

    # tag some debug flags so UI knows what ran
    result["semantic"] = True
    result["vision"] = bool(int(os.getenv("SEMANTICS_ENABLED", "1")))
    result["asr"] = bool(int(os.getenv("ASR_ENABLED", "1")))

    t1 = time.time()
    result["runtime_sec"] = round(t1 - t0, 3)

    return result
