import os
import json
import time
import uuid
import tempfile
import traceback
from typing import Any, Dict, Optional

from .s3_utils import upload_file_presign  # you already have this in repo
from .captions import burn_captions_onto_video  # keep import for now (it's fine if unused)

# helper to read int/float envs safely (strip comments)
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return default
    cleaned = raw.split("#")[0].strip().split()[0]
    try:
        return float(cleaned)
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    cleaned = raw.split("#")[0].strip().split()[0]
    try:
        return int(cleaned)
    except Exception:
        return default

def _safe_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return default
    raw = raw.strip().lower()
    return raw in ("1","true","yes","on")

def _import_pipeline():
    # always import our pipeline module we just wrote
    import pipeline
    return pipeline

def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    This is what RQ runs.
    """
    t0 = time.time()

    session_id = payload.get("session_id", f"session-{uuid.uuid4().hex[:8]}")
    files = payload.get("files", [])
    if not isinstance(files, list) or len(files) == 0:
        return {
            "ok": False,
            "error": "No input files provided",
            "trace": None,
        }

    portrait = bool(payload.get("portrait", True))
    max_duration_req = payload.get("max_duration", None)

    # choose max duration:
    if max_duration_req is not None:
        try:
            max_duration_final = float(str(max_duration_req).strip().split()[0])
        except Exception:
            max_duration_final = _env_float("MAX_DURATION_SEC", 220.0)
    else:
        max_duration_final = _env_float("MAX_DURATION_SEC", 220.0)

    # funnel counts default
    funnel_counts = "1,3,3,3,1"
    if "options" in payload and isinstance(payload["options"], dict):
        fc = payload["options"].get("FUNNEL_COUNTS")
        if isinstance(fc, str) and fc.strip():
            funnel_counts = fc.strip()

    # import pipeline
    try:
        pipe = _import_pipeline()
    except Exception as import_err:
        return {
            "ok": False,
            "error": (
                "Could not import pipeline module.\n"
                f"{repr(import_err)}"
            ),
            "trace": traceback.format_exc(),
        }

    # run pipeline
    try:
        result = pipe.run_pipeline(
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
        return {
            "ok": False,
            "error": "Pipeline returned not-ok or empty result",
            "trace": json.dumps(result, indent=2, default=str),
        }

    t1 = time.time()
    # append runtime info
    result["runtime_sec"] = round(t1 - t0, 3)

    # mirror flags from env for debugging UI
    result["semantic"] = True
    result["vision"] = bool(int(os.getenv("SEMANTICS_ENABLED", "1")))
    result["asr"] = bool(int(os.getenv("ASR_ENABLED", "1")))

    return result
