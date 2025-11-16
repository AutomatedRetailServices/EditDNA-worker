import logging
import traceback
from typing import List, Dict, Any, Optional

from worker import pipeline  # 👈 your V2/V3 pipeline

logger = logging.getLogger(__name__)


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    RQ job entrypoint.

    IMPORTANT:
    The web layer enqueues this as:

        q.enqueue("tasks.job_render", {
            "session_id": "...",
            "files": ["https://...mp4"]
        })

    So RQ passes a SINGLE positional argument: `payload` (a dict).

    We unwrap that dict here and forward to pipeline.run_pipeline(
        session_id=...,
        file_urls=...
    ).
    """
    try:
        logger.info("🎬 job_render called (raw payload)", extra={"payload": payload})

        if not isinstance(payload, dict):
            raise ValueError(f"job_render expected dict payload, got: {type(payload)}")

        session_id: Optional[str] = payload.get("session_id") or payload.get("id")
        files: Optional[List[str]] = payload.get("files") or payload.get("file_urls")

        if not session_id:
            raise ValueError("job_render: missing 'session_id' in payload")

        if not files or not isinstance(files, list):
            raise ValueError("job_render: 'files' must be a non-empty list in payload")

        # ✅ Call the new pipeline signature (no input_local / s3_prefix)
        out = pipeline.run_pipeline(
            session_id=session_id,
            file_urls=files,
        )

        # `out` should already be a dict with clips, slots, urls, etc.
        return {
            "ok": True,
            **out,
        }

    except Exception as e:
        logger.exception("job_render failed")
        return {
            "ok": False,
            "error": "pipeline execution failed",
            "traceback": traceback.format_exc(),
        }
