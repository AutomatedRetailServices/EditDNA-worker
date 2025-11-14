import logging
import traceback
from typing import List, Dict, Any

# 👇 this imports your pipeline logic from worker/pipeline.py
from worker import pipeline

logger = logging.getLogger(__name__)


def job_render(session_id: str, files: List[str]) -> Dict[str, Any]:
    """
    RQ job entrypoint.

    This is what the web API enqueues as "tasks.job_render".
    `files` is a list of video URLs (S3 / HTTPS).
    """
    logger.info("🎬 job_render called", extra={"session_id": session_id, "files": files})

    try:
        # 👇 match your pipeline signature: run_pipeline(session_id=..., file_urls=...)
        out = pipeline.run_pipeline(
            session_id=session_id,
            file_urls=files,
        )

        # `out` should already be a dict with clips, slots, urls, etc.
        # we just wrap it in a stable envelope
        return {
            "ok": True,
            **out,
        }

    except Exception as e:
        logger.exception("job_render failed")
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
