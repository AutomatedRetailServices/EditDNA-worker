import logging
import traceback
from typing import Dict, Any

# import the real pipeline
from worker import pipeline

logger = logging.getLogger(__name__)


def job_render(payload: dict) -> Dict[str, Any]:
    """
    RQ job entrypoint.

    The web API enqueues this as:
    tasks.job_render({"session_id": ..., "files": [...]})

    So we must accept ONE argument: a payload dictionary.
    """
    session_id = payload["session_id"]
    files = payload["files"]

    logger.info("🎬 job_render called", extra={
        "session_id": session_id,
        "files": files
    })

    try:
        out = pipeline.run_pipeline(
            session_id=session_id,
            file_urls=files,
        )

        return {
            "ok": True,
            **out,
        }

    except Exception:
        logger.exception("job_render failed")
        return {
            "ok": False,
            "error": "pipeline execution failed",
            "traceback": traceback.format_exc(),
        }
