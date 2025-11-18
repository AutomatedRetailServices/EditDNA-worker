import logging
import traceback
from typing import List, Dict, Any, Optional

# 👇 make sure this import matches your folder structure
# If your pipeline file is at: EditDNA-worker/worker/pipeline.py
# and /workspace/EditDNA-worker is on PYTHONPATH,
# then this is correct:
from worker import pipeline

logger = logging.getLogger(__name__)


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    RQ job entrypoint.

    The web layer enqueues this as:

        q.enqueue("tasks.job_render", {
            "session_id": "...",
            "files": ["https://...mp4"]
        })

    RQ passes ONE positional arg: `payload` (a dict).

    We unwrap that dict and call:

        pipeline.run_pipeline(
            session_id=...,
            file_urls=...
        )
    """
    try:
        logger.info("🎬 job_render called (raw payload)", extra={"payload": payload})

        # Safety: payload must be a dict
        if not isinstance(payload, dict):
            raise ValueError(f"job_render expected dict payload, got: {type(payload)}")

        # Accept both "session_id" or legacy "id"
        session_id: Optional[str] = payload.get("session_id") or payload.get("id")
        # Accept both "files" and "file_urls" from the web layer
        files: Optional[List[str]] = payload.get("files") or payload.get("file_urls")

        if not session_id:
            raise ValueError("job_render: missing 'session_id' in payload")

        if not files or not isinstance(files, list):
            raise ValueError("job_render: 'files' / 'file_urls' must be a non-empty list in payload")

        logger.info(
            "🚀 Calling pipeline.run_pipeline",
            extra={"session_id": session_id, "num_files": len(files)}
        )

        # ⬇⬇⬇ KEY LINE — must be `file_urls=files`, NOT `files=files`
        result = pipeline.run_pipeline(
            session_id=session_id,
            file_urls=files,
        )

        # `result` should already contain everything (clips, slots, urls, etc.)
        out: Dict[str, Any] = {
            "ok": True,
            **result,
        }

        logger.info(
            "✅ job_render completed successfully",
            extra={"session_id": session_id, "keys": list(out.keys())}
        )
        return out

    except Exception as e:
        logger.exception("❌ job_render failed")
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
