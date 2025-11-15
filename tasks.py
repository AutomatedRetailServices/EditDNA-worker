import logging
import traceback
from typing import Dict, Any, List

import os
import requests

from worker import pipeline  # worker/pipeline.py

logger = logging.getLogger(__name__)


def _download_to_local(url: str, session_id: str) -> str:
    """
    Download the remote video URL to a local temp file
    so pipeline.run_pipeline can read it.
    """
    os.makedirs("/tmp/editdna", exist_ok=True)

    # very simple name: session-id + extension guess
    ext = ".mp4"
    if "." in url.split("?")[0]:
        ext = "." + url.split("?")[0].split(".")[-1]

    local_path = os.path.join("/tmp/editdna", f"{session_id}{ext}")

    logger.info("⬇️ downloading file", extra={"url": url, "local_path": local_path})

    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return local_path


def job_render(payload: dict) -> Dict[str, Any]:
    """
    RQ job entrypoint.

    The web API enqueues this as:
    tasks.job_render({"session_id": ..., "files": [...]})
    """
    session_id = payload["session_id"]
    files: List[str] = payload["files"]

    logger.info("🎬 job_render called", extra={"session_id": session_id, "files": files})

    if not files:
        return {
            "ok": False,
            "error": "no_files",
            "traceback": "payload.files is empty",
        }

    try:
        # 1) download FIRST file to local path
        input_local = _download_to_local(files[0], session_id=session_id)

        # 2) call pipeline with the signature it expects
        out = pipeline.run_pipeline(
            input_local=input_local,
            session_id=session_id,
            s3_prefix=session_id,  # placeholder; your current pipeline doesn't use it
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
