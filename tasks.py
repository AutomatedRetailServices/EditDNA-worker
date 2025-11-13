import os
import logging
from dotenv import load_dotenv

load_dotenv()

from worker.pipeline import run_pipeline  # <-- from worker/pipeline.py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker.tasks")


def job_render(session_id: str,
               files: list,
               s3_prefix: str = "editdna/outputs/"):
    """
    RQ job entrypoint.
    Called from the Web API via: q.enqueue("tasks.job_render", ...)
    """
    logger.info("[worker.tasks] job_render() start")
    logger.info("  session_id=%s", session_id)
    logger.info("  urls=%s", files)
    logger.info("  s3_prefix=%s", s3_prefix)

    if not files:
        raise ValueError("files list is empty")

    # Make sure it's a list of strings
    file_urls = [str(u) for u in files]

    out = run_pipeline(
        session_id=session_id,
        file_urls=file_urls,
        s3_prefix=s3_prefix,
    )

    logger.info("[worker.tasks] job_render() finished OK")
    return out
