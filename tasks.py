import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger("editdna.tasks")
logger.setLevel(logging.INFO)

# Intentamos importar pipeline tanto si está en /worker/pipeline.py
# como si está en /app/pipeline.py
try:
    from worker import pipeline
except ImportError:
    import pipeline


def job_render(
    session_id: str,
    files: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    RQ job entrypoint: esto es lo que RQ busca cuando ve 'tasks.job_render'.

    Siempre llamamos a pipeline.run_pipeline con la firma flexible:
      - run_pipeline(session_id=..., files=..., file_urls=...)
    """
    logger.info(
        f"[job_render] start session_id={session_id} "
        f"files={files} file_urls={file_urls}"
    )

    result = pipeline.run_pipeline(
        session_id=session_id,
        files=files,
        file_urls=file_urls,
    )

    logger.info(f"[job_render] done session_id={session_id}")
    return result
