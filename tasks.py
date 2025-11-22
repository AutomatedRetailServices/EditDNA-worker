import logging
from typing import Any, Dict, List, Optional

from pipeline import run_pipeline

logger = logging.getLogger("editdna.tasks")


def job_render(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point que RQ llama como "tasks.job_render".

    Espera un payload tipo:
      {
        "session_id": "funnel-test-1",
        "files": ["https://...mp4"],      # o
        "file_urls": ["https://...mp4"]   # cualquiera de las dos
      }
    """

    session_id: Optional[str] = data.get("session_id")
    if not session_id:
        raise ValueError("tasks.job_render: falta 'session_id' en el payload")

    files: Optional[List[str]] = data.get("files")
    file_urls: Optional[List[str]] = data.get("file_urls")

    logger.info(
        "tasks.job_render: session_id=%s, files=%d, file_urls=%d",
        session_id,
        len(files or []),
        len(file_urls or []),
    )

    # Delegamos TODO el trabajo real al pipeline
    result = run_pipeline(
        session_id=session_id,
        files=files,
        file_urls=file_urls,
    )

    logger.info("tasks.job_render: terminado ok para session_id=%s", session_id)
    return result
