import logging
from typing import Dict, Any

from pipeline import run_pipeline

logger = logging.getLogger("editdna.tasks")
logger.setLevel(logging.INFO)


def job_render(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point para RQ worker.
    Se encola como "tasks.job_render" desde jobs.py / desde la API.

    Espera un dict con al menos:
      - session_id: str
      - files: [url, ...]  (opcional)
      - file_urls: [url, ...]  (alias opcional)
    """
    logger.info(f"[tasks.job_render] recibido payload: {data}")

    if not isinstance(data, dict):
        raise ValueError("tasks.job_render espera un dict como argumento único.")

    session_id = data.get("session_id")
    if not session_id:
        raise ValueError("tasks.job_render: falta 'session_id' en el payload.")

    files = data.get("files")
    file_urls = data.get("file_urls")

    try:
        result = run_pipeline(
            session_id=session_id,
            files=files,
            file_urls=file_urls,
        )
        logger.info(
            f"[tasks.job_render] completado OK. session_id={session_id}, "
            f"output={result.get('output_video_url') or result.get('output_video_local')}"
        )
        return result
    except Exception as e:
        logger.exception(
            f"[tasks.job_render] ERROR ejecutando run_pipeline para session_id={session_id}: {e}"
        )
        raise
