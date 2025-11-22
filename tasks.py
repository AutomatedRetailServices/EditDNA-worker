import logging
from typing import Any, Dict, List, Optional

from pipeline import run_pipeline

logger = logging.getLogger("editdna.tasks")
logger.setLevel(logging.INFO)


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Job principal que usa RQ.

    RQ lo está llamando así (lo que ves en los logs):
        tasks.job_render({'session_id': 'funnel-test-1', 'files': ['https://...']})

    Por eso aquí recibimos UN solo parámetro: `payload` (dict)
    con claves como: session_id, files, file_urls.
    """
    logger.info(f"[job_render] payload recibido: {payload!r}")

    session_id: str = payload.get("session_id") or "session-unknown"
    files: Optional[List[str]] = payload.get("files")
    file_urls: Optional[List[str]] = payload.get("file_urls")

    try:
        result = run_pipeline(
            session_id=session_id,
            files=files,
            file_urls=file_urls,
        )
        logger.info(f"[job_render] OK session_id={session_id}")
        return result
    except Exception as e:
        logger.exception(f"[job_render] ERROR session_id={session_id}: {e}")
        # Puedes decidir qué devolver al fallar
        return {
            "ok": False,
            "session_id": session_id,
            "error": str(e),
        }
