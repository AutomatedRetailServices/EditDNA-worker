import logging
import traceback
from typing import List, Dict, Any, Optional

from worker import pipeline  # 👈 importa el nuevo pipeline.py

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    🎬 Punto de entrada del worker (RQ Job)

    El web layer hace algo así:
        q.enqueue("tasks.job_render", {
            "session_id": "...",
            "files": ["https://video.mp4"]
        })

    Entonces aquí:
      - validamos entrada
      - ejecutamos pipeline.run_pipeline(session_id, files)
      - devolvemos resultado completo
    """

    try:
        logger.info("🎬 job_render called", extra={"payload": payload})

        if not isinstance(payload, dict):
            raise ValueError(f"job_render expected dict payload, got: {type(payload)}")

        # session_id puede venir como "session_id" o "id"
        session_id: Optional[str] = payload.get("session_id") or payload.get("id")
        files: Optional[List[str]] = payload.get("files") or payload.get("file_urls")

        if not session_id:
            raise ValueError("job_render error: missing 'session_id' in payload")

        if not files or not isinstance(files, list):
            raise ValueError("job_render error: 'files' must be a non-empty list in payload")

        logger.info("🚀 Starting pipeline.run_pipeline", extra={"session_id": session_id})

        # 👇 Llamamos el pipeline con el nombre correcto del parámetro
        result = pipeline.run_pipeline(session_id=session_id, files=files)

        return {
            "ok": True,
            **result
        }

    except Exception as e:
        logger.exception("❌ job_render failed")
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
