import logging
import traceback
from typing import Dict, Any, List, Optional

# 👇 Importa el pipeline nuevo que te pasé
from worker import pipeline

logger = logging.getLogger(__name__)


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    RQ job entrypoint.

    IMPORTANTE:
    El web layer encola esto así:

        q.enqueue("tasks.job_render", {
            "session_id": "...",
            "files": ["https://...mp4"]
        })

    RQ pasa UN solo argumento posicional a la función:
        payload  (un dict)

    Aquí:
    1) Validamos el payload.
    2) Extraemos session_id y files.
    3) Llamamos pipeline.run_pipeline(session_id=..., file_urls=...).
    4) Devolvemos el dict que regresa el pipeline (clips, slots, composer, urls, etc.).
    """
    try:
        logger.info("🎬 job_render called", extra={"payload": payload})

        if not isinstance(payload, dict):
            raise ValueError(f"job_render expected dict payload, got: {type(payload)}")

        # 🧷 session_id (requerido)
        session_id: Optional[str] = payload.get("session_id") or payload.get("id")
        if not session_id:
            raise ValueError("job_render: missing 'session_id' in payload")

        # 🧷 files: lista de URLs de video
        files: Optional[List[str]] = payload.get("files") or payload.get("file_urls")
        if not files or not isinstance(files, list):
            raise ValueError("job_render: 'files' must be a non-empty list in payload")

        # ✅ Llamamos a tu nuevo pipeline (micro-cuts + free-flow + LLM)
        result = pipeline.run_pipeline(
            session_id=session_id,
            file_urls=files,
        )

        if not isinstance(result, dict):
            raise ValueError("pipeline.run_pipeline must return a dict")

        # Aseguramos flag ok en éxito
        result.setdefault("ok", True)

        return result

    except Exception as e:
        logger.exception("job_render failed")
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
