from typing import Any, Dict
import logging

logger = logging.getLogger("editdna.tasks")

try:
    # 👉 IMPORTANTE:
    # Usamos pipeline.py, que ya actualizaste antes y donde
    # definimos job_render(data: Dict[str, Any]) como entrada principal.
    from pipeline import job_render as _job_render_impl
except Exception as e:
    logger.error("[FATAL] No se pudo importar job_render desde pipeline.py: %s", e)
    _job_render_impl = None  # type: ignore


def job_render(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point llamado por RQ (tasks.job_render).

    data viene del web/API y debe tener:
      - session_id
      - files (lista de URLs o S3)
      - cualquier otro campo que pipeline.job_render espere
    """
    if _job_render_impl is None:
        raise RuntimeError(
            "pipeline.job_render no está disponible. "
            "Asegúrate de que pipeline.py existe y define job_render(data)."
        )

    return _job_render_impl(data)
