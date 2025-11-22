import logging
from typing import Any, Dict

logger = logging.getLogger("editdna.tasks")

try:
    # Usamos pipeline.job_render como entrypoint, V3 estándar del worker
    from pipeline import job_render as job_render_impl
except Exception as e:
    logger.error("[FATAL] ❌ No se pudo importar pipeline.job_render: %s", e)
    job_render_impl = None


def job_render(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point llamado por RQ como 'tasks.job_render'.

    Recibe:
      - data['session_id']
      - data['files'] (lista de URLs o S3)
      - cualquier otro campo que pipeline.job_render necesite
    """
    if job_render_impl is None:
        raise RuntimeError(
            "pipeline.job_render no está disponible. Asegúrate "
            "de que pipeline.py existe y define job_render(data)."
        )

    return job_render_impl(data)
