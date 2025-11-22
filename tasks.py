from typing import Any, Dict
import logging

logger = logging.getLogger("editdna.tasks")

# Intentamos importar la función principal del pipeline
try:
    from pipeline import run_pipeline as _run_pipeline
    logger.info("[tasks] pipeline.run_pipeline importado correctamente.")
except Exception as e:
    logger.error(
        "[FATAL] No se pudo importar run_pipeline desde pipeline.py: %s", e
    )
    _run_pipeline = None


def job_render(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Punto de entrada que RQ ejecuta: tasks.job_render(data)

    Espera un dict con:
      - session_id: str
      - files: [url1, url2, ...]  (o file_urls)
    """
    if _run_pipeline is None:
        raise RuntimeError(
            "pipeline.run_pipeline no está disponible. "
            "Asegúrate de que pipeline.py existe y define run_pipeline(session_id, files)."
        )

    session_id = data.get("session_id")
    files = data.get("files") or data.get("file_urls")

    if not session_id or not files:
        raise ValueError(
            "job_render requiere 'session_id' y 'files' (o 'file_urls') con al menos 1 URL."
        )

    logger.info(
        "[tasks.job_render] Iniciando pipeline para session_id=%s, num_files=%s",
        session_id,
        len(files) if isinstance(files, list) else 1,
    )

    # Llamamos directo al pipeline V3
    result = _run_pipeline(session_id=session_id, files=files)

    logger.info(
        "[tasks.job_render] Pipeline completado para session_id=%s (ok=%s)",
        session_id,
        result.get("ok"),
    )
    return result
