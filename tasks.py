import logging
from typing import Dict, Any

# Importamos nuestra función principal del pipeline
try:
    from pipeline import run_pipeline
except Exception as e:
    raise RuntimeError(
        f"No se pudo importar run_pipeline desde pipeline.py. "
        f"Revisa que pipeline.py exista y tenga run_pipeline(session_id, files). Error={e}"
    )

logger = logging.getLogger("editdna.tasks")


def job_render(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Punto de entrada REAL para RQ.
    Este nombre EXACTO debe coincidir con el string 'tasks.job_render'
    que usa el queue.enqueue(...) del server web.
    """

    logger.info(f"[tasks.job_render] payload recibido: {data}")

    if not isinstance(data, dict):
        raise ValueError("data debe ser un dict con session_id y files/file_urls")

    session_id = data.get("session_id")
    files = data.get("files") or data.get("file_urls")

    if not session_id:
        raise ValueError("Falta session_id en data")

    if not files or not isinstance(files, list):
        raise ValueError("Falta files (lista de URLs) en data")

    # Llamamos al pipeline
    result = run_pipeline(session_id=session_id, file_urls=files)

    logger.info(f"[tasks.job_render] finalizado OK session_id={session_id}")
    return result
