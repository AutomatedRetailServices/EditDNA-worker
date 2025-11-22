import os
import logging
from redis import Redis
from rq import get_current_job

from pipeline import run_pipeline  # usamos tu pipeline.py
# Si pipeline.py está en /app/pipeline.py esto es correcto.

logger = logging.getLogger("editdna.tasks")
logger.setLevel(logging.INFO)


def get_redis() -> Redis:
    url = os.environ.get("REDIS_URL")
    if not url:
        raise RuntimeError("REDIS_URL env no está definido")
    return Redis.from_url(url)


def job_render(session_id: str, files=None, file_urls=None):
    """
    Job principal que ejecuta el pipeline y devuelve el resultado.
    Este es el nombre que RQ está intentando importar: tasks.job_render
    """
    logger.info(f"[job_render] session_id={session_id} files={files} file_urls={file_urls}")
    job = get_current_job()
    if job:
        logger.info(f"[job_render] job.id={job.id}")

    # Normalizamos: usamos 'files' como lista de URLs
    effective_files = None
    if files and isinstance(files, list):
        effective_files = files
    elif file_urls and isinstance(file_urls, list):
        effective_files = file_urls

    if not effective_files:
        raise ValueError("job_render: se requiere 'files' o 'file_urls' como lista con al menos 1 URL")

    result = run_pipeline(session_id=session_id, files=effective_files)

    logger.info(f"[job_render] pipeline ok={result.get('ok')} duration={result.get('duration_sec')}")
    return result
