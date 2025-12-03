# EditDNA-worker/tasks.py
import logging
from typing import List, Optional, Dict, Any

# Importa el pipeline REAL que está dentro de Worker/pipeline.py
from Worker.pipeline import run_pipeline

log = logging.getLogger("editdna.tasks")
log.setLevel(logging.INFO)


def job_render(
    session_id: str,
    files: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
    mode: str = "human",
) -> Dict[str, Any]:
    """
    Job principal ejecutado por el worker.

    IMPORTANTE:
      - Este módulo se llama 'tasks'
      - Esta función se llama 'job_render'
      - Así coincide perfecto con el string: 'tasks.job_render'
        que se encola desde EditDNA-web.
    """

    # Normalizar modo
    mode_norm = (mode or "human").lower()
    if mode_norm not in ("human", "clean", "blooper"):
        mode_norm = "human"

    log.info(
        f"[job_render] START session_id={session_id} mode={mode_norm} "
        f"files={files} file_urls={file_urls}"
    )

    # Ejecuta el pipeline real
    result = run_pipeline(
        session_id=session_id,
        files=files,
        file_urls=file_urls,
        mode=mode_norm,
    )

    log.info(
        f"[job_render] DONE session_id={session_id} "
        f"mode={result.get('composer_mode')} "
        f"duration={result.get('duration_sec')}"
    )

    return result
