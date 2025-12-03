import logging
from typing import List, Optional, Dict, Any

from worker.pipeline import run_pipeline  # <- usa el pipeline que pegaste

log = logging.getLogger("editdna.tasks")
log.setLevel(logging.INFO)


def job_render(
    session_id: str,
    files: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
    mode: str = "human",
) -> Dict[str, Any]:
    """
    Punto de entrada que el worker RQ ejecuta como `tasks.job_render`.
    """

    # Normalizar modo para evitar sorpresas
    mode_norm = (mode or "human").lower()
    if mode_norm not in ("human", "clean", "blooper"):
        mode_norm = "human"

    log.info(
        f"[job_render] START session_id={session_id} "
        f"mode={mode_norm} files={files} file_urls={file_urls}"
    )

    # Ejecuta el pipeline REAL
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
