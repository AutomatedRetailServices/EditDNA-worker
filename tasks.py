import logging
from typing import List, Optional, Dict, Any

from pipeline import run_pipeline

log = logging.getLogger("editdna.tasks")
log.setLevel(logging.INFO)


def job_render(
    session_id: str,
    files: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
    mode: str = "human",
) -> Dict[str, Any]:
    """
    Job que ejecuta el pipeline principal.

    Es EXACTAMENTE el nombre que encola el web:
        "tasks.job_render"

    Parámetros:
      - session_id: id de sesión (para carpetas /tmp)
      - files: lista de URLs de video
      - file_urls: alias (por compatibilidad, puedes dejar None)
      - mode: "human" | "clean" | "blooper"
    """
    mode_norm = (mode or "human").lower()
    if mode_norm not in ("human", "clean", "blooper"):
        mode_norm = "human"

    log.info(
        f"[job_render] session_id={session_id} mode={mode_norm} "
        f"files={files} file_urls={file_urls}"
    )

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
