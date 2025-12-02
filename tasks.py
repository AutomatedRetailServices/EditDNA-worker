import logging
from typing import Any, Dict, Optional, List

from worker.pipeline import run_pipeline  # 👈 desde worker/pipeline.py

logger = logging.getLogger("editdna.tasks")
logger.setLevel(logging.INFO)


def _normalize_files_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza el payload que viene del API/web para que siempre
    tengamos:
        session_id: str
        files: List[str] (urls)
        mode: str  ("human" | "clean")

    Aceptamos también file_urls.
    """
    session_id = data.get("session_id") or data.get("session") or "session-unknown"

    files: Optional[List[str]] = None

    # 1) Si viene "files" y es lista, la usamos
    if isinstance(data.get("files"), list):
        files = data["files"]
    # 2) Si viene "file_urls" y es lista, la usamos
    elif isinstance(data.get("file_urls"), list):
        files = data["file_urls"]

    if not files:
        raise ValueError(
            "job_render: se requiere 'files' o 'file_urls' como lista de URLs."
        )

    # 3) Modo de render: "human" (por defecto) o "clean"
    mode_raw = str(data.get("mode") or "human").lower().strip()
    if mode_raw not in ("human", "clean"):
        mode = "human"
    else:
        mode = mode_raw

    return {
        "session_id": session_id,
        "files": files,
        "mode": mode,
    }


def job_render(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point que RQ usa: tasks.job_render

    - Normaliza el payload.
    - Llama a worker.pipeline.run_pipeline(...)
    - Devuelve el dict que arma run_pipeline.
    """
    logger.info(f"[tasks.job_render] payload recibido: {data}")

    normalized = _normalize_files_payload(data)
    session_id = normalized["session_id"]
    files = normalized["files"]
    mode = normalized["mode"]

    logger.info(
        f"[tasks.job_render] Normalizado → session_id={session_id}, files={files}, mode={mode}"
    )

    result = run_pipeline(
        session_id=session_id,
        files=files,
        mode=mode,
    )

    logger.info(
        f"[tasks.job_render] pipeline OK, mode={mode}, output_video_url={result.get('output_video_url')}"
    )
    return result
