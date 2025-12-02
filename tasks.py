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
        mode: str ("clean" | "human" | "blooper")
    """
    session_id = data.get("session_id") or data.get("session") or "session-unknown"

    files: Optional[List[str]] = None

    if isinstance(data.get("files"), list):
        files = data["files"]
    elif isinstance(data.get("file_urls"), list):
        files = data["file_urls"]

    if not files:
        raise ValueError(
            "job_render: se requiere 'files' o 'file_urls' como lista de URLs."
        )

    mode = data.get("mode", "human")  # 👈 DEFAULT si no viene

    return {
        "session_id": session_id,
        "files": files,
        "mode": mode,  # 👈 AÑADIDO
    }


def job_render(data: Dict[str, Any]) -> Dict[str, Any]:
    logger.info(f"[tasks.job_render] payload recibido: {data}")

    normalized = _normalize_files_payload(data)
    session_id = normalized["session_id"]
    files = normalized["files"]
    mode = normalized["mode"]            # 👈 AÑADIDO

    logger.info(
        f"[tasks.job_render] Normalizado → session_id={session_id}, files={files}, mode={mode}"
    )

    result = run_pipeline(
        session_id=session_id,
        files=files,
        mode=mode,    # 👈 AÑADIDO
    )

    logger.info(
        f"[tasks.job_render] pipeline OK, output_video_url={result.get('output_video_url')}"
    )
    return result
