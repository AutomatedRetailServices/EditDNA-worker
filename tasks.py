import logging
from typing import Any, Dict, Optional, List

from worker.pipeline import run_pipeline

logger = logging.getLogger("editdna.tasks")
logger.setLevel(logging.INFO)


def job_render(
    session_id: str,
    files: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
    mode: str = "human",
) -> Dict[str, Any]:
    """
    RQ worker entrypoint.

    - Normaliza 'files' / 'file_urls'
    - Pasa 'mode' directamente al pipeline:
        "human"  | "clean"  | "blooper"
    """
    logger.info(
        f"[job_render] START → session_id={session_id}, mode={mode}, "
        f"files={files}, file_urls={file_urls}"
    )

    # Normalizar lista de archivos
    all_files: List[str] = []
    if isinstance(files, list):
        all_files.extend(files)
    if isinstance(file_urls, list):
        all_files.extend(file_urls)

    if not all_files:
        raise ValueError("job_render: files[] or file_urls[] required")

    # Llamar al pipeline
    result = run_pipeline(
        session_id=session_id,
        files=all_files,
        mode=mode,
    )

    logger.info(
        f"[job_render] DONE → session_id={session_id}, "
        f"composer_mode={result.get('composer_mode')}, "
        f"output_video_url={result.get('output_video_url')}"
    )

    return result
