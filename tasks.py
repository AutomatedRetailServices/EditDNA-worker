import logging
from typing import List, Optional, Dict, Any

from pipeline_errors import JobCanceledError
from job_progress import RQProgressReporter
from rq.exceptions import StopRequested

log = logging.getLogger("editdna.tasks")
log.setLevel(logging.INFO)


def run_pipeline(**kwargs):
    """Load the heavyweight media pipeline only when a render actually runs."""
    from worker.pipeline import run_pipeline as pipeline_run

    return pipeline_run(**kwargs)


def job_render(
    session_id: str,
    files: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
    mode: str = "human",
) -> Dict[str, Any]:
    """
    Punto de entrada que el worker RQ ejecuta como `tasks.job_render`.
    """

    # Direct callers retain the historical signature, but invalid modes are never
    # silently changed into a different editing mode.
    mode_norm = (mode or "human").lower()
    if mode_norm not in ("human", "clean", "blooper"):
        raise ValueError(f"Unsupported render mode: {mode}")

    log.info("[job_render] START session_id=%s mode=%s", session_id, mode_norm)

    reporter = RQProgressReporter()
    reporter.update("downloading", 5, "Preparing source downloads")
    try:
        result = run_pipeline(
            session_id=session_id,
            files=files,
            file_urls=file_urls,
            mode=mode_norm,
            progress=reporter.update,
            check_canceled=reporter.check_canceled,
        )
    except JobCanceledError:
        reporter.update("canceled", reporter.progress, "Render canceled")
        if reporter.job is not None:
            raise StopRequested() from None
        raise
    except Exception:
        reporter.update("failed", reporter.progress, "Render failed")
        raise

    reporter.update("finished", 100, "Render finished")

    log.info(
        f"[job_render] DONE session_id={session_id} "
        f"mode={result.get('composer_mode')} "
        f"duration={result.get('duration_sec')}"
    )

    return result
