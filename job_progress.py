"""Safe RQ progress reporting for render jobs."""

import logging
from typing import Any, Optional

from pipeline_errors import JobCanceledError

log = logging.getLogger("editdna.progress")

STAGES = {
    "queued",
    "downloading",
    "analyzing",
    "selecting",
    "rendering",
    "uploading",
    "finished",
    "failed",
    "canceled",
}


def current_rq_job() -> Optional[Any]:
    """Return the current RQ job, or ``None`` for direct task calls."""
    try:
        from rq import get_current_job

        return get_current_job()
    except Exception:
        return None


class RQProgressReporter:
    """Persist public progress fields without making rendering depend on Redis."""

    def __init__(self, job: Optional[Any] = None):
        self.job = job if job is not None else current_rq_job()
        meta = getattr(self.job, "meta", {}) if self.job is not None else {}
        try:
            self.progress = min(100, max(0, int(meta.get("progress", 0)))) if isinstance(meta, dict) else 0
        except (TypeError, ValueError):
            self.progress = 0

    def update(self, stage: str, progress: int, message: str) -> None:
        if self.job is None or stage not in STAGES:
            return
        progress = max(self.progress, min(100, max(0, int(progress))))
        self.progress = progress
        try:
            self.job.meta.update(stage=stage, progress=progress, message=str(message)[:160])
            self.job.save_meta()
        except Exception:
            log.warning("Unable to save render job progress metadata", exc_info=True)

    def check_canceled(self) -> None:
        """Raise only when a cooperative cancellation flag can be read safely."""
        if self.job is None:
            return
        try:
            self.job.refresh()
            if self.job.meta.get("cancel_requested"):
                raise JobCanceledError("Render job was canceled")
        except JobCanceledError:
            raise
        except Exception:
            log.warning("Unable to refresh render job cancellation metadata", exc_info=True)
