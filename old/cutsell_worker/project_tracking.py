"""Best-effort project state tracking around processing and export jobs.

Project metadata must never become deletion/render authority. A temporary project-store
failure is observable but does not destroy an otherwise valid AI draft or export.
"""
from __future__ import annotations

from typing import Any

from .project_store import update_project


def safe_update_project(**kwargs) -> dict[str, Any]:
    try:
        record = update_project(**kwargs)
        return {"status": "saved", "project_state": record.get("state")}
    except Exception as exc:
        return {"status": "degraded", "reason": exc.__class__.__name__}
