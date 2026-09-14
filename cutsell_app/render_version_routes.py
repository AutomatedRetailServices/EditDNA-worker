"""Render-version recovery routes for CutSell Mobile V1."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from cutsell_worker.render_versions import hydrated_render_versions

router = APIRouter(prefix="/v1/projects", tags=["render-versions"])


@router.get("/{project_id}/renders")
def list_project_renders(project_id: str, user_id: str):
    try:
        return {
            "project_id": project_id,
            "renders": hydrated_render_versions(user_id=user_id, project_id=project_id),
        }
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
