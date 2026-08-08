"""Project library routes for the CutSell mobile app."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from cutsell_worker.project_store import create_project, get_project, list_projects, update_project

router = APIRouter(prefix="/v1/projects", tags=["projects"])


class ProjectCreateRequest(BaseModel):
    user_id: str
    title: str | None = None


class ProjectRenameRequest(BaseModel):
    user_id: str
    title: str


@router.post("")
def create_mobile_project(payload: ProjectCreateRequest):
    try:
        return create_project(user_id=payload.user_id, title=payload.title)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.get("")
def list_mobile_projects(user_id: str, limit: int = Query(default=50, ge=1, le=100)):
    try:
        return {"projects": list_projects(user_id=user_id, limit=limit)}
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.get("/{project_id}")
def get_mobile_project(project_id: str, user_id: str):
    try:
        return get_project(user_id=user_id, project_id=project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="project not found") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.patch("/{project_id}")
def rename_mobile_project(project_id: str, payload: ProjectRenameRequest):
    try:
        return update_project(user_id=payload.user_id, project_id=project_id, title=payload.title)
    except KeyError:
        raise HTTPException(status_code=404, detail="project not found") from None
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
