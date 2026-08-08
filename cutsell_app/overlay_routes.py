"""Photo/video overlay upload and editor routes for CutSell Mobile V1."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cutsell_worker.overlay_edits import add_media_overlay, remove_media_overlay, update_media_overlay
from cutsell_worker.overlay_uploads import create_overlay_presigned_upload, validate_overlay_uri

router = APIRouter(prefix="/v1/overlays", tags=["overlays"])


class OverlayUploadRequest(BaseModel):
    project_id: str
    user_id: str
    original_name: str
    content_type: str | None = None
    size_bytes: int = Field(gt=0)


class OverlayAddRequest(BaseModel):
    project_id: str
    user_id: str
    draft: dict[str, Any]
    kind: str
    uri: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    x: float = Field(default=0.5, ge=0, le=1)
    y: float = Field(default=0.5, ge=0, le=1)
    width: float = Field(default=0.4, ge=0.1, le=1)
    source_start: float = Field(default=0, ge=0)
    source_end: float | None = None
    mute_audio: bool = True


class OverlayUpdateRequest(BaseModel):
    draft: dict[str, Any]
    overlay_id: str
    start: float | None = None
    end: float | None = None
    x: float | None = None
    y: float | None = None
    width: float | None = None
    source_start: float | None = None
    source_end: float | None = None
    mute_audio: bool | None = None


class OverlayRemoveRequest(BaseModel):
    draft: dict[str, Any]
    overlay_id: str


@router.post("/uploads/presign")
def presign_overlay(payload: OverlayUploadRequest):
    try:
        return create_overlay_presigned_upload(
            project_id=payload.project_id,
            user_id=payload.user_id,
            original_name=payload.original_name,
            content_type=payload.content_type,
            size_bytes=payload.size_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.post("/add")
def add_overlay(payload: OverlayAddRequest):
    try:
        _bucket, _key, actual_kind = validate_overlay_uri(
            payload.uri, user_id=payload.user_id, project_id=payload.project_id
        )
        if actual_kind != payload.kind:
            raise ValueError("overlay kind does not match uploaded media")
        return add_media_overlay(
            payload.draft, kind=payload.kind, uri=payload.uri,
            start=payload.start, end=payload.end, x=payload.x, y=payload.y, width=payload.width,
            source_start=payload.source_start, source_end=payload.source_end, mute_audio=payload.mute_audio,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.post("/update")
def update_overlay(payload: OverlayUpdateRequest):
    try:
        return update_media_overlay(
            payload.draft, payload.overlay_id,
            start=payload.start, end=payload.end, x=payload.x, y=payload.y, width=payload.width,
            source_start=payload.source_start, source_end=payload.source_end, mute_audio=payload.mute_audio,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.post("/remove")
def remove_overlay(payload: OverlayRemoveRequest):
    try:
        return remove_media_overlay(payload.draft, payload.overlay_id)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
