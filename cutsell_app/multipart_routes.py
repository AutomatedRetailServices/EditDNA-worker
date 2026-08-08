"""FastAPI routes for resumable direct-to-S3 mobile uploads."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cutsell_worker.multipart_uploads import (
    abort_multipart_upload,
    complete_multipart_upload,
    list_multipart_parts,
    presign_multipart_part,
    start_multipart_upload,
)

router = APIRouter(prefix="/v1/uploads/multipart", tags=["uploads"])


class MultipartStartRequest(BaseModel):
    project_id: str
    user_id: str
    original_name: str
    content_type: str | None = None
    size_bytes: int = Field(gt=0)


class MultipartOwnerRequest(BaseModel):
    project_id: str
    user_id: str


class MultipartPart(BaseModel):
    part_number: int = Field(ge=1)
    etag: str


class MultipartCompleteRequest(MultipartOwnerRequest):
    parts: list[MultipartPart] = Field(min_length=1)


def _translate_error(exc: Exception):
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail="multipart upload session not found") from None
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from None
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=str(exc)) from None
    raise exc


@router.post("/start")
def start_upload(payload: MultipartStartRequest):
    try:
        return start_multipart_upload(
            project_id=payload.project_id,
            user_id=payload.user_id,
            original_name=payload.original_name,
            content_type=payload.content_type,
            size_bytes=payload.size_bytes,
        )
    except Exception as exc:
        _translate_error(exc)


@router.get("/{upload_id}")
def resume_upload(upload_id: str, project_id: str, user_id: str):
    try:
        return list_multipart_parts(
            upload_id=upload_id,
            project_id=project_id,
            user_id=user_id,
        )
    except Exception as exc:
        _translate_error(exc)


@router.post("/{upload_id}/parts/{part_number}/presign")
def sign_part(upload_id: str, part_number: int, payload: MultipartOwnerRequest):
    try:
        return presign_multipart_part(
            upload_id=upload_id,
            project_id=payload.project_id,
            user_id=payload.user_id,
            part_number=part_number,
        )
    except Exception as exc:
        _translate_error(exc)


@router.post("/{upload_id}/complete")
def complete_upload(upload_id: str, payload: MultipartCompleteRequest):
    try:
        return complete_multipart_upload(
            upload_id=upload_id,
            project_id=payload.project_id,
            user_id=payload.user_id,
            parts=[part.model_dump() for part in payload.parts],
        )
    except Exception as exc:
        _translate_error(exc)


@router.delete("/{upload_id}")
def abort_upload(upload_id: str, project_id: str, user_id: str):
    try:
        return abort_multipart_upload(
            upload_id=upload_id,
            project_id=project_id,
            user_id=user_id,
        )
    except Exception as exc:
        _translate_error(exc)
