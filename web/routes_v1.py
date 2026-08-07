"""Stateless V1 API endpoints for editable-draft mutations.

These routes deliberately do not persist projects or drafts. Persistence,
authorization, ownership, and idempotency belong to the product backend; this
worker-facing API only exposes the pure draft transformations recovered for V1.
"""
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from web.contracts_v1 import (
    API_VERSION,
    DraftRemoveRequest,
    DraftReorderRequest,
    DraftRestoreRequest,
    DraftSwapRequest,
)
from worker.draft_edits import (
    DraftEditError,
    remove_clip,
    reorder_clips,
    restore_clip,
    swap_take,
)

router = APIRouter(prefix="/v1", tags=["v1"])


class DraftSwapMutation(DraftSwapRequest):
    draft: Dict[str, Any]


class DraftRemoveMutation(DraftRemoveRequest):
    draft: Dict[str, Any]


class DraftRestoreMutation(DraftRestoreRequest):
    draft: Dict[str, Any]


class DraftReorderMutation(DraftReorderRequest):
    draft: Dict[str, Any]


class DraftMutationResponse(BaseModel):
    api_version: str = API_VERSION
    draft: Dict[str, Any]


def _conflict(exc: DraftEditError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=str(exc),
    )


@router.get("/healthz")
def v1_healthz() -> Dict[str, Any]:
    return {
        "ok": True,
        "api_version": API_VERSION,
        "draft_edits": ["swap", "remove", "restore", "reorder"],
        "persistence": False,
    }


@router.post("/draft-edits/swap", response_model=DraftMutationResponse)
def draft_swap(payload: DraftSwapMutation) -> DraftMutationResponse:
    try:
        draft = swap_take(
            payload.draft,
            payload.selected_clip_id,
            payload.replacement_clip_id,
        )
    except DraftEditError as exc:
        raise _conflict(exc) from exc
    return DraftMutationResponse(draft=draft)


@router.post("/draft-edits/remove", response_model=DraftMutationResponse)
def draft_remove(payload: DraftRemoveMutation) -> DraftMutationResponse:
    try:
        draft = remove_clip(payload.draft, payload.clip_id)
    except DraftEditError as exc:
        raise _conflict(exc) from exc
    return DraftMutationResponse(draft=draft)


@router.post("/draft-edits/restore", response_model=DraftMutationResponse)
def draft_restore(payload: DraftRestoreMutation) -> DraftMutationResponse:
    try:
        draft = restore_clip(payload.draft, payload.clip_id, payload.position)
    except DraftEditError as exc:
        raise _conflict(exc) from exc
    return DraftMutationResponse(draft=draft)


@router.post("/draft-edits/reorder", response_model=DraftMutationResponse)
def draft_reorder(payload: DraftReorderMutation) -> DraftMutationResponse:
    try:
        draft = reorder_clips(payload.draft, payload.ordered_clip_ids)
    except DraftEditError as exc:
        raise _conflict(exc) from exc
    return DraftMutationResponse(draft=draft)
