"""Mobile V1 text-overlay editor routes."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cutsell_worker.text_edits import add_text_overlay, remove_text_overlay, update_text_overlay

router = APIRouter(prefix="/v1/draft-edits/text", tags=["text"])


class TextAddRequest(BaseModel):
    draft: dict[str, Any]
    text: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    x: float = Field(default=0.5, ge=0, le=1)
    y: float = Field(default=0.2, ge=0, le=1)
    scale: float = Field(default=1.0, ge=0.5, le=3.0)


class TextUpdateRequest(BaseModel):
    draft: dict[str, Any]
    overlay_id: str
    text: str | None = None
    start: float | None = None
    end: float | None = None
    x: float | None = None
    y: float | None = None
    scale: float | None = None


class TextRemoveRequest(BaseModel):
    draft: dict[str, Any]
    overlay_id: str


@router.post("/add")
def add_text(payload: TextAddRequest):
    try:
        return add_text_overlay(
            payload.draft,
            text=payload.text,
            start=payload.start,
            end=payload.end,
            x=payload.x,
            y=payload.y,
            scale=payload.scale,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.post("/update")
def update_text(payload: TextUpdateRequest):
    try:
        return update_text_overlay(
            payload.draft,
            payload.overlay_id,
            text=payload.text,
            start=payload.start,
            end=payload.end,
            x=payload.x,
            y=payload.y,
            scale=payload.scale,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.post("/remove")
def remove_text(payload: TextRemoveRequest):
    try:
        return remove_text_overlay(payload.draft, payload.overlay_id)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
