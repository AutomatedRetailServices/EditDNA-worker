"""Standalone FastAPI entrypoint for the clean CutSell mobile backend."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from cutsell_app.multipart_routes import router as multipart_router
from cutsell_app.overlay_routes import router as overlay_router
from cutsell_app.text_routes import router as text_router
from cutsell_worker.caption_settings import patch_caption_settings
from cutsell_worker.config import load_runtime_config
from cutsell_worker.draft_edits import (
    DraftEditError,
    patch_audio,
    patch_captions,
    remove_clip,
    reorder_clips,
    restore_clip,
    split_clip,
    swap_take,
    trim_clip,
)
from cutsell_worker.draft_store import (
    DraftConflictError,
    DraftHistoryError,
    get_draft_snapshot,
    redo_draft_snapshot,
    save_draft_snapshot,
    undo_draft_snapshot,
)
from cutsell_worker.jobs import cancel_job, fetch_job_snapshot
from cutsell_worker.queueing import enqueue_export, enqueue_flow_b
from cutsell_worker.serde import draft_from_dict
from cutsell_worker.source_identity import stable_source_id
from cutsell_worker.timeline_asset_storage import sign_timeline_assets
from cutsell_worker.uploads import create_presigned_upload, validate_product_source_uri

app = FastAPI(title="CutSell API", version="0.1.0")
app.include_router(multipart_router)
app.include_router(text_router)
app.include_router(overlay_router)


class UploadPresignRequest(BaseModel):
    project_id: str
    user_id: str
    original_name: str
    content_type: str | None = None
    size_bytes: int = Field(gt=0)


class UploadPresignResponse(BaseModel):
    method: str
    upload_url: str
    fields: dict[str, str]
    source_uri: str
    object_key: str
    content_type: str
    max_bytes: int
    expires_in: int


class SourceInput(BaseModel):
    original_name: str
    uri: str
    source_order: int = Field(ge=0)
    duration_sec: float = Field(default=0.0, ge=0.0)


class FlowBSubmitRequest(BaseModel):
    project_id: str
    user_id: str
    sources: list[SourceInput] = Field(min_length=1)
    language_hint: str | None = None
    audio_overlap: bool = False


class FlowBSubmitResponse(BaseModel):
    job_id: str
    queue: str
    state: str = "uploaded"


class ExportSourceInput(BaseModel):
    source_asset_id: str
    original_name: str
    uri: str


class ExportSubmitRequest(BaseModel):
    project_id: str
    user_id: str
    draft: dict[str, Any]
    sources: list[ExportSourceInput] = Field(min_length=1)


class ExportSubmitResponse(BaseModel):
    job_id: str
    queue: str
    state: str = "rendering"


class DraftAutosaveRequest(BaseModel):
    user_id: str
    expected_revision: int = Field(ge=1)
    draft: dict[str, Any]


class DraftHistoryRequest(BaseModel):
    user_id: str
    expected_revision: int = Field(ge=1)


class JobStatusResponse(BaseModel):
    job_id: str
    state: str
    progress: int | None = None
    result: object | None = None
    error: str | None = None


class DraftSwapRequest(BaseModel):
    draft: dict[str, Any]
    selected_clip_id: str
    replacement_clip_id: str


class DraftClipRequest(BaseModel):
    draft: dict[str, Any]
    clip_id: str


class DraftRestoreRequest(DraftClipRequest):
    position: int | None = None


class DraftReorderRequest(BaseModel):
    draft: dict[str, Any]
    ordered_clip_ids: list[str]


class DraftTrimRequest(DraftClipRequest):
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class DraftSplitRequest(DraftClipRequest):
    split_time: float = Field(gt=0)


class DraftAudioRequest(DraftClipRequest):
    muted: bool | None = None
    volume: float | None = None


class CaptionEdit(BaseModel):
    clip_id: str
    text: str


class DraftCaptionRequest(BaseModel):
    draft: dict[str, Any]
    edits: list[CaptionEdit] = Field(min_length=1)


class DraftCaptionSettingsRequest(BaseModel):
    draft: dict[str, Any]
    enabled: bool | None = None
    preset: str | None = None


def _draft_edit_error(exc: DraftEditError):
    raise HTTPException(status_code=409, detail=str(exc)) from None


def _draft_history_error(exc: Exception):
    raise HTTPException(status_code=409, detail=str(exc)) from None


def _hydrate_snapshot_assets(snapshot: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(snapshot)
    hydrated_sources = []
    for source in out.get("sources") or ():
        item = dict(source)
        assets = dict(item.get("timeline_assets") or {})
        if assets:
            try:
                item["timeline_assets"] = sign_timeline_assets(assets)
            except Exception as exc:
                item["timeline_assets"] = {
                    **assets,
                    "signed_url_status": "degraded",
                    "signed_url_reason": exc.__class__.__name__,
                }
        hydrated_sources.append(item)
    out["sources"] = hydrated_sources
    return out


@app.get("/v1/healthz")
def healthz():
    config = load_runtime_config()
    return {
        "ok": True,
        "service": "cutsell-api",
        "version": "0.1.0",
        "queue_ready": config.queue_ready,
        "storage_ready": config.storage_ready,
        "semantic_ready": config.semantic_ready,
    }


@app.post("/v1/uploads/presign", response_model=UploadPresignResponse)
def presign_upload(payload: UploadPresignRequest):
    try:
        result = create_presigned_upload(
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
    return UploadPresignResponse(**result)


@app.post("/v1/flow-b/jobs", response_model=FlowBSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_flow_b(payload: FlowBSubmitRequest):
    sources = []
    seen_orders = set()
    for item in payload.sources:
        if item.source_order in seen_orders:
            raise HTTPException(status_code=409, detail="source_order values must be unique")
        seen_orders.add(item.source_order)
        try:
            validate_product_source_uri(item.uri, project_id=payload.project_id, user_id=payload.user_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        sources.append({
            "source_asset_id": stable_source_id(payload.project_id, item.source_order, item.original_name),
            "original_name": item.original_name,
            "source_order": item.source_order,
            "duration_sec": item.duration_sec,
            "uri": item.uri,
        })
    submission = enqueue_flow_b({
        "project_id": payload.project_id,
        "user_id": payload.user_id,
        "sources": sources,
        "language_hint": payload.language_hint,
        "audio_overlap": payload.audio_overlap,
    })
    return FlowBSubmitResponse(job_id=submission.job_id, queue=submission.queue_name)


@app.get("/v1/projects/{project_id}/draft")
def recover_draft(project_id: str, user_id: str):
    try:
        return _hydrate_snapshot_assets(get_draft_snapshot(user_id=user_id, project_id=project_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="draft not found") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@app.put("/v1/projects/{project_id}/draft")
def autosave_draft(project_id: str, payload: DraftAutosaveRequest):
    try:
        draft_from_dict(payload.draft)
        return _hydrate_snapshot_assets(save_draft_snapshot(
            user_id=payload.user_id,
            project_id=project_id,
            draft=payload.draft,
            expected_revision=payload.expected_revision,
        ))
    except DraftConflictError as exc:
        _draft_history_error(exc)
    except KeyError:
        raise HTTPException(status_code=404, detail="draft not found") from None
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@app.post("/v1/projects/{project_id}/draft/undo")
def undo_draft(project_id: str, payload: DraftHistoryRequest):
    try:
        return _hydrate_snapshot_assets(undo_draft_snapshot(
            user_id=payload.user_id, project_id=project_id, expected_revision=payload.expected_revision
        ))
    except (DraftConflictError, DraftHistoryError) as exc:
        _draft_history_error(exc)
    except KeyError:
        raise HTTPException(status_code=404, detail="draft not found") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@app.post("/v1/projects/{project_id}/draft/redo")
def redo_draft(project_id: str, payload: DraftHistoryRequest):
    try:
        return _hydrate_snapshot_assets(redo_draft_snapshot(
            user_id=payload.user_id, project_id=project_id, expected_revision=payload.expected_revision
        ))
    except (DraftConflictError, DraftHistoryError) as exc:
        _draft_history_error(exc)
    except KeyError:
        raise HTTPException(status_code=404, detail="draft not found") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@app.post("/v1/exports/jobs", response_model=ExportSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_export(payload: ExportSubmitRequest):
    try:
        draft = draft_from_dict(payload.draft)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid draft: {exc}") from None
    if draft.project_id != payload.project_id:
        raise HTTPException(status_code=409, detail="draft project does not match export project")
    seen_source_ids = set()
    sources = []
    for item in payload.sources:
        if not item.source_asset_id:
            raise HTTPException(status_code=422, detail="source_asset_id is required")
        if item.source_asset_id in seen_source_ids:
            raise HTTPException(status_code=409, detail="source_asset_id values must be unique")
        seen_source_ids.add(item.source_asset_id)
        try:
            validate_product_source_uri(item.uri, project_id=payload.project_id, user_id=payload.user_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        sources.append(item.model_dump())
    required_source_ids = {clip.source_asset_id for clip in draft.selected}
    if not required_source_ids.issubset(seen_source_ids):
        raise HTTPException(status_code=422, detail="export is missing selected source assets")
    submission = enqueue_export({
        "project_id": payload.project_id,
        "user_id": payload.user_id,
        "draft": payload.draft,
        "sources": sources,
    })
    return ExportSubmitResponse(job_id=submission.job_id, queue=submission.queue_name)


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    try:
        snapshot = fetch_job_snapshot(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found") from None
    return JobStatusResponse(**snapshot.__dict__)


@app.post("/v1/jobs/{job_id}/cancel", response_model=JobStatusResponse)
def cancel_processing_job(job_id: str):
    try:
        snapshot = cancel_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found") from None
    return JobStatusResponse(**snapshot.__dict__)


@app.post("/v1/draft-edits/swap")
def edit_swap_take(payload: DraftSwapRequest):
    try:
        return swap_take(payload.draft, payload.selected_clip_id, payload.replacement_clip_id)
    except DraftEditError as exc:
        _draft_edit_error(exc)


@app.post("/v1/draft-edits/remove")
def edit_remove_clip(payload: DraftClipRequest):
    try:
        return remove_clip(payload.draft, payload.clip_id)
    except DraftEditError as exc:
        _draft_edit_error(exc)


@app.post("/v1/draft-edits/restore")
def edit_restore_clip(payload: DraftRestoreRequest):
    try:
        return restore_clip(payload.draft, payload.clip_id, payload.position)
    except DraftEditError as exc:
        _draft_edit_error(exc)


@app.post("/v1/draft-edits/reorder")
def edit_reorder_clips(payload: DraftReorderRequest):
    try:
        return reorder_clips(payload.draft, payload.ordered_clip_ids)
    except DraftEditError as exc:
        _draft_edit_error(exc)


@app.post("/v1/draft-edits/trim")
def edit_trim_clip(payload: DraftTrimRequest):
    try:
        return trim_clip(payload.draft, payload.clip_id, start=payload.start, end=payload.end)
    except DraftEditError as exc:
        _draft_edit_error(exc)


@app.post("/v1/draft-edits/split")
def edit_split_clip(payload: DraftSplitRequest):
    try:
        return split_clip(payload.draft, payload.clip_id, split_time=payload.split_time)
    except DraftEditError as exc:
        _draft_edit_error(exc)


@app.post("/v1/draft-edits/audio")
def edit_audio(payload: DraftAudioRequest):
    try:
        return patch_audio(payload.draft, payload.clip_id, muted=payload.muted, volume=payload.volume)
    except DraftEditError as exc:
        _draft_edit_error(exc)


@app.post("/v1/draft-edits/captions")
def edit_captions(payload: DraftCaptionRequest):
    try:
        return patch_captions(payload.draft, [edit.model_dump() for edit in payload.edits])
    except DraftEditError as exc:
        _draft_edit_error(exc)


@app.post("/v1/draft-edits/caption-settings")
def edit_caption_settings(payload: DraftCaptionSettingsRequest):
    try:
        return patch_caption_settings(payload.draft, enabled=payload.enabled, preset=payload.preset)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
