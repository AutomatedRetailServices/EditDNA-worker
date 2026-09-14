"""Mobile Timeline API Bridge -- D-282.

Exposes the D-279/D-281 timeline asset + composition service contracts
through a tenant-safe FastAPI surface, using this app's own established
conventions (`project_routes.py`/`multipart_routes.py`/`overlay_routes.py`):
a Pydantic request model carrying `user_id` (validated against the
authenticated principal by the existing, UNMODIFIED `AuthScopeMiddleware`,
exactly like every other route in this file's own siblings), bounded
`HTTPException` mapping, never a raw exception leaked.

## Client asset authority (Stage 4/17/23, binding)

The client references assets and timelines ONLY by server-generated
`asset_id`/`revision_identity` -- never a filesystem path, S3 URI, or
object key. In particular, `base_edit_asset_id` (both on save and on
export) is an ordinary timeline asset (`role=PRIMARY_SOURCE`) ingested
through the SAME `/timeline-assets` route as any B-roll/VO asset; this
route resolves it server-side into `TimelineComposition.base_edit_
identity` (a content-derived opaque string) at save time, and re-
resolves + cross-checks it against the ALREADY-PERSISTED identity at
export time (`_BASE_EDIT_ASSET_MISMATCH`) so a client can never swap in
a different base media than what was actually saved.

## Storage wiring (Stage 31/36, unchanged from D-281)

`_durable_media_dir()` requires an explicit `CUTSELL_TIMELINE_ASSET_
DURABLE_DIR` environment variable -- no silent tempdir default (D-281's
own Stage 34 invariant, restated at the API layer). Tests set this to a
`tmp_path`-backed fake/local durable directory; no real S3 mutation
happens anywhere in this module.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from cutsell_worker import timeline_asset_registry as reg
from cutsell_worker import timeline_asset_registry_store as store
from cutsell_worker import timeline_asset_upload_bridge as bridge
from cutsell_worker import timeline_composition as tc
from cutsell_worker import timeline_composition_executor as tce
from cutsell_worker.project_store import get_project

router = APIRouter(prefix="/v1/projects", tags=["timeline"])

BASE_EDIT_ASSET_MISMATCH = "BASE_EDIT_ASSET_MISMATCH"

# Stage 28: this route's own single additive outcome beyond D-279's 8 --
# never added to D-281's own `store.SERVICE_ERROR_STATUS` (which stays
# untouched), kept local to this route layer instead.
_ROUTE_ERROR_STATUS: dict[str, int] = {
    BASE_EDIT_ASSET_MISMATCH: 409,
}


# =============================================================================
# Storage wiring seam (Stage 31/36) -- fake/local storage this gate's own
# tests exercise; a real production wiring supplies a real S3-backed
# `persist_media`/`materialize` pair through the SAME injection seam.
# =============================================================================

def _durable_media_dir() -> str:
    directory = os.environ.get("CUTSELL_TIMELINE_ASSET_DURABLE_DIR")
    if not directory:
        raise RuntimeError("CUTSELL_TIMELINE_ASSET_DURABLE_DIR is required for timeline asset ingest")
    return directory


def _persist_media():
    return store.local_directory_persister(_durable_media_dir())


# =============================================================================
# Request/response models (Stage 25/26/27)
# =============================================================================

class TimelineAssetCreateRequest(BaseModel):
    user_id: str
    role: str
    media_kind: str
    source_uri: str
    replaces_asset_id: str | None = None

    @field_validator("role")
    @classmethod
    def _valid_role(cls, value: str) -> str:
        try:
            reg.TimelineAssetRole(value)
        except ValueError:
            raise ValueError(f"unknown asset role: {value!r}") from None
        return value

    @field_validator("media_kind")
    @classmethod
    def _valid_media_kind(cls, value: str) -> str:
        try:
            reg.TimelineMediaKind(value)
        except ValueError:
            raise ValueError(f"unknown media_kind: {value!r}") from None
        return value


class TimelineAssetResponse(BaseModel):
    asset_id: str
    role: str
    media_kind: str
    duration_sec: float
    has_audio: bool
    qualification_status: str
    replaces_asset_id: str | None = None
    created_at: str | None = None


class TimelineAssetListResponse(BaseModel):
    project_id: str
    assets: list[TimelineAssetResponse]


class BrollPlacementModel(BaseModel):
    placement_id: str
    asset_id: str
    timeline_start_sec: float = Field(ge=0)
    timeline_end_sec: float = Field(gt=0)
    source_in_sec: float = Field(ge=0)
    source_out_sec: float = Field(gt=0)
    audio_mode: str = tc.TimelineAudioMode.KEEP_PRIMARY_VOICE.value

    @field_validator("audio_mode")
    @classmethod
    def _valid_audio_mode(cls, value: str) -> str:
        try:
            tc.TimelineAudioMode(value)
        except ValueError:
            raise ValueError(f"unknown audio_mode: {value!r}") from None
        return value


class VoiceOverPlacementModel(BaseModel):
    placement_id: str
    asset_id: str
    timeline_start_sec: float = Field(ge=0)
    timeline_end_sec: float = Field(gt=0)
    source_in_sec: float = Field(ge=0)
    source_out_sec: float = Field(gt=0)
    transcript_reference: str | None = None


class TimelineGetResponse(BaseModel):
    contract_version: int
    base_edit_identity: str
    timeline_duration_sec: float
    timeline_revision_identity: str
    broll_placements: list[BrollPlacementModel]
    voice_over_placements: list[VoiceOverPlacementModel]


class TimelineSaveRequest(BaseModel):
    user_id: str
    base_edit_asset_id: str
    timeline_duration_sec: float = Field(gt=0)
    broll_placements: list[BrollPlacementModel] = Field(default_factory=list)
    voice_over_placements: list[VoiceOverPlacementModel] = Field(default_factory=list)
    expected_revision_identity: str | None = None


class TimelineSaveResponse(BaseModel):
    outcome: str
    timeline_revision_identity: str | None = None
    reasons: list[str] = Field(default_factory=list)


class TimelineExportRequest(BaseModel):
    user_id: str
    revision_identity: str
    base_edit_asset_id: str


class TimelineExportResponse(BaseModel):
    outcome: str
    output_path: str | None = None
    format_qc_status: str | None = None
    reasons: list[str] = Field(default_factory=list)


# =============================================================================
# Shared helpers (Stage 2/3/28) -- project ownership + bounded error mapping
# =============================================================================

def _require_project(*, user_id: str, project_id: str) -> None:
    """Stage 2/3: authorization before ANY asset/timeline lookup. Reuses
    the existing `project_store.get_project` ownership check verbatim
    (the same one every other route in this app already relies on) --
    never a new project-ownership mechanism."""
    try:
        get_project(user_id=user_id, project_id=project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="project not found") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


def _bounded_error(outcome: str, reasons: tuple[str, ...] = ()) -> None:
    """Stage 28: maps a D-279/D-281 (or this bridge's own additive
    `BASE_EDIT_ASSET_MISMATCH`) outcome to a bounded HTTP response --
    never a raw exception/stack trace."""
    status_code = _ROUTE_ERROR_STATUS.get(outcome) or store.map_outcome_to_http_status(outcome)
    raise HTTPException(status_code=status_code, detail={"outcome": outcome, "reasons": list(reasons)})


def _resolve_base_edit_identity(*, user_id: str, project_id: str, asset_id: str) -> str:
    """Stage 17/23: the base edit is resolved SERVER-SIDE from an
    already-owned, already-READY `PRIMARY_SOURCE` timeline asset -- the
    client only ever supplies its `asset_id`. Reuses D-279's own
    `authorize_asset_access` directly (no new ownership mechanism)."""
    asset = store.get_timeline_asset(user_id=user_id, project_id=project_id, asset_id=asset_id)
    if asset is None or asset.qualification_status == reg.TimelineAssetQualificationStatus.DELETED:
        _bounded_error(reg.ASSET_NOT_FOUND)
    requesting = reg.AssetOwnershipScope(user_id=user_id, project_id=project_id)
    if not reg.authorize_asset_access(requesting=requesting, record_ownership=asset.ownership):
        _bounded_error(reg.ASSET_NOT_OWNED)
    if asset.qualification_status != reg.TimelineAssetQualificationStatus.READY:
        _bounded_error(reg.ASSET_NOT_READY)
    if asset.media_kind != reg.TimelineMediaKind.VIDEO:
        _bounded_error(reg.ASSET_MEDIA_UNSUPPORTED)
    return asset.source_media_identity


def _asset_to_response(asset: reg.TimelineMediaAsset) -> TimelineAssetResponse:
    view = reg.client_safe_asset_view(asset)
    return TimelineAssetResponse(**view)


# =============================================================================
# Routes
# =============================================================================

@router.post("/{project_id}/timeline-assets", response_model=TimelineAssetResponse)
def create_timeline_asset(project_id: str, payload: TimelineAssetCreateRequest):
    _require_project(user_id=payload.user_id, project_id=project_id)
    role = reg.TimelineAssetRole(payload.role)
    media_kind = reg.TimelineMediaKind(payload.media_kind)
    try:
        if media_kind == reg.TimelineMediaKind.AUDIO or role == reg.TimelineAssetRole.VOICE_OVER:
            asset = bridge.ingest_voice_over_asset_from_upload(
                user_id=payload.user_id, project_id=project_id, source_uri=payload.source_uri,
                persist_media=_persist_media(), replaces_asset_id=payload.replaces_asset_id,
            )
        else:
            asset = bridge.ingest_broll_asset_from_upload(
                user_id=payload.user_id, project_id=project_id, role=role, source_uri=payload.source_uri,
                persist_media=_persist_media(), normalization_output_dir=_durable_media_dir(),
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return _asset_to_response(asset)


@router.get("/{project_id}/timeline-assets", response_model=TimelineAssetListResponse)
def list_timeline_assets(project_id: str, user_id: str):
    _require_project(user_id=user_id, project_id=project_id)
    assets = store.list_timeline_assets(user_id=user_id, project_id=project_id)
    return TimelineAssetListResponse(
        project_id=project_id, assets=[_asset_to_response(asset) for asset in assets],
    )


@router.get("/{project_id}/timeline", response_model=TimelineGetResponse)
def get_timeline(project_id: str, user_id: str):
    _require_project(user_id=user_id, project_id=project_id)
    record = store.get_timeline_revision(user_id=user_id, project_id=project_id)
    if record is None:
        raise HTTPException(status_code=404, detail="timeline not found")
    view = store.client_safe_timeline_view(record)
    return TimelineGetResponse(**view)


@router.put("/{project_id}/timeline", response_model=TimelineSaveResponse)
def save_timeline(project_id: str, payload: TimelineSaveRequest):
    _require_project(user_id=payload.user_id, project_id=project_id)
    base_edit_identity = _resolve_base_edit_identity(
        user_id=payload.user_id, project_id=project_id, asset_id=payload.base_edit_asset_id,
    )

    assets_by_id = {
        asset.asset_id: asset
        for asset in store.list_timeline_assets(user_id=payload.user_id, project_id=project_id, include_deleted=True)
    }

    def _resolve_reference(asset_id: str) -> tc.TimelineAssetReference:
        asset = assets_by_id.get(asset_id)
        if asset is None:
            # A missing asset still needs a placeholder reference so
            # D-279's own registry validation (not this route) reports
            # the real ASSET_NOT_FOUND outcome -- never guessed here.
            return tc.TimelineAssetReference(asset_id=asset_id, source_media_identity="unknown", duration_sec=0.001)
        return reg.to_asset_reference(asset)

    broll_placements = tuple(
        tc.BrollPlacement(
            placement_id=p.placement_id, asset=_resolve_reference(p.asset_id),
            timeline_start_sec=p.timeline_start_sec, timeline_end_sec=p.timeline_end_sec,
            source_in_sec=p.source_in_sec, source_out_sec=p.source_out_sec,
            audio_mode=tc.TimelineAudioMode(p.audio_mode),
        )
        for p in payload.broll_placements
    )
    voice_over_placements = tuple(
        tc.VoiceOverPlacement(
            placement_id=p.placement_id, asset=_resolve_reference(p.asset_id),
            timeline_start_sec=p.timeline_start_sec, timeline_end_sec=p.timeline_end_sec,
            source_in_sec=p.source_in_sec, source_out_sec=p.source_out_sec,
            transcript_reference=p.transcript_reference,
        )
        for p in payload.voice_over_placements
    )
    composition = tc.TimelineComposition(
        contract_version=tc.TIMELINE_CONTRACT_VERSION, base_edit_identity=base_edit_identity,
        timeline_duration_sec=payload.timeline_duration_sec,
        broll_placements=broll_placements, voice_over_placements=voice_over_placements,
    )
    result = store.save_timeline(
        user_id=payload.user_id, project_id=project_id, composition=composition,
        expected_revision_identity=payload.expected_revision_identity,
    )
    if result.outcome != reg.TIMELINE_SAVE_SUCCEEDED:
        _bounded_error(result.outcome, result.reason_codes)
    return TimelineSaveResponse(
        outcome=result.outcome, timeline_revision_identity=result.record.timeline_revision_identity,
    )


@router.post("/{project_id}/timeline/export", response_model=TimelineExportResponse)
def export_timeline(project_id: str, payload: TimelineExportRequest):
    _require_project(user_id=payload.user_id, project_id=project_id)
    record = store.get_timeline_revision(user_id=payload.user_id, project_id=project_id)
    if record is None:
        _bounded_error(reg.TIMELINE_INVALID, ("no timeline saved for this project",))

    # Stage 17/23/24: re-resolve the SAME base_edit_asset_id server-side
    # and cross-check its own content identity against what was actually
    # persisted at save time -- a client cannot swap in a different base
    # media at export time than what the saved revision itself named.
    base_asset = store.get_timeline_asset(
        user_id=payload.user_id, project_id=project_id, asset_id=payload.base_edit_asset_id,
    )
    if base_asset is None or base_asset.qualification_status == reg.TimelineAssetQualificationStatus.DELETED:
        _bounded_error(reg.ASSET_NOT_FOUND)
    if base_asset.source_media_identity != record.composition.base_edit_identity:
        _bounded_error(BASE_EDIT_ASSET_MISMATCH, (
            "base_edit_asset_id does not match the base edit identity recorded in the saved revision",
        ))

    base_resolution = store.resolve_timeline_asset_live(
        user_id=payload.user_id, project_id=project_id, asset_id=payload.base_edit_asset_id,
        required_media_kind=reg.TimelineMediaKind.VIDEO,
    )
    if base_resolution.outcome != reg.ASSET_RESOLVED or base_resolution.resolved is None:
        _bounded_error(base_resolution.outcome, base_resolution.reason_codes)
    base_edit_asset = base_resolution.resolved

    result = store.export_timeline_revision(
        user_id=payload.user_id, project_id=project_id, revision_identity=payload.revision_identity,
        base_edit_asset=base_edit_asset, output_directory=_durable_media_dir(),
    )
    if isinstance(result, tce.CompositionExecutionResult):
        if result.outcome not in (tce.COMPOSITION_SUCCEEDED, tce.BASE_ONLY_BYPASS):
            _bounded_error(result.outcome, tuple(result.errors))
        return TimelineExportResponse(
            outcome=result.outcome, output_path=result.output_path,
            format_qc_status=result.diagnostics.get("format_qc_status"),
        )
    if isinstance(result, reg.TimelineExportRequestResult):
        _bounded_error(result.outcome, result.reason_codes)
    # AssetResolutionResult / TimelineRenderPlanResult -- any other
    # bounded, named outcome from the D-278 bridge.
    outcome = getattr(result, "outcome", "OTHER")
    reasons = getattr(result, "reason_codes", None) or getattr(result, "errors", ())
    _bounded_error(outcome, tuple(reasons))
