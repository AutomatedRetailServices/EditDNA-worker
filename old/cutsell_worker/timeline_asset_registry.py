"""V1 Manual Timeline Asset Ingest / Persistence Contract -- D-279.

Post D-278 (composition executor, Verdict A: the backend can execute the
D-277 `TimelineComposition` contract against already-resolved assets).
Every test in D-278's own suite had to hand-construct a `ResolvedTimeline
Asset` -- there was no durable, authorized notion of "this asset_id belongs
to this user, is qualified, and is safe to place on a timeline." This
module is that missing middle layer.

Desired architecture (restated from the D-279 directive; this gate defines
the CONTRACT, never the wiring):

    MOBILE / API -> authorized asset upload -> media qualification ->
    persistent TimelineAsset record -> timeline references asset_id ->
    secure resolver (THIS MODULE) -> D-278 ResolvedTimelineAsset ->
    composition executor.

Pure types and pure functions only, matching D-277/D-278's own established
discipline for anything that is not literally the render/composition step
itself: no I/O, no S3, no Redis, no ffmpeg, no microphone capture, no
mobile UI, no AI B-roll ranking/suggestion. Persistence wiring (a real
Redis-backed store, real presigned uploads, real FastAPI routes) is a
separate, later, explicitly-authorized integration gate -- exactly how
`timeline_composition.py` defined D-277's contract before D-278 executed
it against real media.

## Ownership -- reused, never reinvented

`tenant_safe_delivery.py` (D-269) already solved "who may touch this
record" for render delivery via `DeliveryOwnershipScope` + `authorize_
delivery_access` + `assert_delivery_access`, and permanently refused to
treat an S3 ETag as a SHA-256 proxy (`is_etag_valid_sha256_proxy`). A
`TimelineMediaAsset` is not job-scoped the way a rendered delivery is --
one asset is uploaded once and may be referenced by many later timeline
revisions and exports, so `DeliveryOwnershipScope`'s mandatory non-empty
`job_id` field does not honestly describe it. Rather than force a
synthetic job_id onto every asset, this module defines `AssetOwnershipScope`
(user_id + project_id only) as the asset-shaped sibling of the SAME
doctrine: same frozen-dataclass/non-empty-fields shape, same full-scope-
equality authorization rule, same `None`-requesting-means-no-authenticated-
principal convention as `assert_delivery_access`. `tenant_safe_delivery.
is_etag_valid_sha256_proxy` is imported and reused DIRECTLY here (never
redefined) via `validate_content_sha256_source`.

V1 policy (recorded here, a Product Owner-reviewable default, not a hidden
assumption): assets are PROJECT-SCOPED ONLY -- no cross-project asset
reuse. This falls out structurally from `AssetOwnershipScope` equality
(user_id AND project_id must both match) rather than from a separate
special-case check, so cross-project isolation cannot be forgotten at one
call site and enforced at another.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from . import timeline_composition as tc
from . import timeline_composition_executor as tce
from .tenant_safe_delivery import is_etag_valid_sha256_proxy

TIMELINE_ASSET_REGISTRY_CONTRACT_VERSION = 1

# =============================================================================
# Error vocabulary -- fixed, closed set (never proliferate ad hoc strings)
# =============================================================================

ASSET_NOT_FOUND = "ASSET_NOT_FOUND"
ASSET_NOT_OWNED = "ASSET_NOT_OWNED"
ASSET_NOT_READY = "ASSET_NOT_READY"
ASSET_MEDIA_UNSUPPORTED = "ASSET_MEDIA_UNSUPPORTED"
ASSET_HAS_NO_AUDIO = "ASSET_HAS_NO_AUDIO"
TIMELINE_REVISION_CONFLICT = "TIMELINE_REVISION_CONFLICT"
TIMELINE_INVALID = "TIMELINE_INVALID"
ASSET_REFERENCED = "ASSET_REFERENCED"

# Success outcomes (distinct from the error vocabulary above, never confused
# with it -- every result dataclass below carries an `outcome` that is
# EITHER one of these OR one of the errors, never a bare bool).
ASSET_RESOLVED = "ASSET_RESOLVED"
ASSET_DELETE_SUCCEEDED = "ASSET_DELETE_SUCCEEDED"
TIMELINE_SAVE_SUCCEEDED = "TIMELINE_SAVE_SUCCEEDED"
TIMELINE_EXPORT_RESOLVED = "TIMELINE_EXPORT_RESOLVED"


# =============================================================================
# Enums
# =============================================================================

class TimelineAssetRole(str, Enum):
    PRIMARY_SOURCE = "PRIMARY_SOURCE"
    SUPPLEMENTAL_BROLL = "SUPPLEMENTAL_BROLL"
    VOICE_OVER = "VOICE_OVER"


class TimelineMediaKind(str, Enum):
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"


class TimelineAssetQualificationStatus(str, Enum):
    UPLOADED = "UPLOADED"
    QUALIFYING = "QUALIFYING"
    READY = "READY"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    DELETED = "DELETED"


# =============================================================================
# Ownership -- the asset-shaped sibling of D-269's DeliveryOwnershipScope
# =============================================================================

@dataclass(frozen=True)
class AssetOwnershipScope:
    user_id: str
    project_id: str

    def __post_init__(self) -> None:
        for name in ("user_id", "project_id"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"AssetOwnershipScope.{name} must be non-empty")


def authorize_asset_access(*, requesting: AssetOwnershipScope, record_ownership: AssetOwnershipScope) -> bool:
    """Pure full-scope-equality decision -- no ID alone (not user_id, not
    project_id) grants access; the WHOLE scope must match, mirroring
    D-269's own `authorize_delivery_access`."""
    return requesting == record_ownership


def assert_asset_access(*, requesting: AssetOwnershipScope | None, record_ownership: AssetOwnershipScope) -> None:
    """`requesting=None` means no authenticated principal is available at
    all (auth disabled for local dev) -- mirrors D-269's own `assert_
    delivery_access(requesting=None)` precedent of skipping the check only
    when there is genuinely no identity to compare against, never when
    there IS one and it disagrees."""
    if requesting is None:
        return
    if not authorize_asset_access(requesting=requesting, record_ownership=record_ownership):
        raise PermissionError("requesting principal does not own this asset")


def validate_content_sha256_source(value: str | None, *, derived_from_etag: bool) -> str | None:
    """The one gate a future ingest pipeline must call before storing
    `TimelineMediaAsset.content_sha256`. `derived_from_etag=True` is
    refused unconditionally -- an S3 ETag is never an acceptable source
    for this field. `tenant_safe_delivery.is_etag_valid_sha256_proxy`
    (permanently `False`) is asserted alongside it as the same doctrine
    stated structurally, reused rather than redefined."""
    if derived_from_etag:
        raise ValueError("content_sha256 must never be derived from an S3 ETag")
    assert is_etag_valid_sha256_proxy(value) is False
    return value


# =============================================================================
# The persisted asset record
# =============================================================================

@dataclass(frozen=True)
class TimelineMediaAsset:
    asset_id: str
    ownership: AssetOwnershipScope
    role: TimelineAssetRole
    media_kind: TimelineMediaKind
    source_media_identity: str
    storage_reference: str
    duration_sec: float
    has_audio: bool
    qualification_status: TimelineAssetQualificationStatus = TimelineAssetQualificationStatus.UPLOADED
    content_sha256: str | None = None
    technical_metadata_reference: str | None = None
    replaces_asset_id: str | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        if not str(self.asset_id or "").strip():
            raise ValueError("TimelineMediaAsset.asset_id must be non-empty")
        if not str(self.source_media_identity or "").strip():
            raise ValueError("TimelineMediaAsset.source_media_identity must be non-empty")
        if not str(self.storage_reference or "").strip():
            raise ValueError("TimelineMediaAsset.storage_reference must be non-empty")
        if float(self.duration_sec) <= 0:
            raise ValueError("TimelineMediaAsset.duration_sec must be positive")


def to_asset_reference(asset: TimelineMediaAsset) -> tc.TimelineAssetReference:
    """The library record's own D-277 `TimelineAssetReference` projection --
    identity + duration only, exactly what a `BrollPlacement`/`VoiceOver
    Placement` is allowed to carry. Never includes `storage_reference`: a
    timeline composition names an asset_id, never a raw path or S3 key."""
    return tc.TimelineAssetReference(
        asset_id=asset.asset_id,
        source_media_identity=asset.source_media_identity,
        duration_sec=asset.duration_sec,
    )


def client_safe_asset_view(asset: TimelineMediaAsset) -> dict:
    """The client-safe projection of a `TimelineMediaAsset` -- never
    includes `storage_reference` (an opaque server-side S3 key/path) or
    `technical_metadata_reference`; a client needs only enough to render
    an asset picker and issue placement operations by `asset_id`."""
    return {
        "asset_id": asset.asset_id,
        "role": asset.role.value,
        "media_kind": asset.media_kind.value,
        "duration_sec": asset.duration_sec,
        "has_audio": asset.has_audio,
        "qualification_status": asset.qualification_status.value,
        "replaces_asset_id": asset.replaces_asset_id,
        "created_at": asset.created_at,
    }


def list_project_assets(
    *,
    requesting: AssetOwnershipScope,
    assets: tuple[TimelineMediaAsset, ...],
    include_deleted: bool = False,
) -> tuple[TimelineMediaAsset, ...]:
    """Project asset listing -- filters to exactly the requesting scope's
    own assets (cross-user and cross-project isolation both fall out of
    the same `ownership == requesting` equality check), excluding DELETED
    entries by default."""
    return tuple(
        asset
        for asset in assets
        if asset.ownership == requesting
        and (include_deleted or asset.qualification_status != TimelineAssetQualificationStatus.DELETED)
    )


def reconcile_asset_duration(*, client_reported_duration_sec: float, server_probed_duration_sec: float) -> float:
    """The server-probed duration is the ONLY authority a persisted
    `TimelineMediaAsset.duration_sec` may ever be built from. A client-
    reported value is accepted as a parameter here only so a caller can
    log/compare the discrepancy -- it is never trusted, matching this
    codebase's standing "server always re-derives, never trusts client-
    declared media facts" convention."""
    del client_reported_duration_sec  # logged by the caller, never authoritative
    return float(server_probed_duration_sec)


# =============================================================================
# The secure resolver -- TimelineMediaAsset -> D-278 ResolvedTimelineAsset
# =============================================================================

@dataclass(frozen=True)
class AssetResolutionResult:
    outcome: str
    resolved: tce.ResolvedTimelineAsset | None = None
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


def resolve_timeline_asset(
    *,
    requesting: AssetOwnershipScope,
    asset: TimelineMediaAsset | None,
    local_path: str,
    required_media_kind: TimelineMediaKind,
    required_audio_mode: "tc.TimelineAudioMode | None" = None,
) -> AssetResolutionResult:
    """The ONE seam between a persisted `TimelineMediaAsset` and a D-278
    `ResolvedTimelineAsset`. Pure: `local_path` is supplied by the
    caller's own already-completed materialization step (download/cache)
    -- this function never touches a filesystem or network itself,
    matching D-277/D-278's own "pure types, pure functions" discipline for
    anything that is not literally the render/composition step. A DELETED
    asset resolves exactly like a missing one (Stage: deletion must not
    be distinguishable from never-having-existed to an unauthorized or
    stale caller)."""
    if asset is None or asset.qualification_status == TimelineAssetQualificationStatus.DELETED:
        return AssetResolutionResult(outcome=ASSET_NOT_FOUND, reason_codes=(ASSET_NOT_FOUND,))
    if not authorize_asset_access(requesting=requesting, record_ownership=asset.ownership):
        return AssetResolutionResult(outcome=ASSET_NOT_OWNED, reason_codes=(ASSET_NOT_OWNED,))
    if asset.qualification_status != TimelineAssetQualificationStatus.READY:
        return AssetResolutionResult(outcome=ASSET_NOT_READY, reason_codes=(ASSET_NOT_READY,))
    if asset.media_kind != required_media_kind:
        return AssetResolutionResult(outcome=ASSET_MEDIA_UNSUPPORTED, reason_codes=(ASSET_MEDIA_UNSUPPORTED,))
    if required_audio_mode == tc.TimelineAudioMode.USE_BROLL_AUDIO and not asset.has_audio:
        return AssetResolutionResult(outcome=ASSET_HAS_NO_AUDIO, reason_codes=(ASSET_HAS_NO_AUDIO,))
    if not str(local_path or "").strip():
        return AssetResolutionResult(outcome=ASSET_NOT_FOUND, reason_codes=(ASSET_NOT_FOUND,))
    resolved = tce.ResolvedTimelineAsset(
        asset_id=asset.asset_id,
        local_path=local_path,
        duration_sec=asset.duration_sec,
        has_audio=asset.has_audio,
    )
    return AssetResolutionResult(outcome=ASSET_RESOLVED, resolved=resolved)


# =============================================================================
# Registry-aware timeline validation -- the layer D-277's own
# `validate_composition` structurally cannot perform (it has no asset
# registry to consult; it only ever sees `TimelineAssetReference`, never
# ownership or qualification state)
# =============================================================================

@dataclass(frozen=True)
class TimelineRegistryValidationResult:
    valid: bool
    errors: tuple[str, ...] = field(default_factory=tuple)


def validate_timeline_against_registry(
    *,
    requesting: AssetOwnershipScope,
    composition: tc.TimelineComposition,
    assets_by_id: dict[str, TimelineMediaAsset],
) -> TimelineRegistryValidationResult:
    """A save/export handler must run this BEFORE `save_timeline_revision`/
    `resolve_timeline_export`: every referenced asset_id must exist, be
    owned by the SAME requesting scope, be READY, and -- for a `USE_
    BROLL_AUDIO` B-roll placement -- actually have audio (the has-audio
    pre-validation belongs at timeline-save time, distinct from D-278's
    own graceful silent-fallback for an asset that is unexpectedly silent
    at execution time)."""
    errors: list[str] = []
    for placement in composition.broll_placements:
        asset = assets_by_id.get(placement.asset.asset_id)
        if asset is None or asset.qualification_status == TimelineAssetQualificationStatus.DELETED:
            errors.append(f"{ASSET_NOT_FOUND}:{placement.placement_id}")
            continue
        if asset.ownership != requesting:
            errors.append(f"{ASSET_NOT_OWNED}:{placement.placement_id}")
            continue
        if asset.qualification_status != TimelineAssetQualificationStatus.READY:
            errors.append(f"{ASSET_NOT_READY}:{placement.placement_id}")
        if placement.audio_mode == tc.TimelineAudioMode.USE_BROLL_AUDIO and not asset.has_audio:
            errors.append(f"{ASSET_HAS_NO_AUDIO}:{placement.placement_id}")
    for placement in composition.voice_over_placements:
        asset = assets_by_id.get(placement.asset.asset_id)
        if asset is None or asset.qualification_status == TimelineAssetQualificationStatus.DELETED:
            errors.append(f"{ASSET_NOT_FOUND}:{placement.placement_id}")
            continue
        if asset.ownership != requesting:
            errors.append(f"{ASSET_NOT_OWNED}:{placement.placement_id}")
            continue
        if asset.qualification_status != TimelineAssetQualificationStatus.READY:
            errors.append(f"{ASSET_NOT_READY}:{placement.placement_id}")
    return TimelineRegistryValidationResult(valid=not errors, errors=tuple(errors))


# =============================================================================
# Asset delete vs placement delete
# =============================================================================

def is_asset_referenced(*, asset_id: str, composition: tc.TimelineComposition) -> bool:
    for placement in composition.broll_placements:
        if placement.asset.asset_id == asset_id:
            return True
    for placement in composition.voice_over_placements:
        if placement.asset.asset_id == asset_id:
            return True
    return False


@dataclass(frozen=True)
class AssetDeleteResult:
    outcome: str
    asset: TimelineMediaAsset | None = None
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


def delete_timeline_asset(
    *,
    requesting: AssetOwnershipScope,
    asset: TimelineMediaAsset | None,
    referencing_compositions: tuple[tc.TimelineComposition, ...] = (),
) -> AssetDeleteResult:
    """Deleting an ASSET (the library entry) is distinct from deleting a
    PLACEMENT (a `delete_broll`/`delete_voice_over` call on a `Timeline
    Composition`, which only removes that one reference and never touches
    the asset library). An asset still referenced by ANY known
    composition/revision is blocked (`ASSET_REFERENCED`) -- the caller is
    expected to pass every revision worth protecting (e.g. the project's
    current saved revision); scanning unbounded history is a persistence-
    layer concern outside this pure function's own scope. A B-roll
    `replace_broll` operation never calls this: the OLD asset stays in the
    library, untouched, exactly as `timeline_composition.replace_broll`
    already only swaps the placement's own asset reference."""
    if asset is None:
        return AssetDeleteResult(outcome=ASSET_NOT_FOUND, reason_codes=(ASSET_NOT_FOUND,))
    if not authorize_asset_access(requesting=requesting, record_ownership=asset.ownership):
        return AssetDeleteResult(outcome=ASSET_NOT_OWNED, reason_codes=(ASSET_NOT_OWNED,))
    for composition in referencing_compositions:
        if is_asset_referenced(asset_id=asset.asset_id, composition=composition):
            return AssetDeleteResult(outcome=ASSET_REFERENCED, reason_codes=(ASSET_REFERENCED,))
    deleted = TimelineMediaAsset(
        asset_id=asset.asset_id,
        ownership=asset.ownership,
        role=asset.role,
        media_kind=asset.media_kind,
        source_media_identity=asset.source_media_identity,
        storage_reference=asset.storage_reference,
        duration_sec=asset.duration_sec,
        has_audio=asset.has_audio,
        qualification_status=TimelineAssetQualificationStatus.DELETED,
        content_sha256=asset.content_sha256,
        technical_metadata_reference=asset.technical_metadata_reference,
        replaces_asset_id=asset.replaces_asset_id,
        created_at=asset.created_at,
    )
    return AssetDeleteResult(outcome=ASSET_DELETE_SUCCEEDED, asset=deleted)


def rerecord_voice_over_asset(
    *,
    previous_asset: TimelineMediaAsset,
    new_asset_id: str,
    new_storage_reference: str,
    new_source_media_identity: str,
    new_duration_sec: float,
    new_has_audio: bool = True,
) -> TimelineMediaAsset:
    """Re-recording a voice-over is ALWAYS a new asset identity, never an
    in-place byte overwrite of the previous asset's storage reference --
    so any timeline revision that still names the OLD asset_id keeps
    resolving to the exact audio it was saved against. `replaces_
    asset_id` records the lineage for a future "revert to previous
    recording" affordance; nothing in this module auto-migrates existing
    placements onto the new asset -- that stays an explicit `replace_
    voice_over` call by the caller, per D-277's own "replace never
    guesses" doctrine."""
    if new_asset_id == previous_asset.asset_id:
        raise ValueError("a re-recorded voice-over must receive a new asset_id")
    return TimelineMediaAsset(
        asset_id=new_asset_id,
        ownership=previous_asset.ownership,
        role=TimelineAssetRole.VOICE_OVER,
        media_kind=TimelineMediaKind.AUDIO,
        source_media_identity=new_source_media_identity,
        storage_reference=new_storage_reference,
        duration_sec=new_duration_sec,
        has_audio=new_has_audio,
        qualification_status=TimelineAssetQualificationStatus.UPLOADED,
        replaces_asset_id=previous_asset.asset_id,
    )


# =============================================================================
# Timeline persistence + optimistic concurrency
# =============================================================================

@dataclass(frozen=True)
class TimelinePersistenceRecord:
    contract_version: int
    ownership: AssetOwnershipScope
    composition: tc.TimelineComposition
    timeline_revision_identity: str

    def __post_init__(self) -> None:
        expected = tc.derive_revision_identity(self.composition)
        if expected.identity != self.timeline_revision_identity:
            raise ValueError("timeline_revision_identity does not match the composition it accompanies")


def build_persistence_record(
    *, ownership: AssetOwnershipScope, composition: tc.TimelineComposition,
) -> TimelinePersistenceRecord:
    revision = tc.derive_revision_identity(composition)
    return TimelinePersistenceRecord(
        contract_version=TIMELINE_ASSET_REGISTRY_CONTRACT_VERSION,
        ownership=ownership,
        composition=composition,
        timeline_revision_identity=revision.identity,
    )


@dataclass(frozen=True)
class TimelineSaveResult:
    outcome: str
    record: TimelinePersistenceRecord | None = None
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


def save_timeline_revision(
    *,
    ownership: AssetOwnershipScope,
    composition: tc.TimelineComposition,
    current_record: TimelinePersistenceRecord | None,
    expected_revision_identity: str | None,
) -> TimelineSaveResult:
    """Optimistic concurrency: `expected_revision_identity` is the
    revision the CALLER last read. If a persisted record already exists
    and either its ownership disagrees or its current revision has moved
    on from what the caller expected, the save is rejected rather than
    silently overwriting a concurrent edit (`TIMELINE_REVISION_CONFLICT`).
    A brand-new timeline (no `current_record` at all) never conflicts,
    regardless of what the caller passed for `expected_revision_
    identity` -- there is nothing yet to disagree with."""
    validation = tc.validate_composition(composition)
    if not validation.valid:
        return TimelineSaveResult(outcome=TIMELINE_INVALID, reason_codes=validation.errors or (TIMELINE_INVALID,))
    if current_record is not None:
        if current_record.ownership != ownership:
            return TimelineSaveResult(outcome=ASSET_NOT_OWNED, reason_codes=(ASSET_NOT_OWNED,))
        if expected_revision_identity != current_record.timeline_revision_identity:
            return TimelineSaveResult(outcome=TIMELINE_REVISION_CONFLICT, reason_codes=(TIMELINE_REVISION_CONFLICT,))
    record = build_persistence_record(ownership=ownership, composition=composition)
    return TimelineSaveResult(outcome=TIMELINE_SAVE_SUCCEEDED, record=record)


# =============================================================================
# Export reproducibility -- export resolves the EXACT revision, never
# "whatever happens to be currently saved"
# =============================================================================

@dataclass(frozen=True)
class TimelineExportRequestResult:
    outcome: str
    composition: tc.TimelineComposition | None = None
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


def resolve_timeline_export(
    *,
    requesting: AssetOwnershipScope,
    record: TimelinePersistenceRecord | None,
    expected_revision_identity: str,
) -> TimelineExportRequestResult:
    """An export request always names the exact revision it means to
    render. If the project's persisted record has since moved to a
    different revision (someone kept editing after the export was
    queued), this refuses rather than silently rendering a different
    timeline than the one the caller asked for."""
    if record is None:
        return TimelineExportRequestResult(outcome=TIMELINE_INVALID, reason_codes=(TIMELINE_INVALID,))
    if record.ownership != requesting:
        return TimelineExportRequestResult(outcome=ASSET_NOT_OWNED, reason_codes=(ASSET_NOT_OWNED,))
    if record.timeline_revision_identity != expected_revision_identity:
        return TimelineExportRequestResult(
            outcome=TIMELINE_REVISION_CONFLICT, reason_codes=(TIMELINE_REVISION_CONFLICT,),
        )
    return TimelineExportRequestResult(outcome=TIMELINE_EXPORT_RESOLVED, composition=record.composition)
