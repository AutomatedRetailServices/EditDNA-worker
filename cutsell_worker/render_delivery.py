"""Render Identity + Delivery Contract Foundation -- D-267.

Post D-266 (structured render failure observability + atomic temp->final
promotion) / D-266A (bounded 1200s ffmpeg execution timeout). Renderer
execution safety is CLOSED; this module adds the commercial delivery
identity/integrity layer on top of it, answering:

    "What exact render is this?"                  -> RENDER_IDENTITY
    "Did it come from the plans/versions expected?" -> RENDER_IDENTITY inputs
    "Did the final file pass technical QC?"        -> technical_qc_status
    "Was the file hashed?"                          -> output_sha256
    "Was upload/delivery completed?"                -> upload_status
    "Is this safe to present to the user?"          -> delivery_status

## RENDER_IDENTITY vs OUTPUT_SHA256 (D-267 Stage 2, binding distinction)

`RENDER_IDENTITY` (`compute_render_identity`) is a SEMANTIC EXECUTION
identity: a deterministic function of the render PLAN (selected clip
identities/order, timing, captions, visual/audio finishing plan
identities, output geometry, renderer contract version) -- never of any
local filesystem path, and never of the actual encoded bytes. Two
renders from the identical plan mint the identical `render_identity`
even if they run on different machines or at different times.

`OUTPUT_SHA256` (`compute_output_sha256`) is the actual FINAL FILE
integrity hash -- computed only after D-266's atomic temp->final
promotion, from real decoded/encoded bytes on disk. D-267 Stage 13:
`libx264`'s `veryfast` preset is multi-threaded and this codebase's own
render command never pins `-threads 1` (see D-265's own determinism
finding), so two separate encodes of the IDENTICAL plan can legitimately
produce DIFFERENT `output_sha256` values while sharing the SAME
`render_identity` -- this is expected container/codec non-determinism,
not a defect, and this module never asserts otherwise.

These two identities are never conflated, never derived from each other,
and never substituted for one another anywhere in this module.

## Authority boundary (D-267 Stage 8, binding)

This module NEVER re-derives, re-computes, or overrides technical QC.
`technical_qc_status_from_live_render_qc` reads ONLY
`live_render_qc.LiveRenderQCResult.deliverable` -- the one authoritative
delivery gate that module's own docstring already names (D-036 item 7).
That module's own `delivery_status` property (a narrow "DELIVERABLE" /
"NOT_DELIVERABLE_<qc status>" diagnostic string) is a DIFFERENT, narrower
concept than THIS module's `RenderDeliveryRecord.delivery_status` (the
broader identity+hash+QC+upload commercial contract vocabulary) -- never
conflate the two `delivery_status` names across these two modules.

## No S3/upload wiring (D-267 Stage 15)

This module models the UPLOAD STATE TRANSITION contract only
(`with_upload_result`) -- it never calls `multipart_uploads.py` or any
other network/storage code, never constructs credentials, and never
performs an actual upload. A future integration gate feeds a real upload
outcome into `with_upload_result`; this foundation only defines what
happens to the delivery record once that outcome is known.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover -- type-only, avoids a hard runtime
    # dependency from this module onto live_render_qc.py (matches render_
    # plan.py's own existing TYPE_CHECKING-only import precedent).
    from .live_render_qc import LiveRenderQCResult
    from .render_plan import RenderSegment

# --- Stage 11: renderer contract version -----------------------------------
# Bumped only when a change to THIS module's own identity-computation
# contract (which fields feed RENDER_IDENTITY, or how) would otherwise
# silently collide two semantically-different renders under the same
# identity. Never the engine/product version, never invented for an
# unrelated subsystem.
RENDER_CONTRACT_VERSION = 1

# --- Stage 5: bounded delivery-status vocabulary ----------------------------
# NOT_READY is a caller-tracked PRE-render sentinel (the natural initial
# value a caller holds before this module is ever invoked) -- it is
# never produced by `build_render_delivery_record` itself, since that
# function is only ever called once a render execution outcome exists.
DELIVERY_STATUS_NOT_READY = "NOT_READY"
DELIVERY_STATUS_RENDER_FAILED = "RENDER_FAILED"
DELIVERY_STATUS_QC_FAILED = "QC_FAILED"
DELIVERY_STATUS_HASH_FAILED = "HASH_FAILED"
DELIVERY_STATUS_READY_FOR_UPLOAD = "READY_FOR_UPLOAD"
DELIVERY_STATUS_UPLOAD_IN_PROGRESS = "UPLOAD_IN_PROGRESS"
DELIVERY_STATUS_UPLOAD_FAILED = "UPLOAD_FAILED"
DELIVERY_STATUS_DELIVERY_READY = "DELIVERY_READY"
DELIVERY_STATUS_DELIVERY_BLOCKED = "DELIVERY_BLOCKED"
DELIVERY_STATUS_UNKNOWN = "UNKNOWN"

RENDER_EXECUTION_STATUS_SUCCEEDED = "SUCCEEDED"
RENDER_EXECUTION_STATUS_FAILED = "FAILED"

TECHNICAL_QC_STATUS_PASS = "PASS"
TECHNICAL_QC_STATUS_FAIL = "FAIL"
TECHNICAL_QC_STATUS_NOT_RUN = "NOT_RUN"

UPLOAD_STATUS_NOT_ATTEMPTED = "NOT_ATTEMPTED"
UPLOAD_STATUS_IN_PROGRESS = "IN_PROGRESS"
UPLOAD_STATUS_SUCCEEDED = "SUCCEEDED"
UPLOAD_STATUS_FAILED = "FAILED"

# --- Stage 3/9: bounded, streamed hashing -----------------------------------
_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB -- never loads a full render into memory


# =============================================================================
# Stage 1/2/10/12 -- RENDER IDENTITY (semantic execution identity)
# =============================================================================

def _segment_identity_fields(segment: "RenderSegment") -> dict:
    """The identity-relevant fields of one `RenderSegment` -- deliberately
    EXCLUDES `source_path` (Stage 10: identity must be independent of any
    local/machine-specific path). `source_asset_id` (a stable logical
    source identity, not a filesystem path) carries the real source
    identity instead."""
    fields: dict = {
        "clip_id": segment.clip_id,
        "source_asset_id": segment.source_asset_id,
        "start": round(float(segment.start), 3),
        "end": round(float(segment.end), 3),
        "audio_muted": bool(segment.audio_muted),
        "audio_volume": round(float(segment.audio_volume), 3),
        "caption_text": str(segment.caption_text or ""),
        "caption_preset": str(segment.caption_preset or ""),
    }
    if segment.audio_start is not None:
        fields["audio_start"] = round(float(segment.audio_start), 3)
    if segment.audio_end is not None:
        fields["audio_end"] = round(float(segment.audio_end), 3)
    visual_transform = segment.visual_transform
    if visual_transform is not None:
        fields["visual_transform"] = {
            "action": visual_transform.action,
            "source_width": visual_transform.source_width,
            "source_height": visual_transform.source_height,
            "scale_factor": round(float(visual_transform.scale_factor), 6),
            "scaled_width": visual_transform.scaled_width,
            "scaled_height": visual_transform.scaled_height,
            "crop_x": visual_transform.crop_x,
            "crop_y": visual_transform.crop_y,
            "crop_width": visual_transform.crop_width,
            "crop_height": visual_transform.crop_height,
        }
    return fields


def compute_render_identity(
    segments: Sequence["RenderSegment"],
    *,
    width: int,
    height: int,
    fps: int,
    audio_finishing_plan_identity: str | None = None,
    visual_finishing_plan_identity: str | None = None,
    renderer_contract_version: int = RENDER_CONTRACT_VERSION,
) -> str:
    """D-267 Stage 1/2/10/12: deterministic SEMANTIC execution identity for
    one render -- independent of any local filesystem path or temp/final
    filename (Stage 10). Binds, in order: renderer contract version,
    every selected segment's own identity-relevant fields IN ORDER
    (selection AND order both matter -- Stage 12), output geometry, and
    the caller-supplied Audio/Visual Finishing plan identities (opaque
    strings this module never inspects the shape of -- it only binds
    them, matching this codebase's own established "one owner mints the
    id, everyone else only carries it" discipline from `canonical_
    identity.py`). Reuses the existing `json.dumps(payload, sort_keys=
    True, default=str)` -> SHA-256 -> bounded-digest identity convention
    already established for D-263/D-264's own composition/execution
    identities (`[:24]`), distinct from `canonical_identity.py`'s own
    `[:20]` convention for a different subsystem -- never a new scheme."""
    payload = {
        "renderer_contract_version": int(renderer_contract_version),
        "segments": [_segment_identity_fields(segment) for segment in segments],
        "output_geometry": {"width": int(width), "height": int(height), "fps": int(fps)},
        "audio_finishing_plan_identity": audio_finishing_plan_identity,
        "visual_finishing_plan_identity": visual_finishing_plan_identity,
    }
    normalized = json.dumps(payload, sort_keys=True, default=str)
    return "render_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


# =============================================================================
# Stage 3/9 -- OUTPUT SHA-256 (actual final file integrity)
# =============================================================================

def compute_output_sha256(path: str | Path) -> str:
    """D-267 Stage 3: the real, final-file integrity hash. Callers must
    only ever call this AFTER D-266's atomic promotion has confirmed the
    file at `path` is the real, complete, promoted output -- never on a
    `.rendering.` temp artifact (Stage 3's own explicit instruction).
    Streamed in 1 MiB chunks so this scales to a long-form render without
    loading it fully into memory. Raises `OSError` (never silently
    returns a placeholder) if the file cannot be read -- callers must map
    that to `DELIVERY_STATUS_HASH_FAILED` (Stage 9), never to a fabricated
    hash or a fabricated PASS."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# Stage 8 -- technical QC integration (consumes, never re-derives, authority)
# =============================================================================

def technical_qc_status_from_live_render_qc(result: "LiveRenderQCResult | None") -> str:
    """D-267 Stage 8: reads the ALREADY-DECIDED technical QC verdict from
    `live_render_qc.LiveRenderQCResult` -- never re-derives PASS/FAIL from
    `.status` string-matching, finding counts, or any other signal; only
    ever reads `.deliverable`, exactly as that module's own docstring
    requires every caller to (D-036 item 7). `None` (no QC run yet) maps
    to `TECHNICAL_QC_STATUS_NOT_RUN`, never a fabricated PASS."""
    if result is None:
        return TECHNICAL_QC_STATUS_NOT_RUN
    return TECHNICAL_QC_STATUS_PASS if result.deliverable else TECHNICAL_QC_STATUS_FAIL


# =============================================================================
# Stage 16 -- S3/remote identity semantics (no live upload wiring)
# =============================================================================

def is_etag_valid_sha256_proxy(etag: str | None) -> bool:  # noqa: ARG001 -- signature is the contract
    """D-267 Stage 16/19: an S3 ETag is NEVER treated as a SHA-256 proxy
    by this module. For a real multipart upload an ETag is not even a
    hash of the object's own bytes (it is a hash-of-part-hashes plus a
    part-count suffix); for a single-part upload it MAY equal the MD5 of
    the object, never the SHA-256. This helper therefore always returns
    `False` -- it exists so a future integration has one explicit, named
    place to encode a real remote-hash verification method instead of
    ever silently assuming ETag == SHA-256."""
    return False


# =============================================================================
# Stage 4/5/6/7/14/20/21 -- the delivery record itself
# =============================================================================

@dataclass(frozen=True)
class RenderDeliveryRecord:
    """D-267 Stage 4/14: the structured, immutable delivery contract for
    one render. A frozen dataclass -- no historical record is ever
    mutated in place; a state transition (e.g. an upload outcome becoming
    known) produces a NEW record via `with_upload_result` (Stage 14)."""

    render_identity: str
    render_contract_version: int

    final_path: str | None
    output_sha256: str | None
    output_size_bytes: int | None

    render_execution_status: str
    technical_qc_status: str
    upload_status: str
    delivery_status: str

    created_at: float

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)

    # Stage 16/17/18 -- remote identity, populated only once a real upload
    # outcome is known via `with_upload_result`; never fabricated here.
    remote_reference: str | None = None
    remote_etag: str | None = None
    remote_size_bytes: int | None = None
    remote_sha256_verified: bool = False

    # Stage 21 -- ownership linkage, exposed only where an existing
    # contract already names one (`project_id` matches `contracts.
    # DraftTimeline.project_id` exactly); `job_id` is a generic optional
    # slot for whatever job-tracking system a future integration has --
    # this module invents no auth/tenant system of its own.
    project_id: str | None = None
    job_id: str | None = None

    @property
    def ready_for_delivery(self) -> bool:
        """D-267 Stage 6: the ONE authoritative gate a caller should
        branch on -- true if and only if `delivery_status ==
        DELIVERY_READY`. Never derive "safe to present to the user" from
        any other field on this record."""
        return self.delivery_status == DELIVERY_STATUS_DELIVERY_READY


def build_render_delivery_record(
    *,
    render_identity: str,
    render_execution_status: str,
    final_path: str | None,
    technical_qc_status: str,
    require_upload: bool = False,
    upload_status: str = UPLOAD_STATUS_NOT_ATTEMPTED,
    renderer_contract_version: int = RENDER_CONTRACT_VERSION,
    blocking_error: str | None = None,
    remote_reference: str | None = None,
    remote_etag: str | None = None,
    remote_size_bytes: int | None = None,
    remote_sha256_verified: bool = False,
    project_id: str | None = None,
    job_id: str | None = None,
    provenance: dict | None = None,
) -> RenderDeliveryRecord:
    """D-267 Stage 4/5/6/7/9: builds one `RenderDeliveryRecord`, computing
    `output_sha256`/`output_size_bytes` from the real file at `final_path`
    (Stage 3: only after the caller's own D-266 atomic promotion has
    already happened -- this function does not know or care HOW
    `final_path` was produced, only that it must already be the promoted
    final file) and deriving `delivery_status` deterministically from
    every gate below. `DELIVERY_READY` requires ALL of: render execution
    succeeded, final file exists and is non-empty, `output_sha256` was
    successfully computed, technical QC PASSed, and (only when
    `require_upload=True`) the upload step succeeded (Stage 6/7 -- a
    local-only render with `require_upload=False` stops at
    `READY_FOR_UPLOAD`, never fakes `DELIVERY_READY`)."""
    errors: list[str] = []
    output_sha256: str | None = None
    output_size_bytes: int | None = None
    status: str

    if blocking_error:
        status = DELIVERY_STATUS_DELIVERY_BLOCKED
        errors.append(str(blocking_error))
    elif render_execution_status == RENDER_EXECUTION_STATUS_FAILED:
        status = DELIVERY_STATUS_RENDER_FAILED
        errors.append("render_execution_failed")
    elif render_execution_status != RENDER_EXECUTION_STATUS_SUCCEEDED:
        status = DELIVERY_STATUS_UNKNOWN
        errors.append(f"unrecognized_render_execution_status:{render_execution_status}")
    elif not final_path or not Path(final_path).exists():
        status = DELIVERY_STATUS_RENDER_FAILED
        errors.append("final_output_missing")
    elif Path(final_path).stat().st_size <= 0:
        status = DELIVERY_STATUS_RENDER_FAILED
        errors.append("final_output_empty")
    else:
        output_size_bytes = Path(final_path).stat().st_size
        try:
            output_sha256 = compute_output_sha256(final_path)
        except OSError as exc:
            status = DELIVERY_STATUS_HASH_FAILED
            errors.append(f"hash_computation_failed:{exc}")
        else:
            if technical_qc_status != TECHNICAL_QC_STATUS_PASS:
                status = DELIVERY_STATUS_QC_FAILED
                errors.append(f"technical_qc_not_pass:{technical_qc_status}")
            elif not require_upload:
                status = DELIVERY_STATUS_READY_FOR_UPLOAD  # Stage 7: local-only case
            elif upload_status == UPLOAD_STATUS_SUCCEEDED:
                status = DELIVERY_STATUS_DELIVERY_READY
            elif upload_status == UPLOAD_STATUS_IN_PROGRESS:
                status = DELIVERY_STATUS_UPLOAD_IN_PROGRESS
            elif upload_status == UPLOAD_STATUS_FAILED:
                status = DELIVERY_STATUS_UPLOAD_FAILED
                errors.append("upload_failed")
            else:
                status = DELIVERY_STATUS_READY_FOR_UPLOAD

    return RenderDeliveryRecord(
        render_identity=render_identity,
        render_contract_version=renderer_contract_version,
        final_path=str(final_path) if final_path else None,
        output_sha256=output_sha256,
        output_size_bytes=output_size_bytes,
        render_execution_status=render_execution_status,
        technical_qc_status=technical_qc_status,
        upload_status=upload_status,
        delivery_status=status,
        created_at=time.time(),
        errors=tuple(errors),
        warnings=(),
        provenance=dict(provenance or {}),
        remote_reference=remote_reference,
        remote_etag=remote_etag,
        remote_size_bytes=remote_size_bytes,
        remote_sha256_verified=bool(remote_sha256_verified),
        project_id=project_id,
        job_id=job_id,
    )


def with_upload_result(
    record: RenderDeliveryRecord,
    *,
    upload_status: str,
    remote_reference: str | None = None,
    remote_etag: str | None = None,
    remote_size_bytes: int | None = None,
) -> RenderDeliveryRecord:
    """D-267 Stage 14/15/17/18: produces a NEW `RenderDeliveryRecord`
    reflecting a real upload outcome -- never mutates `record` in place,
    never performs the upload itself. Fails closed (Stage 18): a record
    not already in an upload-eligible state (`READY_FOR_UPLOAD`,
    `UPLOAD_IN_PROGRESS`, or a retried `UPLOAD_FAILED`) is returned
    unchanged (still a new object) rather than silently forced into a
    transition it never qualified for; `UPLOAD_FAILED` can never itself
    produce `DELIVERY_READY`."""
    eligible = (
        DELIVERY_STATUS_READY_FOR_UPLOAD,
        DELIVERY_STATUS_UPLOAD_IN_PROGRESS,
        DELIVERY_STATUS_UPLOAD_FAILED,
    )
    if record.delivery_status not in eligible:
        return dataclasses.replace(record)

    if upload_status == UPLOAD_STATUS_SUCCEEDED:
        new_status = DELIVERY_STATUS_DELIVERY_READY
        new_errors = record.errors
    elif upload_status == UPLOAD_STATUS_IN_PROGRESS:
        new_status = DELIVERY_STATUS_UPLOAD_IN_PROGRESS
        new_errors = record.errors
    elif upload_status == UPLOAD_STATUS_FAILED:
        new_status = DELIVERY_STATUS_UPLOAD_FAILED
        new_errors = record.errors + ("upload_failed",)
    else:
        new_status = DELIVERY_STATUS_UNKNOWN
        new_errors = record.errors + (f"unrecognized_upload_status:{upload_status}",)

    return dataclasses.replace(
        record,
        upload_status=upload_status,
        delivery_status=new_status,
        errors=new_errors,
        remote_reference=remote_reference if remote_reference is not None else record.remote_reference,
        remote_etag=remote_etag if remote_etag is not None else record.remote_etag,
        remote_size_bytes=remote_size_bytes if remote_size_bytes is not None else record.remote_size_bytes,
    )


# =============================================================================
# Stage 20 -- observability diagnostics (no secrets: none exist on this record)
# =============================================================================

def render_delivery_diagnostics(record: RenderDeliveryRecord) -> dict:
    """D-267 Stage 20: a bounded, structured, additive diagnostics view.
    Every field here is already public on `RenderDeliveryRecord` -- this
    is a stable, named projection for logs/reports, not a new source of
    truth."""
    hash_status = "NOT_COMPUTED"
    if record.output_sha256:
        hash_status = "OK"
    elif record.delivery_status == DELIVERY_STATUS_HASH_FAILED:
        hash_status = "FAILED"
    return {
        "render_identity": record.render_identity,
        "render_contract_version": record.render_contract_version,
        "render_status": record.render_execution_status,
        "qc_status": record.technical_qc_status,
        "hash_status": hash_status,
        "upload_status": record.upload_status,
        "delivery_status": record.delivery_status,
        "output_size_bytes": record.output_size_bytes,
        "output_sha256": record.output_sha256,
        "errors": list(record.errors),
        "warnings": list(record.warnings),
        "created_at": record.created_at,
    }
