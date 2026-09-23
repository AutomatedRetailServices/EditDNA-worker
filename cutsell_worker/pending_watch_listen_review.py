"""D-288 (findings 1/2): recoverable, persistent storage for a candidate
whose perceptual `watch_listen_status` is `HUMAN_REVIEW_REQUIRED`, and the
real, authenticated approval flow that connects a human decision back to
resumed delivery.

## Finding 1 -- the render must survive local cleanup

Before this module, `export_job.run_export_job` rendered inside a
`tempfile.TemporaryDirectory()` and, on anything other than a fully
successful delivery, let that directory (and the rendered MP4 in it) be
deleted when the `with` block exited -- a `HUMAN_REVIEW_REQUIRED` candidate
was indistinguishable from a genuine render failure and the file itself
was simply gone. `persist_pending_review` uploads the ACTUAL rendered file
to a PRIVATE, non-tenant-facing S3 location (`PENDING_REVIEW_PREFIX`,
structurally distinct from `tenant_safe_delivery`'s customer-facing export
key -- a pending-review artifact can never be confused with, or
accidentally served as, a `DELIVERY_READY` object) BEFORE the caller's own
`TemporaryDirectory` can close, and records bounded, recoverable metadata
in Redis. Same "S3 holds bytes, Redis holds bounded metadata" split this
codebase already establishes in `render_versions.py` -- no new storage
pattern invented.

## Finding 2 -- the real approval flow

`apply_human_approval` is the ONE place a human decision can promote a
pending record. It is deliberately strict and fails closed on every axis
named by the finding:

- **Authentication/authorization**: `requesting` must be a real, non-empty
  `tenant_safe_delivery.DeliveryOwnershipScope` (that dataclass's own
  `__post_init__` already rejects empty identities -- reused, not
  reinvented) that matches the pending record's own ownership, checked via
  the SAME `tenant_safe_delivery.assert_delivery_access` the tenant-safe
  delivery gate already trusts for exactly this question.
- **File identity**: the approval must name the EXACT `render_identity`
  AND `output_sha256` of the CURRENTLY persisted pending record (D-267's
  own documented non-determinism: two encodes of the identical plan can
  legitimately hash differently, so binding both is required, not just
  one) -- AND the exact `plan_id`/`plan_version` of the CanonicalEditPlan
  that was actually executed (a distinct, semantic identity from the
  physical render identity, D-025) -- a mismatch on ANY of the four is a
  rejected, stale approval.
- **BLOCKED is never approvable**: mirrors `render_delivery.resolve_
  watch_listen_status_for_delivery`'s own contract -- a confirmed
  perceptual defect is a root-authority fix or a re-render, never
  something a human approves away through this gate.
- **The gate stays mandatory**: `apply_human_approval` only ever produces
  an APPROVED or REJECTED record; nothing in this module can make a
  pending record silently become deliverable without an explicit,
  validated call here.

`resume_delivery_after_approval` (in `export_job.py`, which owns the real
tenant-safe delivery seam) is the "reanudación de entrega" this finding
also requires -- see that module for the actual resumed upload.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from . import tenant_safe_delivery as tsd
from .config import load_runtime_config
from .exports import store_export

# Deliberately a DIFFERENT prefix from `tenant_safe_delivery.
# DEFAULT_TENANT_SAFE_EXPORT_PREFIX` ("cutsell/exports/") and `exports.
# EXPORT_PREFIX` -- a pending-review object must never live under, or be
# reachable through, the customer-facing export namespace.
PENDING_REVIEW_PREFIX = "cutsell/pending-review/"

APPROVAL_STATUS_NONE = "NONE"
APPROVAL_STATUS_APPROVED = "APPROVED"
APPROVAL_STATUS_REJECTED = "REJECTED"


class PendingReviewError(ValueError):
    """Every rejection reason below raises this, always naming the exact
    reason in the message -- never a silent no-op, never a generic
    exception a caller could accidentally swallow."""


def _redis_client(client=None):
    if client is not None:
        return client
    config = load_runtime_config()
    if not config.redis_url:
        raise RuntimeError("REDIS_URL is required for pending watch-listen review storage")
    from redis import Redis
    return Redis.from_url(config.redis_url)


def _scope_hash(value: str) -> str:
    if not value or len(value) > 200:
        raise ValueError("pending review scope identifiers must contain 1 to 200 characters")
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def pending_review_key(*, user_id: str, project_id: str, job_id: str) -> str:
    if not str(user_id or "").strip() or not str(project_id or "").strip() or not str(job_id or "").strip():
        raise PendingReviewError("pending_review_key_requires_non_empty_user_project_job_id")
    return f"cutsell:v1:pending_watch_listen:{_scope_hash(user_id)}:{_scope_hash(project_id)}:{_scope_hash(job_id)}"


@dataclass(frozen=True)
class PendingWatchListenRecord:
    record_id: str
    user_id: str
    project_id: str
    job_id: str
    render_identity: str
    output_sha256: str
    plan_id: str
    plan_version: int
    pending_s3_uri: str
    # D-288.2 (correction): the ORIGINAL, real automated perceptual verdict
    # at persist time -- set ONCE here and NEVER mutated afterward by
    # `apply_human_approval`. `watch_listen_status` below is the CURRENT
    # EFFECTIVE status a human decision can move; a REJECTION (including a
    # revocation of a prior APPROVAL) always reverts `watch_listen_status`
    # back to THIS field, never leaves it stuck at HUMAN_APPROVED. See the
    # module's own D-288.2 section for the exact bug this closes.
    automated_watch_listen_status: str = ""
    watch_listen_status: str = ""
    perceptual_review: dict = field(default_factory=dict)
    created_at: float = 0.0
    # D-288.2 (blocker 3): carried from the original render so a LATER
    # resumed delivery can reuse the exact same finalization (render-
    # version registration) the same-job path uses, without needing the
    # original in-memory `draft` object (which no longer exists by the
    # time a human approves this later).
    selected_count: int = 0
    text_overlay_count: int = 0
    media_overlay_count: int = 0
    approval_status: str = APPROVAL_STATUS_NONE
    approver: str | None = None
    approved_at: float | None = None
    resumed_delivery_at: float | None = None
    # D-288.2 (finalization idempotency): set ONCE, the first time `export_
    # job.resume_delivery_after_approval` successfully finalizes delivery
    # for this record. A repeated resume call for an already-resumed
    # record returns THIS stored result instead of re-registering a render
    # version or re-firing a "finished" notification.
    resumed_delivery_result: dict | None = None

    @property
    def ownership(self) -> "tsd.DeliveryOwnershipScope":
        return tsd.DeliveryOwnershipScope(user_id=self.user_id, project_id=self.project_id, job_id=self.job_id)


def _to_dict(record: PendingWatchListenRecord) -> dict[str, Any]:
    return dataclasses.asdict(record)


def _from_dict(data: dict[str, Any]) -> PendingWatchListenRecord:
    known = {f.name for f in dataclasses.fields(PendingWatchListenRecord)}
    return PendingWatchListenRecord(**{k: v for k, v in data.items() if k in known})


def persist_pending_review(
    *,
    local_path: str,
    user_id: str,
    project_id: str,
    job_id: str,
    render_identity: str,
    output_sha256: str,
    plan_id: str,
    plan_version: int,
    watch_listen_status: str,
    perceptual_review: dict | None,
    client=None,
    s3_client=None,
    store_export_fn=None,
    selected_count: int = 0,
    text_overlay_count: int = 0,
    media_overlay_count: int = 0,
) -> PendingWatchListenRecord:
    """Finding 1: uploads the ACTUAL rendered file to a PRIVATE location
    and records recoverable metadata, so it survives the caller's own
    local cleanup. Called with whatever real verdict it is given -- this
    function does not itself gate on `watch_listen_status`; the caller
    (`export_job.py`) decides which verdicts are worth persisting for
    review (HUMAN_REVIEW_REQUIRED -- a confirmed BLOCKED defect has no
    review value and is not persisted here).

    `store_export_fn` (default: this module's own imported `store_export`)
    lets a caller with its own monkeypatchable/test-faked upload function
    (e.g. `export_job.py`'s own module-level `store_export` name, which
    its existing D-269A test suite already monkeypatches) route uploads
    through that SAME function instead of a second, independent
    reference -- one real upload implementation, never two."""
    if not str(render_identity or "").strip() or not str(output_sha256 or "").strip():
        raise PendingReviewError("persist_requires_non_empty_render_identity_and_output_sha256")
    upload = store_export_fn or store_export
    ownership = tsd.DeliveryOwnershipScope(user_id=user_id, project_id=project_id, job_id=job_id)
    pending_key = tsd.build_tenant_safe_export_key(
        ownership=ownership, render_identity=render_identity, prefix=PENDING_REVIEW_PREFIX,
    )
    upload_kwargs: dict[str, Any] = {
        "project_id": project_id, "user_id": user_id, "object_key": pending_key,
        "object_metadata": {
            "render_identity": render_identity, "sha256": output_sha256, "job_id": job_id,
            "kind": "pending_watch_listen_review",
        },
    }
    # Only forwarded when explicitly given -- a caller's own `store_export_
    # fn` may already be a `functools.partial(..., client=fake)`, which
    # would collide with a second `client=` keyword here.
    if s3_client is not None:
        upload_kwargs["client"] = s3_client
    stored = upload(local_path, **upload_kwargs)
    record = PendingWatchListenRecord(
        record_id=f"pwl_{uuid4().hex}",
        user_id=user_id, project_id=project_id, job_id=job_id,
        render_identity=render_identity, output_sha256=output_sha256,
        plan_id=plan_id, plan_version=int(plan_version),
        pending_s3_uri=stored["export_uri"],
        automated_watch_listen_status=watch_listen_status,
        watch_listen_status=watch_listen_status,
        perceptual_review=dict(perceptual_review or {}),
        created_at=time.time(),
        selected_count=int(selected_count),
        text_overlay_count=int(text_overlay_count),
        media_overlay_count=int(media_overlay_count),
    )
    target = _redis_client(client)
    target.set(
        pending_review_key(user_id=user_id, project_id=project_id, job_id=job_id),
        json.dumps(_to_dict(record), ensure_ascii=False),
    )
    return record


def load_pending_review(*, user_id: str, project_id: str, job_id: str, client=None) -> PendingWatchListenRecord | None:
    target = _redis_client(client)
    raw = target.get(pending_review_key(user_id=user_id, project_id=project_id, job_id=job_id))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return _from_dict(json.loads(raw))


def _save(record: PendingWatchListenRecord, *, client=None) -> None:
    target = _redis_client(client)
    target.set(
        pending_review_key(user_id=record.user_id, project_id=record.project_id, job_id=record.job_id),
        json.dumps(_to_dict(record), ensure_ascii=False),
    )


def _require_authenticated_requesting(requesting: "tsd.DeliveryOwnershipScope | None") -> "tsd.DeliveryOwnershipScope":
    """D-288.2 (finding 2 correction): `tenant_safe_delivery.assert_
    delivery_access` itself silently BYPASSES its own check when
    `requesting is None` (a deliberate, documented legacy behavior for a
    genuinely auth-disabled local/test context elsewhere in this
    codebase). This module's own operations -- querying, approving, and
    resuming delivery of a pending review -- are never allowed that
    bypass: `requesting=None` is rejected outright, here, before
    `assert_delivery_access` is ever reached, so this gate can never be
    silently skipped regardless of what that shared helper's own contract
    allows for other callers."""
    if requesting is None:
        raise PendingReviewError("requesting_identity_required_no_none_bypass")
    return requesting


def get_pending_review_for_authenticated_caller(
    *, user_id: str, project_id: str, job_id: str, requesting: "tsd.DeliveryOwnershipScope | None", client=None,
) -> PendingWatchListenRecord | None:
    """Finding 2: "consulta/revisión privada" -- the ONE real, authenticated
    entry point for READING a pending review (its perceptual findings,
    current status). `requesting` must be a real, authenticated identity
    (never `None`, never derived from the `user_id`/`project_id`/`job_id`
    being looked up -- those are the RESOURCE being addressed, not the
    caller's own identity) and must match the record's own ownership."""
    _require_authenticated_requesting(requesting)
    record = load_pending_review(user_id=user_id, project_id=project_id, job_id=job_id, client=client)
    if record is None:
        return None
    tsd.assert_delivery_access(requesting=requesting, record_ownership=record.ownership)
    return record


def apply_human_approval(
    *,
    user_id: str,
    project_id: str,
    job_id: str,
    requesting: "tsd.DeliveryOwnershipScope | None",
    approved: bool,
    approver: str,
    expected_render_identity: str,
    expected_output_sha256: str,
    expected_plan_id: str,
    expected_plan_version: int,
    client=None,
) -> PendingWatchListenRecord:
    """Finding 2: the ONE real approval entry point. See module docstring
    for the exact fail-closed contract. Returns the updated record on
    success; raises `PendingReviewError`/`PermissionError` naming the
    exact reason on any rejection -- never a silent no-op.

    `requesting` MUST be a real, authenticated identity resolved by the
    caller's own auth layer (in production, `cutsell_app.pending_review_
    routes`'s handlers resolve it from `request.state.auth_user_id`,
    itself set by `AuthScopeMiddleware` from a verified bearer session --
    never from a request body/query field). `None` is rejected outright
    (`_require_authenticated_requesting`), never silently bypassed.

    `approver` MUST likewise be the caller's own resolved authenticated
    identity, never a client-supplied free-text label -- this function
    cannot itself verify that (it has no access to the request), which is
    exactly why "no declares flujo autenticado basándote únicamente en
    pruebas de helpers" requires the REAL route handler (not this
    function's own unit tests) to be the thing that actually enforces it;
    see `cutsell_app/pending_review_routes.py`."""
    from .perceptual_watch_listen import (
        WATCH_LISTEN_BLOCKED, WATCH_LISTEN_HUMAN_APPROVED, WATCH_LISTEN_HUMAN_REVIEW_REQUIRED, WATCH_LISTEN_SYSTEM_PASS,
    )

    _require_authenticated_requesting(requesting)

    record = load_pending_review(user_id=user_id, project_id=project_id, job_id=job_id, client=client)
    if record is None:
        raise PendingReviewError("no_pending_review_found")

    # Authentication/authorization -- reuses the EXISTING tenant-safe
    # delivery ownership-match authority, never a second one.
    tsd.assert_delivery_access(requesting=requesting, record_ownership=record.ownership)

    approver_clean = str(approver or "").strip()
    if not approver_clean:
        raise PendingReviewError("approval_requires_non_empty_approver_identity")

    if not str(expected_render_identity or "").strip() or not str(expected_output_sha256 or "").strip():
        raise PendingReviewError("approval_requires_non_empty_render_identity_and_output_sha256")
    if not str(expected_plan_id or "").strip():
        raise PendingReviewError("approval_requires_non_empty_plan_id")

    if record.watch_listen_status == WATCH_LISTEN_BLOCKED:
        raise PendingReviewError("blocked_never_approvable")

    artifact_matches = (
        record.render_identity == expected_render_identity
        and record.output_sha256 == expected_output_sha256
        and record.plan_id == expected_plan_id
        and int(record.plan_version) == int(expected_plan_version)
    )
    if not artifact_matches:
        raise PendingReviewError("approval_bound_to_different_artifact_or_plan")

    if not approved:
        # D-288.2 (blocker 1, the revocation bug): a rejection -- whether
        # this is a first-time decline of a HUMAN_REVIEW_REQUIRED record
        # or a REVOCATION of a PRIOR approval -- always reverts `watch_
        # listen_status` back to the record's own immutable `automated_
        # watch_listen_status`. Before this fix, rejecting an ALREADY-
        # APPROVED record left `watch_listen_status` unchanged at HUMAN_
        # APPROVED (only `approval_status` moved to REJECTED), which
        # `resume_delivery_after_approval`'s own single-field check would
        # have silently accepted as still-deliverable.
        updated = dataclasses.replace(
            record,
            watch_listen_status=record.automated_watch_listen_status,
            approval_status=APPROVAL_STATUS_REJECTED,
            approver=approver_clean,
            approved_at=time.time(),
        )
        _save(updated, client=client)
        return updated

    if record.watch_listen_status not in (WATCH_LISTEN_HUMAN_REVIEW_REQUIRED, WATCH_LISTEN_SYSTEM_PASS):
        raise PendingReviewError(f"automated_status_not_promotable:{record.watch_listen_status}")

    updated = dataclasses.replace(
        record,
        watch_listen_status=WATCH_LISTEN_HUMAN_APPROVED,
        approval_status=APPROVAL_STATUS_APPROVED,
        approver=approver_clean,
        approved_at=time.time(),
    )
    _save(updated, client=client)
    return updated
