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

## D-288.3 -- atomic, versioned transitions (second correction pass)

Product Owner review of D-288.2 found the real remaining race: `resume_
delivery_after_approval` read this record ONCE at the top and used that
SAME stale snapshot both to decide "is this approved?" and, much later
(after a real S3 download/upload), to build its FINAL save via
`dataclasses.replace(stale_record, ...)` -- a concurrent revocation (or a
second concurrent resume) landing in that window was silently clobbered
by the stale write. Fixed with two mechanisms, both applied to EVERY
mutating entry point in this module (`apply_human_approval` included, not
only resume):

1. **`version`**: a monotonic counter, incremented on every successful
   write. `_cas_save` writes a record ONLY if the currently-stored
   record's `version` still equals the `expected_version` the caller read
   -- one atomic Redis Lua script (`_CAS_LUA`), never a separate GET-then-
   SET (which is exactly the race being closed). A conflict returns
   `None` (never raises by itself) so a caller can decide: retry with a
   fresh read (`apply_human_approval`), or treat it as "someone else
   already reached the terminal state I was about to write" (`export_job.
   resume_delivery_after_approval`, which rereads and returns the
   concurrent winner's own result rather than erroring on a benign race).
2. **Re-validate live state immediately before the irreversible action**:
   `resume_delivery_after_approval` re-reads this record FRESH right
   before the tenant-safe (customer-facing) upload -- the actual point of
   no return -- rather than trusting the read from the top of the
   function. A revocation that lands after the initial read is caught
   here, before anything customer-visible happens, never after.

See `export_job.resume_delivery_after_approval`'s own docstring for how
these combine with render-identity-keyed idempotency (`render_versions.
add_render_version`, `notifications.publish_notification`) to make a
retry after a mid-flight interruption safe, and with `job_started_at`
(now carried on this record from the ORIGINAL job, never `None`) to keep
a late approval of a since-superseded job from regressing the project's
current result.

## D-288.4 -- the claim comes BEFORE the effects (third correction pass)

Review of D-288.3 showed its atomic commit still came AFTER the tenant-
safe upload and finalize -- a revocation landing after resume's last
read was only detected once the effects had happened, and the cached
`resumed_delivery_result` was returned without checking whether the
approval still stood. Closed here:

- **`delivery_state`** (`NONE` -> `PUBLISHING` -> `DELIVERED`): `claim_
  delivery_publication` is a versioned CAS taken off a fresh read
  IMMEDIATELY BEFORE the first external effect; a revocation that landed
  after that read makes the claim fail, so nothing is published. While
  `PUBLISHING` is held, `apply_human_approval` REFUSES a rejection
  (`PendingReviewConflict`), so the two can never interleave; `commit_
  delivered` closes the window.
- **Current approval is checked before any cached result is returned**:
  aprobar -> entregar -> rechazar -> reanudar refuses the new request
  (`pending_review_not_approved`) while the delivered history stays on
  the record.
- **Atomic version/notification reuse** lives in `redis_atomic_list`
  (used by `render_versions`/`notifications`), and `project_store.update_
  project` no longer appends the same `render_version_id` twice.
- **Immutable reviewed bytes**: the pending object key now carries the
  exact `output_sha256`, so render B can never overwrite render A's
  object even when both share a render identity.
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

# D-288.4: the publication state machine, separate from the human
# approval state. NONE -> PUBLISHING (the atomic claim `resume_delivery_
# after_approval` takes BEFORE any external effect) -> DELIVERED (the
# atomic commit after finalize). A rejection is refused while PUBLISHING
# (the publish is already committed to), recorded as history once
# DELIVERED, and always blocks any NEW resume request.
DELIVERY_STATE_NONE = "NONE"
DELIVERY_STATE_PUBLISHING = "PUBLISHING"
DELIVERY_STATE_DELIVERED = "DELIVERED"


class PendingReviewError(ValueError):
    """Every rejection reason below raises this, always naming the exact
    reason in the message -- never a silent no-op, never a generic
    exception a caller could accidentally swallow."""


class PendingReviewConflict(PendingReviewError):
    """D-288.3: a `_cas_save` write lost to a genuinely incompatible
    concurrent change -- exhausted retries in `apply_human_approval`, or
    `resume_delivery_after_approval`'s own final commit conflicted with a
    change that was NOT simply another resume reaching the same terminal
    result first (that benign case is handled by rereading and returning
    the concurrent winner's result, never raised as a conflict)."""


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
    # D-288.3 (blockers 1/2): a monotonic write counter -- see `_cas_save`.
    # Every mutating write increments this; a write is only ever applied
    # if the currently-stored record's own `version` still matches what
    # the writer last read, so a stale in-memory snapshot can never
    # silently clobber a concurrent change.
    version: int = 1
    # D-288.3 (blocker 3): the ORIGINAL job's own real start timestamp
    # (`export_job._job_started_epoch`), carried from `persist_pending_
    # review` time so a LATER resumed delivery can preserve the ORIGINAL
    # job's identity/order for `tenant_safe_delivery.is_job_still_current`
    # -- never `None` (which that guard treats as "no ordering evidence,
    # always allow", i.e. would let a late approval of a stale job
    # silently regress a project whose "latest" pointer already moved on
    # to a newer job).
    job_started_at: float | None = None
    # D-288.4 (blocker 1): publication state, see `DELIVERY_STATE_*`.
    delivery_state: str = DELIVERY_STATE_NONE
    publishing_claimed_at: float | None = None

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
    job_started_at: float | None = None,
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
    # D-288.4 (blocker 3): the exact bytes' own hash is part of the key,
    # so a later render for the same job/render identity with different
    # bytes lands at a DIFFERENT object -- the reviewed MP4 is immutable
    # and a link already issued for it keeps showing it.
    pending_key = tsd.build_tenant_safe_export_key(
        ownership=ownership, render_identity=render_identity, prefix=PENDING_REVIEW_PREFIX,
        content_sha256=output_sha256,
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
        job_started_at=job_started_at,
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


# D-288.3: a single atomic Lua script -- the check-and-write is ONE round
# trip to Redis, never a separate GET-then-SET a concurrent writer could
# interleave with. `cjson` is built into real Redis's Lua sandbox (no
# extension needed). Returns a literal status string, never raises for an
# ordinary version conflict -- the caller decides how to react.
_CAS_LUA = """
local current = redis.call('GET', KEYS[1])
if current == false then
  return 'missing'
end
local ok, decoded = pcall(cjson.decode, current)
if not ok then
  return 'corrupt'
end
if tostring(decoded['version']) ~= ARGV[1] then
  return 'conflict'
end
redis.call('SET', KEYS[1], ARGV[2])
return 'ok'
"""

_MAX_CAS_RETRIES = 5


def _cas_save(
    record: PendingWatchListenRecord, *, expected_version: int, client=None,
) -> PendingWatchListenRecord | None:
    """D-288.3: atomically writes `record` (with `version` bumped to
    `expected_version + 1`) ONLY IF the record currently stored under this
    key still has `version == expected_version`. Returns the newly-
    written record (with its bumped `version`) on success, or `None` on a
    conflict (someone else wrote first) -- never raises by itself, so a
    caller can retry with a fresh read (`apply_human_approval`) or treat
    "someone else already reached the terminal state I was about to
    write" as a benign race rather than an error (`export_job.resume_
    delivery_after_approval`)."""
    target = _redis_client(client)
    key = pending_review_key(user_id=record.user_id, project_id=record.project_id, job_id=record.job_id)
    new_record = dataclasses.replace(record, version=int(expected_version) + 1)
    new_json = json.dumps(_to_dict(new_record), ensure_ascii=False)
    result = target.eval(_CAS_LUA, 1, key, str(int(expected_version)), new_json)
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    return new_record if result == "ok" else None


def is_approved_for_delivery(record: PendingWatchListenRecord) -> bool:
    """D-288.2/D-288.4: BOTH the current watch-listen status AND the
    approval status must agree -- never one field alone."""
    from .perceptual_watch_listen import WATCH_LISTEN_HUMAN_APPROVED
    return (
        record.watch_listen_status == WATCH_LISTEN_HUMAN_APPROVED
        and record.approval_status == APPROVAL_STATUS_APPROVED
    )


def claim_delivery_publication(record: PendingWatchListenRecord, *, client=None) -> PendingWatchListenRecord | None:
    """D-288.4 (blocker 1): the atomic transition that must succeed BEFORE
    any external effect of a resumed delivery (tenant-safe upload, render
    version, notification). Only an approved record in `NONE` (first
    attempt) or `PUBLISHING` (a retry after an interrupted attempt -- the
    effects downstream are idempotent by render identity) can be claimed;
    the write is a versioned `_cas_save` off the caller's own fresh read,
    so a revocation that landed after that read makes the claim fail
    (`None`) instead of being raced. Once PUBLISHING is held, `apply_
    human_approval` refuses a rejection until the commit lands."""
    if not is_approved_for_delivery(record):
        return None
    if record.delivery_state not in (DELIVERY_STATE_NONE, DELIVERY_STATE_PUBLISHING):
        return None
    return _cas_save(
        dataclasses.replace(record, delivery_state=DELIVERY_STATE_PUBLISHING, publishing_claimed_at=time.time()),
        expected_version=record.version, client=client,
    )


def commit_delivered(record: PendingWatchListenRecord, result: dict, *, client=None) -> PendingWatchListenRecord | None:
    """D-288.4: PUBLISHING -> DELIVERED with the finalized result stored
    (the history a later rejection preserves). Versioned like every other
    write; `None` on a conflict."""
    return _cas_save(
        dataclasses.replace(
            record, delivery_state=DELIVERY_STATE_DELIVERED,
            resumed_delivery_at=time.time(), resumed_delivery_result=dict(result),
        ),
        expected_version=record.version, client=client,
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


def get_pending_review_media_access(
    *,
    user_id: str,
    project_id: str,
    job_id: str,
    requesting: "tsd.DeliveryOwnershipScope | None",
    expected_render_identity: str,
    expected_output_sha256: str,
    client=None,
    s3_client=None,
    expires_in: int = 900,
) -> dict:
    """D-288.3 (blocker 4): "acceso autenticado al MP4 privado para
    revisarlo antes de aprobar" -- a short-lived, read-only presigned URL
    for the ACTUAL pending MP4 (never a redirect through `tenant_safe_
    delivery`, and never anything that could be mistaken for a real
    `download_url`: this object lives under `PENDING_REVIEW_PREFIX`, the
    same private namespace `persist_pending_review` uploaded to, and this
    function's own return value carries no `delivery_status` field and
    never promotes the record -- reading it has zero effect on `watch_
    listen_status`/`approval_status`).

    "Ligado al artefacto exacto": the caller MUST name the render's
    CURRENT `render_identity`/`output_sha256` (the SAME two-field binding
    `apply_human_approval` already requires) -- a mismatch (e.g. a stale
    client reference to a PREVIOUS pending record for this same job_id,
    since a job_id's pending-review slot is reused/overwritten by a
    later export attempt) is refused rather than silently handing back
    access to whatever file happens to occupy this job's CURRENT slot
    now."""
    _require_authenticated_requesting(requesting)
    if not str(expected_render_identity or "").strip() or not str(expected_output_sha256 or "").strip():
        raise PendingReviewError("media_access_requires_non_empty_render_identity_and_output_sha256")
    record = load_pending_review(user_id=user_id, project_id=project_id, job_id=job_id, client=client)
    if record is None:
        raise PendingReviewError("no_pending_review_found")
    tsd.assert_delivery_access(requesting=requesting, record_ownership=record.ownership)
    if record.render_identity != expected_render_identity or record.output_sha256 != expected_output_sha256:
        raise PendingReviewError("media_access_bound_to_different_artifact")

    parsed_bucket, parsed_key = record.pending_s3_uri[5:].split("/", 1)
    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3")
    url = s3_client.generate_presigned_url(
        "get_object", Params={"Bucket": parsed_bucket, "Key": parsed_key}, ExpiresIn=int(expires_in),
    )
    return {
        "record_id": record.record_id,
        "render_identity": record.render_identity,
        "output_sha256": record.output_sha256,
        "preview_url": url,
        "expires_in": int(expires_in),
    }


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
    see `cutsell_app/pending_review_routes.py`.

    D-288.3: every read-decide-write cycle below uses a FRESH read and an
    atomic, versioned `_cas_save` -- never a stale snapshot blindly
    overwritten. A conflict (another approval/rejection/resume landed
    between this call's read and write) retries with a fresh read, up to
    `_MAX_CAS_RETRIES` times, rather than clobbering whatever the
    concurrent writer just committed."""
    from .perceptual_watch_listen import (
        WATCH_LISTEN_BLOCKED, WATCH_LISTEN_HUMAN_APPROVED, WATCH_LISTEN_HUMAN_REVIEW_REQUIRED, WATCH_LISTEN_SYSTEM_PASS,
    )

    _require_authenticated_requesting(requesting)

    approver_clean = str(approver or "").strip()
    if not approver_clean:
        raise PendingReviewError("approval_requires_non_empty_approver_identity")
    if not str(expected_render_identity or "").strip() or not str(expected_output_sha256 or "").strip():
        raise PendingReviewError("approval_requires_non_empty_render_identity_and_output_sha256")
    if not str(expected_plan_id or "").strip():
        raise PendingReviewError("approval_requires_non_empty_plan_id")

    for _attempt in range(_MAX_CAS_RETRIES):
        record = load_pending_review(user_id=user_id, project_id=project_id, job_id=job_id, client=client)
        if record is None:
            raise PendingReviewError("no_pending_review_found")

        # Authentication/authorization -- reuses the EXISTING tenant-safe
        # delivery ownership-match authority, never a second one.
        tsd.assert_delivery_access(requesting=requesting, record_ownership=record.ownership)

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

        if not approved and record.delivery_state == DELIVERY_STATE_PUBLISHING:
            # D-288.4 (blocker 1): the publish claim is already held --
            # external effects are in flight or done. Refuse, never race
            # them; once the publisher commits (DELIVERED) a rejection is
            # accepted as history and blocks every later resume.
            raise PendingReviewConflict("delivery_in_progress_revocation_refused")

        if not approved:
            # D-288.2 (blocker 1, the revocation bug): a rejection --
            # whether this is a first-time decline of a HUMAN_REVIEW_
            # REQUIRED record or a REVOCATION of a PRIOR approval -- always
            # reverts `watch_listen_status` back to the record's own
            # immutable `automated_watch_listen_status`. Before this fix,
            # rejecting an ALREADY-APPROVED record left `watch_listen_
            # status` unchanged at HUMAN_APPROVED (only `approval_status`
            # moved to REJECTED), which `resume_delivery_after_approval`'s
            # own single-field check would have silently accepted as
            # still-deliverable.
            updated = dataclasses.replace(
                record,
                watch_listen_status=record.automated_watch_listen_status,
                approval_status=APPROVAL_STATUS_REJECTED,
                approver=approver_clean,
                approved_at=time.time(),
            )
        else:
            if record.watch_listen_status not in (WATCH_LISTEN_HUMAN_REVIEW_REQUIRED, WATCH_LISTEN_SYSTEM_PASS):
                raise PendingReviewError(f"automated_status_not_promotable:{record.watch_listen_status}")
            updated = dataclasses.replace(
                record,
                watch_listen_status=WATCH_LISTEN_HUMAN_APPROVED,
                approval_status=APPROVAL_STATUS_APPROVED,
                approver=approver_clean,
                approved_at=time.time(),
            )

        saved = _cas_save(updated, expected_version=record.version, client=client)
        if saved is not None:
            return saved
        # D-288.3: a concurrent write landed between our read and our
        # write -- retry with a fresh read rather than overwrite it.

    raise PendingReviewConflict("pending_review_concurrent_update_conflict")
