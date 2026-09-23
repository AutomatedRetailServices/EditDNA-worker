"""Tenant-Safe Remote Delivery Foundation -- D-269.

Post D-268 (export/storage delivery integration audit, Verdict A: storage/
delivery foundation functional, clear tenant/remote P0 gaps identified).
D-267 built RENDER_IDENTITY + OUTPUT_SHA256 + a local `RenderDeliveryRecord`
but explicitly did not enforce tenant ownership, cross-user isolation, or
remote-object ownership. This module binds OWNER + PROJECT + JOB + RENDER
IDENTITY + OUTPUT SHA-256 + REMOTE OBJECT REFERENCE into one immutable
record, and makes it structurally impossible to call a remote object
DELIVERY_READY unless all of those correspond to the SAME job/output.

## No real S3 mutation (binding constraint)

Nothing in this module performs a network call, constructs AWS
credentials, or issues a real presigned URL. `RemoteObjectMetadataFixture`
stands in for what a real `boto3.head_object` response would look like;
`verify_remote_delivery` is fully offline-testable against it. A future
integration gate feeds this module a REAL fixture built from a real S3
response -- this foundation never performs that call itself.

## Relationship to D-267 (never re-derived)

`evaluate_tenant_safe_delivery` takes D-267's own `render_delivery.
RenderDeliveryRecord` as an input and NEVER re-computes or overrides its
QC/hash verdict -- a render-level blocker (`RENDER_FAILED`/`QC_FAILED`/
`HASH_FAILED`/etc.) always passes straight through unchanged. This module
only ever ADDS ownership/render-identity/remote-object binding on top of
an already-decided local delivery status.

## Relationship to D-268's findings (each addressed narrowly)

- P0 (auth fails open by default): closed in `cutsell_app/auth_middleware.
  py` (`_auth_required`), not in this module -- this module supplies the
  ownership-scope TYPE that a hardened auth layer's resolved identity
  should be compared against.
- P1 (D-267 unwired): this module is the wiring seam -- `evaluate_tenant_
  safe_delivery` is exactly what a future `export_job.py` integration
  would call.
- P1 (no post-upload remote verification): `verify_remote_delivery`.
- P1 (IDOR protection invisible at the handler seam): `assert_delivery_
  access`, mirroring `jobs.py`'s own `_assert_job_owner` precedent.
- P1 (stale "latest job" pointer race): `is_job_still_current`, wired
  (as an optional, backward-compatible parameter) into `project_store.
  update_project`.
- P2 (raw S3 URI exposure, no job/render binding on the remote object):
  `RemoteDeliveryObject`/`build_tenant_safe_export_key` bind job_id and
  render_identity into the object's own identity; nothing in this module
  ever models a raw, unsigned `s3://` URI as something to hand to a client.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from . import render_delivery as rd

# =============================================================================
# Stage 4/5 -- deterministic, tenant-safe S3 key construction
# =============================================================================

DEFAULT_TENANT_SAFE_EXPORT_PREFIX = "cutsell/exports/"

_SAFE_KEY_SEGMENT_RE = re.compile(r"^[a-z0-9_]+$")


def _scope_component(value: str) -> str:
    """D-269 Stage 4: the same SHA-256-hashed-scope-component convention
    already established in `exports.py`/`uploads.py`/`project_store.py`
    (each with their own local `_scope_hash`/`_scope` helper) -- this
    module owns its own copy rather than reaching into another module's
    private helper, matching this codebase's existing per-module
    convention. Bounded input length, never empty."""
    text = str(value or "").strip()
    if not text or len(text) > 200:
        raise ValueError("identity component must contain 1 to 200 characters")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _validate_key_segment(segment: str) -> str:
    """D-269 Stage 5: structural key-normalization proof -- every segment
    this module ever places in a remote key must already match this bounded
    charset. Since every segment is EITHER a `_scope_component` hash
    (pure lowercase hex) OR a `compute_render_identity` value (`render_`
    + lowercase hex), this can never legitimately fail; it exists as a
    fail-closed assertion, not a sanitizer -- this module never tries to
    "clean up" an unsafe value, it only ever refuses one."""
    if not segment or not _SAFE_KEY_SEGMENT_RE.match(segment):
        raise ValueError(f"unsafe key segment: {segment!r}")
    return segment


@dataclass(frozen=True)
class DeliveryOwnershipScope:
    """D-269 Stage 2: the ONE immutable ownership identity a delivery
    record and a requesting principal are compared against. Deliberately
    has NO `tenant_id`/`organization_id` field -- this product has no
    tenant/org model today (D-268's own Stage 3 finding); inventing one
    here would be exactly the kind of ungrounded addition this gate's own
    Stage 2 instruction forbids ("Do NOT invent tenant_id if product has
    no tenant model yet")."""

    user_id: str
    project_id: str
    job_id: str

    def __post_init__(self) -> None:
        for name in ("user_id", "project_id", "job_id"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"DeliveryOwnershipScope.{name} must be non-empty")


def build_tenant_safe_export_key(
    *,
    ownership: DeliveryOwnershipScope,
    render_identity: str,
    prefix: str = DEFAULT_TENANT_SAFE_EXPORT_PREFIX,
    content_sha256: str | None = None,
) -> str:
    """D-269 Stage 4/5/6/7: a deterministic, server-side-only remote key
    binding user + project + job + render identity. `content_sha256`
    (D-288.4, pending-review objects only) appends the exact rendered
    bytes' own hash as one more segment, so two encodes that share a
    render identity but differ in bytes (D-267's own documented non-
    determinism) can never overwrite each other -- a review link issued
    for A keeps showing A after B is generated. Deliberately NOT
    `uuid4()`-based (Stage 4: "Do not rely solely on uuid4") -- every
    segment is derived from trusted server-side identity (Stage 4's own
    "generated from trusted server-side identities"), never a raw user-
    controlled filename (Stage 4/18). Being fully deterministic is a
    FEATURE, not an oversight: re-running the exact same job against the
    exact same render plan naturally produces the exact same key, which is
    D-269 Stage 23's own duplicate-job-idempotence requirement satisfied
    structurally (a duplicate delivery overwrites the SAME object rather
    than accumulating a new one) -- collision-freedom comes from the
    user+project+job+render-identity binding itself, not from randomness.
    """
    if not render_identity or not render_identity.startswith("render_"):
        raise ValueError("render_identity must be a compute_render_identity() value")
    segments = [
        _validate_key_segment(_scope_component(ownership.user_id)),
        _validate_key_segment(_scope_component(ownership.project_id)),
        _validate_key_segment(_scope_component(ownership.job_id)),
        _validate_key_segment(render_identity),
    ]
    if content_sha256 is not None:
        segments.append(_validate_key_segment(str(content_sha256)))
    normalized_prefix = prefix if prefix.endswith("/") else f"{prefix}/"
    if ".." in normalized_prefix or normalized_prefix.startswith("/") or "\\" in normalized_prefix:
        raise ValueError("prefix must be a safe relative path")
    return f"{normalized_prefix}{'/'.join(segments)}.mp4"


# =============================================================================
# Stage 3 -- remote object identity
# =============================================================================

@dataclass(frozen=True)
class RemoteDeliveryObject:
    """D-269 Stage 3: what THIS server believes/intends the remote object
    to be -- every field here is locally trusted (computed by this server,
    never read back from an untrusted remote response). Carries no
    credentials (Stage 27); `bucket`/`object_key` are the only storage-
    identifying fields, never a raw presigned URL or secret."""

    ownership: DeliveryOwnershipScope
    render_identity: str
    output_sha256: str
    output_size_bytes: int
    bucket: str
    object_key: str
    remote_size_bytes: int | None = None
    remote_checksum: str | None = None
    upload_status: str = rd.UPLOAD_STATUS_NOT_ATTEMPTED


# =============================================================================
# Stage 9 -- offline-verifiable remote verification contract
# =============================================================================

REMOTE_HASH_STATUS_UNVERIFIED = "REMOTE_HASH_UNVERIFIED"
REMOTE_HASH_STATUS_MATCHED = "REMOTE_HASH_MATCHED"
REMOTE_HASH_STATUS_MISMATCHED = "REMOTE_HASH_MISMATCHED"
REMOTE_HASH_STATUS_NOT_AVAILABLE = "REMOTE_HASH_NOT_AVAILABLE"

VERIFICATION_STATUS_PASS = "PASS"
VERIFICATION_STATUS_FAIL = "FAIL"


@dataclass(frozen=True)
class RemoteObjectMetadataFixture:
    """D-269 Stage 9: a stand-in for a real S3 `head_object` response --
    "no live S3 request required" (Stage 9's own instruction). A future
    real integration constructs this from an actual boto3 call; nothing
    in THIS module ever performs that call. `metadata_sha256`/`metadata_
    render_identity` model an object-metadata header (Stage 15's own
    "metadata header with sha256" option) -- absent (`None`) on any object
    that predates this feature, which is honestly represented as
    NOT_AVAILABLE, never fabricated as a match (Stage 8)."""

    exists: bool
    key: str | None = None
    size_bytes: int | None = None
    metadata_render_identity: str | None = None
    metadata_sha256: str | None = None


@dataclass(frozen=True)
class RemoteVerificationResult:
    object_exists: bool
    remote_key_matches_expected: bool
    remote_size_matches: bool
    render_identity_matches: bool
    local_hash_present: bool
    remote_hash_status: str
    verification_status: str
    errors: tuple[str, ...] = ()


def verify_remote_delivery(
    expected: RemoteDeliveryObject,
    remote: RemoteObjectMetadataFixture,
) -> RemoteVerificationResult:
    """D-269 Stage 9/16/19: compares an EXPECTED (locally, trustedly known)
    remote object against a (real or, here, fixture) remote HEAD-style
    response. Never uses an ETag as a SHA-256 proxy anywhere (Stage 9/16 --
    `remote.metadata_sha256` is an explicit content-hash metadata field,
    never an S3 ETag; see `is_etag_valid_sha256_proxy` below for the
    explicit, permanent refusal to conflate the two)."""
    local_hash_present = bool(expected.output_sha256)

    if not remote.exists:
        return RemoteVerificationResult(
            object_exists=False, remote_key_matches_expected=False, remote_size_matches=False,
            render_identity_matches=False, local_hash_present=local_hash_present,
            remote_hash_status=REMOTE_HASH_STATUS_NOT_AVAILABLE,
            verification_status=VERIFICATION_STATUS_FAIL, errors=("remote_object_missing",),
        )

    errors: list[str] = []
    key_matches = remote.key == expected.object_key
    if not key_matches:
        errors.append("remote_key_mismatch")

    size_matches = remote.size_bytes is not None and int(remote.size_bytes) == int(expected.output_size_bytes)
    if not size_matches:
        errors.append("remote_size_mismatch")

    identity_known = remote.metadata_render_identity is not None
    render_identity_matches = (not identity_known) or (remote.metadata_render_identity == expected.render_identity)
    if identity_known and not render_identity_matches:
        errors.append("remote_render_identity_mismatch")

    if remote.metadata_sha256 is None:
        remote_hash_status = REMOTE_HASH_STATUS_NOT_AVAILABLE if local_hash_present else REMOTE_HASH_STATUS_UNVERIFIED
    elif remote.metadata_sha256 == expected.output_sha256:
        remote_hash_status = REMOTE_HASH_STATUS_MATCHED
    else:
        remote_hash_status = REMOTE_HASH_STATUS_MISMATCHED
        errors.append("remote_hash_mismatch")

    overall_pass = (
        key_matches and size_matches and render_identity_matches
        and remote_hash_status != REMOTE_HASH_STATUS_MISMATCHED
    )
    return RemoteVerificationResult(
        object_exists=True, remote_key_matches_expected=key_matches, remote_size_matches=size_matches,
        render_identity_matches=render_identity_matches, local_hash_present=local_hash_present,
        remote_hash_status=remote_hash_status,
        verification_status=VERIFICATION_STATUS_PASS if overall_pass else VERIFICATION_STATUS_FAIL,
        errors=tuple(errors),
    )


def is_etag_valid_sha256_proxy(etag: str | None) -> bool:  # noqa: ARG001 -- signature is the contract
    """D-269 Stage 9/16: restates `render_delivery.is_etag_valid_sha256_
    proxy`'s own permanent refusal -- an S3 ETag is NEVER a SHA-256 proxy
    in this codebase, in either module. Always `False`."""
    return False


# =============================================================================
# Stage 11/12 -- tenant-safe delivery status vocabulary + the bridge
# =============================================================================

# D-269 Stage 11: only what D-267's own vocabulary cannot already express.
# Reused verbatim everywhere else (RENDER_FAILED/QC_FAILED/HASH_FAILED/
# READY_FOR_UPLOAD/UPLOAD_IN_PROGRESS/UPLOAD_FAILED/DELIVERY_READY/UNKNOWN).
TENANT_DELIVERY_STATUS_OWNERSHIP_MISMATCH = "OWNERSHIP_MISMATCH"
TENANT_DELIVERY_STATUS_WRONG_OBJECT = "WRONG_OBJECT"
TENANT_DELIVERY_STATUS_REMOTE_VERIFY_FAILED = "REMOTE_VERIFY_FAILED"
# STALE_JOB is realized structurally by `is_job_still_current` silently
# skipping the stale update (Stage 22: "old output remains auditable but
# not current") rather than as a delivery-record status -- a stale job's
# own delivery record, if one exists, is not itself invalid; it is simply
# never allowed to become the project's CURRENT pointer. No separate
# STALE_JOB delivery_status value is needed or added (Stage 11: "Do not
# proliferate states unnecessarily").


@dataclass(frozen=True)
class TenantSafeDeliveryRecord:
    """D-269 Stage 10/12: the tenant-safe bridge output. Carries D-267's
    own, UNTOUCHED `RenderDeliveryRecord` (`delivery`) alongside the
    ownership/remote binding this module adds -- never a re-derivation."""

    ownership: DeliveryOwnershipScope
    expected_render_identity: str
    delivery: rd.RenderDeliveryRecord
    remote: RemoteDeliveryObject | None
    remote_verification: RemoteVerificationResult | None
    delivery_status: str
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ready_for_delivery(self) -> bool:
        """D-269 Stage 12: the ONE gate. `DELIVERY_READY` here requires
        everything D-267's own `ready_for_delivery` required (render
        success, real non-empty file, successful hash, QC PASS, upload
        success if required) PLUS render-identity/ownership/remote-object
        binding all matching -- never fabricated from any subset."""
        return self.delivery_status == rd.DELIVERY_STATUS_DELIVERY_READY


def evaluate_tenant_safe_delivery(
    *,
    ownership: DeliveryOwnershipScope,
    expected_render_identity: str,
    delivery: rd.RenderDeliveryRecord,
    remote: RemoteDeliveryObject | None = None,
    remote_verification: RemoteVerificationResult | None = None,
    require_remote: bool = False,
) -> TenantSafeDeliveryRecord:
    """D-269 Stage 10/12: the tenant-safe bridge. Never re-derives D-267's
    own render/QC/hash verdict -- a `delivery.delivery_status` that is
    already anything OTHER than `READY_FOR_UPLOAD`/`DELIVERY_READY`
    (i.e. D-267 already found a render/QC/hash-level blocker) passes
    straight through unchanged. `require_remote=False` (the default)
    mirrors D-267's own local-only contract (its Stage 7): a caller with
    no remote-delivery requirement stops at whatever `delivery.delivery_
    status` already allows. `require_remote=True` additionally requires a
    verified, ownership-matching, render-identity-matching remote object
    with a successful upload before `DELIVERY_READY`."""
    errors: list[str] = list(delivery.errors)

    if delivery.render_identity != expected_render_identity:
        return TenantSafeDeliveryRecord(
            ownership=ownership, expected_render_identity=expected_render_identity, delivery=delivery,
            remote=remote, remote_verification=remote_verification,
            delivery_status=TENANT_DELIVERY_STATUS_WRONG_OBJECT,
            errors=tuple(errors + ["render_identity_mismatch"]),
        )

    if delivery.delivery_status not in (rd.DELIVERY_STATUS_DELIVERY_READY, rd.DELIVERY_STATUS_READY_FOR_UPLOAD):
        return TenantSafeDeliveryRecord(
            ownership=ownership, expected_render_identity=expected_render_identity, delivery=delivery,
            remote=remote, remote_verification=remote_verification,
            delivery_status=delivery.delivery_status, errors=tuple(errors),
        )

    if remote is None:
        if require_remote:
            return TenantSafeDeliveryRecord(
                ownership=ownership, expected_render_identity=expected_render_identity, delivery=delivery,
                remote=None, remote_verification=None,
                delivery_status=TENANT_DELIVERY_STATUS_REMOTE_VERIFY_FAILED,
                errors=tuple(errors + ["remote_object_required_but_absent"]),
            )
        return TenantSafeDeliveryRecord(
            ownership=ownership, expected_render_identity=expected_render_identity, delivery=delivery,
            remote=None, remote_verification=None,
            delivery_status=rd.DELIVERY_STATUS_READY_FOR_UPLOAD, errors=tuple(errors),
        )

    if remote.ownership != ownership:
        return TenantSafeDeliveryRecord(
            ownership=ownership, expected_render_identity=expected_render_identity, delivery=delivery,
            remote=remote, remote_verification=remote_verification,
            delivery_status=TENANT_DELIVERY_STATUS_OWNERSHIP_MISMATCH,
            errors=tuple(errors + ["remote_ownership_mismatch"]),
        )

    if remote.render_identity != expected_render_identity:
        return TenantSafeDeliveryRecord(
            ownership=ownership, expected_render_identity=expected_render_identity, delivery=delivery,
            remote=remote, remote_verification=remote_verification,
            delivery_status=TENANT_DELIVERY_STATUS_WRONG_OBJECT,
            errors=tuple(errors + ["remote_render_identity_mismatch"]),
        )

    if remote_verification is None or remote_verification.verification_status != VERIFICATION_STATUS_PASS:
        verify_errors = remote_verification.errors if remote_verification else ("remote_verification_missing",)
        return TenantSafeDeliveryRecord(
            ownership=ownership, expected_render_identity=expected_render_identity, delivery=delivery,
            remote=remote, remote_verification=remote_verification,
            delivery_status=TENANT_DELIVERY_STATUS_REMOTE_VERIFY_FAILED,
            errors=tuple(errors) + tuple(verify_errors),
        )

    if remote.upload_status != rd.UPLOAD_STATUS_SUCCEEDED:
        status = (
            rd.DELIVERY_STATUS_UPLOAD_FAILED if remote.upload_status == rd.UPLOAD_STATUS_FAILED
            else rd.DELIVERY_STATUS_UPLOAD_IN_PROGRESS
        )
        return TenantSafeDeliveryRecord(
            ownership=ownership, expected_render_identity=expected_render_identity, delivery=delivery,
            remote=remote, remote_verification=remote_verification, delivery_status=status, errors=tuple(errors),
        )

    return TenantSafeDeliveryRecord(
        ownership=ownership, expected_render_identity=expected_render_identity, delivery=delivery,
        remote=remote, remote_verification=remote_verification,
        delivery_status=rd.DELIVERY_STATUS_DELIVERY_READY, errors=tuple(errors),
    )


# =============================================================================
# Stage 13/14/26/27 -- authorization + defense-in-depth + presign contract
# =============================================================================

def authorize_delivery_access(*, requesting: DeliveryOwnershipScope, record_ownership: DeliveryOwnershipScope) -> bool:
    """D-269 Stage 13: pure ownership-match decision. No ID alone grants
    access (Stage 17) -- the FULL scope (user+project+job) must match."""
    return requesting == record_ownership


def assert_delivery_access(*, requesting: DeliveryOwnershipScope | None, record_ownership: DeliveryOwnershipScope) -> None:
    """D-269 Stage 14: defense-in-depth ownership check for a handler/
    service layer that must never rely solely on middleware ordering.
    `requesting=None` means no authenticated principal is available at
    all (auth disabled for local dev) -- mirrors `jobs.py`'s own existing
    `_assert_job_owner(user_id=None)` precedent of skipping the check
    only when there is genuinely no identity to compare against, never
    when there IS one and it disagrees."""
    if requesting is None:
        return
    if not authorize_delivery_access(requesting=requesting, record_ownership=record_ownership):
        raise PermissionError("requesting principal does not own this delivery")


ALLOWED_PRESIGN_DELIVERY_STATUSES = (rd.DELIVERY_STATUS_DELIVERY_READY,)


def authorize_presign_issuance(
    *, requesting: DeliveryOwnershipScope, record: TenantSafeDeliveryRecord,
) -> tuple[bool, str | None]:
    """D-269 Stage 13/26: pure authorization decision for presigned-URL
    issuance -- NO real URL is ever created here (Stage 26: "no real
    network"). Returns `(authorized, denial_reason)`; a real issuance
    call site should refuse to proceed unless this returns `(True, None)`."""
    if not authorize_delivery_access(requesting=requesting, record_ownership=record.ownership):
        return False, "ownership_mismatch"
    if record.delivery_status not in ALLOWED_PRESIGN_DELIVERY_STATUSES:
        return False, f"delivery_not_ready:{record.delivery_status}"
    return True, None


# =============================================================================
# Stage 21/22 -- stale-job / "latest" pointer protection
# =============================================================================

def is_job_still_current(
    *,
    current_latest_job_id: str | None,
    current_latest_job_started_at: float | None,
    candidate_job_id: str,
    candidate_job_started_at: float | None,
) -> bool:
    """D-269 Stage 21/22: should `candidate_job_id`'s own completion be
    allowed to become a project's new "latest" pointer? Uses the
    candidate's own REAL job start timestamp (e.g. RQ's `Job.started_at`/
    `enqueued_at` -- already-existing infrastructure) rather than
    inventing a new sequence number (this gate's own Stage 21 instruction).
    The SAME job is always allowed to re-record itself (idempotent). A
    candidate with no timestamp info, or a project with nothing recorded
    yet, is allowed through -- this guard can only ever be MORE
    conservative than doing nothing when it has real ordering evidence; it
    never blocks the very first completion or a caller that supplies no
    ordering signal at all (preserving every existing caller's current
    behavior exactly)."""
    if current_latest_job_id is not None and str(current_latest_job_id) == str(candidate_job_id):
        return True
    if candidate_job_started_at is None or current_latest_job_started_at is None:
        return True
    return float(candidate_job_started_at) >= float(current_latest_job_started_at)


# =============================================================================
# Stage 20/30 -- observability
# =============================================================================

def tenant_safe_delivery_diagnostics(record: TenantSafeDeliveryRecord) -> dict:
    """D-269 Stage 20/30: machine-readable event/state surface -- ownership
    denied, wrong object, integrity mismatch, remote verification failed,
    delivery ready, etc. No media contents, no credentials (nothing on
    this record ever carries either)."""
    base = rd.render_delivery_diagnostics(record.delivery)
    return {
        **base,
        "tenant_delivery_status": record.delivery_status,
        "ready_for_delivery": record.ready_for_delivery,
        "user_id": record.ownership.user_id,
        "project_id": record.ownership.project_id,
        "job_id": record.ownership.job_id,
        "expected_render_identity": record.expected_render_identity,
        "remote_object_key": record.remote.object_key if record.remote else None,
        "remote_verification_status": (
            record.remote_verification.verification_status if record.remote_verification else None
        ),
        "remote_hash_status": (
            record.remote_verification.remote_hash_status if record.remote_verification else None
        ),
        "tenant_errors": list(record.errors),
    }
