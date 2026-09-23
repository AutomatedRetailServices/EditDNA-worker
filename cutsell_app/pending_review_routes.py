"""D-288.2 (finding 2, blocker 2): the REAL, authenticated HTTP handlers
for a HUMAN_REVIEW_REQUIRED pending render -- querying it privately,
approving/rejecting it, and resuming its delivery once approved.

Every handler below derives the caller's identity EXCLUSIVELY from
`request.state.auth_user_id` -- the value `AuthScopeMiddleware` sets from
a verified bearer session (see `auth_middleware.py`) -- and NEVER from a
request body or query parameter. This is the exact correction the D-288.1
audit named: "Deriva la identidad de la autenticación existente, nunca de
IDs o etiquetas aportadas como sustituto."

`auth_user_id` is `None` only when this deployment's auth enforcement is
disabled (local/dev, see `auth_middleware._auth_required`). Every other
route in this codebase tolerates that (`main.py`'s own `get_job`/`cancel_
processing_job`) by passing `None` through to a downstream ownership
check that then no-ops. These three routes deliberately do NOT tolerate
it -- "Rechaza requesting=None en estas operaciones" is enforced HERE,
at the transport boundary (a 401 before any pending-review function is
ever called), AND AGAIN inside `pending_watch_listen_review.py`'s own
functions (`_require_authenticated_requesting`) -- neither layer trusts
the other to be the only place the check exists.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from cutsell_worker import export_job
from cutsell_worker import pending_watch_listen_review as pwl
from cutsell_worker import tenant_safe_delivery as tsd

router = APIRouter(prefix="/v1/projects/{project_id}/jobs/{job_id}/pending-review", tags=["pending-review"])


def _authenticated_requesting(request: Request, *, project_id: str, job_id: str) -> tsd.DeliveryOwnershipScope:
    """The ONE place these routes resolve the caller's identity. `auth_
    user_id` is the authenticated principal `AuthScopeMiddleware` already
    verified for this request -- never a path/query/body value. A missing
    identity (auth disabled, or -- should the middleware ever be bypassed
    for this path -- no session at all) is a 401, not a silent `None`
    passed through to a downstream check that might no-op on it."""
    auth_user_id = getattr(request.state, "auth_user_id", None)
    if not auth_user_id:
        raise HTTPException(status_code=401, detail="authenticated session required")
    return tsd.DeliveryOwnershipScope(user_id=str(auth_user_id), project_id=project_id, job_id=job_id)


def _record_to_dict(record: pwl.PendingWatchListenRecord) -> dict:
    return {
        "record_id": record.record_id,
        "render_identity": record.render_identity,
        "output_sha256": record.output_sha256,
        "plan_id": record.plan_id,
        "plan_version": record.plan_version,
        "watch_listen_status": record.watch_listen_status,
        "approval_status": record.approval_status,
        "perceptual_review": record.perceptual_review,
        "created_at": record.created_at,
        "approver": record.approver,
        "approved_at": record.approved_at,
        "resumed_delivery_at": record.resumed_delivery_at,
    }


@router.get("")
def get_pending_review(project_id: str, job_id: str, request: Request):
    """"Consulta/revisión privada" -- the perceptual review a human needs
    to see before deciding. Never returns another user's record: `user_id`
    in the lookup is the AUTHENTICATED caller's own id, so a mismatched
    `job_id`/`project_id` combination simply finds no record (404), it can
    never leak a different user's pending review by URL alone."""
    requesting = _authenticated_requesting(request, project_id=project_id, job_id=job_id)
    try:
        record = pwl.get_pending_review_for_authenticated_caller(
            user_id=requesting.user_id, project_id=project_id, job_id=job_id, requesting=requesting,
        )
    except pwl.PendingReviewError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    if record is None:
        raise HTTPException(status_code=404, detail="no pending review found")
    return _record_to_dict(record)


class PendingReviewDecisionRequest(BaseModel):
    approved: bool
    expected_render_identity: str
    expected_output_sha256: str
    expected_plan_id: str
    expected_plan_version: int
    # Deliberately NO `approver`/`user_id` field: the approver identity is
    # ALWAYS the authenticated caller (`request.state.auth_user_id`),
    # never a client-supplied label.


@router.get("/media")
def get_pending_review_media(
    project_id: str, job_id: str, expected_render_identity: str, expected_output_sha256: str, request: Request,
):
    """D-288.3 (blocker 4): "acceso autenticado al MP4 privado para
    revisarlo antes de aprobar" -- a short-lived presigned URL for the
    ACTUAL pending file, so the authenticated owner can watch/listen to
    it before deciding. Deliberately a SEPARATE endpoint from `GET
    ""` (metadata only): never returns a `delivery_status`, never touches
    `watch_listen_status`/`approval_status`, and lives under this job's
    private review namespace only -- reading it can never, by itself,
    promote anything toward Ready. `expected_render_identity`/`expected_
    output_sha256` are REQUIRED query params (the same two-field binding
    `PendingReviewDecisionRequest` already requires for a decision) so a
    stale client reference to a PREVIOUS pending record for this job_id
    is refused rather than silently handed a preview of whatever file
    occupies this job's CURRENT slot now."""
    requesting = _authenticated_requesting(request, project_id=project_id, job_id=job_id)
    try:
        return pwl.get_pending_review_media_access(
            user_id=requesting.user_id, project_id=project_id, job_id=job_id,
            requesting=requesting,
            expected_render_identity=expected_render_identity,
            expected_output_sha256=expected_output_sha256,
        )
    except pwl.PendingReviewError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.post("/decision")
def submit_pending_review_decision(project_id: str, job_id: str, payload: PendingReviewDecisionRequest, request: Request):
    """Approve or reject (including REVOKE an earlier approval -- the
    same endpoint, `approved=False` again) a pending review. `approver`
    is always the authenticated caller's own id -- never accepted from
    `payload`."""
    requesting = _authenticated_requesting(request, project_id=project_id, job_id=job_id)
    try:
        record = pwl.apply_human_approval(
            user_id=requesting.user_id, project_id=project_id, job_id=job_id,
            requesting=requesting,
            approved=payload.approved,
            approver=requesting.user_id,
            expected_render_identity=payload.expected_render_identity,
            expected_output_sha256=payload.expected_output_sha256,
            expected_plan_id=payload.expected_plan_id,
            expected_plan_version=payload.expected_plan_version,
        )
    except pwl.PendingReviewError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return _record_to_dict(record)


@router.post("/resume-delivery")
def resume_pending_review_delivery(project_id: str, job_id: str, request: Request):
    """"Reanudación de entrega" -- only ever succeeds for a record this
    SAME endpoint's own approval decision already promoted to HUMAN_
    APPROVED. Idempotent: a repeat call after a successful resume returns
    the SAME stored result, never a second delivery/notification."""
    requesting = _authenticated_requesting(request, project_id=project_id, job_id=job_id)
    try:
        return export_job.resume_delivery_after_approval(
            user_id=requesting.user_id, project_id=project_id, job_id=job_id, requesting=requesting,
        )
    except pwl.PendingReviewError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except export_job.TenantSafeDeliveryBlocked as exc:
        raise HTTPException(status_code=422, detail=f"tenant_safe_delivery_blocked:{exc.record.delivery_status}") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
