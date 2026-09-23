"""RQ export job: edited Draft Timeline -> final MP4 -> scoped S3 URL."""
from __future__ import annotations

import dataclasses
from pathlib import Path
import tempfile
import time

from . import pending_watch_listen_review as pwl
from . import render_delivery as rd
from . import tenant_safe_delivery as tsd
from .draft_edits import DraftEditError
from .editorial_slot_resolution_install import reset_editorial_slot_resolution_evidence
from .exports import store_export
from .live_render_qc import PostRenderQCFailure, render_with_post_render_qc
from .media_overlay_render import LocalMediaOverlay
from .notifications import publish_notification
from .overlay_uploads import validate_overlay_uri
from .perceptual_watch_listen import WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
from .project_tracking import safe_update_project
from .render import RENDER_FPS_DEFAULT
from .render_plan import build_render_plan
from .render_versions import add_render_version
from .serde import draft_from_dict
from .storage import download_source
from .universal_clean_cut_validation import perceptual_review_for_rendered_candidate
from .uploads import validate_product_source_uri

# D-269A Stage 2/4: the SAME output geometry `render.render_preview`'s own
# defaults use -- never a new decision, just threading the existing
# constants so `render_delivery.compute_render_identity` sees exactly the
# geometry this job actually renders at (it never passes width/height
# explicitly to `render_with_post_render_qc`, so these are its real
# effective values).
_RENDER_WIDTH = 1080
_RENDER_HEIGHT = 1920


class TenantSafeDeliveryBlocked(RuntimeError):
    """D-269A: the rendered/QC-passed candidate could not be safely bound
    to a tenant-scoped remote delivery (ownership/render-identity/remote-
    verification/upload invariant not fully satisfied). Never delivered,
    never published as the project's current/latest result -- the
    existing generic `except Exception` handler below treats this exactly
    like any other export failure (state="failed", render_failed
    notification), reusing all existing failure bookkeeping rather than
    duplicating it."""

    def __init__(self, record: "tsd.TenantSafeDeliveryRecord", *, perceptual_review: dict | None = None):
        super().__init__(f"tenant_safe_delivery_blocked:{record.delivery_status}")
        self.record = record
        # D-288: the perceptual System Watch+Listen review that produced
        # this block (when the block was watch-listen-caused) -- carried
        # separately from `record` so a caller/notification can distinguish
        # "this candidate never technically rendered/uploaded" from "this
        # candidate rendered fine but is perceptually BLOCKED/HUMAN_REVIEW_
        # REQUIRED", never conflating the two under one generic failure.
        self.perceptual_review = perceptual_review


class PendingHumanWatchListenReview(RuntimeError):
    """D-288 (finding 1): a technically-passing candidate is held at
    `WATCH_LISTEN_PENDING` (perceptual `HUMAN_REVIEW_REQUIRED`) -- NOT a
    render/delivery failure. The actual rendered file has already been
    persisted to private, recoverable storage (`pending_watch_listen_
    review.persist_pending_review`, called BEFORE this is raised, so the
    bytes are safely in S3 before the caller's own `TemporaryDirectory`
    can ever close) by the time this is raised. Callers must handle this
    SEPARATELY from `TenantSafeDeliveryBlocked` -- a pending review is a
    recoverable, resumable state, never `state="failed"`."""

    def __init__(self, record: "pwl.PendingWatchListenRecord"):
        super().__init__(f"pending_human_watch_listen_review:{record.record_id}")
        self.record = record


def _job_started_epoch(job) -> float | None:
    """D-269A Stage 9/10: use the job's OWN existing, already-real start
    timestamp (RQ's `Job.started_at`, set by the worker before this
    function runs) as the stale-job guard's ordering evidence -- never an
    invented sequence number. Handles a real `datetime`, a test double
    supplying a plain float/int, or nothing at all (returns `None`, which
    `project_store.update_project`'s own guard already treats as "no
    ordering evidence, allow through" -- preserving every existing
    caller's behavior when no timestamp is available)."""
    started = getattr(job, "started_at", None)
    if started is None:
        return None
    if hasattr(started, "timestamp"):
        try:
            return float(started.timestamp())
        except Exception:
            return None
    try:
        return float(started)
    except (TypeError, ValueError):
        return None


def _tenant_safe_deliver(
    *,
    output_path: str,
    plan,
    project_id: str,
    user_id: str,
    job_id: str,
    draft=None,
    local_paths: dict | None = None,
    qc_result=None,
    requesting: "tsd.DeliveryOwnershipScope | None" = None,
    pending_review_redis_client=None,
) -> dict:
    """D-269A: the live activation seam -- binds the actual rendered/QC-
    passed local file to a tenant-safe remote delivery. Computes D-267's
    own `render_identity` from the SAME `plan` the renderer used (never
    from a filename/path), uploads via the EXISTING `store_export`
    interface (Stage 5: no parallel uploader) but with the D-269 tenant-
    safe key (Stage 4), consumes the real (or, in tests, faked) post-
    upload HEAD response through `verify_remote_delivery` (Stage 6), and
    only ever returns a dict implying delivery once D-267's own render/
    QC/hash gate AND D-269's ownership/render-identity/remote-
    verification/upload invariant are ALL satisfied (Stage 7) -- anything
    less raises `TenantSafeDeliveryBlocked` rather than fabricating a
    ready state. `requesting` defaults to this export's own ownership
    scope: the real RQ export flow is always its own job's owner and can
    never diverge from it; a caller (a test) may pass a different
    `requesting` scope to prove the presign-authorization seam (Stage 13)
    denies a mismatched principal."""
    ownership = tsd.DeliveryOwnershipScope(user_id=user_id, project_id=project_id, job_id=job_id)
    if requesting is None:
        requesting = ownership

    render_identity = rd.compute_render_identity(
        tuple(plan), width=_RENDER_WIDTH, height=_RENDER_HEIGHT, fps=RENDER_FPS_DEFAULT,
    )

    # D-288: run the perceptual System Watch+Listen review on the ACTUAL
    # rendered file before this candidate can reach DELIVERY_READY. Before
    # this fix, `run_export_job` never called `perceptual_watch_listen`
    # (or anything in `universal_clean_cut_validation.py`) at all -- a
    # technical-QC PASS alone was sufficient for this job to mark the
    # project "finished" and hand back a real `download_url`. Never raises
    # (`review_rendered_candidate`'s own "never raises" contract, an
    # internal exception is reported as capability ERROR); an ERROR
    # capability resolves `watch_listen_status` to BLOCKED (D-154/D-155),
    # never silently skipped or treated as PASS.
    #
    # `draft is None` is the pre-D-288 calling convention (this function's
    # own lower-level test coverage exercises ONLY the D-269A remote/
    # ownership/upload contract and never supplies one) -- the D-288 gate
    # is additive/opt-in exactly like `build_render_delivery_record`'s own
    # `watch_listen_status=None` contract, so it is left UNAPPLIED for such
    # a caller rather than fabricating a verdict with no real review input.
    # The REAL caller, `run_export_job` below, always supplies a real
    # `draft`/`local_paths`/`qc_result`, so the production path is always
    # gated.
    if draft is None:
        perceptual = None
        watch_listen_status = None
    else:
        perceptual = perceptual_review_for_rendered_candidate(output_path, draft, local_paths or {}, qc_result)
        watch_listen_status = (perceptual or {}).get("watch_listen_status") or WATCH_LISTEN_HUMAN_REVIEW_REQUIRED

    local_delivery = rd.build_render_delivery_record(
        render_identity=render_identity,
        render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=output_path,
        technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
        require_upload=True,
        upload_status=rd.UPLOAD_STATUS_NOT_ATTEMPTED,
        project_id=project_id,
        job_id=job_id,
        watch_listen_status=watch_listen_status,
    )
    if local_delivery.delivery_status == rd.DELIVERY_STATUS_WATCH_LISTEN_PENDING:
        # D-288 (finding 1): HUMAN_REVIEW_REQUIRED -- a technically clean
        # candidate held for human review, NOT a failure. Persist the
        # ACTUAL rendered file to private, recoverable storage BEFORE
        # this function returns/raises, so it survives the caller's own
        # `TemporaryDirectory` cleanup. `plan_id`/`plan_version` come from
        # the SAME `qc_result` the technical loop already produced --
        # never re-derived, never guessed.
        pending_record = pwl.persist_pending_review(
            local_path=output_path,
            user_id=user_id, project_id=project_id, job_id=job_id,
            render_identity=render_identity, output_sha256=local_delivery.output_sha256 or "",
            plan_id=str(getattr(qc_result, "plan_id", "") or ""),
            plan_version=int(getattr(qc_result, "plan_version", 0) or 0),
            watch_listen_status=watch_listen_status,
            perceptual_review=perceptual,
            # D-288: route through THIS module's own `store_export` name
            # (monkeypatchable by tests, exactly like the tenant-safe
            # upload path below) -- never a second, independent reference.
            store_export_fn=store_export,
            client=pending_review_redis_client,
            # D-288.2 (blocker 3): carried so a LATER resumed delivery can
            # reuse the exact same finalization this job's own same-job
            # path uses, without needing this `draft` object again.
            selected_count=len(draft.selected) if draft is not None else 0,
            text_overlay_count=len(getattr(draft, "text_overlays", ()) or ()) if draft is not None else 0,
            media_overlay_count=len(getattr(draft, "media_overlays", ()) or ()) if draft is not None else 0,
        )
        raise PendingHumanWatchListenReview(pending_record)

    if local_delivery.delivery_status != rd.DELIVERY_STATUS_READY_FOR_UPLOAD:
        # D-267's own render/hash/QC gate (or, as of D-288, a WATCH_LISTEN_
        # BLOCKED perceptual verdict) already found a confirmed blocker --
        # never re-derived, never overridden here, only wrapped as a
        # tenant-safe-delivery block so the caller has one exception type.
        # Deliberately NOT persisted to pending-review storage (unlike the
        # PENDING branch above): a confirmed BLOCKED defect has no review
        # value, and "BLOCKED nunca entrega" holds regardless of storage.
        # `perceptual_review` is attached so the caller can tell a
        # watch-listen hold apart from a genuine render/hash/upload
        # failure.
        blocked = tsd.evaluate_tenant_safe_delivery(
            ownership=ownership, expected_render_identity=render_identity, delivery=local_delivery,
        )
        raise TenantSafeDeliveryBlocked(blocked, perceptual_review=perceptual)

    return _upload_verify_and_finalize_delivery(
        local_path=output_path, ownership=ownership, requesting=requesting,
        render_identity=render_identity, local_delivery=local_delivery,
        project_id=project_id, user_id=user_id, job_id=job_id,
        watch_listen_status=watch_listen_status,
    )


def _upload_verify_and_finalize_delivery(
    *,
    local_path: str,
    ownership: "tsd.DeliveryOwnershipScope",
    requesting: "tsd.DeliveryOwnershipScope",
    render_identity: str,
    local_delivery: "rd.RenderDeliveryRecord",
    project_id: str,
    user_id: str,
    job_id: str,
    watch_listen_status: str | None,
) -> dict:
    """D-288: the real tenant-safe upload/verify/evaluate/result-build
    seam, shared by `_tenant_safe_deliver` (same-job delivery) and
    `resume_delivery_after_approval` (finding 2's "reanudación de
    entrega" -- a LATER job resuming delivery of an already-approved
    pending review) -- one implementation, never a second guess. The
    caller is responsible for the D-267/D-288 local gate (`local_
    delivery.delivery_status == READY_FOR_UPLOAD`) having already passed
    before calling this."""
    tenant_key = tsd.build_tenant_safe_export_key(ownership=ownership, render_identity=render_identity)
    stored = store_export(
        local_path,
        project_id=project_id,
        user_id=user_id,
        object_key=tenant_key,
        object_metadata={
            "render_identity": render_identity,
            "sha256": local_delivery.output_sha256 or "",
            "job_id": job_id,
        },
    )

    uploaded_delivery = rd.with_upload_result(
        local_delivery,
        upload_status=rd.UPLOAD_STATUS_SUCCEEDED,
        remote_reference=stored.get("export_uri"),
        remote_size_bytes=stored.get("size_bytes"),
    )
    remote_head = stored.get("remote_head") or {}
    remote_object = tsd.RemoteDeliveryObject(
        ownership=ownership,
        render_identity=render_identity,
        output_sha256=uploaded_delivery.output_sha256,
        output_size_bytes=uploaded_delivery.output_size_bytes,
        bucket=str(stored.get("bucket") or ""),
        object_key=str(stored.get("object_key") or tenant_key),
        remote_size_bytes=remote_head.get("size_bytes"),
        upload_status=rd.UPLOAD_STATUS_SUCCEEDED,
    )
    remote_metadata = dict(remote_head.get("metadata") or {})
    remote_fixture = tsd.RemoteObjectMetadataFixture(
        exists=bool(remote_head.get("exists")),
        key=remote_head.get("key"),
        size_bytes=remote_head.get("size_bytes"),
        metadata_render_identity=remote_metadata.get("render_identity") or None,
        metadata_sha256=remote_metadata.get("sha256") or None,
    )
    verification = tsd.verify_remote_delivery(remote_object, remote_fixture)
    record = tsd.evaluate_tenant_safe_delivery(
        ownership=ownership,
        expected_render_identity=render_identity,
        delivery=uploaded_delivery,
        remote=remote_object,
        remote_verification=verification,
        require_remote=True,
    )

    if not record.ready_for_delivery:
        raise TenantSafeDeliveryBlocked(record)

    result = {
        "delivery_status": record.delivery_status,
        # D-288: real value, real trace -- this candidate reached delivery
        # only because `watch_listen_status` cleared the D-288 gate above.
        "watch_listen_status": watch_listen_status,
        "render_identity": render_identity,
        "output_sha256": uploaded_delivery.output_sha256,
        "remote_reference": stored.get("export_uri"),
        "expires_in": stored.get("expires_in"),
        "size_bytes": stored.get("size_bytes"),
        # D-269A Stage 24/25: the raw `export_uri` is preserved for
        # existing client/API compatibility (Stage 25 -- no security
        # finding requires removing it yet) alongside the safer,
        # structured `remote_reference` field (Stage 24), which carries
        # the identical value under a name that does not read as a raw
        # internal storage path.
        "export_uri": stored.get("export_uri"),
    }
    authorized, denial_reason = tsd.authorize_presign_issuance(requesting=requesting, record=record)
    if authorized:
        result["download_url"] = stored.get("download_url")
    else:
        # Stage 13: no presigned download reference is ever published to
        # a mismatched requesting principal, even though `store_export`'s
        # own existing upload+presign call already ran as one atomic
        # legacy interface call (Stage 5 keeps that interface unchanged;
        # this gate gates PUBLICATION of its result, not its generation).
        result["download_url"] = None
        result["presign_denied_reason"] = denial_reason
    return result


def resume_delivery_after_approval(
    *,
    user_id: str,
    project_id: str,
    job_id: str,
    requesting: "tsd.DeliveryOwnershipScope | None",
    client=None,
    s3_client=None,
) -> dict:
    """D-288.2 (finding 2, blockers 1 and 3): "reanudación de entrega" --
    given a pending review record that has ALREADY been approved via
    `pending_watch_listen_review.apply_human_approval` (this function
    performs NO approval logic itself and re-derives nothing about that
    decision; it only trusts the persisted record's own `watch_listen_
    status` AND `approval_status`, which `apply_human_approval` is the
    ONE place that can ever set), downloads the private pending artifact
    and completes the SAME tenant-safe upload/verify/evaluate/finalize
    seam `run_export_job`'s own same-job path uses -- one implementation,
    never a second guess, so a resumed delivery reaches the exact same
    finished state (registered render version, project `state="finished"`,
    a real `render_finished` notification).

    `requesting` is REQUIRED (no default, `None` explicitly rejected by
    `pending_watch_listen_review._require_authenticated_requesting`) --
    this is a LATER, separate operation triggered by a real human's own
    authenticated request, never the trusted same-job worker context
    `_tenant_safe_deliver`'s own `requesting=None -> requesting=ownership`
    default is deliberately scoped to (see that function's own docstring
    for why that default is safe there and not here). Never derive it
    from `user_id`/`project_id`/`job_id` -- those name the RESOURCE being
    acted on, not the caller's own identity; the real route handler
    (`cutsell_app/pending_review_routes.py`) resolves `requesting` from
    `request.state.auth_user_id`, never from a request body/query field.

    Idempotent (blocker 3, "evita duplicados al repetir la petición"): a
    record whose `resumed_delivery_result` is already set (a PRIOR
    successful resume) returns that STORED result directly -- no re-
    upload, no second render-version registration, no duplicate
    notification."""
    pwl._require_authenticated_requesting(requesting)

    from .perceptual_watch_listen import WATCH_LISTEN_HUMAN_APPROVED

    ownership = tsd.DeliveryOwnershipScope(user_id=user_id, project_id=project_id, job_id=job_id)

    record = pwl.load_pending_review(user_id=user_id, project_id=project_id, job_id=job_id, client=client)
    if record is None:
        raise pwl.PendingReviewError("no_pending_review_found")
    tsd.assert_delivery_access(requesting=requesting, record_ownership=record.ownership)

    if record.resumed_delivery_result is not None:
        return dict(record.resumed_delivery_result)

    # D-288.2 (blocker 1): BOTH fields must agree -- never trust `watch_
    # listen_status` alone (the exact single-field check the revocation
    # bug slipped through). A record whose approval was later revoked has
    # `watch_listen_status` reverted (by `apply_human_approval`'s own
    # fix) but this redundant check means even a hypothetical future bug
    # in that revert would still be caught here, never silently deliver.
    if record.watch_listen_status != WATCH_LISTEN_HUMAN_APPROVED or record.approval_status != pwl.APPROVAL_STATUS_APPROVED:
        raise pwl.PendingReviewError(
            f"pending_review_not_approved:{record.watch_listen_status}:{record.approval_status}"
        )

    with tempfile.TemporaryDirectory(prefix="cutsell-resume-delivery-") as directory:
        local_path = str(Path(directory) / "cutsell-resume-export.mp4")
        parsed_bucket, parsed_key = record.pending_s3_uri[5:].split("/", 1)
        if s3_client is None:
            import boto3
            s3_client = boto3.client("s3")
        s3_client.download_file(parsed_bucket, parsed_key, local_path)

        local_delivery = rd.build_render_delivery_record(
            render_identity=record.render_identity,
            render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
            final_path=local_path,
            technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
            require_upload=True,
            upload_status=rd.UPLOAD_STATUS_NOT_ATTEMPTED,
            project_id=project_id,
            job_id=job_id,
            watch_listen_status=record.watch_listen_status,
        )
        if local_delivery.output_sha256 != record.output_sha256:
            # The downloaded bytes do not match what was approved -- never
            # deliver them. This is a hard integrity failure, not a
            # watch-listen state; it never reaches the tenant-safe upload.
            raise pwl.PendingReviewError("resumed_artifact_hash_mismatch")
        if local_delivery.delivery_status != rd.DELIVERY_STATUS_READY_FOR_UPLOAD:
            blocked = tsd.evaluate_tenant_safe_delivery(
                ownership=ownership, expected_render_identity=record.render_identity, delivery=local_delivery,
            )
            raise TenantSafeDeliveryBlocked(blocked)

        delivery = _upload_verify_and_finalize_delivery(
            local_path=local_path, ownership=ownership, requesting=requesting,
            render_identity=record.render_identity, local_delivery=local_delivery,
            project_id=project_id, user_id=user_id, job_id=job_id,
            watch_listen_status=record.watch_listen_status,
        )

        # D-288.2 (blocker 3): reuse the SAME finalization the same-job
        # path uses -- render version registered, project "finished",
        # real notification fired. Job-start timestamp is unknown here
        # (this is not itself a fresh render job); `None` is `project_
        # store.update_project`'s own documented "no ordering evidence,
        # allow through" sentinel, not a fabricated value.
        finalized = _finalize_successful_delivery(
            user_id=user_id, project_id=project_id, job_id=job_id, job_started_at=None,
            delivery=delivery, selected_count=record.selected_count,
            text_overlay_count=record.text_overlay_count, media_overlay_count=record.media_overlay_count,
        )

        pwl._save(
            dataclasses.replace(record, resumed_delivery_at=time.time(), resumed_delivery_result=finalized),
            client=client,
        )
        return finalized


def _finalize_successful_delivery(
    *,
    user_id: str,
    project_id: str,
    job_id: str,
    job_started_at: float | None,
    delivery: dict,
    selected_count: int,
    text_overlay_count: int,
    media_overlay_count: int,
) -> dict:
    """D-288.2 (blocker 3): the ONE finalization sequence for ANY
    successfully, verifiably delivered candidate -- registers the render
    version, moves the project to `state="finished"`, and fires the
    `render_finished` notification. Shared by `run_export_job`'s own
    same-job delivery and `resume_delivery_after_approval`'s later-job
    resumed delivery -- one implementation, never a second guess, so a
    resumed delivery reaches the EXACT SAME finished state a same-job
    delivery does (registered render version, "finished" project state,
    a real notification an approving human/mobile client can observe)."""
    version_payload: dict = {}
    version = None
    try:
        version = add_render_version(
            user_id=user_id,
            project_id=project_id,
            export_uri=delivery["export_uri"],
            size_bytes=delivery["size_bytes"],
            selected_count=selected_count,
            text_overlay_count=text_overlay_count,
            media_overlay_count=media_overlay_count,
        )
        version_payload = {
            "render_version_status": "saved",
            "render_version_id": version["render_version_id"],
        }
    except Exception as exc:
        version_payload = {
            "render_version_status": "degraded",
            "render_version_reason": exc.__class__.__name__,
        }

    project_tracking = safe_update_project(
        user_id=user_id,
        project_id=project_id,
        state="finished",
        latest_job_id=job_id,
        latest_job_started_at=job_started_at,
        render_version=(
            {
                "render_version_id": version["render_version_id"],
                "created_at": version["created_at"],
                "size_bytes": version["size_bytes"],
            }
            if version else None
        ),
    )
    notification = _safe_notify(
        user_id=user_id,
        project_id=project_id,
        kind="render_finished",
        payload={
            "job_id": job_id,
            "render_version_id": version["render_version_id"] if version else None,
        },
    )
    return {
        "project_id": project_id,
        "state": "finished",
        "selected_count": selected_count,
        "text_overlay_count": text_overlay_count,
        "media_overlay_count": media_overlay_count,
        "project_tracking": project_tracking,
        "notification": notification,
        **version_payload,
        **delivery,
    }


def _safe_notify(*, user_id: str, project_id: str, kind: str, payload: dict | None = None) -> dict:
    try:
        event = publish_notification(
            user_id=user_id,
            project_id=project_id,
            kind=kind,
            payload=payload,
        )
        return {"status": "queued", "notification_id": event["notification_id"]}
    except Exception as exc:
        return {"status": "degraded", "reason": exc.__class__.__name__}


def run_export_job(payload: dict) -> dict:
    # D-288 (finding 5): the real per-job init boundary for the
    # EditorialSlotResolution evidence ContextVar -- see universal_clean_
    # cut_validation.run_single_universal_clean_cut_validation's own D-288
    # comment for why this must run before any arbiter call this job
    # could make.
    reset_editorial_slot_resolution_evidence()
    from rq import get_current_job

    job = get_current_job()

    def publish(stage: str, percent: int) -> None:
        if job is None:
            return
        job.meta["stage"] = stage
        job.meta["progress_percent"] = max(0, min(100, int(percent)))
        job.save_meta()

    project_id = str(payload["project_id"])
    user_id = str(payload["user_id"])
    job_id = str(getattr(job, "id", "") or "") or None
    draft = draft_from_dict(dict(payload["draft"]))
    if draft.project_id != project_id:
        raise DraftEditError("draft project does not match export project")
    sources = list(payload.get("sources") or ())
    if not sources:
        raise ValueError("export requires source metadata")

    job_started_at = _job_started_epoch(job)

    tracking_start = safe_update_project(
        user_id=user_id,
        project_id=project_id,
        state="rendering",
        latest_job_id=job_id,
        latest_job_started_at=job_started_at,
    )

    try:
        publish("rendering", 2)
        with tempfile.TemporaryDirectory(prefix="cutsell-export-") as directory:
            local_paths = {}
            seen_source_ids = set()
            for index, item in enumerate(sources):
                source_id = str(item["source_asset_id"])
                if source_id in seen_source_ids:
                    raise ValueError("export source_asset_id values must be unique")
                seen_source_ids.add(source_id)
                uri = str(item["uri"])
                validate_product_source_uri(uri, project_id=project_id, user_id=user_id)
                suffix = Path(str(item.get("original_name") or "source.mp4")).suffix or ".mp4"
                destination = str(Path(directory) / f"source-{index:03d}-{source_id}{suffix}")
                local_paths[source_id] = download_source(uri, destination)
                publish("rendering", min(25, 5 + int((index + 1) * 20 / len(sources))))

            required_ids = {clip.source_asset_id for clip in draft.selected}
            missing = required_ids - set(local_paths)
            if missing:
                raise ValueError("export is missing selected source assets")

            local_overlays = []
            overlay_count = max(1, len(draft.media_overlays))
            for index, overlay in enumerate(draft.media_overlays):
                _bucket, _key, actual_kind = validate_overlay_uri(
                    overlay.uri, user_id=user_id, project_id=project_id
                )
                if actual_kind != overlay.kind:
                    raise ValueError("media overlay kind does not match its S3 object")
                suffix = Path(overlay.uri).suffix or (".jpg" if overlay.kind == "photo" else ".mp4")
                destination = str(Path(directory) / f"overlay-{index:03d}{suffix}")
                download_source(overlay.uri, destination)
                local_overlays.append(LocalMediaOverlay(overlay=overlay, path=destination))
                publish("rendering", min(32, 26 + int((index + 1) * 6 / overlay_count)))

            plan = build_render_plan(draft, local_paths)
            output = str(Path(directory) / "cutsell-export.mp4")
            publish("rendering", 35)
            # D-030: live PostRenderWatchListenQC + bounded physical repair,
            # run against the ACTUAL local rendered file, before this job
            # ever calls store_export -- never a downloaded-back artifact.
            # A SEMANTIC_MISMATCH_INVALIDATED or NEEDS_HUMAN_REVIEW result
            # raises PostRenderQCFailure below, which this job's own
            # except-block already treats as a hard failure (state="failed",
            # render_failed notification) -- this candidate is never
            # delivered/uploaded.
            qc_result = render_with_post_render_qc(
                draft,
                plan,
                output,
                text_overlays=draft.text_overlays,
                media_overlays=tuple(local_overlays),
            )
            if qc_result.status != "PASS":
                raise PostRenderQCFailure(qc_result)
            publish("rendering", 85)
            # D-269A: the live tenant-safe delivery seam -- render_identity
            # binding, D-269 tenant-safe key, real upload via the existing
            # `store_export` interface, and post-upload remote verification
            # all happen here. Raises (caught by the generic except-block
            # below, exactly like any other export failure) rather than
            # ever fabricating a ready/finished state. A real RQ worker
            # always assigns `job.id` before invoking this function --
            # `DeliveryOwnershipScope` itself requires a non-empty job_id
            # (Stage 2), so a genuinely job-less invocation surfaces as a
            # plain, clearly-labelled failure here rather than silently
            # skipping tenant-safe delivery.
            if not job_id:
                raise ValueError("tenant_safe_delivery_requires_job_id")
            delivery = _tenant_safe_deliver(
                output_path=output, plan=plan, draft=draft, local_paths=local_paths, qc_result=qc_result,
                project_id=project_id, user_id=user_id, job_id=job_id,
            )
            finalized = _finalize_successful_delivery(
                user_id=user_id, project_id=project_id, job_id=job_id, job_started_at=job_started_at,
                delivery=delivery, selected_count=len(draft.selected),
                text_overlay_count=len(draft.text_overlays), media_overlay_count=len(draft.media_overlays),
            )
            publish("finished", 100)
            return {
                **finalized,
                "project_tracking_start": tracking_start,
                # D-030: the delivered candidate's exact plan identity and
                # post-render QC/repair history.
                "post_render_qc_status": qc_result.status,
                "plan_id": qc_result.plan_id,
                "plan_version": qc_result.plan_version,
                "semantic_hash": qc_result.semantic_hash,
                "render_attempt_count": len(qc_result.attempts),
            }
    except PendingHumanWatchListenReview as exc:
        # D-288 (finding 1): a technically-clean candidate held for human
        # review -- NOT a render/delivery failure. The rendered file is
        # already safely persisted (see `PendingHumanWatchListenReview`'s
        # own docstring); this job completes with a recoverable pending
        # result rather than an unhandled failure, and the project state
        # is a distinct "pending_review", never "failed".
        safe_update_project(
            user_id=user_id,
            project_id=project_id,
            state="pending_review",
            latest_job_id=job_id,
            latest_job_started_at=job_started_at,
        )
        notification = _safe_notify(
            user_id=user_id,
            project_id=project_id,
            kind="render_pending_human_watch_listen",
            payload={
                "job_id": job_id,
                "record_id": exc.record.record_id,
                "watch_listen_status": exc.record.watch_listen_status,
                "render_identity": exc.record.render_identity,
                "plan_id": exc.record.plan_id,
                "plan_version": exc.record.plan_version,
            },
        )
        return {
            "project_id": project_id,
            "state": "pending_review",
            "watch_listen_status": exc.record.watch_listen_status,
            "pending_review_record_id": exc.record.record_id,
            "pending_review_render_identity": exc.record.render_identity,
            "pending_review_output_sha256": exc.record.output_sha256,
            "pending_review_plan_id": exc.record.plan_id,
            "pending_review_plan_version": exc.record.plan_version,
            "project_tracking_start": tracking_start,
            "notification": notification,
        }
    except PostRenderQCFailure as exc:
        # Never delivered: PostRenderWatchListenQC (or the bounded physical
        # repair loop) did not reach PASS on this candidate. Record exactly
        # which plan it was and why, per D-030's observability requirement --
        # this candidate's plan_id/version/hash is recorded even on failure.
        safe_update_project(
            user_id=user_id,
            project_id=project_id,
            state="failed",
            latest_job_id=job_id,
            latest_job_started_at=job_started_at,
        )
        _safe_notify(
            user_id=user_id,
            project_id=project_id,
            kind="render_failed",
            payload={
                "job_id": job_id,
                "error": exc.__class__.__name__,
                "post_render_qc_status": exc.result.status,
                "plan_id": exc.result.plan_id,
                "plan_version": exc.result.plan_version,
                "semantic_hash": exc.result.semantic_hash,
                "render_attempt_count": len(exc.result.attempts),
            },
        )
        raise
    except TenantSafeDeliveryBlocked as exc:
        # D-288: never delivered -- distinct from PostRenderQCFailure above
        # (that candidate never even reached a passing technical QC) and
        # from the generic Exception branch below (an infra/upload/hash
        # failure). This branch specifically covers a technically-PASSing
        # render that the D-288 watch-listen gate (or D-269's own remote/
        # upload verification) refused to mark ready -- the notification
        # payload carries `watch_listen_status`/`delivery_status` so this
        # is never silently indistinguishable from a genuine render
        # failure, per this gate's own "distingue el repair loop editorial
        # previo a Freeze, el técnico y el perceptual posterior al render"
        # requirement.
        safe_update_project(
            user_id=user_id,
            project_id=project_id,
            state="failed",
            latest_job_id=job_id,
            latest_job_started_at=job_started_at,
        )
        perceptual = exc.perceptual_review or {}
        _safe_notify(
            user_id=user_id,
            project_id=project_id,
            kind="render_failed",
            payload={
                "job_id": job_id,
                "error": exc.__class__.__name__,
                "delivery_status": exc.record.delivery_status,
                "watch_listen_status": perceptual.get("watch_listen_status"),
                "watch_listen_gated": perceptual.get("watch_listen_status") is not None,
            },
        )
        raise
    except Exception as exc:
        safe_update_project(
            user_id=user_id,
            project_id=project_id,
            state="failed",
            latest_job_id=job_id,
            latest_job_started_at=job_started_at,
        )
        _safe_notify(
            user_id=user_id,
            project_id=project_id,
            kind="render_failed",
            payload={"job_id": job_id, "error": exc.__class__.__name__},
        )
        raise
