"""RQ export job: edited Draft Timeline -> final MP4 -> scoped S3 URL."""
from __future__ import annotations

from pathlib import Path
import tempfile

from . import render_delivery as rd
from . import tenant_safe_delivery as tsd
from .draft_edits import DraftEditError
from .exports import store_export
from .live_render_qc import PostRenderQCFailure, render_with_post_render_qc
from .media_overlay_render import LocalMediaOverlay
from .notifications import publish_notification
from .overlay_uploads import validate_overlay_uri
from .project_tracking import safe_update_project
from .render import RENDER_FPS_DEFAULT
from .render_plan import build_render_plan
from .render_versions import add_render_version
from .serde import draft_from_dict
from .storage import download_source
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

    def __init__(self, record: "tsd.TenantSafeDeliveryRecord"):
        super().__init__(f"tenant_safe_delivery_blocked:{record.delivery_status}")
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
    requesting: "tsd.DeliveryOwnershipScope | None" = None,
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
    local_delivery = rd.build_render_delivery_record(
        render_identity=render_identity,
        render_execution_status=rd.RENDER_EXECUTION_STATUS_SUCCEEDED,
        final_path=output_path,
        technical_qc_status=rd.TECHNICAL_QC_STATUS_PASS,
        require_upload=True,
        upload_status=rd.UPLOAD_STATUS_NOT_ATTEMPTED,
        project_id=project_id,
        job_id=job_id,
    )
    if local_delivery.delivery_status != rd.DELIVERY_STATUS_READY_FOR_UPLOAD:
        # D-267's own render/hash/QC gate already found a blocker -- never
        # re-derived, never overridden here, only wrapped as a tenant-
        # safe-delivery block so the caller has one exception type.
        blocked = tsd.evaluate_tenant_safe_delivery(
            ownership=ownership, expected_render_identity=render_identity, delivery=local_delivery,
        )
        raise TenantSafeDeliveryBlocked(blocked)

    tenant_key = tsd.build_tenant_safe_export_key(ownership=ownership, render_identity=render_identity)
    stored = store_export(
        output_path,
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
                output_path=output, plan=plan, project_id=project_id, user_id=user_id, job_id=job_id,
            )
            version_payload = {}
            version = None
            try:
                version = add_render_version(
                    user_id=user_id,
                    project_id=project_id,
                    export_uri=delivery["export_uri"],
                    size_bytes=delivery["size_bytes"],
                    selected_count=len(draft.selected),
                    text_overlay_count=len(draft.text_overlays),
                    media_overlay_count=len(draft.media_overlays),
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
            publish("finished", 100)
            return {
                "project_id": project_id,
                "state": "finished",
                "selected_count": len(draft.selected),
                "text_overlay_count": len(draft.text_overlays),
                "media_overlay_count": len(draft.media_overlays),
                "project_tracking_start": tracking_start,
                "project_tracking": project_tracking,
                "notification": notification,
                # D-030: the delivered candidate's exact plan identity and
                # post-render QC/repair history.
                "post_render_qc_status": qc_result.status,
                "plan_id": qc_result.plan_id,
                "plan_version": qc_result.plan_version,
                "semantic_hash": qc_result.semantic_hash,
                "render_attempt_count": len(qc_result.attempts),
                **version_payload,
                **delivery,
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
