"""RQ export job: edited Draft Timeline -> final MP4 -> scoped S3 URL."""
from __future__ import annotations

import re
import time

import threading

from pathlib import Path
import tempfile

from . import pending_watch_listen_review as pwl
from . import audio_clean
from . import keyframes as keyframes_lib
from . import sfx_library
from . import timeline_asset_registry as timeline_registry
from . import timeline_asset_registry_store as timeline_store
from . import timeline_asset_s3
from .audio_mix_render import AudioMix, LocalAudioClip
from . import render_delivery as rd
from . import tenant_safe_delivery as tsd
from .draft_edits import DraftEditError
from .editorial_slot_resolution_install import reset_editorial_slot_resolution_evidence
from .exports import store_export
from .live_render_qc import PostRenderQCFailure, render_with_post_render_qc
from .media_overlay_render import LocalMediaOverlay
from .notifications import publish_notification
from .overlay_uploads import validate_overlay_uri
from .config import EXPORT_GATE_TECHNICAL_QC, export_delivery_gate
from .perceptual_watch_listen import WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
from .project_tracking import safe_update_project
from .render import RENDER_FPS_DEFAULT
from .render_plan import bound_draft_to_sources, build_render_plan
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


class AudioClipUnavailable(RuntimeError):
    """Editor v5: an audio-lane clip of the draft could not be resolved to a file this project may
    use (missing, not owned, not ready, not audio, or the download failed). The export fails with
    this clear code -- an audio clip is never silently dropped from the creator's video."""

    error_code = "audio_clip_unavailable"

    def __init__(self, audio_id: str, reason: str):
        super().__init__(f"audio_clip_unavailable:{audio_id}:{reason}")
        self.audio_id = audio_id
        self.reason = reason


def _resolve_voice_or_music(clip, *, user_id: str, project_id: str, directory: str) -> str:
    """A voice/music clip's file, ONLY through the tenant-safe timeline asset registry (the
    requesting user/project must own a READY audio asset); never a URI from the client."""
    scratch = str(Path(directory) / "audio-assets")
    try:
        result = timeline_store.resolve_timeline_asset_live(
            user_id=user_id, project_id=project_id, asset_id=str(clip.asset_id),
            required_media_kind=timeline_registry.TimelineMediaKind.AUDIO,
            materialize=timeline_asset_s3.s3_fetcher(user_id=user_id, project_id=project_id, scratch_dir=scratch),
        )
    except Exception as exc:  # registry/storage outage: still a clear, attributed failure
        raise AudioClipUnavailable(clip.audio_id, exc.__class__.__name__) from exc
    if result.outcome != timeline_registry.ASSET_RESOLVED or result.resolved is None:
        raise AudioClipUnavailable(clip.audio_id, str(result.outcome))
    return str(result.resolved.local_path)


def _resolve_sound_effect(clip, *, directory: str, index: int) -> str:
    """An official library sound (sfx_library), from the configured media bucket."""
    try:
        # download_source is routed (storage_routing): the configured storage (R2 after the move),
        # falling back to the old Amazon bucket while a sound is not there yet.
        uri = sfx_library.sfx_uri(str(clip.sfx_id))
        suffix = Path(sfx_library.sound_effect(str(clip.sfx_id)).file).suffix or ".m4a"
        return download_source(uri, str(Path(directory) / f"sfx-{index:03d}{suffix}"))
    except Exception as exc:
        raise AudioClipUnavailable(clip.audio_id, exc.__class__.__name__) from exc


def resolve_audio_mix(draft, *, user_id: str, project_id: str, directory: str) -> AudioMix | None:
    """Editor v5: the draft's audio lane resolved to local files, plus the whole-video volume and
    the whole sound's volume / mute. None when the draft uses none of them (the render then skips
    the mix pass).

    Build 17: Clean up voice plays the stored cleaned copy (audio_clean) -- of a voice / music
    recording here, of the video's own sound in the render plan (export_clean_audio_paths) -- the
    same file the iPhone preview played; the mix never re-filters it. A muted clip is silent."""
    clips = tuple(getattr(draft, "audio_clips", ()) or ())
    video_volume = float(getattr(draft, "video_volume", 1.0))
    master = 0.0 if bool(getattr(draft, "master_muted", False)) else float(getattr(draft, "master_volume", 1.0))
    if not clips and video_volume == 1.0 and master == 1.0:
        return None
    local = []
    sfx_files: dict[str, str] = {}
    for index, clip in enumerate(clips):
        if bool(getattr(clip, "muted", False)) or float(clip.volume) <= 0.0:
            continue  # silent: nothing to mix (and nothing to download)
        if clip.kind == "sfx":
            path = sfx_files.get(str(clip.sfx_id))
            if path is None:
                path = sfx_files[str(clip.sfx_id)] = _resolve_sound_effect(clip, directory=directory, index=index)
        else:
            # Tenant-safe registry first (clear codes for a missing / foreign recording) ...
            path = _resolve_voice_or_music(clip, user_id=user_id, project_id=project_id, directory=directory)
            if bool(clip.clean):
                # ... then its stored cleaned copy (made from that file if it is not stored yet).
                try:
                    path = audio_clean.ensure_clean_audio(
                        user_id=user_id, project_id=project_id, target="asset", ident=str(clip.asset_id),
                        directory=str(Path(directory) / "clean"), local_input=path,
                    )
                except Exception as exc:
                    raise AudioClipUnavailable(clip.audio_id, f"clean:{exc.__class__.__name__}") from exc
        local.append(LocalAudioClip(
            path=path, start=float(clip.start), source_in=float(clip.source_in), source_out=float(clip.source_out),
            volume=float(clip.volume), duck_video=bool(clip.duck_video), clean=False,
        ))
    return AudioMix(video_volume=video_volume, video_clean=False, clips=tuple(local), master_volume=master)


def export_clean_audio_paths(draft, local_paths: dict, *, user_id: str, project_id: str,
                             directory: str) -> dict:
    """Build 17: source_asset_id -> the cleaned copy of that source's sound, for every source with
    a clip that plays it cleaned (the whole video's Clean up, or a clip's own). Made now (and
    stored) when the creator exported before the preview's copy was ready."""
    whole = bool(getattr(draft, "video_clean_voice", False))
    wanted = {clip.source_asset_id for clip in draft.selected if whole or bool(getattr(clip, "clean_voice", False))}
    paths = {}
    for source_id in sorted(wanted):
        paths[source_id] = audio_clean.ensure_clean_audio(
            user_id=user_id, project_id=project_id, target="source", ident=source_id,
            directory=str(Path(directory) / "clean"), local_input=local_paths.get(source_id),
        )
    return paths


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
    job_started_at: float | None = None,
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
        if (
            watch_listen_status == WATCH_LISTEN_HUMAN_REVIEW_REQUIRED
            and export_delivery_gate() == EXPORT_GATE_TECHNICAL_QC
        ):
            # CUTSELL_EXPORT_DELIVERY_GATE=technical_qc (see config.py): "needs a human look" is
            # not held for approval -- the D-288 gate is left unapplied, exactly like the
            # `watch_listen_status=None` opt-in contract above. Only this one verdict is relaxed:
            # BLOCKED keeps its value and still never delivers. The review itself is kept in the
            # result for observability.
            if isinstance(perceptual, dict):
                perceptual = {**perceptual, "delivery_gate_policy": EXPORT_GATE_TECHNICAL_QC}
            watch_listen_status = None

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
            # D-288.3 (blocker 3): the ORIGINAL job's own real start
            # timestamp, so a LATER resumed delivery can preserve this
            # job's identity/order for the stale-job guard instead of
            # disabling it with `None` -- see `resume_delivery_after_
            # approval`'s own docstring.
            job_started_at=job_started_at,
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
    """D-288.2/D-288.3/D-288.4: "reanudación de entrega" -- given a pending
    review record ALREADY approved via `pending_watch_listen_review.apply_
    human_approval` (this function performs NO approval logic itself),
    download the private pending artifact and complete the SAME tenant-
    safe upload/verify/evaluate/finalize seam `run_export_job`'s own same-
    job path uses -- one implementation, never a second guess.

    `requesting` is REQUIRED (no default, `None` explicitly rejected by
    `pending_watch_listen_review._require_authenticated_requesting`) --
    this is a LATER, separate operation triggered by a real human's own
    authenticated request, never the trusted same-job worker context
    `_tenant_safe_deliver`'s own `requesting=None -> requesting=ownership`
    default is deliberately scoped to. The real route handler (`cutsell_
    app/pending_review_routes.py`) resolves it from `request.state.auth_
    user_id`, never from a request body/query field.

    Ordering contract (D-288.4, "coordinar revocación y publicación
    mediante una transición atómica ANTES de los efectos externos"):

    1. CURRENT approval is checked first -- BEFORE any cached result is
       ever returned. aprobar -> entregar -> rechazar -> reanudar refuses
       the new request while the delivered history stays on the record.
    2. Download + hash-verify (reads only, no external effect).
    3. A FRESH read, then `claim_delivery_publication`: a versioned CAS
       moving `delivery_state` to PUBLISHING. A revocation that landed
       after that read makes the claim fail -> nothing is published. Once
       held, `apply_human_approval` refuses a rejection until the commit
       below, so the two can never interleave. A concurrent resume that
       already holds PUBLISHING is allowed to re-claim (retry after an
       interruption looks the same) because every effect below is
       idempotent by render identity.
    4. Tenant-safe upload + `_finalize_successful_delivery` -- `add_
       render_version`/`publish_notification` are atomic-idempotent, so
       concurrent or retried callers all receive the ONE stored id;
       `job_started_at` is the ORIGINAL job's own timestamp (D-288.3
       blocker 3), never `None`.
    5. `commit_delivered` (PUBLISHING -> DELIVERED, result stored). On a
       conflict (only another concurrent resume, or a re-approval, can
       move the version now), reread: return the result the other party
       already committed, or retry the commit while the record is still
       PUBLISHING and approved -- never a lost update, never a duplicate.
    """
    pwl._require_authenticated_requesting(requesting)

    ownership = tsd.DeliveryOwnershipScope(user_id=user_id, project_id=project_id, job_id=job_id)

    def _require_approved(candidate: "pwl.PendingWatchListenRecord") -> None:
        if not pwl.is_approved_for_delivery(candidate):
            raise pwl.PendingReviewError(
                f"pending_review_not_approved:{candidate.watch_listen_status}:{candidate.approval_status}"
            )

    def _load() -> "pwl.PendingWatchListenRecord":
        loaded = pwl.load_pending_review(user_id=user_id, project_id=project_id, job_id=job_id, client=client)
        if loaded is None:
            raise pwl.PendingReviewError("no_pending_review_found")
        return loaded

    record = _load()
    tsd.assert_delivery_access(requesting=requesting, record_ownership=record.ownership)

    # D-288.4 (blocker 1): the CURRENT approval gates everything -- a
    # cached result is history, never a bypass of a later rejection.
    _require_approved(record)
    if record.resumed_delivery_result is not None:
        return dict(record.resumed_delivery_result)

    with tempfile.TemporaryDirectory(prefix="cutsell-resume-delivery-") as directory:
        local_path = str(Path(directory) / "cutsell-resume-export.mp4")
        parsed_bucket, parsed_key = record.pending_s3_uri[5:].split("/", 1)
        if s3_client is None:
            import boto3
            s3_client = boto3.client("s3")
        from .storage_routing import download as routed_download
        routed_download(parsed_bucket, parsed_key, local_path, client=s3_client)

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
            # deliver them. A hard integrity failure, never reaches upload.
            raise pwl.PendingReviewError("resumed_artifact_hash_mismatch")
        if local_delivery.delivery_status != rd.DELIVERY_STATUS_READY_FOR_UPLOAD:
            blocked = tsd.evaluate_tenant_safe_delivery(
                ownership=ownership, expected_render_identity=record.render_identity, delivery=local_delivery,
            )
            raise TenantSafeDeliveryBlocked(blocked)

        # D-288.4 (blocker 1): the atomic publish claim, BEFORE the first
        # external effect, off a fresh read.
        claimed = None
        for _attempt in range(pwl._MAX_CAS_RETRIES):
            live = _load()
            _require_approved(live)
            if live.resumed_delivery_result is not None:
                return dict(live.resumed_delivery_result)
            if live.render_identity != record.render_identity or live.output_sha256 != record.output_sha256:
                # A later export replaced this job's pending slot with a
                # different render -- the bytes we hold are not what the
                # CURRENT record approves.
                raise pwl.PendingReviewError("resumed_artifact_identity_changed")
            claimed = pwl.claim_delivery_publication(live, client=client)
            if claimed is not None:
                break
        if claimed is None:
            raise pwl.PendingReviewConflict("publish_claim_conflicted_with_concurrent_change")

        delivery = _upload_verify_and_finalize_delivery(
            local_path=local_path, ownership=ownership, requesting=requesting,
            render_identity=claimed.render_identity, local_delivery=local_delivery,
            project_id=project_id, user_id=user_id, job_id=job_id,
            watch_listen_status=claimed.watch_listen_status,
        )
        finalized = _finalize_successful_delivery(
            user_id=user_id, project_id=project_id, job_id=job_id, job_started_at=claimed.job_started_at,
            delivery=delivery, selected_count=claimed.selected_count,
            text_overlay_count=claimed.text_overlay_count, media_overlay_count=claimed.media_overlay_count,
        )

        current = claimed
        for _attempt in range(pwl._MAX_CAS_RETRIES):
            if pwl.commit_delivered(current, finalized, client=client) is not None:
                return finalized
            current = _load()
            if current.resumed_delivery_result is not None:
                return dict(current.resumed_delivery_result)
            if current.delivery_state != pwl.DELIVERY_STATE_PUBLISHING or not pwl.is_approved_for_delivery(current):
                break
        raise pwl.PendingReviewConflict("resume_commit_conflicted_with_concurrent_change")


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
    a real notification an approving human/mobile client can observe).

    D-288.3 (blocker 2, "idempotencia por identidad del render... en
    registro de versión y notificación"): `add_render_version` and `_safe_
    notify` below are both idempotent, keyed by `delivery["render_
    identity"]`/`delivery["export_uri"]` (a render's own `export_uri` is
    derived deterministically from its `render_identity`, so the two are
    equivalent keys). Calling this function AGAIN for the SAME delivery
    (e.g. `resume_delivery_after_approval` recovering after a crash that
    happened right after a first, successful call to this function but
    before its OWN caller could record that fact) reuses the existing
    render version and skips re-firing the notification, rather than
    duplicating either -- `job_started_at` is the ONLY thing that then
    still matters to re-supply correctly (see that function's own D-288.3
    docstring for why it must be the ORIGINAL job's timestamp, never
    `None`)."""
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
        idempotency_key=str(delivery.get("render_identity") or "") or None,
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


def _safe_notify(
    *, user_id: str, project_id: str, kind: str, payload: dict | None = None, idempotency_key: str | None = None,
) -> dict:
    try:
        event = publish_notification(
            user_id=user_id,
            project_id=project_id,
            kind=kind,
            payload=payload,
            idempotency_key=idempotency_key,
        )
        return {"status": "queued", "notification_id": event["notification_id"]}
    except Exception as exc:
        return {"status": "degraded", "reason": exc.__class__.__name__}


class _RenderProgressTicker:
    """Publishes the export % from 35 to 84 while the render and its QC run. The pace is the
    measured cost of this worker: about RENDER_SECONDS_PER_VIDEO_SECOND per second of video
    (more with overlays, which need their own pass), never past 84 before the work ends."""

    RENDER_SECONDS_PER_VIDEO_SECOND = 1.1
    BASE_SECONDS = 6.0

    def __init__(self, publish, *, seconds_of_video: float, media_overlays: int = 0):
        self._publish = publish
        factor = self.RENDER_SECONDS_PER_VIDEO_SECOND * (1.6 if media_overlays else 1.0)
        self.expected = max(4.0, self.BASE_SECONDS + factor * max(0.0, float(seconds_of_video)))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="export-progress", daemon=True)

    @staticmethod
    def percent(elapsed: float, expected: float) -> int:
        # Linear to 80% of the expected time, then slower; capped at 84 until the work ends.
        fraction = elapsed / max(0.1, expected)
        if fraction <= 0.8:
            value = 35 + 45 * (fraction / 0.8)
        else:
            value = 80 + 4 * (1 - 1 / (1 + (fraction - 0.8) * 3))
        return int(min(84, max(35, value)))

    def _run(self) -> None:
        started = time.monotonic()
        while not self._stop.wait(1.0):
            try:
                self._publish("rendering", self.percent(time.monotonic() - started, self.expected))
            except Exception:
                return

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


_EXPORT_STEPS = ("download_sources", "download_overlays", "download_audio", "clean_audio", "plan", "render_and_qc",
                 "upload_and_verify", "finalize")


def failure_code(timings: dict, exc: BaseException) -> str:
    """Build 27: why an export failed, short and safe to show in the app's Diagnostics (export_failed step=...
    reason=...): the step that was running (the first one not finished) and the error's own code word -- never a
    path, a key or a URL."""
    step = next((name for name in _EXPORT_STEPS if name not in timings), "finalize")
    code = getattr(exc, "error_code", None) or str(exc).split(":", 1)[0]
    reason = re.sub(r"[^a-z0-9_]+", "_", f"{exc.__class__.__name__}_{code}".lower()).strip("_")[:90]
    return f"{step}:{reason}"


def _remember_failure(job, timings: dict, exc: BaseException) -> None:
    if job is None:
        return
    try:
        job.meta["error_code"] = failure_code(timings, exc)
        job.save_meta()
    except Exception:
        pass


def run_export_job(payload: dict) -> dict:
    # D-288 (finding 5): the real per-job init boundary for the
    # EditorialSlotResolution evidence ContextVar -- see universal_clean_
    # cut_validation.run_single_universal_clean_cut_validation's own D-288
    # comment for why this must run before any arbiter call this job
    # could make.
    reset_editorial_slot_resolution_evidence()
    from rq import get_current_job

    job = get_current_job()

    published = {"percent": 0}
    publish_lock = threading.Lock()

    def publish(stage: str, percent: int) -> None:
        # Never backwards (the render ticker and the stage marks both publish).
        with publish_lock:
            value = max(published["percent"], max(0, min(100, int(percent))))
            published["percent"] = value
            if job is None:
                return
            job.meta["stage"] = stage
            job.meta["progress_percent"] = value
            job.meta["timings"] = dict(timings)
            try:
                job.save_meta()
            except Exception:
                pass

    # Seconds spent in each export stage (returned with the result and kept in job.meta, so the
    # app and the export measurements can see where the time goes).
    timings: dict[str, float] = {}
    stage_clock = {"t": time.monotonic()}

    def mark(stage: str) -> None:
        now = time.monotonic()
        timings[stage] = round(now - stage_clock["t"], 3)
        stage_clock["t"] = now

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
            mark("download_sources")
            # A cut made from the voice file can end a few hundredths of a second past the video's own end
            # (19 Oct: 60.070 vs 60.012 s). Bound the draft itself so the render and its QC agree.
            draft = bound_draft_to_sources(draft, local_paths)

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
                local_overlays.append(LocalMediaOverlay(
                    overlay=overlay, path=destination,
                    keyframes=keyframes_lib.overlay_track(getattr(draft, "keyframes", ()) or (),
                                                          overlay_id=str(overlay.overlay_id))
                    if str(getattr(overlay, "layout", "float") or "float") == "float" else (),
                ))
                publish("rendering", min(32, 26 + int((index + 1) * 6 / overlay_count)))

            mark("download_overlays")
            # Editor v5: voice/music through the tenant-safe registry, sounds from the official
            # library; a clip that cannot be resolved fails the export (AudioClipUnavailable).
            audio_mix = resolve_audio_mix(draft, user_id=user_id, project_id=project_id, directory=directory)
            mark("download_audio")
            clean_paths = export_clean_audio_paths(draft, local_paths, user_id=user_id, project_id=project_id,
                                                   directory=directory)
            mark("clean_audio")
            plan = build_render_plan(draft, local_paths, clean_paths) if clean_paths else build_render_plan(draft, local_paths)
            output = str(Path(directory) / "cutsell-export.mp4")
            publish("rendering", 35)
            mark("plan")
            # The render + QC used to sit at 35% until it ended: move the % on from 35 to 84
            # as the work goes, from how long renders of this length take on this machine.
            ticker = _RenderProgressTicker(
                publish, seconds_of_video=sum(max(0.0, seg.end - seg.start) for seg in plan),
                media_overlays=len(local_overlays),
            )
            ticker.start()
            # D-030: live PostRenderWatchListenQC + bounded physical repair,
            # run against the ACTUAL local rendered file, before this job
            # ever calls store_export -- never a downloaded-back artifact.
            # A SEMANTIC_MISMATCH_INVALIDATED or NEEDS_HUMAN_REVIEW result
            # raises PostRenderQCFailure below, which this job's own
            # except-block already treats as a hard failure (state="failed",
            # render_failed notification) -- this candidate is never
            # delivered/uploaded.
            try:
                qc_result = render_with_post_render_qc(
                    draft,
                    plan,
                    output,
                    text_overlays=draft.text_overlays,
                    media_overlays=tuple(local_overlays),
                    **({"audio_mix": audio_mix} if audio_mix is not None else {}),
                )
            finally:
                ticker.stop()
            mark("render_and_qc")
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
                job_started_at=job_started_at,
            )
            mark("upload_and_verify")
            finalized = _finalize_successful_delivery(
                user_id=user_id, project_id=project_id, job_id=job_id, job_started_at=job_started_at,
                delivery=delivery, selected_count=len(draft.selected),
                text_overlay_count=len(draft.text_overlays), media_overlay_count=len(draft.media_overlays),
            )
            mark("finalize")
            publish("finished", 100)
            return {
                **finalized,
                "timings": dict(timings),
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
        _remember_failure(job, timings, exc)
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
        _remember_failure(job, timings, exc)
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
            payload={"job_id": job_id, "error": exc.__class__.__name__,
                     **({"error_code": exc.error_code} if getattr(exc, "error_code", None) else {})},
        )
        _remember_failure(job, timings, exc)
        raise
