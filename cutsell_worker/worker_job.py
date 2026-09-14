"""RQ entry point for real CutSell Flow B processing."""
from __future__ import annotations

import tempfile
from pathlib import Path

from .asr import FasterWhisperASR
from .brain_runtime import build_brain_runtime
from .config import load_runtime_config
from .draft_store import create_initial_draft
from .flow_b import process_local_sources
from .media_probe import probe_media
from .notifications import publish_notification
from .project_tracking import safe_update_project
from .serde import request_from_dict, result_to_dict
from . import output_format_qc as ofq
from . import source_format_policy as sfp
from . import source_media_profile as smp
from . import source_normalization_executor as sne
from . import source_normalization_plan as snp
from .source_format_policy import DECISION_ACCEPT, evaluate_source_format_policy
from .storage import download_source
from .timeline_asset_storage import store_timeline_assets
from .timeline_assets import generate_filmstrip, waveform_peaks
from .uploads import validate_product_source_uri
from .usage_limits import record_processing_minutes, release_processing_slot

# =============================================================================
# D-274F Stage 18 -- user-facing normalization outcome error codes. Bounded,
# never polished copy -- mirrors `source_format_policy.py`'s own
# `USER_FACING_*` vocabulary convention, scoped to auto-normalization
# TERMINAL outcomes specifically (D-272's own codes stop at "requires
# normalization", the pre-D-274F terminal state for a NORMALIZE_REQUIRED
# source). `USER_FACING_RUNTIME_CODEC_SUPPORT_UNVERIFIED` is D-272's own
# existing code, reused directly rather than re-declared (Stage 18's own
# "use existing error-contract conventions").
# =============================================================================
USER_FACING_VIDEO_NORMALIZATION_FAILED = "VIDEO_NORMALIZATION_FAILED"
USER_FACING_VIDEO_NORMALIZATION_UNSUPPORTED = "VIDEO_NORMALIZATION_UNSUPPORTED"
USER_FACING_VIDEO_NORMALIZATION_VERIFICATION_FAILED = "VIDEO_NORMALIZATION_VERIFICATION_FAILED"
USER_FACING_VIDEO_NORMALIZATION_TIMEOUT_POLICY_REQUIRED = "VIDEO_NORMALIZATION_TIMEOUT_POLICY_REQUIRED"
USER_FACING_RUNTIME_CODEC_SUPPORT_UNVERIFIED = sfp.USER_FACING_RUNTIME_CODEC_SUPPORT_UNVERIFIED

# D-274F Stage 16: the two resolved-source-kind labels a per-source
# lifecycle diagnostic ever carries -- bounded, never proliferated.
RESOLVED_SOURCE_KIND_ORIGINAL = "ORIGINAL"
RESOLVED_SOURCE_KIND_NORMALIZED = "NORMALIZED"


class SourceFormatGateBlocked(RuntimeError):
    """D-272A: raised when the D-272 early source-format policy blocks one or
    more of a job's sources before expensive editorial processing starts.

    Carries structured, machine-readable diagnostics (never raw filesystem
    paths -- Stage 10's own instruction) so the caller can map this into the
    existing failed-job contract (Stage 12: no new parallel job lifecycle)
    while preserving the D-272 decision/reason vocabulary for the mobile/app
    layer instead of collapsing it into a generic exception class name.
    """

    def __init__(self, blocked_sources: list[dict]):
        self.blocked_sources = blocked_sources
        first = blocked_sources[0]
        super().__init__(
            "source_format_gate_blocked: "
            f"{len(blocked_sources)} source(s) blocked before editorial "
            f"processing; first={first['source_asset_id']} "
            f"decision={first['decision']} "
            f"error_code={first['user_facing_error_code']}"
        )

    @property
    def primary_error_code(self) -> str | None:
        return self.blocked_sources[0].get("user_facing_error_code")


def _source_format_diagnostic(source_asset_id: str, profile, decision) -> dict:
    """D-272A Stage 10/23: bounded, machine-readable diagnostic for one
    source -- source_asset_id only, never the local filesystem path."""
    return {
        "source_asset_id": source_asset_id,
        "source_profile_status": decision.source_profile_status,
        "source_format_class": decision.source_format_class,
        "decision": decision.decision,
        "policy_version": decision.policy_version,
        "reason_codes": list(decision.reason_codes),
        "blocking_reasons": list(decision.blocking_reasons),
        "normalization_reasons": list(decision.normalization_reasons),
        "warnings": list(decision.warnings),
        "user_facing_error_code": decision.user_facing_error_code,
        "container": profile.container_name,
        "video_codec": profile.video_codec,
        "rotation_degrees": profile.rotation_degrees,
        "hdr_status": profile.hdr_status,
        "vfr_status": profile.vfr_status,
        "bit_depth": profile.bit_depth,
    }


def evaluate_source_format_gate(local_paths: dict[str, str]) -> list[dict]:
    """D-272A Stage 1/2: the live early-gate seam. For every already-
    downloaded local source, reuse D-271's own `probe_source_media_profile`
    and D-272's own `evaluate_source_format_policy` -- the SAME two
    functions `evaluate_source_for_editorial_entry` composes -- so this
    call site introduces zero duplicated codec/HDR/rotation/VFR/stream
    policy. D-272 remains the sole policy authority.

    D-274C-A Stage 6: now passes a REAL `RuntimeCapabilityInput`, built
    from this worker process's own established capability snapshot
    (`worker_runtime_capability.get_worker_runtime_capability_input`) --
    memoized once per process, never re-probed per source/job (Stage 5).
    That bridge fails closed on its own authority (production_
    verification_status must be genuinely ESTABLISHED, decoder AND
    canonical H264 encoder both present -- D-274C-A Stage 7): whenever
    establishment did not genuinely succeed, this call is byte-identical
    to every prior gate's own all-`False` default, so H.264 sources and
    every existing test remain completely unaffected. This function still
    introduces ZERO duplicated HEVC/HDR/codec logic of its own -- it only
    supplies D-272's own existing capability input parameter with real
    evidence instead of an always-empty default.
    """
    from . import worker_runtime_capability as wrc

    runtime_capability = wrc.get_worker_runtime_capability_input()
    diagnostics: list[dict] = []
    for source_asset_id, local_path in local_paths.items():
        profile = smp.probe_source_media_profile(local_path)
        decision = evaluate_source_format_policy(profile, runtime_capability=runtime_capability)
        diagnostics.append(_source_format_diagnostic(source_asset_id, profile, decision))
    return diagnostics


def _source_normalization_diagnostic(
    source_asset_id: str,
    profile,
    decision,
    *,
    plan_result: "snp.SourceNormalizationPlanResult",
    exec_result: "sne.NormalizationExecutionResult | None",
    resolved_source_kind: str,
) -> dict:
    """D-274F Stage 16: bounded per-source normalization lifecycle
    diagnostic -- extends `_source_format_diagnostic`'s own base shape
    (Stage 10/23 discipline: source_asset_id only, never a local
    filesystem path) with the exact lifecycle fields Stage 16 names:
    original policy decision, normalization-required flag, plan identity,
    plan actions, normalization outcome, normalized SHA, reprobe status,
    re-evaluated D-272 decision, format QC status, resolved source kind.
    Built once for every NORMALIZE_REQUIRED source, whether the
    normalization ultimately succeeded or failed."""
    base = _source_format_diagnostic(source_asset_id, profile, decision)
    plan = plan_result.plan if plan_result is not None else None

    normalization_outcome = plan_result.outcome if plan_result is not None else None
    normalized_sha256 = None
    reprobe_status = None
    reevaluated_decision = None
    format_qc_status = None
    if exec_result is not None:
        normalization_outcome = exec_result.outcome
        format_qc_status = exec_result.diagnostics.get("format_qc_status")
        if exec_result.normalized_reference is not None:
            normalized_sha256 = exec_result.normalized_reference.normalized_output_sha256
        if exec_result.normalized_profile is not None:
            reprobe_status = exec_result.normalized_profile.probe_status
        if exec_result.verification is not None:
            reevaluated_decision = exec_result.verification.normalized_policy_decision.decision

    base.update(
        {
            "normalization_required": True,
            "plan_identity": plan.plan_identity if plan is not None else None,
            "plan_actions": (
                {
                    "container": plan.container_action,
                    "codec": plan.codec_action,
                    "rotation": plan.rotation_action,
                    "frame_rate": plan.frame_rate_action,
                    "hdr": plan.hdr_action,
                    "bit_depth": plan.bit_depth_action,
                    "pixel_format": plan.pixel_format_action,
                    "timeline": plan.timeline_action,
                    "audio": plan.audio_action,
                }
                if plan is not None
                else None
            ),
            "normalization_outcome": normalization_outcome,
            "normalized_sha256": normalized_sha256,
            "reprobe_status": reprobe_status,
            "reevaluated_d272_decision": reevaluated_decision,
            "format_qc_status": format_qc_status,
            "resolved_source_kind": resolved_source_kind,
        }
    )
    return base


def _normalization_user_facing_error_code(
    outcome: str,
    *,
    blocking_capability_gaps: tuple[str, ...] = (),
    failure_category: str | None = None,
    format_qc_status: str | None = None,
) -> str:
    """D-274F Stage 18: bounded mapping from a normalization plan/execution
    outcome to a user-facing error code -- never invents polished copy.
    Capability gaps (HEVC decode / tonemap unverified, whether surfaced at
    the plan or the executor's own defense-in-depth pre-check) reuse
    D-272's own existing `RUNTIME_CODEC_SUPPORT_UNVERIFIED` code (Stage
    18's own "use existing error-contract conventions"); everything else
    is one of this gate's three genuinely-new, equally bounded codes."""
    if outcome == sne.PRODUCT_OWNER_NORMALIZATION_TIMEOUT_REQUIRED:
        return USER_FACING_VIDEO_NORMALIZATION_TIMEOUT_POLICY_REQUIRED
    if blocking_capability_gaps or failure_category in (
        sne.FAILURE_CODEC_UNAVAILABLE,
        sne.FAILURE_HDR_CAPABILITY_UNAVAILABLE,
    ):
        return USER_FACING_RUNTIME_CODEC_SUPPORT_UNVERIFIED
    if outcome == snp.NORMALIZATION_UNSUPPORTED:
        return USER_FACING_VIDEO_NORMALIZATION_UNSUPPORTED
    if outcome == snp.NORMALIZATION_VERIFICATION_FAILED:
        return USER_FACING_VIDEO_NORMALIZATION_VERIFICATION_FAILED
    if format_qc_status is not None and format_qc_status != ofq.STATUS_PASS:
        # D-274F Stage 9/38: the executor's own outcome reached
        # NORMALIZATION_SUCCEEDED (D-272 re-evaluated ACCEPT, format QC
        # did not reach a hard FAIL), but this gate's own stricter AND
        # requirement (policy ACCEPT *and* format QC == PASS) still blocks
        # on a non-PASS (e.g. PARTIAL) format QC result -- distinctly from
        # a genuine D-271/D-272 re-evaluation failure, but the same
        # user-facing terminal state (verification did not fully clear).
        return USER_FACING_VIDEO_NORMALIZATION_VERIFICATION_FAILED
    return USER_FACING_VIDEO_NORMALIZATION_FAILED


def resolve_sources_for_editorial_entry(
    local_paths: dict[str, str],
    *,
    output_directory: str,
    normalization_timeout_sec: float | None = None,
) -> tuple[dict[str, str], list[dict], list[dict], list[dict]]:
    """D-274F Stages 1-20/38-40: the live auto-normalization activation
    seam. For every already-downloaded local source:

      ACCEPT                       -> original path, ZERO normalization
                                       calls (Stage 2 -- unchanged).
      NORMALIZE_REQUIRED           -> build a D-274A plan, execute it
                                       (D-274B/C/D executor), and require
                                       BOTH the re-evaluated D-272 policy
                                       to reach ACCEPT AND the D-274E
                                       normalized-source format QC to
                                       reach real PASS (Stage 9/38) before
                                       substituting the normalized path
                                       into the returned resolution --
                                       never falling back to the original
                                       source on any other outcome
                                       (Stage 19).
      REJECT / INSUFFICIENT_EVIDENCE -> never attempts normalization
                                       (Stage 5/6); blocks the whole job.

    Exactly ONE normalization attempt per source, ever (Stage 4/21/D-274A's
    own `MAX_NORMALIZATION_ATTEMPTS = 1`) -- `attempt_count=0` on every
    call here, no retry loop of any kind, no second pass on any failure.

    `normalization_timeout_sec` is a raw pass-through to `source_
    normalization_executor.execute_source_normalization`'s own `timeout_
    sec` kwarg. Stage 10: D-274B's own audit found NO canonical
    normalization/media-operation timeout anywhere in this repository, and
    explicitly forbids silently reusing the renderer's own 1200s
    (`render.RENDER_FFMPEG_TIMEOUT_SEC`). `run_flow_b_job` (the real,
    production call site) NEVER overrides this parameter's `None` default
    -- so, in production, every real NORMALIZE_REQUIRED source currently
    resolves to `PRODUCT_OWNER_NORMALIZATION_TIMEOUT_REQUIRED` until a
    Product Owner decision activates a concrete number at a future gate
    (D-274F-A). Only test code injects a small, bounded override to prove
    the rest of this chain end-to-end (Stage 41's own explicit
    instruction) -- this is never a silently-invented production default.

    Preserves original source ordering (Stage 13): this function only
    ever assigns `resolved_local_paths[source_asset_id]`, on the SAME
    keys `local_paths` already carries, in the SAME iteration order; no
    source is ever reordered or silently dropped (Stage 12).

    Returns `(resolved_local_paths, source_format_diagnostics,
    normalization_diagnostics, blocked_sources)`. `source_format_
    diagnostics` is BYTE-IDENTICAL in shape to `evaluate_source_format_
    gate`'s own return value (Stage 40/regression firewall: the existing
    `source_format_diagnostics` job-result field and its own tests are
    unaffected). `normalization_diagnostics` is a NEW, additive,
    public-safe list (Stage 16/17) -- one entry per NORMALIZE_REQUIRED
    source, whatever its outcome. `blocked_sources` carries every
    diagnostic (of either shape) for a source that must block the whole
    job before `process_local_sources` is ever called (Stage 39: REJECT/
    INSUFFICIENT_EVIDENCE sources, unexecutable plans, and any
    normalization failure/timeout-policy-absence)."""
    from . import worker_runtime_capability as wrc

    runtime_capability = wrc.get_worker_runtime_capability_input()
    tonemap_available = wrc.get_worker_tonemap_available()
    # Stage 7/8/9: the SAME real, process-memoized capability snapshot
    # feeds BOTH the executor's own H264-encoder and HDR-tonemap
    # defense-in-depth pre-checks (Stage 9's own "no local-sandbox
    # optimism") -- one real capability source, never two independently
    # drifting ones.
    worker_capability = wrc.get_worker_runtime_capability()

    resolved_local_paths: dict[str, str] = {}
    source_format_diagnostics: list[dict] = []
    normalization_diagnostics: list[dict] = []
    blocked_sources: list[dict] = []

    for source_asset_id, local_path in local_paths.items():
        profile = smp.probe_source_media_profile(local_path)
        decision = evaluate_source_format_policy(profile, runtime_capability=runtime_capability)
        diagnostic = _source_format_diagnostic(source_asset_id, profile, decision)
        source_format_diagnostics.append(diagnostic)

        if decision.decision == DECISION_ACCEPT:
            # Stage 2: unchanged ACCEPT behavior.
            resolved_local_paths[source_asset_id] = local_path
            continue

        if decision.decision != sfp.DECISION_NORMALIZE_REQUIRED:
            # Stage 5/6: REJECT / INSUFFICIENT_EVIDENCE -- never attempt
            # normalization; blocks the whole job exactly as before D-274F.
            blocked_sources.append(diagnostic)
            continue

        # Stage 3/7/8/9 -- NORMALIZE_REQUIRED: build the plan with real
        # runtime/tonemap capability evidence (never sandbox optimism).
        plan_result = snp.build_source_normalization_plan(
            source_asset_id,
            profile,
            decision,
            runtime_capability=runtime_capability,
            tonemap_available=tonemap_available,
        )
        if plan_result.plan is None or not plan_result.plan.is_executable:
            # Stage 9/33/34/35/36/37: capability-unverified / unsupported
            # (Dolby Vision, HDR_OTHER) / invalid source state -- never
            # launch the executor, never touch downstream.
            norm_diag = _source_normalization_diagnostic(
                source_asset_id,
                profile,
                decision,
                plan_result=plan_result,
                exec_result=None,
                resolved_source_kind=RESOLVED_SOURCE_KIND_ORIGINAL,
            )
            norm_diag["user_facing_error_code"] = _normalization_user_facing_error_code(
                plan_result.outcome,
                blocking_capability_gaps=plan_result.blocking_capability_gaps,
            )
            normalization_diagnostics.append(norm_diag)
            blocked_sources.append(norm_diag)
            continue

        # Stage 3/4/21 -- execute exactly ONE normalization pass. Stage 7/8/9:
        # real worker capability supplied as BOTH the H264-encoder and the
        # HDR-tonemap defense-in-depth pre-checks.
        exec_result = sne.execute_source_normalization(
            local_path,
            plan_result.plan,
            output_directory=output_directory,
            timeout_sec=normalization_timeout_sec,
            runtime_capability=runtime_capability,
            attempt_count=0,
            codec_capability=worker_capability,
            tonemap_capability=worker_capability,
        )
        format_qc_status = exec_result.diagnostics.get("format_qc_status")
        # Stage 9/38: the ONLY condition under which a normalized source
        # ever reaches the editorial pipeline -- D-272 re-evaluates ACCEPT
        # AND D-274E format QC reaches real PASS. Deliberately stricter
        # than the executor's OWN internal success gate (which treats a
        # non-FAIL, e.g. PARTIAL, format QC as still NORMALIZATION_
        # SUCCEEDED so it never regresses a pre-D-274F rotation/VFR-only
        # normalization -- see source_normalization_executor.py's own
        # D-274E Stage 19/20 comment): this live activation seam is a
        # SEPARATE, additive AND-requirement layered on top, never a
        # change to the executor's own return contract.
        succeeded = (
            exec_result.outcome == snp.NORMALIZATION_SUCCEEDED
            and format_qc_status == ofq.STATUS_PASS
        )
        resolved_kind = RESOLVED_SOURCE_KIND_NORMALIZED if succeeded else RESOLVED_SOURCE_KIND_ORIGINAL
        norm_diag = _source_normalization_diagnostic(
            source_asset_id,
            profile,
            decision,
            plan_result=plan_result,
            exec_result=exec_result,
            resolved_source_kind=resolved_kind,
        )
        if succeeded:
            norm_diag["user_facing_error_code"] = None
            # Stage 10: substitute the normalized artifact's path -- every
            # downstream consumer of `local_paths` (ASR, attempts/retries,
            # BestTake, P1/P2, Boundary, Pacing, render, and the existing
            # timeline-asset/filmstrip/waveform generation) now sees ONLY
            # the normalized media for this source (Stage 15: no mixed
            # timeline).
            resolved_local_paths[source_asset_id] = exec_result.normalized_path
        else:
            failure_category = exec_result.failure.error_category if exec_result.failure else None
            norm_diag["user_facing_error_code"] = _normalization_user_facing_error_code(
                exec_result.outcome,
                failure_category=failure_category,
                format_qc_status=format_qc_status,
            )
            # Stage 19/20: normalization failure or format-QC-not-PASS
            # blocks the whole job before editorial processing -- never a
            # fallback to the original source, never a second attempt.
            blocked_sources.append(norm_diag)
        normalization_diagnostics.append(norm_diag)

    return resolved_local_paths, source_format_diagnostics, normalization_diagnostics, blocked_sources


def _build_timeline_assets(request, local_paths: dict[str, str], directory: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for source in request.sources:
        try:
            source_path = local_paths[source.source_asset_id]
            probe = probe_media(source_path)
            asset_dir = str(Path(directory) / "timeline-assets" / source.source_asset_id)
            filmstrip = generate_filmstrip(
                source_path,
                asset_dir,
                duration_sec=probe.duration_sec,
                max_frames=24,
                width=160,
            )
            waveform = waveform_peaks(source_path, buckets=256)
            output[source.source_asset_id] = store_timeline_assets(
                user_id=request.user_id,
                project_id=request.project_id,
                source_asset_id=source.source_asset_id,
                filmstrip=filmstrip,
                waveform=waveform,
            )
        except Exception as exc:
            output[source.source_asset_id] = {
                "status": "degraded",
                "reason": exc.__class__.__name__,
                "filmstrip": [],
                "waveform_uri": None,
                "waveform_bucket_count": 0,
            }
    return output


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


def run_flow_b_job(payload: dict) -> dict:
    """Run the CutSell Flow B brain inside the RunPod RQ worker.

    Local ASR/vision/timing remains the backbone. Gemini semantic reasoning can be
    added only through brain_runtime's explicit Hybrid feature flag + approved model +
    per-edit dollar guard; stored keys alone cannot activate paid inference.
    """
    from rq import get_current_job

    job = get_current_job()

    def publish(stage: str, percent: int) -> None:
        if job is None:
            return
        job.meta["stage"] = stage
        job.meta["progress_percent"] = max(0, min(100, int(percent)))
        job.save_meta()

    request = request_from_dict(payload)
    job_id = str(getattr(job, "id", "") or "") or None
    basic_sources = [
        {
            "source_asset_id": source.source_asset_id,
            "original_name": source.original_name,
            "source_order": source.source_order,
            "duration_sec": source.duration_sec,
            "uri": source.uri,
        }
        for source in request.sources
    ]
    tracking_start = safe_update_project(
        user_id=request.user_id,
        project_id=request.project_id,
        state="processing",
        sources=basic_sources,
        latest_job_id=job_id,
    )

    config = load_runtime_config()
    brain = build_brain_runtime(config)
    asr = FasterWhisperASR(model_name=config.asr_model)

    measured_seconds = 0.0
    outcome = "failed"
    try:
        publish("preparing", 1)
        with tempfile.TemporaryDirectory(prefix="cutsell-flow-b-") as directory:
            local_paths = {}
            measured_by_source: dict[str, float] = {}
            for index, source in enumerate(request.sources):
                validate_product_source_uri(
                    source.uri,
                    project_id=request.project_id,
                    user_id=request.user_id,
                )
                suffix = Path(source.original_name).suffix or ".mp4"
                destination = str(Path(directory) / f"{source.source_order:03d}-{source.source_asset_id}{suffix}")
                local_paths[source.source_asset_id] = download_source(source.uri, destination)
                probe = probe_media(local_paths[source.source_asset_id])
                measured_by_source[source.source_asset_id] = float(probe.duration_sec)
                measured_seconds += max(0.0, float(probe.duration_sec))
                publish("preparing", min(10, 2 + int((index + 1) * 8 / len(request.sources))))

            (
                local_paths,
                source_format_diagnostics,
                source_normalization_diagnostics,
                blocked_sources,
            ) = resolve_sources_for_editorial_entry(local_paths, output_directory=directory)
            if blocked_sources:
                # D-272A Stage 3-9, extended by D-274F Stage 5/6/19/20:
                # every source must reach ACCEPT -- either natively or via
                # a successful, format-QC-PASS-verified auto-normalization
                # -- before this job enters expensive editorial processing
                # (ASR/GPU/semantic reasoning). No existing product
                # contract allows silently skipping a bad source, so any
                # unresolved source blocks the whole job here -- before
                # process_local_sources is called.
                raise SourceFormatGateBlocked(blocked_sources)

            result = process_local_sources(
                request,
                local_paths,
                asr_provider=asr,
                semantic_provider=brain.semantic_provider,
                whole_video_provider=brain.whole_video_provider,
                visual_provider=brain.visual_provider,
                take_grouping_provider=brain.take_grouping_provider,
                take_judge_provider=brain.take_judge_provider,
                clean_cut_provider=brain.clean_cut_provider,
                composer_provider=brain.composer_provider,
                draft_review_provider=brain.draft_review_provider,
                editorial_judge=brain.editorial_judge,
                progress=publish,
            )
            serialized = result_to_dict(result)
            serialized["brain_backend"] = brain.backend
            serialized["external_brain_calls_enabled"] = brain.external_calls_enabled
            serialized["hybrid_provider"] = brain.hybrid_settings.provider
            serialized["hybrid_primary_model"] = brain.hybrid_settings.primary_model
            serialized["source_format_diagnostics"] = source_format_diagnostics
            # D-274F Stage 16/17: additive, public-safe per-source
            # normalization lifecycle diagnostics -- never a local
            # filesystem path, never an ffmpeg command, never a
            # credential (Stage 17's own explicit exclusions). Existing
            # `source_format_diagnostics` consumers/tests are unaffected.
            serialized["source_normalization_diagnostics"] = source_normalization_diagnostics

            publish("draft_ready", 94)
            timeline_assets = _build_timeline_assets(request, local_paths, directory)
            source_records = [
                {
                    "source_asset_id": source.source_asset_id,
                    "original_name": source.original_name,
                    "source_order": source.source_order,
                    "duration_sec": measured_by_source.get(source.source_asset_id, source.duration_sec),
                    "uri": source.uri,
                    "timeline_assets": timeline_assets.get(source.source_asset_id, {"status": "not_generated"}),
                }
                for source in request.sources
            ]
            create_initial_draft(
                user_id=request.user_id,
                project_id=request.project_id,
                draft=dict(serialized["draft"]),
                sources=source_records,
            )
            serialized["timeline_assets"] = timeline_assets
            serialized["project_tracking_start"] = tracking_start
            serialized["project_tracking"] = safe_update_project(
                user_id=request.user_id,
                project_id=request.project_id,
                state="draft_ready",
                sources=source_records,
                latest_job_id=job_id,
            )
            serialized["notification"] = _safe_notify(
                user_id=request.user_id,
                project_id=request.project_id,
                kind="draft_ready",
                payload={"job_id": job_id, "selected_count": len(serialized["draft"].get("selected") or [])},
            )
            outcome = "draft_ready"
            publish("draft_ready", 100)
            return serialized
    except Exception as exc:
        # D-272A Stage 12: map the early source-format gate into the existing
        # failed-job contract (no parallel job lifecycle) while preserving
        # the D-272 decision/reason vocabulary in the error payload/meta
        # instead of collapsing it into the generic exception class name.
        error_code = exc.__class__.__name__
        if isinstance(exc, SourceFormatGateBlocked):
            error_code = exc.primary_error_code or error_code
        error_payload: dict = {"job_id": job_id, "error": error_code}
        if isinstance(exc, SourceFormatGateBlocked):
            error_payload["source_format_gate"] = {"blocked_sources": exc.blocked_sources}
        safe_update_project(
            user_id=request.user_id,
            project_id=request.project_id,
            state="failed",
            latest_job_id=job_id,
        )
        _safe_notify(
            user_id=request.user_id,
            project_id=request.project_id,
            kind="processing_failed",
            payload=error_payload,
        )
        if job is not None:
            job.meta["stage"] = "failed"
            job.meta["error_code"] = error_code
            if isinstance(exc, SourceFormatGateBlocked):
                job.meta["source_format_gate"] = {"blocked_sources": exc.blocked_sources}
            job.save_meta()
        raise
    finally:
        if measured_seconds > 0:
            record_processing_minutes(
                user_id=request.user_id,
                project_id=request.project_id,
                minutes=measured_seconds / 60.0,
                metadata={"job_id": job_id, "outcome": outcome, "source_count": len(request.sources)},
            )
        release_processing_slot(user_id=request.user_id)