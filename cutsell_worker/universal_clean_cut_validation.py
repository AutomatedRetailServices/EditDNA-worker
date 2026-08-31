"""Real-video validation harness for Universal Clean Cut only.

D-035 (single-path rule): the preview render below goes through the exact
same `live_render_qc.render_with_post_render_qc` the real mobile-app export
job (`export_job.run_export_job`) uses -- there is no separate
"Video00RenderQC"/"AppRenderQC" implementation. This benchmark harness only
supplies validation-specific storage/output handling (a local preview path
instead of an uploaded export); the semantic/physical editing behavior --
render, PostRenderWatchListenQC, bounded physical repair, re-render -- is
one shared production-grade service. See docs/CUTSELL_DECISIONS.md D-035.
"""
from __future__ import annotations

import dataclasses
from dataclasses import replace
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Any

from .asr import FasterWhisperASR
from .brain_runtime import build_brain_runtime
from .config import load_runtime_config
from .contracts import ProcessingRequest, SourceAsset
from .live_render_qc import LiveRenderQCResult, render_with_post_render_qc
from .media_probe import probe_media
from .render_plan import build_render_plan
from .source_identity import stable_source_id
from .storage import download_source
from .universal_clean_cut import process_universal_clean_cut_sources
from .validation import _is_real_video_key


def _render_validation_preview(
    draft,
    local_paths,
    *,
    preview_output: str | None,
    preview_captions: bool,
    freeze_blocked: bool = False,
) -> tuple[str | None, str | None, LiveRenderQCResult | None]:
    """Render the validation preview through the SAME live render/QC service
    the real export job uses (D-030/D-035): Boundary (already applied to
    `draft.selected` upstream, before this is ever called) -> render actual
    MP4 -> PostRenderWatchListenQC on that actual local file -> PASS, or a
    bounded physical repair + re-render, or an invalidated semantic mismatch
    that this harness must never deliver as a preview.

    `freeze_blocked=True` means Final Story Coherence Validation / the repair
    loop already determined this draft must not be frozen -- Selection
    Freeze and Boundary never ran for it upstream, so there is nothing safe
    to render here either. Per the canonical live order, a semantic failure
    must never reach render at all.
    """
    if not preview_output:
        return None, None, None
    if not draft.selected:
        return None, "empty_draft", None
    if freeze_blocked:
        return None, "freeze_blocked_no_render", None

    plan = build_render_plan(draft, local_paths)
    if not preview_captions:
        plan = tuple(replace(segment, caption_text="") for segment in plan)
    qc_result = render_with_post_render_qc(draft, plan, preview_output)
    if qc_result.status != "PASS":
        return None, f"post_render_qc_{qc_result.status.lower()}", qc_result
    return qc_result.output_path, None, qc_result


def _live_render_qc_diagnostics(
    qc_result: LiveRenderQCResult | None, *, skipped_reason: str | None
) -> dict[str, Any]:
    if qc_result is None:
        return {
            "status": "not_attempted",
            "reason": skipped_reason,
            "output_path": None,
            "plan_id": None,
            "plan_version": None,
            "semantic_hash": None,
            "render_attempt_count": 0,
            "attempts": [],
        }
    return {
        "status": qc_result.status,
        "reason": None,
        "output_path": qc_result.output_path,
        "plan_id": qc_result.plan_id,
        "plan_version": qc_result.plan_version,
        "semantic_hash": qc_result.semantic_hash,
        "render_attempt_count": len(qc_result.attempts),
        "attempts": [dataclasses.asdict(a) for a in qc_result.attempts],
    }


def run_single_universal_clean_cut_validation(
    key: str,
    *,
    project_id: str = "cutsell-universal-clean-cut-validation",
    language_hint: str | None = None,
    preview_output: str | None = None,
    preview_captions: bool = False,
) -> dict[str, Any]:
    """Run one full S3 raw through local perception plus the active Selection authority."""
    if not _is_real_video_key(key):
        raise ValueError("unsupported validation video")

    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")

    brain = build_brain_runtime(config)
    source_id = stable_source_id(project_id, 0, PurePosixPath(key).name)
    source = SourceAsset(
        source_asset_id=source_id,
        project_id=project_id,
        user_id="validation",
        original_name=PurePosixPath(key).name,
        source_order=0,
        duration_sec=0.0,
        uri=f"s3://{config.s3_bucket}/{key}",
    )
    request = ProcessingRequest(
        project_id=project_id,
        user_id="validation",
        sources=(source,),
        language_hint=language_hint,
    )

    asr = FasterWhisperASR(model_name=config.asr_model)

    started = time.monotonic()
    preview_path = None
    preview_skipped_reason = None
    with tempfile.TemporaryDirectory(prefix="cutsell-universal-clean-cut-") as directory:
        destination = str(Path(directory) / source.original_name)
        local = download_source(source.uri, destination)
        source_duration_sec = float(probe_media(local).duration_sec)
        local_paths = {source_id: local}

        result = process_universal_clean_cut_sources(
            request,
            local_paths,
            asr_provider=asr,
            whole_video_provider=brain.whole_video_provider,
            visual_provider=brain.visual_provider,
            take_grouping_provider=brain.take_grouping_provider,
            take_judge_provider=brain.take_judge_provider,
            clean_cut_provider=brain.clean_cut_provider,
            editorial_judge=brain.editorial_judge,
            selection_reasoner=brain.selection_reasoner,
            deterministic_best_take_authority_enabled=brain.deterministic_best_take_authority_enabled,
            semantic_equivalence_arbiter=brain.semantic_equivalence_arbiter,
            clean_cut_core_v1_enabled=brain.clean_cut_core_v1_enabled,
        )

        freeze_blocked = bool(result.stage_status.get("freeze_blocked_pending_coherence_review"))
        preview_path, preview_skipped_reason, live_render_qc_result = _render_validation_preview(
            result.draft,
            local_paths,
            preview_output=preview_output,
            preview_captions=preview_captions,
            freeze_blocked=freeze_blocked,
        )

    elapsed = round(time.monotonic() - started, 3)
    selected_duration_sec = round(
        sum(max(0.0, clip.end - clip.start) for clip in result.draft.selected), 3
    )
    diagnostics = result.draft.diagnostics
    clean_decisions = list(diagnostics.get("clean_cut_decisions") or ())
    temporal = list(diagnostics.get("temporal_performance_trims") or ())
    hybrid_chunks = list(diagnostics.get("hybrid_editorial_chunks") or ())
    unified_diag = diagnostics.get("unified_selection_reasoner") or {}

    return {
        "schema_version": result.schema_version,
        "benchmark_mode": "universal_clean_cut",
        "brain_backend": brain.backend,
        "external_brain_calls_enabled": brain.external_calls_enabled,
        "selection_reasoner_enabled": brain.selection_reasoner is not None,
        "selection_reasoner_status": unified_diag.get("status") if isinstance(unified_diag, dict) else None,
        "selection_reasoner_provider": unified_diag.get("provider") if isinstance(unified_diag, dict) else None,
        "selection_reasoner_model": unified_diag.get("model") if isinstance(unified_diag, dict) else None,
        "hybrid_provider": brain.hybrid_settings.provider if brain.external_calls_enabled else None,
        "hybrid_primary_model": brain.hybrid_settings.primary_model if brain.external_calls_enabled else None,
        "hybrid_requested_group_count": int(diagnostics.get("hybrid_editorial_requested_chunk_count") or 0),
        "hybrid_available_group_count": int(diagnostics.get("hybrid_editorial_available_chunk_count") or 0),
        "hybrid_deleted_count": int(diagnostics.get("hybrid_editorial_deleted_count") or 0),
        "hybrid_group_diagnostic_count": len(hybrid_chunks),
        "project_id": result.project_id,
        "source_key": key,
        "source_duration_sec": round(source_duration_sec, 3),
        "selected_duration_sec": selected_duration_sec,
        "selected_to_input_ratio": round(selected_duration_sec / source_duration_sec, 4) if source_duration_sec else None,
        "elapsed_sec": elapsed,
        "preview_path": preview_path,
        "preview_captions": bool(preview_captions),
        "preview_skipped_reason": preview_skipped_reason,
        # D-030/D-035: the exact same live render/QC/repair-loop service the
        # real mobile export job uses -- render attempt history, the
        # PostRenderWatchListenQC findings for each attempt, and the exact
        # frozen plan id/version/hash the delivered (or invalidated) output
        # corresponds to.
        "live_render_qc": _live_render_qc_diagnostics(live_render_qc_result, skipped_reason=preview_skipped_reason),
        "empty_draft": not bool(result.draft.selected),
        "selected_count": len(result.draft.selected),
        "alternate_count": len(result.draft.alternates),
        "discarded_count": len(result.draft.discarded),
        "clean_cut_removed_count": sum(1 for item in clean_decisions if not bool(item.get("keep", True))),
        "temporal_trimmed_count": sum(1 for item in temporal if bool(item.get("applied"))),
        "models": {
            "brain_backend": brain.backend,
            "asr": config.asr_model,
            "whole_video": "runpod_local_asr_context",
            "visual": "runpod_local_mediapipe_opencv",
            "take_grouping": "deterministic_local_evidence",
            "take_judge": "deterministic_local_evidence",
            "clean_cut_judge": "deterministic_local_evidence",
            "hybrid_editorial": brain.hybrid_settings.primary_model if brain.editorial_judge is not None else None,
            "unified_selection": brain.hybrid_settings.primary_model if brain.selection_reasoner is not None else None,
            "semantic_sales": None,
            "composer": None,
            "draft_review": None,
        },
        "stage_status": result.stage_status,
        "diagnostics": diagnostics,
        "selected": [
            {"clip_id": clip.clip_id, "start": clip.start, "end": clip.end, "text": clip.text, "take_group_id": clip.take_group_id}
            for clip in result.draft.selected
        ],
        "alternates": [
            {"clip_id": clip.clip_id, "start": clip.start, "end": clip.end, "text": clip.text, "take_group_id": clip.take_group_id}
            for clip in result.draft.alternates
        ],
        "discarded": [
            {"clip_id": clip.clip_id, "start": clip.start, "end": clip.end, "text": clip.text}
            for clip in result.draft.discarded
        ],
    }
