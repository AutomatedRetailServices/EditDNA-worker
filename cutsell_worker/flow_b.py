"""Real-video Flow B orchestration for CutSell Milestone 1."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping

from .asr import ASRProvider
from .attempt_reconstruction import reconstruct_delivery_attempts
from .clean_cut_provider import CleanCutProvider
from .composer_provider import ComposerProvider
from .contracts import ProcessingRequest, ProcessingResult
from .draft_review_provider import DraftReviewProvider
from .frame_sampling import sample_take_frames
from .hybrid_editorial import EditorialJudge
from .local_performance import (
    analyze_local_performance,
    apply_local_performance_to_takes,
    merge_local_events_into_context,
)
from .media_probe import probe_media
from .observability import ExecutionTrace
from .performance_confirmation import confirm_local_performance_events
from .pipeline import build_flow_b_draft
from .providers import NoopSemanticProvider, SemanticProvider, safe_semantic_classify
from .silence_analysis import word_silence_gaps
from .source_sampling import sample_source_frames
from .take_grouping_provider import TakeGroupingProvider
from .take_judge_provider import TakeJudgeProvider
from .take_segmentation import segment_takes
from .usage_limits import check_processing_allowance
from .visual_analysis import VisualProvider, apply_visual_observations, safe_visual_analyze
from .whole_video_analysis import WholeVideoProvider, safe_whole_video_analyze

ProgressCallback = Callable[[str, int], None]


def _resolve_editorial_mode(value: str | None = None) -> str:
    mode = str(value or os.environ.get("CUTSELL_EDITORIAL_MODE") or "clean_cut").strip().lower()
    if mode not in {"clean_cut", "full"}:
        raise ValueError("CUTSELL_EDITORIAL_MODE must be clean_cut or full")
    return mode


def process_local_sources(
    request: ProcessingRequest,
    local_paths: Mapping[str, str],
    *,
    asr_provider: ASRProvider,
    semantic_provider: SemanticProvider | None = None,
    visual_provider: VisualProvider | None = None,
    take_judge_provider: TakeJudgeProvider | None = None,
    clean_cut_provider: CleanCutProvider | None = None,
    composer_provider: ComposerProvider | None = None,
    take_grouping_provider: TakeGroupingProvider | None = None,
    draft_review_provider: DraftReviewProvider | None = None,
    whole_video_provider: WholeVideoProvider | None = None,
    editorial_judge: EditorialJudge | None = None,
    editorial_mode: str | None = None,
    progress: ProgressCallback | None = None,
) -> ProcessingResult:
    """Process registered sources from raw media to an editable Flow B draft.

    Perception and exact edit boundaries remain local. When explicitly enabled, the
    Hybrid Editorial judge receives only already-bounded retry groups and may classify
    semantic intent (winner/alternate/failed/BTS); it never creates timestamps.
    """
    mode = _resolve_editorial_mode(editorial_mode)
    notify = progress or (lambda _stage, _percent: None)
    trace = ExecutionTrace()
    hydrated_sources = []
    transcripts = []

    notify("preparing", 2)
    for source in sorted(request.sources, key=lambda item: item.source_order):
        path = local_paths.get(source.source_asset_id)
        if not path:
            raise ValueError(f"missing local path for source {source.source_asset_id}")
        if not Path(path).exists():
            raise FileNotFoundError(path)
        probe = probe_media(path)
        hydrated_sources.append(replace(
            source,
            duration_sec=probe.duration_sec,
            has_audio=probe.has_audio,
            metadata={**source.metadata, "width": probe.width, "height": probe.height, "fps": probe.fps},
        ))
    trace.complete("media_probe", source_count=len(hydrated_sources))

    usage = check_processing_allowance(
        user_id=request.user_id,
        durations_sec=[source.duration_sec for source in hydrated_sources],
    )
    if not usage.allowed:
        trace.fail("usage_guard", reason=usage.reason)
        raise ValueError(f"processing denied: {usage.reason}")
    trace.complete(
        "usage_guard",
        reason=usage.reason,
        requested_minutes=round(usage.requested_minutes, 4),
        monthly_used_minutes=usage.monthly_used_minutes,
        monthly_limit_minutes=usage.monthly_limit_minutes,
    )
    notify("transcribing", 12)

    source_by_id = {source.source_asset_id: source for source in hydrated_sources}
    for source in sorted(hydrated_sources, key=lambda item: item.source_order):
        if source.has_audio:
            transcripts.extend(asr_provider.transcribe(
                local_paths[source.source_asset_id],
                source_asset_id=source.source_asset_id,
                language_hint=request.language_hint,
            ))
    transcript_tuple = tuple(transcripts)
    trace.complete("asr", segment_count=len(transcript_tuple))
    notify("analyzing", 27)

    local_performance = analyze_local_performance(local_paths, target_fps=12.0)
    local_frame_count = sum(len(item.observations) for item in local_performance.timelines)
    local_event_count = sum(len(item.events) for item in local_performance.timelines)
    if local_performance.status.available:
        trace.complete(
            "local_performance",
            status=local_performance.status.status,
            source_count=len(local_performance.timelines),
            frame_count=local_frame_count,
            candidate_event_count=local_event_count,
            target_fps=12.0,
        )
    else:
        trace.degraded(
            "local_performance",
            reason=local_performance.status.reason or local_performance.status.status,
            frame_count=local_frame_count,
            candidate_event_count=local_event_count,
        )
    notify("analyzing", 36)

    if whole_video_provider is not None and hydrated_sources:
        with tempfile.TemporaryDirectory(prefix="cutsell-whole-video-") as whole_dir:
            whole_samples = []
            for source in hydrated_sources:
                source_path = local_paths[source.source_asset_id]
                whole_samples.extend(sample_source_frames(
                    source_path,
                    source_asset_id=source.source_asset_id,
                    duration_sec=source.duration_sec,
                    output_dir=str(Path(whole_dir) / source.source_asset_id),
                ))
            whole_context = safe_whole_video_analyze(
                whole_video_provider,
                tuple(hydrated_sources),
                transcript_tuple,
                tuple(whole_samples),
            )
        if whole_context.status.status == "provider_error":
            trace.degraded("whole_video_context", reason=whole_context.status.reason or "provider_error")
        else:
            trace.complete(
                "whole_video_context",
                status=whole_context.status.status,
                edit_mode=whole_context.dominant_edit_mode,
                source_count=len(whole_context.sources),
                event_count=sum(len(item.events) for item in whole_context.sources),
                frame_count=len(whole_samples),
            )
    else:
        whole_context = safe_whole_video_analyze(None, tuple(hydrated_sources), transcript_tuple, ())
        trace.complete("whole_video_context", status="not_requested", source_count=0, event_count=0, frame_count=0)

    whole_context = merge_local_events_into_context(whole_context, local_performance.timelines)
    notify("analyzing", 45)

    gaps = word_silence_gaps(transcript_tuple)
    trace.complete("silence_analysis", gap_count=len(gaps))
    notify("analyzing", 50)

    takes = segment_takes(transcript_tuple, hydrated_sources, gaps)
    trace.complete("take_segmentation", candidate_count=len(takes))

    whole_context, confirmation_diagnostics = confirm_local_performance_events(
        takes,
        local_performance.timelines,
        whole_context,
    )
    confirmed_wrong_takes = sum(1 for item in confirmation_diagnostics if item.get("confirmed_kind") == "wrong_take")
    confirmed_retry_setups = sum(1 for item in confirmation_diagnostics if item.get("confirmed_kind") == "retry_setup")
    trace.complete(
        "performance_confirmation",
        confirmed_count=len(confirmation_diagnostics),
        confirmed_wrong_take_count=confirmed_wrong_takes,
        confirmed_retry_setup_count=confirmed_retry_setups,
    )
    notify("analyzing", 58)

    if visual_provider is not None and takes:
        with tempfile.TemporaryDirectory(prefix="cutsell-frames-") as frame_dir:
            samples = []
            for take in takes:
                source_path = local_paths.get(take.source_asset_id)
                if not source_path or take.source_asset_id not in source_by_id:
                    raise ValueError("candidate take lost source identity")
                samples.extend(sample_take_frames(source_path, take, frame_dir))
            visual = safe_visual_analyze(visual_provider, takes, tuple(samples))
        if visual.status.status in {"provider_error", "provider_unavailable"}:
            trace.degraded("visual", reason=visual.status.reason or visual.status.status)
        else:
            takes = apply_visual_observations(takes, visual.observations)
            trace.complete(
                "visual",
                status=visual.status.status,
                observation_count=len(visual.observations),
                frame_count=len(samples),
            )
    else:
        trace.complete("visual", status="not_requested", observation_count=0, frame_count=0)

    takes = apply_local_performance_to_takes(takes, local_performance.timelines)
    trace.complete(
        "local_performance_fusion",
        candidate_count=len(takes),
        local_source_count=sum(1 for item in local_performance.timelines if item.observations),
    )
    notify("analyzing", 69)

    # Reconstruct the creator's complete delivery attempts before any destructive Clean
    # Cut/Best Take decision. ASR chunks are evidence, not editorial takes. This stage
    # never deletes speech; absent a clear retry/reset boundary it preserves neighboring
    # fragments as one paragraph/delivery-level candidate.
    takes, attempt_reconstruction_diagnostics = reconstruct_delivery_attempts(takes, whole_context)
    trace.complete(
        "attempt_reconstruction",
        input_candidate_count=attempt_reconstruction_diagnostics.get("input_take_count", 0),
        attempt_count=attempt_reconstruction_diagnostics.get("attempt_count", 0),
        merged_fragment_count=attempt_reconstruction_diagnostics.get("merged_fragment_count", 0),
        boundary_count=len(attempt_reconstruction_diagnostics.get("boundaries") or ()),
    )
    notify("analyzing", 74)

    if mode == "full":
        semantic = safe_semantic_classify(semantic_provider or NoopSemanticProvider(), takes)
        if semantic.status.status in {"provider_error", "provider_unavailable"}:
            trace.degraded("semantic", reason=semantic.status.reason or semantic.status.status)
        else:
            trace.complete("semantic", status=semantic.status.status, label_count=len(semantic.labels))
        semantic_labels = semantic.labels
        active_composer = composer_provider
        active_reviewer = draft_review_provider
    else:
        semantic_labels = ()
        active_composer = None
        active_reviewer = None
        trace.complete("semantic", status="not_requested_clean_cut", label_count=0)
        trace.complete("editorial_composition", status="bypassed_clean_cut")

    notify("composing", 84)
    hydrated_request = replace(request, sources=tuple(hydrated_sources))
    result = build_flow_b_draft(
        hydrated_request,
        takes,
        semantic_labels,
        take_judge_provider=take_judge_provider,
        clean_cut_provider=clean_cut_provider,
        composer_provider=active_composer,
        take_grouping_provider=take_grouping_provider,
        draft_review_provider=active_reviewer,
        editorial_judge=editorial_judge,
        whole_video_context=whole_context,
        attempt_reconstruction_diagnostics=attempt_reconstruction_diagnostics,
        performance_confirmation_diagnostics=confirmation_diagnostics,
    )
    notify("draft_ready", 100)
    return replace(
        result,
        stage_status={"editorial_mode": mode, **result.stage_status, **trace.as_dict()},
    )