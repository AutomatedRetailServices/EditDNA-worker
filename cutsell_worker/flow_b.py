"""Real-video Flow B orchestration for CutSell Milestone 1."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Callable, Mapping

from .asr import ASRProvider
from .clean_cut_provider import CleanCutProvider
from .composer_provider import ComposerProvider
from .contracts import ProcessingRequest, ProcessingResult
from .frame_sampling import sample_take_frames
from .media_probe import probe_media
from .observability import ExecutionTrace
from .pipeline import build_flow_b_draft
from .providers import NoopSemanticProvider, SemanticProvider, safe_semantic_classify
from .silence_analysis import word_silence_gaps
from .source_sampling import sample_source_frames
from .take_grouping_provider import TakeGroupingProvider
from .take_judge_provider import TakeJudgeProvider
from .take_segmentation import segment_takes
from .temporal_editing import refine_takes_with_temporal_context
from .usage_limits import check_processing_allowance
from .visual_analysis import VisualProvider, apply_visual_observations, safe_visual_analyze
from .whole_video_analysis import WholeVideoProvider, safe_whole_video_analyze

ProgressCallback = Callable[[str, int], None]


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
    whole_video_provider: WholeVideoProvider | None = None,
    progress: ProgressCallback | None = None,
) -> ProcessingResult:
    """Process registered sources all the way from local media to editable draft."""
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
    notify("analyzing", 30)

    # Pass 1: understand the complete source before destructive editing. This
    # creates product/topic/story logic plus timestamped performance events.
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
    notify("analyzing", 43)

    gaps = word_silence_gaps(transcript_tuple)
    trace.complete("silence_analysis", gap_count=len(gaps))
    notify("analyzing", 49)

    takes = segment_takes(transcript_tuple, hydrated_sources, gaps)
    trace.complete("take_segmentation", candidate_count=len(takes))
    notify("analyzing", 58)

    # Pass 2: take-level visual scoring complements the global timeline.
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
    notify("analyzing", 69)

    # Global context now becomes actionable. Safely trim high-confidence body
    # resets/reactions/fumbles at take edges while preserving interior uncertainty.
    takes, temporal_trim_diagnostics = refine_takes_with_temporal_context(takes, whole_context)
    applied_trim_count = sum(1 for item in temporal_trim_diagnostics if item.get("applied"))
    interior_event_count = sum(len(item.get("interior_bad_events") or ()) for item in temporal_trim_diagnostics)
    trace.complete(
        "temporal_performance",
        candidate_count=len(takes),
        trimmed_take_count=applied_trim_count,
        interior_bad_event_count=interior_event_count,
    )
    notify("analyzing", 74)

    semantic = safe_semantic_classify(semantic_provider or NoopSemanticProvider(), takes)
    if semantic.status.status in {"provider_error", "provider_unavailable"}:
        trace.degraded("semantic", reason=semantic.status.reason or semantic.status.status)
    else:
        trace.complete("semantic", status=semantic.status.status, label_count=len(semantic.labels))
    notify("composing", 84)

    hydrated_request = replace(request, sources=tuple(hydrated_sources))
    result = build_flow_b_draft(
        hydrated_request,
        takes,
        semantic.labels,
        take_judge_provider=take_judge_provider,
        clean_cut_provider=clean_cut_provider,
        composer_provider=composer_provider,
        take_grouping_provider=take_grouping_provider,
        whole_video_context=whole_context,
        temporal_trim_diagnostics=temporal_trim_diagnostics,
    )
    notify("draft_ready", 100)
    return replace(result, stage_status={**result.stage_status, **trace.as_dict()})
