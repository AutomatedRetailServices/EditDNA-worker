"""Real-video Flow B orchestration for CutSell Milestone 1."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping, Tuple

from .asr import ASRProvider
from .contracts import ProcessingRequest, ProcessingResult, SourceAsset
from .media_probe import probe_media
from .observability import ExecutionTrace
from .pipeline import build_flow_b_draft
from .providers import NoopSemanticProvider, SemanticProvider, safe_semantic_classify
from .silence_analysis import word_silence_gaps
from .take_segmentation import segment_takes


def process_local_sources(
    request: ProcessingRequest,
    local_paths: Mapping[str, str],
    *,
    asr_provider: ASRProvider,
    semantic_provider: SemanticProvider | None = None,
) -> ProcessingResult:
    """Process registered sources all the way from local media to editable draft.

    ``local_paths`` is keyed by source_asset_id. Download/upload concerns remain
    outside this function so the engine stays deterministic and testable.
    """
    trace = ExecutionTrace()
    hydrated_sources = []
    transcripts = []

    for source in sorted(request.sources, key=lambda item: item.source_order):
        path = local_paths.get(source.source_asset_id)
        if not path:
            raise ValueError(f"missing local path for source {source.source_asset_id}")
        if not Path(path).exists():
            raise FileNotFoundError(path)

        probe = probe_media(path)
        hydrated = replace(
            source,
            duration_sec=probe.duration_sec,
            has_audio=probe.has_audio,
            metadata={
                **source.metadata,
                "width": probe.width,
                "height": probe.height,
                "fps": probe.fps,
            },
        )
        hydrated_sources.append(hydrated)
        if not probe.has_audio:
            continue
        transcripts.extend(asr_provider.transcribe(
            path,
            source_asset_id=source.source_asset_id,
            language_hint=request.language_hint,
        ))

    trace.complete("media_probe", source_count=len(hydrated_sources))
    trace.complete("asr", segment_count=len(transcripts))

    gaps = word_silence_gaps(transcripts)
    trace.complete("silence_analysis", gap_count=len(gaps))

    takes = segment_takes(transcripts, hydrated_sources, gaps)
    trace.complete("take_segmentation", candidate_count=len(takes))

    semantic = safe_semantic_classify(semantic_provider or NoopSemanticProvider(), takes)
    if semantic.status.status in {"provider_error", "provider_unavailable"}:
        trace.degraded("semantic", reason=semantic.status.reason or semantic.status.status)
    else:
        trace.complete("semantic", status=semantic.status.status, label_count=len(semantic.labels))

    hydrated_request = replace(request, sources=tuple(hydrated_sources))
    result = build_flow_b_draft(hydrated_request, takes, semantic.labels)
    return replace(
        result,
        stage_status={
            **result.stage_status,
            **trace.as_dict(),
        },
    )
