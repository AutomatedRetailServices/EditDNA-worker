"""Universal Clean Cut orchestration.

This module intentionally separates recording cleanup from downstream editorial
objectives. It answers one question only: can CutSell turn messy raw recording
footage into a clean, naturally ordered source timeline without sales/narrative
recomposition?

Allowed intelligence:
- ASR / silence analysis
- whole-video context for performance/retry evidence
- dense local performance tracking
- take-level visual evidence
- multimodal performance confirmation
- temporal boundary trimming
- deterministic Clean Cut
- retry grouping + Best Take
- optional bounded Hybrid Editorial classification for recording intent/BTS/failures

Explicitly disabled here:
- commercial semantic funnel labeling
- Sales/Natural story composition
- global draft rewriting/reordering/removal
"""
from __future__ import annotations

from typing import Mapping

from .asr import ASRProvider
from .clean_cut_provider import CleanCutProvider
from .contracts import ProcessingRequest, ProcessingResult
from .flow_b import ProgressCallback, process_local_sources
from .human_boundary_polish import polish_human_boundaries
from .hybrid_editorial import EditorialJudge
from .providers import NoopSemanticProvider
from .take_grouping_provider import TakeGroupingProvider
from .take_judge_provider import TakeJudgeProvider
from .visual_analysis import VisualProvider
from .whole_video_analysis import WholeVideoProvider


def process_universal_clean_cut_sources(
    request: ProcessingRequest,
    local_paths: Mapping[str, str],
    *,
    asr_provider: ASRProvider,
    visual_provider: VisualProvider | None = None,
    take_judge_provider: TakeJudgeProvider | None = None,
    clean_cut_provider: CleanCutProvider | None = None,
    take_grouping_provider: TakeGroupingProvider | None = None,
    whole_video_provider: WholeVideoProvider | None = None,
    editorial_judge: EditorialJudge | None = None,
    progress: ProgressCallback | None = None,
) -> ProcessingResult:
    """Run only the universal recording-cleanup brain.

    Natural source order is preserved after Best Take selection. The optional Hybrid
    judge is constrained to already-bounded creator mini-sessions and may classify
    failed/BTS attempts; it cannot activate Sales Funnel composition or change timing.
    """
    result = process_local_sources(
        request,
        local_paths,
        asr_provider=asr_provider,
        semantic_provider=NoopSemanticProvider(),
        visual_provider=visual_provider,
        take_judge_provider=take_judge_provider,
        clean_cut_provider=clean_cut_provider,
        composer_provider=None,
        take_grouping_provider=take_grouping_provider,
        draft_review_provider=None,
        whole_video_provider=whole_video_provider,
        editorial_judge=editorial_judge,
        progress=progress,
    )

    # Final physical-boundary authority.  Selection/meaning is already frozen; this
    # source-evidenced pass only fixes sentence-completion cuts, real dead air and
    # repeated tails.  With no dense timeline supplied, interior deletion still requires
    # >=1.35 s of source-proven silence, so uncertain gestures remain fail-open.
    result = polish_human_boundaries(result, local_paths, ())

    return ProcessingResult(
        schema_version=result.schema_version,
        project_id=result.project_id,
        state=result.state,
        draft=result.draft,
        stage_status={
            **result.stage_status,
            "brain_mode": "universal_clean_cut",
            "semantic": "not_requested_clean_cut_only",
            "composer": "not_requested_clean_cut_only",
            "draft_review": "not_requested_clean_cut_only",
            "human_boundary_polish": "source_evidenced_complete",
        },
    )
