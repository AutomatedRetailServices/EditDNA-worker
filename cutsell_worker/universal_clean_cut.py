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

from dataclasses import replace
from typing import Mapping

from .asr import ASRProvider
from .clean_cut_provider import CleanCutProvider
from .contracts import ProcessingRequest, ProcessingResult
from .final_boundary_authority import enforce_complete_idea_boundaries
from .flow_b import ProgressCallback, process_local_sources
from .human_boundary_polish_v5 import polish_human_boundaries_v5
from .hybrid_editorial import EditorialJudge
from .providers import NoopSemanticProvider
from .selection_boundary_contract import enforce_selection_contract, freeze_selection_contract
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

    Phase ownership is enforced here. Any pass that can restore/add spoken words belongs
    to Selection/semantic recovery and must run before the freeze. Only speech-preserving
    timing/fragment operations may run after the freeze.
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

    has_draft_contract = hasattr(result.draft, "selected") and hasattr(result.draft, "discarded")
    if has_draft_contract:
        # Complete-idea recovery can restore missing spoken leading/trailing words. That is
        # semantic Selection recovery, not Boundary timing, so it MUST run before freeze.
        result = enforce_complete_idea_boundaries(
            result,
            local_paths,
            asr_provider=asr_provider,
        )
        boundary_stage = "complete_idea_word_lock_overlap_guard_before_freeze"

        # Hard Selection/Boundary phase barrier after every operation allowed to change
        # spoken membership/content.
        result = replace(result, draft=freeze_selection_contract(result.draft))

        # Boundary-only polish: may split/remove source-evidenced non-speech gaps, but may
        # not add, remove, substitute, or reorder spoken tokens.
        result = polish_human_boundaries_v5(result, local_paths)
        polish_stage = "source_evidenced_multimodal_v5_boundary_only_complete"

        # Unavoidable output invariant. Any post-freeze mutation of ordered spoken content
        # is a pipeline bug and fails closed before a renderable result can escape.
        result = replace(result, draft=enforce_selection_contract(result.draft))
        contract_stage = "selection_semantic_stream_verified_after_boundary"
    else:
        polish_stage = "not_applicable_missing_draft_contract"
        boundary_stage = "not_applicable_missing_draft_contract"
        contract_stage = "not_applicable_missing_draft_contract"

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
            "selection_boundary_contract": contract_stage,
            "human_boundary_polish": polish_stage,
            "final_boundary_authority": boundary_stage,
        },
    )
