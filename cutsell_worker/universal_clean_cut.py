"""Universal Clean Cut orchestration.

This module separates semantic Selection from physical Boundary.  Local perception and
legacy grouping may generate evidence and provisional buckets, but when the Unified
Selection Reasoner is enabled one whole-video semantic authority decides final spoken
membership before the hard freeze. Boundary can then change timing/fragment structure
only, never the selected spoken stream.
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
from .selection_conflicted_bridge_guard import apply_selection_conflicted_bridge_guard
from .selection_phase_authority import apply_selection_phase_authority
from .take_grouping_provider import TakeGroupingProvider
from .take_judge_provider import TakeJudgeProvider
from .unified_selection_reasoner import UnifiedSelectionReasoner, apply_unified_selection_reasoner
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
    selection_reasoner: UnifiedSelectionReasoner | None = None,
    progress: ProgressCallback | None = None,
) -> ProcessingResult:
    """Run the Universal Clean Cut brain with explicit Selection/Boundary ownership."""
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
        if selection_reasoner is not None:
            # PIVOT: one whole-video semantic authority sees Selected + SWAP + Discarded
            # together. Legacy local/group decisions become evidence rather than final
            # membership authorities. No per-benchmark semantic guard runs after this.
            result = replace(
                result,
                draft=apply_unified_selection_reasoner(result.draft, selection_reasoner),
            )
            reasoner_diag = (result.draft.diagnostics or {}).get("unified_selection_reasoner") or {}
            reasoner_status = str(reasoner_diag.get("status") or "unknown")
            selection_stage = f"unified_whole_video_selection_{reasoner_status}"
        else:
            # Legacy fallback remains available while Unified Selection is feature-gated.
            result = replace(result, draft=apply_selection_phase_authority(result.draft))
            result = replace(result, draft=apply_selection_conflicted_bridge_guard(result.draft))
            selection_stage = "legacy_explicit_final_selection_authority_executed"

        # Complete-idea recovery may restore source-proven leading/trailing spoken words.
        # It therefore belongs before Selection freeze regardless of semantic authority.
        result = enforce_complete_idea_boundaries(
            result,
            local_paths,
            asr_provider=asr_provider,
        )
        recovery_stage = "complete_idea_word_lock_overlap_guard_before_freeze"

        # Hard semantic phase barrier. Everything after this line is Boundary-only.
        result = replace(result, draft=freeze_selection_contract(result.draft))

        result = polish_human_boundaries_v5(result, local_paths)
        polish_stage = "source_evidenced_multimodal_v5_boundary_only_complete"

        # Fail closed if Boundary changed ordered spoken content after the freeze.
        result = replace(result, draft=enforce_selection_contract(result.draft))
        contract_stage = "selection_semantic_stream_verified_after_boundary"
    else:
        selection_stage = "not_applicable_missing_draft_contract"
        polish_stage = "not_applicable_missing_draft_contract"
        recovery_stage = "not_applicable_missing_draft_contract"
        contract_stage = "not_applicable_missing_draft_contract"

    return ProcessingResult(
        schema_version=result.schema_version,
        project_id=result.project_id,
        state=result.state,
        draft=result.draft,
        stage_status={
            **result.stage_status,
            "brain_mode": "universal_clean_cut",
            "semantic": "whole_video_selection" if selection_reasoner is not None else "legacy_clean_cut_selection",
            "composer": "not_requested_clean_cut_only",
            "draft_review": "not_requested_clean_cut_only",
            "selection_phase_authority": selection_stage,
            "unified_selection_reasoner": "enabled" if selection_reasoner is not None else "disabled",
            "selection_boundary_contract": contract_stage,
            "human_boundary_polish": polish_stage,
            "final_boundary_authority": recovery_stage,
        },
    )

# Raw benchmark trigger marker: unified whole-video Selection reasoner pivot.
