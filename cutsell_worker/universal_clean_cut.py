"""Universal Clean Cut orchestration -- Clean Cut Core V1.

This module separates semantic Selection from physical Boundary. Clean Cut Core V1
(clean_cut_core_v1_enabled=True, the default) reasons idea-first: local perception,
attempt reconstruction, and idea/retry-family clustering (pipeline.py, take_grouping_
provider.py, semantic_idea_equivalence.py) already ran before this module is reached;
here the deterministic Best-Take authority and Final Story Coherence Validation are
the semantic authorities that decide final KEEP/DISCARD membership before the hard
freeze. Gemini participates only as a bounded semantic arbiter at specific points
(idea-equivalence during grouping, residual-ambiguity resolution during coherence
validation) -- it is never the primary editor, and the old whole-video Unified
Selection reasoner is deactivated in this path (kept only behind
clean_cut_core_v1_enabled=False for rollback). SWAP is out of scope for Clean Cut
Core V1: everything not SELECTed is DISCARDed, never parked as an alternate. Boundary
can then change timing/fragment structure only, never the selected spoken stream, and
must never repair a semantic membership mistake.
"""
from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Mapping

from .asr import ASRProvider
from .claim_coverage_best_take import apply_claim_coverage_best_take
from .clean_cut_provider import CleanCutProvider
from .contracts import ProcessingRequest, ProcessingResult
from .deterministic_best_take_authority import apply_deterministic_best_take_authority
from .final_boundary_authority import enforce_complete_idea_boundaries
from .final_story_coherence_validation import apply_final_story_coherence_validation
from .causal_order_validator import CausalOrderArbiter
from .semantic_atom_importance import SemanticAtomImportanceArbiter
from .semantic_claims import ClaimEquivalenceArbiter
from .repair_loop import run_repair_loop
from .flow_b import ProgressCallback, process_local_sources
from .human_boundary_polish_v5 import polish_human_boundaries_v5
from .hybrid_editorial import EditorialJudge
from .providers import NoopSemanticProvider
from .selection_boundary_contract import enforce_selection_contract, freeze_selection_contract
from .selection_conflicted_bridge_guard import apply_selection_conflicted_bridge_guard
from .selection_phase_authority import apply_selection_phase_authority
from .semantic_idea_equivalence import SemanticEquivalenceArbiter
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
    deterministic_best_take_authority_enabled: bool = True,
    semantic_equivalence_arbiter: SemanticEquivalenceArbiter | None = None,
    causal_order_arbiter: CausalOrderArbiter | None = None,
    semantic_atom_importance_arbiter: SemanticAtomImportanceArbiter | None = None,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
    clean_cut_core_v1_enabled: bool = True,
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
        semantic_equivalence_arbiter=semantic_equivalence_arbiter,
        progress=progress,
    )

    has_draft_contract = hasattr(result.draft, "selected") and hasattr(result.draft, "discarded")
    if has_draft_contract:
        if clean_cut_core_v1_enabled:
            # Clean Cut Core V1 (see CLAUDE.md / docs/CUTSELL_DECISIONS.md):
            # idea-first deterministic pipeline is the one active path.
            # Gemini is a bounded semantic arbiter (semantic_idea_equivalence,
            # invoked from pipeline.py's grouping stage and again from Final
            # Story Coherence Validation below), never the primary editor --
            # the whole-video Unified Selection reasoner is deactivated here
            # regardless of whether a selection_reasoner instance was passed
            # in; it is retained only behind clean_cut_core_v1_enabled=False
            # for rollback. SWAP is out of scope for this path: a legitimate
            # losing retry is DISCARDed, not parked as an alternate.
            result = replace(result, draft=apply_selection_phase_authority(result.draft))
            result = replace(result, draft=apply_selection_conflicted_bridge_guard(result.draft))
            result = replace(
                result,
                draft=apply_deterministic_best_take_authority(result.draft, swap_enabled=False),
            )
            # D-038: a visually/performance-clean take must not beat a
            # semantically complete one -- runs strictly after the
            # deterministic ranker's own verdict, before it becomes final,
            # so it can still correct a clear-winner decision that drops a
            # critical audience-facing claim another family member carried.
            result = replace(
                result,
                draft=apply_claim_coverage_best_take(
                    result.draft, claim_equivalence_arbiter=claim_equivalence_arbiter,
                ),
            )
            result = replace(
                result,
                draft=apply_final_story_coherence_validation(
                    result.draft,
                    semantic_equivalence_arbiter=semantic_equivalence_arbiter,
                    semantic_atom_importance_arbiter=semantic_atom_importance_arbiter,
                    claim_equivalence_arbiter=claim_equivalence_arbiter,
                ),
            )
            selection_stage = "clean_cut_core_v1_idea_first_keep_discard"
            semantic_status = "clean_cut_core_v1_idea_first"
            reasoner_status_label = "disabled_clean_cut_core_v1"
        elif selection_reasoner is not None:
            # One whole-video semantic authority sees Selected + SWAP + Discarded
            # together and decides. Local/group decisions inform its payload as
            # evidence, but no longer have unconditional final say afterward --
            # see the deterministic Best-Take pass immediately below.
            result = replace(
                result,
                draft=apply_unified_selection_reasoner(result.draft, selection_reasoner),
            )
            reasoner_diag = (result.draft.diagnostics or {}).get("unified_selection_reasoner") or {}
            reasoner_status = str(reasoner_diag.get("status") or "unknown")
            selection_stage = f"unified_whole_video_selection_{reasoner_status}"

            # Architecture rebalance Phase 0/1: Unified Selection and the
            # deterministic take_judge Best-Take layer are now sequential
            # rather than Unified Selection having unconditional final say.
            # For a retry-family contest the local ranker was genuinely
            # decisive about, its verdict becomes authoritative here; an
            # ambiguous (thin score-gap) contest is left exactly as Unified
            # Selection decided it. Rollback: set
            # CUTSELL_DETERMINISTIC_BEST_TAKE_AUTHORITY=0 to restore the
            # previous pure-whole-video-reasoner behavior unmodified.
            if deterministic_best_take_authority_enabled:
                result = replace(
                    result,
                    draft=apply_deterministic_best_take_authority(result.draft, swap_enabled=True),
                )
                selection_stage = f"{selection_stage}+deterministic_best_take_authority"
            semantic_status = "whole_video_selection"
            reasoner_status_label = "enabled"
        else:
            # Legacy fallback remains available while Unified Selection is
            # feature-gated. Untouched by this phase: this path already has its
            # own, more targeted Best-Take reconciliation (pipeline.py's
            # _semantic_best_take plus these Hybrid-vote-informed guards); the
            # new deterministic override above is scoped to Unified Selection
            # mode only, so it can never undo a legitimate Hybrid semantic
            # override made here.
            result = replace(result, draft=apply_selection_phase_authority(result.draft))
            result = replace(result, draft=apply_selection_conflicted_bridge_guard(result.draft))
            selection_stage = "legacy_explicit_final_selection_authority_executed"
            semantic_status = "not_requested_clean_cut_only"
            reasoner_status_label = "disabled"

        # CanonicalEditPlan (D-024) + bounded targeted repair loop (D-026) +
        # general causal/story order validation (D-027): build v1, review it
        # (review now also runs CAUSAL_ORDER_BREAK's general cross-idea
        # dependency check, see causal_order_validator.py), and -- only for
        # finding types with a safe, content-preserving repair strategy
        # (today: STORY_ORDER_BREAK's composite reordering; CAUSAL_ORDER_
        # BREAK has none by design -- a cross-idea reorder risks undoing an
        # intentional Composer pacing choice, see repair_loop.py's own
        # docstring) -- apply bounded, targeted repairs and re-review. Never
        # invents semantic judgment; never mutates an unrelated Idea.
        repair_result = run_repair_loop(result.draft, causal_order_arbiter=causal_order_arbiter)
        edit_plan = repair_result.final_plan
        review_result = repair_result.final_review
        result = replace(result, draft=repair_result.final_draft)
        diagnostics = dict(result.draft.diagnostics or {})
        diagnostics["canonical_edit_plan"] = dataclasses.asdict(edit_plan)
        diagnostics["final_edit_reviewer"] = {
            "status": review_result.status,
            "findings": [dataclasses.asdict(f) for f in review_result.findings],
            "warnings": [dataclasses.asdict(f) for f in review_result.warnings],
        }
        diagnostics["repair_loop"] = {
            "status": repair_result.status,
            "attempt_count": len(repair_result.attempts),
            "attempts": [dataclasses.asdict(a) for a in repair_result.attempts],
        }
        result = replace(result, draft=replace(result.draft, diagnostics=diagnostics))
        final_edit_reviewer_status = review_result.status

        # Hard pre-Freeze gate: Final Story Coherence Validation may find a
        # high-confidence semantic failure (an unresolved factual
        # contradiction between still-co-selected same-retry-family members,
        # or an entire intended idea losing every member from the final
        # selected set) that must never reach Selection Freeze. The repair
        # loop's own NEEDS_HUMAN_REVIEW outcome (FinalEditReviewer still FAILs
        # after exhausting any safe repair) is the same kind of finding --
        # not something Boundary could ever repair -- so this skips freeze/
        # boundary entirely and surfaces the draft as-is (still selected/
        # discarded, just unfrozen) for human review rather than silently
        # producing a bad video.
        coherence_diag = (result.draft.diagnostics or {}).get("final_story_coherence_validation") or {}
        freeze_blocked = bool(coherence_diag.get("freeze_blocked")) or repair_result.status == "NEEDS_HUMAN_REVIEW"

        if freeze_blocked:
            recovery_stage = "not_applicable_freeze_blocked_by_coherence_validation"
            polish_stage = "not_applicable_freeze_blocked_by_coherence_validation"
            contract_stage = "not_applicable_freeze_blocked_by_coherence_validation"
            selection_stage = f"{selection_stage}+freeze_blocked_pending_human_review"

            # D-025 (Issue 2): install_selection_freeze()/install_boundary_
            # selection_invariant() unconditionally freeze+verify inside
            # process_local_sources -> build_flow_b_draft, BEFORE StoryValidator/
            # CanonicalEditPlan/FinalEditReviewer ever run in this module -- a
            # holdover from the pre-V1 architecture where build_flow_b_draft's
            # own output was the final answer. That leaves diagnostics.
            # selection_boundary_contract.status stuck at "frozen"/"verified"
            # from that premature, pre-StoryValidator freeze even though this
            # gate has just determined the real final draft must NOT be frozen
            # -- a direct evidence-level contradiction (RAW 33366538992: the
            # result JSON reported freeze_blocked=true AND selection_boundary_
            # contract.status=frozen simultaneously). This is the one place
            # that authoritatively knows the true state, so it corrects the
            # record rather than leaving that stale, misleading key in place.
            stale_contract = dict((result.draft.diagnostics or {}).get("selection_boundary_contract") or {})
            diagnostics = dict(result.draft.diagnostics or {})
            diagnostics["selection_boundary_contract"] = {
                "schema_version": "cutsell.selection_boundary_contract.v1",
                "status": "not_frozen_freeze_blocked_by_coherence_review",
                "plan_id": edit_plan.plan_id,
                "plan_version": edit_plan.plan_version,
                "semantic_hash": edit_plan.semantic_hash,
                "superseded_premature_freeze_status": stale_contract.get("status"),
            }
            result = replace(result, draft=replace(result.draft, diagnostics=diagnostics))
        else:
            # Complete-idea recovery may restore source-proven leading/trailing spoken words.
            # It therefore belongs before Selection freeze regardless of semantic authority.
            result = enforce_complete_idea_boundaries(
                result,
                local_paths,
                asr_provider=asr_provider,
            )
            recovery_stage = "complete_idea_word_lock_overlap_guard_before_freeze"

            # Hard semantic phase barrier. Everything after this line is Boundary-only.
            # Freezes the specific plan FinalEditReviewer PASSed (D-025) --
            # see freeze_selection_contract's own docstring for why this is
            # observability (matches_reviewed_plan), not a hard equality gate.
            result = replace(result, draft=freeze_selection_contract(result.draft, plan=edit_plan))

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
        semantic_status = "not_requested_clean_cut_only"
        reasoner_status_label = "disabled"
        freeze_blocked = False
        final_edit_reviewer_status = "not_applicable_missing_draft_contract"

    return ProcessingResult(
        schema_version=result.schema_version,
        project_id=result.project_id,
        state=result.state,
        draft=result.draft,
        stage_status={
            **result.stage_status,
            "freeze_blocked_pending_coherence_review": freeze_blocked,
            "final_edit_reviewer": final_edit_reviewer_status,
            "brain_mode": "universal_clean_cut",
            "semantic": semantic_status,
            "composer": "not_requested_clean_cut_only",
            "draft_review": "not_requested_clean_cut_only",
            "selection_phase_authority": selection_stage,
            "unified_selection_reasoner": reasoner_status_label,
            "selection_boundary_contract": contract_stage,
            "human_boundary_polish": polish_stage,
            "final_boundary_authority": recovery_stage,
        },
    )

# Raw benchmark trigger marker: unified whole-video Selection reasoner pivot.
