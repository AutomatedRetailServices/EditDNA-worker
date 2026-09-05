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
from .semantic_claims import ClaimEquivalenceArbiter, ClauseRoleArbiter
from .canonical_edit_plan import authoritative_plan_source_to_diagnostics, build_authoritative_plan_source
from .repair_loop import run_repair_loop
from .flow_b import ProgressCallback, process_local_sources
from .semantic_ledger import build_ledger_parity_report, build_semantic_ledger_diagnostics, build_semantic_ledger_shadow
from .realization_resolver import (
    apply_authoritative_realization_resolution,
    AUTHORITATIVE_REVIEW_REQUIRED,
    build_authoritative_resolution_diagnostics,
    build_authoritative_semantic_state,
    build_authoritative_semantic_state_diagnostics,
    build_realization_resolver_diagnostics,
    build_preserved_claim_id_index,
    build_semantic_preservation_proofs,
    build_semantic_preservation_proofs_diagnostics,
    resolve_intra_idea_semantic_preservation_shadow,
    resolve_pre_group_semantic_preservation_shadow,
    resolve_realizations_shadow,
)
from .resolver_mode import RESOLVER_MODE_AUTHORITATIVE, resolve_resolver_mode
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
    clause_role_arbiter: ClauseRoleArbiter | None = None,
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
                    result.draft,
                    claim_equivalence_arbiter=claim_equivalence_arbiter,
                    clause_role_arbiter=clause_role_arbiter,
                ),
            )
            result = replace(
                result,
                draft=apply_final_story_coherence_validation(
                    result.draft,
                    semantic_equivalence_arbiter=semantic_equivalence_arbiter,
                    semantic_atom_importance_arbiter=semantic_atom_importance_arbiter,
                    claim_equivalence_arbiter=claim_equivalence_arbiter,
                    clause_role_arbiter=clause_role_arbiter,
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
        #
        # D-050C3 Section 1/4: this FIRST pass runs on whatever the 3-way
        # selection branch above produced -- the pre-cutover ("legacy")
        # draft. In LEGACY/SHADOW mode (and always for the two non-Clean-
        # Cut-Core-V1 selection branches, which the Unified Realization
        # Resolver was never designed for) this pass's output IS the real,
        # final diagnostics -- nothing below ever touches it again. In
        # AUTHORITATIVE mode it is ALSO still needed here: (a) the Semantic
        # Ledger's own reconstruction reads `final_story_coherence_
        # validation`/`canonical_edit_plan` diagnostics for one enrichment
        # (see semantic_ledger.py Section 11), so this pass has to exist
        # before the Ledger is built, and (b) it becomes this run's LEGACY
        # EVIDENCE for comparison once the authoritative pass below
        # recomputes these same three diagnostics keys on the resolver's
        # OWN resolved draft and takes over as the real ones Freeze reads.
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

        # D-050B: Semantic Ledger. Built here -- after every stage it
        # observes (grouping, DeliveryScorer, semantic best-take,
        # ClaimCoverage, StoryValidator, CanonicalEditPlan, FinalEditReviewer)
        # has already run and already written its own diagnostics -- as a
        # pure, read-only reconstruction. In LEGACY/SHADOW mode this remains
        # purely observational: nothing below this line (Freeze, Boundary,
        # complete-idea recovery, Render/QC) reads `diagnostics
        # ["semantic_ledger"]`. In AUTHORITATIVE mode it is ALSO the
        # Unified Realization Resolver's own input two steps below -- see
        # semantic_ledger.py's module docstring for the full contract.
        ledger = build_semantic_ledger_shadow(result.draft)
        ledger_parity = build_ledger_parity_report(ledger, result.draft)
        diagnostics = dict(result.draft.diagnostics or {})
        diagnostics["semantic_ledger"] = build_semantic_ledger_diagnostics(ledger, ledger_parity)
        result = replace(result, draft=replace(result.draft, diagnostics=diagnostics))

        # D-050C1/D-050C1.5/D-050C1.6: Unified Realization Resolver. Consumes
        # the Semantic Ledger built immediately above and computes what ONE
        # unified resolver decides per semantic idea. `resolve_realizations_
        # shadow` itself NEVER writes to a DraftTimeline -- it is pure
        # observation, same as every prior D-050C1.x directive. Whether that
        # decision is APPLIED depends entirely on `resolver_mode` below.
        resolver_mode = resolve_resolver_mode()
        resolver_report = resolve_realizations_shadow(ledger)
        diagnostics = dict(result.draft.diagnostics or {})
        diagnostics["realization_resolver_shadow"] = build_realization_resolver_diagnostics(resolver_report)
        result = replace(result, draft=replace(result.draft, diagnostics=diagnostics))

        # D-050C2/D-050C3 CONTROLLED AUTHORITY CUTOVER -- see resolver_mode.py's
        # own module docstring for the 3-state contract (LEGACY/SHADOW/
        # AUTHORITATIVE, default LEGACY, one environment variable, no code
        # revert to roll back) and realization_resolver.py's
        # `apply_authoritative_realization_resolution` docstring for exactly
        # what gets applied. This is THE ONE explicit point in the pipeline
        # the resolver's decision is ever applied (Section 3) -- no other
        # module below this line, or above it, mutates selection membership
        # on the resolver's behalf. Gated on `clean_cut_core_v1_enabled` too:
        # the resolver's per-idea/retry-family model only makes sense against
        # Clean Cut Core V1's own idea-first grouping.
        #
        # D-050C3 Section 4 (the C2 evidence-only exemption removed): where
        # D-050C2 left CanonicalEditPlan/StoryValidator/FinalEditReviewer's
        # FIRST-pass output (computed above, on the pre-cutover draft) as the
        # diagnostics Freeze actually reads even in AUTHORITATIVE mode, this
        # phase re-runs all three -- StoryValidator, then CanonicalEditPlan
        # + FinalEditReviewer + bounded repair -- a SECOND time, strictly on
        # the resolver's own resolved draft, and THAT second pass becomes
        # the real `diagnostics["canonical_edit_plan"]`/`["final_edit_
        # reviewer"]`/`["final_story_coherence_validation"]`/["repair_loop"]`
        # keys Freeze reads below. The first pass's output for those same
        # four keys is relabeled `*_legacy_evidence` -- still present, still
        # fully computed, but structurally unable to block or approve
        # anything: no code below this point ever reads those `_legacy_
        # evidence` keys for a decision, only for comparison/observability.
        # In `LEGACY`/`SHADOW` mode (and the two non-Clean-Cut-V1 selection
        # branches) none of this runs: identical to every prior D-050C1.x/
        # D-050C2 directive -- the first pass above is the only pass, full
        # stop.
        authoritative_result = None
        authoritative_semantic_state = None
        if clean_cut_core_v1_enabled and resolver_mode == RESOLVER_MODE_AUTHORITATIVE:
            authoritative_result = apply_authoritative_realization_resolution(
                result.draft, ledger, resolver_report, claim_equivalence_arbiter=claim_equivalence_arbiter,
            )

            pre_authority_diagnostics = dict(result.draft.diagnostics or {})
            legacy_evidence_keys = (
                "canonical_edit_plan", "final_edit_reviewer", "repair_loop",
                "final_story_coherence_validation",
            )
            authoritative_diagnostics = {
                key: value for key, value in pre_authority_diagnostics.items()
                if key not in legacy_evidence_keys
            }
            for key in legacy_evidence_keys:
                if key in pre_authority_diagnostics:
                    authoritative_diagnostics[f"{key}_legacy_evidence"] = pre_authority_diagnostics[key]
            # D-087 SINGLE-TRUTH HANDOFF: the resolver's own per-idea verdict
            # (winner / composite / review-required, with its coverage
            # evidence) becomes the canonical source CanonicalEditPlan
            # represents in AUTHORITATIVE mode -- built once here from the
            # very same `authoritative_result` + `ledger` every other
            # authoritative stage consumes, stored as diagnostics (Section
            # 15) and handed to the repair loop below explicitly. LEGACY/
            # SHADOW never reach this branch, so they never carry the key
            # and CanonicalEditPlan keeps its pre-D-087 path there.
            authoritative_plan_source = build_authoritative_plan_source(authoritative_result, ledger)
            authoritative_diagnostics["authoritative_plan_source"] = authoritative_plan_source_to_diagnostics(
                authoritative_plan_source
            )
            authoritative_draft = replace(authoritative_result.draft, diagnostics=authoritative_diagnostics)

            # D-076: SEMANTIC_PRESERVATION_PROOF -- built from the same
            # `ledger`/`resolver_report` already computed above, ONLY here
            # (AUTHORITATIVE mode's own second StoryValidator pass) since
            # this is structurally the first point in the pipeline both
            # the Ledger and a resolved draft exist together. StoryValidator
            # only ever consumes this map (one dict lookup per discarded
            # clip) -- see final_story_coherence_validation.py's own
            # consumption comment; it discovers no candidate, extracts no
            # claim, and invokes no arbiter of its own for this decision.
            pre_group_semantic_preservation_proofs = resolve_pre_group_semantic_preservation_shadow(
                ledger, claim_equivalence_arbiter=claim_equivalence_arbiter,
            )
            # D-079 Phase 1/2: the third, remaining discard population --
            # a realization that DID reach grouping and lost, within its
            # own idea, to that idea's own resolved winner/composite.
            # Reuses the ALREADY-COMPUTED `resolver_report` from this same
            # function's own diagnostics pass above (no redundant second
            # per-idea resolution).
            intra_idea_semantic_preservation_proofs = resolve_intra_idea_semantic_preservation_shadow(
                ledger, claim_equivalence_arbiter=claim_equivalence_arbiter, resolver_report=resolver_report,
            )
            semantic_preservation_proofs = build_semantic_preservation_proofs(
                ledger, claim_equivalence_arbiter=claim_equivalence_arbiter,
                pre_group_proofs=pre_group_semantic_preservation_proofs,
                intra_idea_proofs=intra_idea_semantic_preservation_proofs,
            )
            # D-079 Phase 1/2: the single, CLAIM-scoped index `_lost_
            # critical_claims` consumes -- built from ALL verified proofs
            # (hybrid_editorial PATH A/B reframed, pre-group, and this
            # directive's own intra-idea pass), never from a coarse clip-
            # or idea-level credit. See `build_preserved_claim_id_index`'s
            # own docstring for the full contract.
            critical_claim_preservation_index = build_preserved_claim_id_index(
                pre_group_semantic_preservation_proofs, intra_idea_semantic_preservation_proofs,
            )

            # StoryValidator, AUTHORITATIVELY: re-validated on the resolver's
            # own resolved selection, never the pre-cutover one.
            authoritative_draft = apply_final_story_coherence_validation(
                authoritative_draft,
                semantic_equivalence_arbiter=semantic_equivalence_arbiter,
                semantic_atom_importance_arbiter=semantic_atom_importance_arbiter,
                claim_equivalence_arbiter=claim_equivalence_arbiter,
                clause_role_arbiter=clause_role_arbiter,
                semantic_preservation_proofs=semantic_preservation_proofs,
                critical_claim_preservation_index=critical_claim_preservation_index,
            )

            # CanonicalEditPlan + FinalEditReviewer + bounded repair,
            # AUTHORITATIVELY: same call as the first pass above, now
            # operating on the resolved draft -- this becomes the plan
            # Freeze actually consumes.
            repair_result = run_repair_loop(
                authoritative_draft,
                causal_order_arbiter=causal_order_arbiter,
                authoritative_source=authoritative_plan_source,
            )
            edit_plan = repair_result.final_plan
            review_result = repair_result.final_review
            result = replace(result, draft=repair_result.final_draft)
            final_edit_reviewer_status = review_result.status

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
            authoritative_semantic_state = build_authoritative_semantic_state(authoritative_result, ledger)
            diagnostics["authoritative_semantic_state"] = build_authoritative_semantic_state_diagnostics(
                authoritative_semantic_state
            )
            # D-076 Section 14: every PRE_GROUP_SEMANTIC_PRESERVATION
            # attempt, verified or not -- LEXICAL_REPLACEMENT/SEMANTIC_
            # REPLACEMENT's own full evidence stays in `realization_
            # resolver_shadow`/`realization_resolver_authority`'s existing
            # `orphan_reviews`, unchanged.
            diagnostics["semantic_preservation_proofs"] = build_semantic_preservation_proofs_diagnostics(
                pre_group_semantic_preservation_proofs
            )
            result = replace(result, draft=replace(result.draft, diagnostics=diagnostics))
        diagnostics = dict(result.draft.diagnostics or {})
        diagnostics["realization_resolver_authority"] = (
            build_authoritative_resolution_diagnostics(authoritative_result, mode=resolver_mode)
            if authoritative_result is not None
            else {"schema_version": "cutsell.realization_resolver_authority.v1", "mode": resolver_mode, "status": None, "ideas": []}
        )
        result = replace(result, draft=replace(result.draft, diagnostics=diagnostics))

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
        # producing a bad video. D-050C2 Section 11 (Freeze contract): in
        # AUTHORITATIVE mode, the resolver's own REVIEW_REQUIRED status is
        # an equally hard gate -- OR'd in here, never allowed to be
        # silently overridden by a legacy coherence check that ran on the
        # PRE-cutover selection and has no visibility into the resolver's
        # own verdict.
        coherence_diag = (result.draft.diagnostics or {}).get("final_story_coherence_validation") or {}
        freeze_blocked = bool(coherence_diag.get("freeze_blocked")) or repair_result.status == "NEEDS_HUMAN_REVIEW"
        if authoritative_result is not None and authoritative_result.status == AUTHORITATIVE_REVIEW_REQUIRED:
            freeze_blocked = True

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
