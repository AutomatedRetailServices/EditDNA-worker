"""Clean orchestration for CutSell Flow B Milestone 1."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace as dataclass_replace
import hashlib
from typing import Dict, Iterable

from .canonical_identity import (
    build_identity_chain_diagnostics,
    mint_realization_id,
    mint_retry_family_id,
    mint_semantic_idea_id,
)
from .claim_coverage_best_take import critical_coverage_sets, resolve_critical_coverage_dominance
from .clean_cut import apply_clean_cut
from .clean_cut_provider import CleanCutProvider, apply_provider_judgements, safe_clean_cut_judge
from .contradiction_signal import any_pair_contradicts
from .composer import compose_selected
from .composer_provider import ComposerProvider, safe_compose_order
from .contracts import (
    CandidateTake,
    DraftClip,
    DraftTimeline,
    JobState,
    ProcessingRequest,
    ProcessingResult,
    SCHEMA_VERSION,
    SemanticLabel,
    SemanticRole,
    TakeGroup,
)
from .composite_resolver import (
    apply_composite_family_stabilization,
    apply_composite_group_split,
    apply_composite_resolution,
)
from .draft_review_provider import DraftReviewProvider, safe_review_draft
from .hybrid_editorial import EditorialJudge
from .semantic_compute_planner import build_cost_contract_report
from .semantic_idea_equivalence import SemanticEquivalenceArbiter
from .session_boundaries import safe_group_takes_by_sessions
from .strategy import choose_strategy
from .take_grouping_provider import (
    TakeGroupingProvider,
    reconcile_semantic_idea_equivalence,
    split_incohesive_retry_groups,
)
from .take_judge_provider import TakeJudgeProvider, safe_rank_takes
from .temporal_editing import refine_takes_with_temporal_context
from .whole_video_analysis import WholeVideoContext


def _group_id(project_id: str, key: str) -> str:
    return "tg_" + hashlib.sha256(f"{project_id}|{key}".encode()).hexdigest()[:18]


def _draft_clip(take: CandidateTake, *, role: SemanticRole, group_id: str | None, selected: bool) -> DraftClip:
    # D-050A: `group_id` here is already the FINAL, post-semantic-
    # equivalence take-group id (pipeline.py is its one minting owner --
    # see canonical_identity.py's ID OWNERSHIP table). semantic_idea_id
    # and retry_family_id are minted from it, additively; take_group_id
    # itself is left completely unchanged for every existing consumer.
    semantic_idea_id = mint_semantic_idea_id(group_id) if group_id else None
    retry_family_id = mint_retry_family_id(group_id) if group_id else None
    return DraftClip(
        clip_id=take.clip_id,
        source_asset_id=take.source_asset_id,
        source_order=take.source_order,
        start=take.start,
        end=take.end,
        text=take.text,
        caption_text=take.text,
        words=take.words,
        semantic_role=role,
        take_group_id=group_id,
        selected=selected,
        # Carry the take's local face/pose/motion evidence through to the
        # draft so downstream Selection authorities (Hybrid or Unified) can
        # actually see it instead of it being silently dropped at this
        # conversion. See local_performance.py / MediaSignals.
        signals=take.signals,
        # D-050A: carried unchanged from the CandidateTake this clip was
        # built from -- never recomputed here.
        realization_id=take.realization_id,
        semantic_idea_id=semantic_idea_id,
        retry_family_id=retry_family_id,
        # D-050C1.6: carried unchanged from the CandidateTake -- see
        # DraftClip.complete_idea's own docstring.
        complete_idea=take.complete_idea,
        # D-076: carried unchanged from the CandidateTake -- see
        # DraftClip.source_span_id/.attempt_id's own docstring.
        source_span_id=take.source_span_id,
        attempt_id=take.attempt_id,
    )


def _semantic_best_take(
    members: tuple[CandidateTake, ...],
    semantic_decisions: dict[str, tuple[str, float]],
    local_selected_clip_id: str,
    ranked: tuple = (),
    *,
    winner_confidence: float = 0.85,
    semantic_delete_recommended: dict[str, bool] | None = None,
) -> tuple[str, str | None, str]:
    """Honor one clear semantic winner only inside an already-proven retry group.

    Hybrid session cleanup sees the full message and may recognize which delivery is the
    intended final take. The local Watch+Listen ranker still establishes the fallback,
    but a unique medium-high semantic winner may override a tiny local score difference.
    This can never create a group: it only chooses among members the deterministic
    grouping stage already proved to be competing retries.

    D-082: when semantic labels are NOT decisive (zero or 2+ "winner" labels
    -- the exact D-080 sonography shape, where labels degraded from a
    decisive {"failed", "winner"} to a non-actionable {"keep", "keep"}
    across two live runs on byte-identical DeliveryScorer scores), this no
    longer falls straight to `local_selected_clip_id` (the raw,
    completeness-blind DeliveryScorer rank -- D-080's proven root cause).
    It instead consults, in order, deterministic evidence this codebase
    already computes elsewhere -- no new authority, no weighted scoring:

      1. D-081 `semantic_delete_recommended` evidence (soft: a candidate
         carrying it is excluded from consideration unless that would
         eliminate every candidate -- same "WHEN UNCERTAIN, KEEP" fail-open
         rule as every other check below, and never a second irreversible
         delete authority -- D-081/D-082 Section 12);
      2. attempt completeness (`CandidateTake.complete_idea`, the exact
         signal `take_judge.score_take` already weights for delivery
         scoring -- soft, same fail-open rule: only an EXPLICIT False
         excludes, never an unset/unknown value);
      3. D-063/D-065/D-066 CRITICAL_COVERAGE_DOMINANCE, reused verbatim via
         `claim_coverage_best_take.resolve_critical_coverage_dominance` --
         never reimplemented, including its own safety gates (never prefers
         a proven-incomplete candidate, never overrides a real
         contradiction);
      4. only once dominance finds no single winner AND the survivors'
         CRITICAL-claim coverage sets are genuinely IDENTICAL (a true tie)
         does the local DeliveryScorer ranking get to decide among them --
         its proper role once content is effectively tied (D-082 Section
         8/10). A genuinely disjoint/asymmetric coverage split (distinct
         unique facts, neither a superset of the other) is left exactly as
         it was -- `local_selected_clip_id`, this function's own pre-D-082
         safe default -- rather than forcing delivery to pick a side
         (D-082 Section 7: "remain unresolved / preserve safe behavior").

    Any step that finds nothing decisive falls open to the next one; the
    final fallback is always `local_selected_clip_id`, never worse than
    today's behavior for a genuinely unresolved family.
    """
    winners = []
    for member in members:
        label, confidence = semantic_decisions.get(member.clip_id, ("", 0.0))
        if label == "winner" and confidence >= winner_confidence:
            winners.append((member.clip_id, confidence))
    if len(winners) == 1:
        preferred_id, _ = winners[0]
        if preferred_id == local_selected_clip_id:
            return local_selected_clip_id, preferred_id, "single_semantic_winner"
        return preferred_id, preferred_id, "single_semantic_winner"

    member_ids = [member.clip_id for member in members]
    by_id = {member.clip_id: member for member in members}
    if len(member_ids) < 2:
        return local_selected_clip_id, None, "single_member_no_contest"

    def _exclude_unless_all(ids: list[str], excluded: set[str]) -> list[str]:
        survivors = [cid for cid in ids if cid not in excluded]
        return survivors if survivors else list(ids)

    # Step 1: D-081 semantic-delete-recommended evidence.
    delete_recommended_ids = {
        cid for cid in member_ids if (semantic_delete_recommended or {}).get(cid, False)
    }
    survivors = _exclude_unless_all(member_ids, delete_recommended_ids)

    # Step 2: attempt completeness.
    incomplete_ids = {cid for cid in survivors if by_id[cid].complete_idea is False}
    survivors = _exclude_unless_all(survivors, incomplete_ids)

    if len(survivors) >= 2:
        members_pairs = [(cid, by_id[cid]) for cid in member_ids]

        # Step 3/4: D-063/D-065/D-066 CRITICAL_COVERAGE_DOMINANCE, reused.
        dominant_id, _hindsight_rows = resolve_critical_coverage_dominance(members_pairs, survivors)
        if dominant_id is not None:
            if dominant_id == local_selected_clip_id:
                return local_selected_clip_id, dominant_id, "critical_coverage_dominance"
            return dominant_id, dominant_id, "critical_coverage_dominance"

        # Step 5: unique-required-fact safety -- delivery may only settle a
        # GENUINE tie, never an asymmetric/disjoint split. A defensive
        # second check reuses `any_pair_contradicts` (the SAME safety gate
        # `_critical_coverage_dominant_candidate` itself already applies
        # internally) directly on the surviving texts: two candidates that
        # factually contradict each other must never be handed to delivery
        # merely because their CRITICAL-claim coverage sets happened to
        # look identical (e.g. a negated and non-negated claim minted under
        # colliding canonical ids).
        coverage = critical_coverage_sets(members_pairs, survivors)
        if coverage:
            coverage_values = list(coverage.values())
            if any(value != coverage_values[0] for value in coverage_values[1:]):
                return local_selected_clip_id, None, "unresolved_unique_fact_asymmetry"
        if any_pair_contradicts([str(by_id[cid].text or "") for cid in survivors]):
            return local_selected_clip_id, None, "unresolved_contradiction"

    # Steps 6-9: delivery score / richness tie-break among the surviving,
    # safe candidate set -- delivery's proper role once content is
    # effectively tied.
    rank_by_id = {row.clip_id: row.score for row in ranked}
    survivor_ranked = [cid for cid in survivors if cid in rank_by_id]
    if survivor_ranked:
        best = max(survivor_ranked, key=lambda cid: rank_by_id[cid])
        if best == local_selected_clip_id:
            return local_selected_clip_id, None, "delivery_tie_break_among_survivors"
        return best, best, "delivery_tie_break_among_survivors"

    return local_selected_clip_id, None, "local_fallback"


def build_flow_b_draft(
    request: ProcessingRequest,
    takes: Iterable[CandidateTake],
    semantic_labels: Iterable[SemanticLabel] = (),
    take_judge_provider: TakeJudgeProvider | None = None,
    clean_cut_provider: CleanCutProvider | None = None,
    composer_provider: ComposerProvider | None = None,
    take_grouping_provider: TakeGroupingProvider | None = None,
    draft_review_provider: DraftReviewProvider | None = None,
    editorial_judge: EditorialJudge | None = None,
    whole_video_context: WholeVideoContext | None = None,
    temporal_trim_diagnostics: Iterable[dict] = (),
    attempt_reconstruction_diagnostics: dict | None = None,
    performance_confirmation_diagnostics: Iterable[dict] = (),
    semantic_equivalence_arbiter: SemanticEquivalenceArbiter | None = None,
) -> ProcessingResult:
    """Build an editable draft after understanding the complete source context."""
    take_tuple = tuple(takes)
    temporal_trim_diagnostics = tuple(temporal_trim_diagnostics)
    performance_confirmation_diagnostics = tuple(performance_confirmation_diagnostics)
    attempt_reconstruction_diagnostics = dict(attempt_reconstruction_diagnostics or {})
    label_map: dict[str, SemanticLabel] = {label.clip_id: label for label in semantic_labels}
    context_text = whole_video_context.compact_text() if whole_video_context is not None else ""

    # D-050D1: mint `realization_id` here -- on the COMPLETE candidate pool
    # (AttemptReconstructor output plus any `preserved_subspan_candidates`,
    # already merged into `takes` by flow_b.py before this function is
    # ever called), before ANY editorial stage (clean_cut, provider
    # judgements, hybrid/composite resolution) can keep, discard, or
    # transform a candidate. This used to run only over the survivors of
    # all three of those stages (see D-050C3/D-050D's own audit), which
    # meant anything they removed never received a canonical identity at
    # all -- the exact mechanism behind every orphan realization traced in
    # that audit. `apply_clean_cut`/`apply_provider_judgements` are pure
    # partitions of the SAME CandidateTake objects (never rebuild them),
    # and `apply_composite_resolution`'s own kept/deleted split is the
    # same shape -- so minting here, once, is carried forward unchanged
    # into every branch, kept or discarded, by construction. Still the
    # single canonical owner (`mint_realization_id`, canonical_identity.py)
    # -- no second minting implementation anywhere else in this function.
    take_tuple = tuple(
        take if take.realization_id else dataclass_replace(
            take, realization_id=mint_realization_id(take.source_asset_id, take.attempt_id, take.text),
        )
        for take in take_tuple
    )

    # Pass 1: deterministic/local cleanup remains the backbone and removes obvious
    # recording garbage before optional semantic reasoning spends anything.
    kept, deterministic_discarded, decisions = apply_clean_cut(take_tuple, whole_video_context)
    clean_judged = safe_clean_cut_judge(clean_cut_provider, kept)
    kept, provider_discarded, clean_judge_diagnostics = apply_provider_judgements(kept, clean_judged)
    discarded = tuple(deterministic_discarded) + tuple(provider_discarded)

    for item in clean_judge_diagnostics:
        if not item.get("applied_mixed_trim"):
            continue
        parent_id = str(item.get("clip_id") or "")
        parent_label = label_map.get(parent_id)
        if parent_label is None:
            continue
        child_ids = [item.get("kept_clip_id"), *(item.get("discarded_clip_ids") or [])]
        for child_id in child_ids:
            if child_id:
                label_map[str(child_id)] = SemanticLabel(
                    str(child_id), parent_label.role, parent_label.confidence, parent_label.reason
                )

    # Pass 2: batch semantic intent by bounded creator mini-session. This catches BTS,
    # self-review and failed attempts with context while avoiding one paid call for every
    # singleton/retry group. Semantic winner/alternate evidence is retained for Pass 3.
    #
    # CompositeResolver (composite_resolver.py, see D-023): the single, directly-
    # callable authority for delivery restoration/rescue/composite marking. Owns
    # what used to be 14 separately-monkeypatched hybrid_* authorities layered
    # onto this one call, in the same order, same algorithms -- now one explicit
    # composition instead of an implicit import-time chain.
    hybrid_cleanup, composite_split_ids = apply_composite_resolution(
        kept,
        whole_video_context,
        editorial_judge,
    )
    kept = hybrid_cleanup.kept
    discarded = (*discarded, *hybrid_cleanup.deleted)
    hybrid_semantic_decisions = {
        clip_id: (label, float(confidence))
        for clip_id, label, confidence in hybrid_cleanup.semantic_decisions
    }
    # D-082 Section 12: surface D-081's semantic_delete_recommended evidence
    # (recorded per-window inside hybrid_cleanup.diagnostics, never
    # destructive on its own) so _semantic_best_take's non-decisive-label
    # fallback can treat it as soft negative evidence. OR-across windows:
    # if any window flagged the candidate, that evidence is never silently
    # dropped, matching D-081's own "never discard the evidence" posture.
    hybrid_semantic_delete_recommended: dict[str, bool] = {}
    for diagnostic in hybrid_cleanup.diagnostics:
        for decision in diagnostic.get("decisions") or ():
            clip_id = decision.get("clip_id")
            if not clip_id:
                continue
            if decision.get("semantic_delete_recommended"):
                hybrid_semantic_delete_recommended[clip_id] = True
            else:
                hybrid_semantic_delete_recommended.setdefault(clip_id, False)

    # D-050D1: `realization_id` is minted once, above, before Pass 1 even
    # starts -- every member of `kept` here already carries it (see the
    # single minting pass at the top of this function). No second minting
    # pass; no reminting.

    # Pass 3: deterministic retry grouping + Best Take runs after semantic garbage is
    # removed. The local ranker remains the fallback. If Hybrid already identified one
    # clear winner among members of the same proven retry group, that editorial winner
    # takes precedence over a marginal local score difference.
    take_by_id = {take.clip_id: take for take in kept}
    grouping = safe_group_takes_by_sessions(
        take_grouping_provider,
        kept,
        whole_video_context,
        context_text=context_text,
    )
    # CompositeResolver's composite-marked pairs (see above) are forced into
    # singleton groups here so BestTakeResolver's one-winner competition
    # cannot re-collapse an intended composite delivery. Direct call, no
    # ContextVar, no monkeypatch of safe_group_takes_by_sessions.
    grouping = apply_composite_group_split(grouping, kept, composite_split_ids)

    # Phase 2 of the architecture rebalance: a narrow, gated semantic-
    # equivalence arbiter may confirm that two groups the lexical layer left
    # separate are recording attempts of the same intended idea, merging
    # them into one retry contest BEFORE the completeness/performance
    # ranking (safe_rank_takes) and deterministic Best Take run below. This
    # runs here, directly on safe_group_takes_by_sessions's resolved output,
    # rather than being threaded as a parameter through that call -- see
    # take_grouping_provider.safe_group_takes's docstring for why: this
    # function is already wrapped by several production monkeypatch layers
    # that hardcode its current signature, and this is the one choke point
    # every one of those layers' output must pass through regardless.
    # D-025: composite_split_ids are protected here too, not just at the
    # grouping-split step above -- otherwise this call's own, separate
    # arbiter invocation can re-merge an accepted composite's pieces (or
    # merge one into an unrelated group), silently discarding a decision
    # CompositeResolver already made. See reconcile_semantic_idea_
    # equivalence's own docstring for the exact RAW that exposed this.
    semantic_equivalence_groups, semantic_equivalence_diagnostics = reconcile_semantic_idea_equivalence(
        grouping.groups, kept, semantic_equivalence_arbiter,
        protected_ids=composite_split_ids,
    )

    # D-058 Phase 1: one final cohesion-validation pass -- see
    # take_grouping_provider.split_incohesive_retry_groups's own module
    # comment for the full defect/fix rationale (docs/CUTSELL_DECISIONS.md
    # D-057/D-058). Runs after every merging step above, on whatever groups
    # they produced, and before Best Take ranking below ever treats a group
    # as one mutually-exclusive contest. Same `protected_ids` contract as
    # the arbiter merge immediately above -- an accepted composite's pieces
    # are never re-examined here either.
    semantic_equivalence_groups, cohesion_diagnostics = split_incohesive_retry_groups(
        semantic_equivalence_groups, kept, semantic_equivalence_arbiter,
        protected_ids=composite_split_ids,
    )
    group_members = [tuple(take_by_id[clip_id] for clip_id in ids) for ids in semantic_equivalence_groups]

    groups = []
    clip_to_group: Dict[str, str] = {}
    judge_statuses = Counter()
    judge_reasons = Counter()
    alternate_group_count = 0
    semantic_best_take_override_count = 0
    judge_group_diagnostics = []

    for members in group_members:
        if not members:
            continue
        if len(members) >= 2:
            alternate_group_count += 1
        judged = safe_rank_takes(members, take_judge_provider)
        ranked = judged.ranked
        judge_statuses[judged.status.status] += 1
        if judged.status.reason:
            judge_reasons[judged.status.reason] += 1
        local_selected_clip_id = ranked[0].clip_id
        selected_clip_id, semantic_preferred_clip_id, semantic_best_take_reason = _semantic_best_take(
            members,
            hybrid_semantic_decisions,
            local_selected_clip_id,
            ranked,
            semantic_delete_recommended=hybrid_semantic_delete_recommended,
        )
        if semantic_preferred_clip_id and selected_clip_id != local_selected_clip_id:
            semantic_best_take_override_count += 1
        membership_key = "semantic:" + hashlib.sha256(
            "|".join(sorted(member.clip_id for member in members)).encode()
        ).hexdigest()[:16]
        gid = _group_id(request.project_id, membership_key)
        groups.append(TakeGroup(
            group_id=gid,
            semantic_key=membership_key,
            candidate_ids=tuple(member.clip_id for member in members),
            ranked=ranked,
            selected_clip_id=selected_clip_id,
        ))
        if len(members) >= 2:
            judge_group_diagnostics.append({
                "group_id": gid,
                "selected_clip_id": selected_clip_id,
                "local_selected_clip_id": local_selected_clip_id,
                "semantic_preferred_clip_id": semantic_preferred_clip_id,
                "semantic_override_applied": selected_clip_id != local_selected_clip_id,
                "semantic_best_take_reason": semantic_best_take_reason,
                "semantic_candidates": [
                    {
                        "clip_id": member.clip_id,
                        "label": hybrid_semantic_decisions.get(member.clip_id, ("", 0.0))[0],
                        "confidence": hybrid_semantic_decisions.get(member.clip_id, ("", 0.0))[1],
                    }
                    for member in members
                ],
                "execution_status": judged.status.status,
                "execution_reason": judged.status.reason,
                "ranked": [
                    {"clip_id": item.clip_id, "score": item.score, "reason": item.reason}
                    for item in ranked
                ],
            })
        for member in members:
            clip_to_group[member.clip_id] = gid

    # Pass 4: only after retry families have been judged and a logical winner has been
    # chosen do we touch physical edit boundaries. Keep the logical clip IDs stable so
    # semantic labels, retry-group membership and the already-made Best Take decision
    # cannot be invalidated by a later timestamp adjustment.
    kept, post_best_take_trim_diagnostics = refine_takes_with_temporal_context(
        kept,
        whole_video_context,
        preserve_clip_id=True,
    )
    temporal_trim_diagnostics = (
        *temporal_trim_diagnostics,
        *post_best_take_trim_diagnostics,
    )

    surviving_labels = tuple(label_map[take.clip_id] for take in kept if take.clip_id in label_map)
    strategy = choose_strategy(surviving_labels, kept)
    natural_selected = compose_selected(kept, groups, surviving_labels)

    composition = safe_compose_order(
        composer_provider,
        natural_selected,
        surviving_labels,
        strategy,
        context_text=context_text,
    )
    selected_map = {take.clip_id: take for take in natural_selected}
    composed_takes = tuple(selected_map[clip_id] for clip_id in composition.ordered_clip_ids)

    review = safe_review_draft(
        draft_review_provider,
        composed_takes,
        surviving_labels,
        strategy,
        context_text=context_text,
    )
    composed_map = {take.clip_id: take for take in composed_takes}
    selected_takes = tuple(composed_map[clip_id] for clip_id in review.ordered_clip_ids)
    selected_ids = {take.clip_id for take in selected_takes}

    initially_removed_ids = set(composed_map) - selected_ids
    removed_group_ids = {
        clip_to_group[clip_id]
        for clip_id in initially_removed_ids
        if clip_id in clip_to_group
    }
    review_removed_ids = {
        take.clip_id
        for take in kept
        if take.clip_id in initially_removed_ids or clip_to_group.get(take.clip_id) in removed_group_ids
    }
    review_removed = tuple(take for take in kept if take.clip_id in review_removed_ids)

    selected = tuple(
        _draft_clip(
            take,
            role=label_map.get(take.clip_id, SemanticLabel(take.clip_id, SemanticRole.OTHER, 0.0)).role,
            group_id=clip_to_group.get(take.clip_id),
            selected=True,
        )
        for take in selected_takes
    )
    alternates = tuple(
        _draft_clip(
            take,
            role=label_map.get(take.clip_id, SemanticLabel(take.clip_id, SemanticRole.OTHER, 0.0)).role,
            group_id=clip_to_group.get(take.clip_id),
            selected=False,
        )
        for take in kept
        if take.clip_id not in selected_ids and take.clip_id not in review_removed_ids
    )
    discarded_clips = tuple(
        _draft_clip(
            take,
            role=label_map.get(take.clip_id, SemanticLabel(take.clip_id, SemanticRole.OTHER, 0.0)).role,
            # D-050C3 Section 5: this used to hardcode group_id=None
            # unconditionally, unlike the selected/alternates buckets above
            # which correctly consult clip_to_group. `discarded` (pre-
            # grouping clean_cut/hybrid-cleanup rejects) never had a group
            # to begin with, so clip_to_group.get() is a no-op there -- but
            # `review_removed` (post-grouping draft_review rejects, see
            # removed_group_ids above) DID go through grouping and have a
            # real group_id, which the hardcoded None silently discarded,
            # stripping semantic_idea_id/retry_family_id (see _draft_clip)
            # from every such realization regardless of whether grouping
            # found a real retry family for it. Fixed to the same lookup
            # selected/alternates already use -- no clip-id hardcoding,
            # general to any discard path that reaches this constructor.
            group_id=clip_to_group.get(take.clip_id),
            selected=False,
        )
        for take in (*discarded, *review_removed)
    )

    whole_video_diag = {
        "status": whole_video_context.status.__dict__ if whole_video_context is not None else None,
        "dominant_edit_mode": whole_video_context.dominant_edit_mode if whole_video_context is not None else "natural",
        "sources": [
            {
                "source_asset_id": source.source_asset_id,
                "summary": source.summary,
                "dominant_style": source.dominant_style,
                "creator_intent": source.creator_intent,
                "edit_mode": source.edit_mode,
                "sales_intent": source.sales_intent,
                "main_topic": source.main_topic,
                "product_or_subject": source.product_or_subject,
                "story_logic": source.story_logic,
                "events": [event.__dict__ for event in source.events],
            }
            for source in (whole_video_context.sources if whole_video_context is not None else ())
        ],
    }

    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id=request.project_id,
        strategy=strategy,
        selected=selected,
        alternates=alternates,
        discarded=discarded_clips,
        diagnostics={
            "whole_video_context": whole_video_diag,
            "attempt_reconstruction": attempt_reconstruction_diagnostics,
            "performance_confirmation": list(performance_confirmation_diagnostics)[:300],
            "temporal_performance_trims": list(temporal_trim_diagnostics)[:300],
            "clean_cut_decisions": [decision.__dict__ for decision in decisions],
            "clean_cut_judge_status": clean_judged.status.__dict__,
            "clean_cut_judge": list(clean_judge_diagnostics)[:100],
            "clean_cut_judge_deleted_count": sum(1 for item in clean_judge_diagnostics if item.get("applied_delete")),
            "clean_cut_judge_mixed_count": sum(1 for item in clean_judge_diagnostics if item["action"] == "mixed"),
            "clean_cut_judge_mixed_trimmed_count": sum(1 for item in clean_judge_diagnostics if item.get("applied_mixed_trim")),
            "hybrid_editorial_requested_chunk_count": hybrid_cleanup.requested_chunk_count,
            "hybrid_editorial_available_chunk_count": hybrid_cleanup.available_chunk_count,
            "hybrid_editorial_deleted_count": len(hybrid_cleanup.deleted),
            "hybrid_editorial_semantic_decision_count": len(hybrid_cleanup.semantic_decisions),
            "hybrid_editorial_chunks": list(hybrid_cleanup.diagnostics)[:100],
            # D-052 Part B: present only when CUTSELL_SEMANTIC_COMPUTE_PLANNER
            # was enabled for this run -- None otherwise (today's default).
            "semantic_compute_plan": (
                build_cost_contract_report(hybrid_cleanup.semantic_compute_plan)
                if hybrid_cleanup.semantic_compute_plan is not None else None
            ),
            "hybrid_semantic_best_take_override_count": semantic_best_take_override_count,
            "take_group_count": len(groups),
            "alternate_group_count": alternate_group_count,
            "take_grouping_status": grouping.status.__dict__,
            "take_grouping_reason": grouping.reason,
            "take_group_members": [list(group) for group in semantic_equivalence_groups][:100],
            "semantic_idea_equivalence": semantic_equivalence_diagnostics,
            "distinct_idea_grouping_safety": cohesion_diagnostics,
            "source_count": len(request.sources),
            "take_judge_status_counts": dict(judge_statuses),
            "take_judge_fallback_reasons": dict(judge_reasons),
            "take_judge_groups": judge_group_diagnostics[:50],
            "composer_status": composition.status.__dict__,
            "composer_reason": composition.reason,
            "composer_order": list(composition.ordered_clip_ids),
            "draft_review_status": review.status.__dict__,
            "draft_review_postable": review.postable,
            "draft_review_issues": list(review.issues),
            "draft_review_reason": review.reason,
            "draft_review_order": list(review.ordered_clip_ids),
            "draft_review_removed_ids": [take.clip_id for take in review_removed],
            "draft_review_removed_group_ids": sorted(removed_group_ids),
        },
    )

    # CompositeResolver step 16 (composite_resolver.py, D-023): the one
    # genuinely downstream extension, operating on the built draft rather
    # than raw takes -- repairs a concise discarded delivery + later winner
    # that jointly cover a redundant selected monolith better than the
    # monolith alone. Called explicitly here instead of via a monkeypatch
    # on this function.
    draft = apply_composite_family_stabilization(draft)

    # D-050A observability (Section 7): the full identity chain for every
    # selected clip, in one place, computed AFTER the last stage in this
    # function that can change `selected` membership -- read-only, never
    # fed back into any decision here or downstream.
    draft = dataclass_replace(
        draft,
        diagnostics={
            **(draft.diagnostics or {}),
            "canonical_identity_chain": build_identity_chain_diagnostics(draft),
        },
    )

    if alternate_group_count == 0:
        judge_stage = "not_applicable_no_alternates"
    elif judge_statuses.get("applied"):
        judge_stage = "provider_complete"
    elif judge_statuses.get("provider_error_fallback"):
        judge_stage = "degraded_fallback"
    else:
        judge_stage = "baseline_complete"

    if clean_cut_provider is None:
        clean_cut_stage = "context_aware_deterministic_complete"
    elif clean_judged.status.status == "applied":
        clean_cut_stage = "provider_complete"
    elif clean_judged.status.status == "provider_error":
        clean_cut_stage = "degraded_fail_open"
    else:
        clean_cut_stage = clean_judged.status.status

    if editorial_judge is None:
        hybrid_stage = "disabled_local_only"
    elif hybrid_cleanup.requested_chunk_count and hybrid_cleanup.available_chunk_count:
        hybrid_stage = "provider_complete"
    elif hybrid_cleanup.requested_chunk_count:
        hybrid_stage = "degraded_fail_open"
    else:
        hybrid_stage = "confidence_gate_local"

    if composer_provider is None or len(natural_selected) <= 1:
        composer_stage = "natural_order"
    elif composition.status.status == "applied":
        composer_stage = "provider_complete"
    else:
        composer_stage = "degraded_natural_order_fallback"

    if take_grouping_provider is None or len(kept) <= 1:
        grouping_stage = "baseline_complete"
    elif grouping.status.status == "applied":
        grouping_stage = "provider_complete"
    else:
        grouping_stage = "degraded_baseline_fallback"

    if draft_review_provider is None or len(composed_takes) <= 1:
        review_stage = "not_requested"
    elif review.status.status == "applied":
        review_stage = "postable" if review.postable else "needs_attention"
    else:
        review_stage = "degraded_fallback"

    temporal_applied_count = sum(
        1 for item in temporal_trim_diagnostics if item.get("applied")
    )

    return ProcessingResult(
        schema_version=SCHEMA_VERSION,
        project_id=request.project_id,
        state=JobState.DRAFT_READY,
        draft=draft,
        stage_status={
            "whole_video_context": whole_video_context.status.status if whole_video_context is not None else "not_requested",
            "edit_mode": whole_video_context.dominant_edit_mode if whole_video_context is not None else "natural",
            "attempt_reconstruction": "applied" if attempt_reconstruction_diagnostics else "not_requested",
            "temporal_performance": "applied" if temporal_applied_count else "no_edge_trim_needed",
            "clean_cut": clean_cut_stage,
            "hybrid_editorial": hybrid_stage,
            "take_grouping": grouping_stage,
            "take_judge": judge_stage,
            "semantic": "provided" if label_map else "not_provided",
            "composer": composer_stage,
            "draft_review": review_stage,
        },
    )