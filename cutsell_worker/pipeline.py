"""Clean orchestration for CutSell Flow B Milestone 1."""
from __future__ import annotations

from collections import Counter
import hashlib
from typing import Dict, Iterable

from .clean_cut import apply_clean_cut
from .clean_cut_provider import CleanCutProvider, apply_provider_judgements, safe_clean_cut_judge
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
from .semantic_idea_equivalence import SemanticEquivalenceArbiter
from .session_boundaries import safe_group_takes_by_sessions
from .strategy import choose_strategy
from .take_grouping_provider import TakeGroupingProvider, reconcile_semantic_idea_equivalence
from .take_judge_provider import TakeJudgeProvider, safe_rank_takes
from .temporal_editing import refine_takes_with_temporal_context
from .whole_video_analysis import WholeVideoContext


def _group_id(project_id: str, key: str) -> str:
    return "tg_" + hashlib.sha256(f"{project_id}|{key}".encode()).hexdigest()[:18]


def _draft_clip(take: CandidateTake, *, role: SemanticRole, group_id: str | None, selected: bool) -> DraftClip:
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
    )


def _semantic_best_take(
    members: tuple[CandidateTake, ...],
    semantic_decisions: dict[str, tuple[str, float]],
    local_selected_clip_id: str,
    *,
    winner_confidence: float = 0.85,
) -> tuple[str, str | None]:
    """Honor one clear semantic winner only inside an already-proven retry group.

    Hybrid session cleanup sees the full message and may recognize which delivery is the
    intended final take. The local Watch+Listen ranker still establishes the fallback,
    but a unique medium-high semantic winner may override a tiny local score difference.
    This can never create a group: it only chooses among members the deterministic
    grouping stage already proved to be competing retries.
    """
    winners = []
    for member in members:
        label, confidence = semantic_decisions.get(member.clip_id, ("", 0.0))
        if label == "winner" and confidence >= winner_confidence:
            winners.append((member.clip_id, confidence))
    if len(winners) != 1:
        return local_selected_clip_id, None
    preferred_id, _ = winners[0]
    if preferred_id == local_selected_clip_id:
        return local_selected_clip_id, preferred_id
    return preferred_id, preferred_id


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
        selected_clip_id, semantic_preferred_clip_id = _semantic_best_take(
            members,
            hybrid_semantic_decisions,
            local_selected_clip_id,
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
            group_id=None,
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
            "hybrid_semantic_best_take_override_count": semantic_best_take_override_count,
            "take_group_count": len(groups),
            "alternate_group_count": alternate_group_count,
            "take_grouping_status": grouping.status.__dict__,
            "take_grouping_reason": grouping.reason,
            "take_group_members": [list(group) for group in semantic_equivalence_groups][:100],
            "semantic_idea_equivalence": semantic_equivalence_diagnostics,
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