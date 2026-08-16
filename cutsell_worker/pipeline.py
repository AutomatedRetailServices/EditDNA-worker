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
from .draft_review_provider import DraftReviewProvider, safe_review_draft
from .hybrid_editorial import EditorialJudge
from .hybrid_group_cleanup import apply_hybrid_group_cleanup
from .session_boundaries import safe_group_takes_by_sessions
from .strategy import choose_strategy
from .take_grouping_provider import TakeGroupingProvider
from .take_judge_provider import TakeJudgeProvider, safe_rank_takes
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
    )


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
) -> ProcessingResult:
    """Build an editable draft after understanding the complete source context."""
    take_tuple = tuple(takes)
    temporal_trim_diagnostics = tuple(temporal_trim_diagnostics)
    label_map: dict[str, SemanticLabel] = {label.clip_id: label for label in semantic_labels}
    context_text = whole_video_context.compact_text() if whole_video_context is not None else ""

    # Pass 1 remains deterministic and context-aware. It removes obvious recording
    # garbage before any optional paid semantic reasoning is considered.
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

    # Retry grouping is already scoped by conservative creator/session walls. Semantic
    # reasoning is applied only *inside* those bounded groups and cannot invent or move
    # timestamps. This is the layer that can understand BTS/self-talk and failed human
    # attempts that deterministic rules should not try to enumerate forever.
    take_by_id = {take.clip_id: take for take in kept}
    grouping = safe_group_takes_by_sessions(
        take_grouping_provider,
        kept,
        whole_video_context,
        context_text=context_text,
    )
    group_members = [tuple(take_by_id[clip_id] for clip_id in ids) for ids in grouping.groups]

    groups = []
    clip_to_group: Dict[str, str] = {}
    judge_statuses = Counter()
    judge_reasons = Counter()
    alternate_group_count = 0
    judge_group_diagnostics = []
    hybrid_cleanup_diagnostics = []
    hybrid_deleted = []
    hybrid_requested_groups = 0
    hybrid_available_groups = 0

    for original_members in group_members:
        cleanup = apply_hybrid_group_cleanup(original_members, editorial_judge)
        if cleanup.requested:
            hybrid_requested_groups += 1
        if cleanup.available:
            hybrid_available_groups += 1
        hybrid_deleted.extend(cleanup.deleted)
        if cleanup.diagnostics:
            hybrid_cleanup_diagnostics.append({
                "source_asset_id": original_members[0].source_asset_id if original_members else None,
                "original_member_ids": [member.clip_id for member in original_members],
                "kept_ids": [member.clip_id for member in cleanup.kept],
                "deleted_ids": [member.clip_id for member in cleanup.deleted],
                "preferred_winner_id": cleanup.preferred_winner_id,
                "provider": cleanup.provider,
                "model": cleanup.model,
                "decisions": list(cleanup.diagnostics),
            })

        members = cleanup.kept
        if not members:
            continue
        if len(members) >= 2:
            alternate_group_count += 1

        judged = safe_rank_takes(members, take_judge_provider)
        ranked = judged.ranked
        # The semantic pass already paid for one classification of this bounded group.
        # If it named exactly one high-confidence winner, reuse that decision rather
        # than making a second LLM request through the Best Take provider.
        if cleanup.preferred_winner_id and ranked and ranked[0].clip_id != cleanup.preferred_winner_id:
            ranked = (
                next(item for item in ranked if item.clip_id == cleanup.preferred_winner_id),
                *(item for item in ranked if item.clip_id != cleanup.preferred_winner_id),
            )

        judge_statuses[judged.status.status] += 1
        if judged.status.reason:
            judge_reasons[judged.status.reason] += 1
        selected_clip_id = ranked[0].clip_id
        membership_key = "semantic:" + hashlib.sha256(
            "|".join(sorted(member.clip_id for member in members)).encode()
        ).hexdigest()[:16]
        gid = _group_id(request.project_id, membership_key)
        group = TakeGroup(
            group_id=gid,
            semantic_key=membership_key,
            candidate_ids=tuple(member.clip_id for member in members),
            ranked=ranked,
            selected_clip_id=selected_clip_id,
        )
        groups.append(group)
        if len(members) >= 2 or cleanup.requested:
            judge_group_diagnostics.append({
                "group_id": gid,
                "selected_clip_id": selected_clip_id,
                "execution_status": judged.status.status,
                "execution_reason": judged.status.reason,
                "hybrid_requested": cleanup.requested,
                "hybrid_provider": cleanup.provider,
                "hybrid_model": cleanup.model,
                "ranked": [
                    {"clip_id": item.clip_id, "score": item.score, "reason": item.reason}
                    for item in ranked
                ],
            })
        for member in members:
            clip_to_group[member.clip_id] = gid

    hybrid_deleted_ids = {take.clip_id for take in hybrid_deleted}
    kept = tuple(take for take in kept if take.clip_id not in hybrid_deleted_ids)
    discarded = (*discarded, *tuple(hybrid_deleted))

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
            "temporal_performance_trims": list(temporal_trim_diagnostics)[:300],
            "clean_cut_decisions": [decision.__dict__ for decision in decisions],
            "clean_cut_judge_status": clean_judged.status.__dict__,
            "clean_cut_judge": list(clean_judge_diagnostics)[:100],
            "clean_cut_judge_deleted_count": sum(1 for item in clean_judge_diagnostics if item.get("applied_delete")),
            "clean_cut_judge_mixed_count": sum(1 for item in clean_judge_diagnostics if item["action"] == "mixed"),
            "clean_cut_judge_mixed_trimmed_count": sum(1 for item in clean_judge_diagnostics if item.get("applied_mixed_trim")),
            "hybrid_editorial_requested_group_count": hybrid_requested_groups,
            "hybrid_editorial_available_group_count": hybrid_available_groups,
            "hybrid_editorial_deleted_count": len(hybrid_deleted),
            "hybrid_editorial_groups": hybrid_cleanup_diagnostics[:100],
            "take_group_count": len(groups),
            "alternate_group_count": alternate_group_count,
            "take_grouping_status": grouping.status.__dict__,
            "take_grouping_reason": grouping.reason,
            "take_group_members": [list(group) for group in grouping.groups][:100],
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
    elif hybrid_requested_groups and hybrid_available_groups:
        hybrid_stage = "provider_complete"
    elif hybrid_requested_groups:
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

    return ProcessingResult(
        schema_version=SCHEMA_VERSION,
        project_id=request.project_id,
        state=JobState.DRAFT_READY,
        draft=draft,
        stage_status={
            "whole_video_context": whole_video_context.status.status if whole_video_context is not None else "not_requested",
            "edit_mode": whole_video_context.dominant_edit_mode if whole_video_context is not None else "natural",
            "temporal_performance": "applied" if temporal_trim_diagnostics else "no_edge_trim_needed",
            "clean_cut": clean_cut_stage,
            "hybrid_editorial": hybrid_stage,
            "take_grouping": grouping_stage,
            "take_judge": judge_stage,
            "semantic": "provided" if label_map else "not_provided",
            "composer": composer_stage,
            "draft_review": review_stage,
        },
    )
