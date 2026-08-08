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
from .strategy import choose_strategy
from .take_grouping import group_takes
from .take_judge_provider import TakeJudgeProvider, safe_rank_takes


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
) -> ProcessingResult:
    """Build an editable draft from already-transcribed/segmented takes."""
    take_tuple = tuple(takes)
    label_map: dict[str, SemanticLabel] = {label.clip_id: label for label in semantic_labels}

    # Tier 1 only removes very high-certainty recording errors.
    kept, deterministic_discarded, decisions = apply_clean_cut(take_tuple)

    # Tier 2 is optional and separate from commercial labels. It fails open.
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
                    str(child_id),
                    parent_label.role,
                    parent_label.confidence,
                    parent_label.reason,
                )

    grouped = group_takes(kept)
    groups = []
    clip_to_group: Dict[str, str] = {}
    judge_statuses = Counter()
    judge_reasons = Counter()
    alternate_group_count = 0
    judge_group_diagnostics = []

    for key, members in grouped.items():
        if len(members) >= 2:
            alternate_group_count += 1
        judged = safe_rank_takes(members, take_judge_provider)
        ranked = judged.ranked
        judge_statuses[judged.status.status] += 1
        if judged.status.reason:
            judge_reasons[judged.status.reason] += 1
        selected_clip_id = ranked[0].clip_id
        gid = _group_id(request.project_id, key)
        group = TakeGroup(
            group_id=gid,
            semantic_key=key,
            candidate_ids=tuple(member.clip_id for member in members),
            ranked=ranked,
            selected_clip_id=selected_clip_id,
        )
        groups.append(group)
        if len(members) >= 2:
            judge_group_diagnostics.append({
                "group_id": gid,
                "selected_clip_id": selected_clip_id,
                "execution_status": judged.status.status,
                "execution_reason": judged.status.reason,
                "ranked": [
                    {"clip_id": item.clip_id, "score": item.score, "reason": item.reason}
                    for item in ranked
                ],
            })
        for member in members:
            clip_to_group[member.clip_id] = gid

    surviving_labels = tuple(label_map[take.clip_id] for take in kept if take.clip_id in label_map)
    strategy = choose_strategy(surviving_labels, kept)

    # Baseline selection remains conservative: one winner per take group.
    natural_selected = compose_selected(kept, groups, surviving_labels)

    # Flexible composer may only reorder those already-selected real clips.
    # It has no deletion authority and cannot fabricate/duplicate speech.
    composition = safe_compose_order(
        composer_provider,
        natural_selected,
        surviving_labels,
        strategy,
    )
    selected_map = {take.clip_id: take for take in natural_selected}
    selected_takes = tuple(selected_map[clip_id] for clip_id in composition.ordered_clip_ids)
    selected_ids = {take.clip_id for take in selected_takes}

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
        if take.clip_id not in selected_ids
    )
    discarded_clips = tuple(
        _draft_clip(
            take,
            role=label_map.get(take.clip_id, SemanticLabel(take.clip_id, SemanticRole.OTHER, 0.0)).role,
            group_id=None,
            selected=False,
        )
        for take in discarded
    )

    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id=request.project_id,
        strategy=strategy,
        selected=selected,
        alternates=alternates,
        discarded=discarded_clips,
        diagnostics={
            "clean_cut_decisions": [decision.__dict__ for decision in decisions],
            "clean_cut_judge_status": clean_judged.status.__dict__,
            "clean_cut_judge": list(clean_judge_diagnostics)[:100],
            "clean_cut_judge_deleted_count": sum(1 for item in clean_judge_diagnostics if item.get("applied_delete")),
            "clean_cut_judge_mixed_count": sum(1 for item in clean_judge_diagnostics if item["action"] == "mixed"),
            "clean_cut_judge_mixed_trimmed_count": sum(1 for item in clean_judge_diagnostics if item.get("applied_mixed_trim")),
            "take_group_count": len(groups),
            "alternate_group_count": alternate_group_count,
            "source_count": len(request.sources),
            "take_judge_status_counts": dict(judge_statuses),
            "take_judge_fallback_reasons": dict(judge_reasons),
            "take_judge_groups": judge_group_diagnostics[:50],
            "composer_status": composition.status.__dict__,
            "composer_reason": composition.reason,
            "composer_order": list(composition.ordered_clip_ids),
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
        clean_cut_stage = "deterministic_complete"
    elif clean_judged.status.status == "applied":
        clean_cut_stage = "provider_complete"
    elif clean_judged.status.status == "provider_error":
        clean_cut_stage = "degraded_fail_open"
    else:
        clean_cut_stage = clean_judged.status.status

    if composer_provider is None or len(natural_selected) <= 1:
        composer_stage = "natural_order"
    elif composition.status.status == "applied":
        composer_stage = "provider_complete"
    else:
        composer_stage = "degraded_natural_order_fallback"

    return ProcessingResult(
        schema_version=SCHEMA_VERSION,
        project_id=request.project_id,
        state=JobState.DRAFT_READY,
        draft=draft,
        stage_status={
            "clean_cut": clean_cut_stage,
            "take_grouping": "complete",
            "take_judge": judge_stage,
            "semantic": "provided" if label_map else "not_provided",
            "composer": composer_stage,
        },
    )
