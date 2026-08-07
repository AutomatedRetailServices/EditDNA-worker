"""Clean orchestration for CutSell Flow B Milestone 1."""
from __future__ import annotations

from collections import Counter
import hashlib
from typing import Dict, Iterable, Mapping

from .clean_cut import apply_clean_cut
from .composer import compose_selected
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
        semantic_role=role,
        take_group_id=group_id,
        selected=selected,
    )


def build_flow_b_draft(
    request: ProcessingRequest,
    takes: Iterable[CandidateTake],
    semantic_labels: Iterable[SemanticLabel] = (),
    take_judge_provider: TakeJudgeProvider | None = None,
) -> ProcessingResult:
    """Build an editable draft from already-transcribed/segmented takes."""
    take_tuple = tuple(takes)
    label_map: Mapping[str, SemanticLabel] = {label.clip_id: label for label in semantic_labels}

    kept, discarded, decisions = apply_clean_cut(take_tuple)
    grouped = group_takes(kept)
    groups = []
    clip_to_group: Dict[str, str] = {}
    judge_statuses = Counter()
    judge_reasons = Counter()
    alternate_group_count = 0

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
        for member in members:
            clip_to_group[member.clip_id] = gid

    selected_takes = compose_selected(kept, groups, label_map.values())
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

    # Strategy is descriptive only. It may use both semantic and visual evidence,
    # but it has no authority to delete speech or force a funnel order.
    strategy = choose_strategy(label_map.values(), kept)
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id=request.project_id,
        strategy=strategy,
        selected=selected,
        alternates=alternates,
        discarded=discarded_clips,
        diagnostics={
            "clean_cut_decisions": [decision.__dict__ for decision in decisions],
            "take_group_count": len(groups),
            "alternate_group_count": alternate_group_count,
            "source_count": len(request.sources),
            "take_judge_status_counts": dict(judge_statuses),
            "take_judge_fallback_reasons": dict(judge_reasons),
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
    return ProcessingResult(
        schema_version=SCHEMA_VERSION,
        project_id=request.project_id,
        state=JobState.DRAFT_READY,
        draft=draft,
        stage_status={
            "clean_cut": "complete",
            "take_grouping": "complete",
            "take_judge": judge_stage,
            "semantic": "provided" if label_map else "not_provided",
            "composer": "complete",
        },
    )
