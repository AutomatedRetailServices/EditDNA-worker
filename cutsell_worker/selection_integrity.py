"""Final selection guard for weak retry fragments stranded as singleton groups.

Retry grouping is intentionally conservative. In long-form creator footage, cleanup can
remove some retry evidence before Best Take, leaving one short false start as a singleton
while a later full delivery is correctly selected in another group. This guard never
creates a new winner and never deletes substantive speech. It only removes a singleton
from the selected draft when the existing Best Take ranker independently classifies it
as a fragment/restart relative to another already-selected fuller delivery.
"""
from __future__ import annotations

from typing import Iterable, Tuple

from .contracts import CandidateTake, SemanticLabel, TakeGroup
from .take_judge import rank_takes

_FRAGMENT_PENALTIES = (
    "material_prefix_fragment_penalty",
    "repetitive_restart_fragment_penalty",
    "restart_tail_fragment_penalty",
)


def _within_retry_window(left: CandidateTake, right: CandidateTake, *, maximum_gap_sec: float = 30.0) -> bool:
    if left.source_asset_id != right.source_asset_id:
        return False
    if left.end < right.start:
        gap = right.start - left.end
    elif right.end < left.start:
        gap = left.start - right.end
    else:
        gap = 0.0
    return gap <= maximum_gap_sec


def _fragment_penalized_against(candidate: CandidateTake, fuller: CandidateTake) -> bool:
    if not _within_retry_window(candidate, fuller):
        return False
    if fuller.duration_sec <= candidate.duration_sec + 0.5:
        return False
    ranked = rank_takes((candidate, fuller))
    if not ranked or ranked[0].clip_id != fuller.clip_id:
        return False
    by_id = {item.clip_id: item for item in ranked}
    candidate_rank = by_id.get(candidate.clip_id)
    if candidate_rank is None:
        return False
    reason = str(candidate_rank.reason or "")
    return any(marker in reason for marker in _FRAGMENT_PENALTIES)


def filter_stranded_retry_singletons(
    selected: Iterable[CandidateTake],
    groups: Iterable[TakeGroup],
) -> Tuple[CandidateTake, ...]:
    selected_tuple = tuple(selected)
    groups_tuple = tuple(groups)
    group_size_by_selected = {
        group.selected_clip_id: len(group.candidate_ids)
        for group in groups_tuple
    }
    suppressed = set()

    for candidate in selected_tuple:
        if group_size_by_selected.get(candidate.clip_id, 1) != 1:
            continue
        for fuller in selected_tuple:
            if fuller.clip_id == candidate.clip_id:
                continue
            if _fragment_penalized_against(candidate, fuller):
                suppressed.add(candidate.clip_id)
                break

    if not suppressed:
        return selected_tuple
    return tuple(take for take in selected_tuple if take.clip_id not in suppressed)


def install_selection_integrity() -> None:
    from . import composer

    original = composer.compose_selected
    if getattr(original, "_cutsell_selection_integrity", False):
        return

    def compose_selected_with_integrity(
        takes: Iterable[CandidateTake],
        groups: Iterable[TakeGroup],
        labels: Iterable[SemanticLabel],
    ) -> Tuple[CandidateTake, ...]:
        take_tuple = tuple(takes)
        group_tuple = tuple(groups)
        label_tuple = tuple(labels)
        selected = original(take_tuple, group_tuple, label_tuple)
        return filter_stranded_retry_singletons(selected, group_tuple)

    compose_selected_with_integrity._cutsell_selection_integrity = True
    composer.compose_selected = compose_selected_with_integrity
