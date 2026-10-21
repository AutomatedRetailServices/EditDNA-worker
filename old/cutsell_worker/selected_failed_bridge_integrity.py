"""Strengthen final-draft retry collapse when a failed bridge repeats an opening verbatim.

Round 7 showed a Spanish retry where the failed bridge contains mostly function words:
``Ahí fue cuando me mandaron``. Content-token overlap alone therefore counts only two
meaningful words even though the bridge is an exact five-word restart of the prior take.
This guard treats a long contiguous opening repeat as structural retry evidence, but only
when the bridge is semantically failed, the later selected take covers the same message,
and dense physical reset evidence exists between the attempts.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .contracts import DraftClip
from . import final_draft_retry_integrity as base


def _opening_repeat_coverage(bridge_text: str, earlier_text: str) -> tuple[int, float]:
    bridge = base._tokens(bridge_text)
    earlier = base._tokens(earlier_text)
    if len(bridge) < 4 or len(earlier) < len(bridge):
        return 0, 0.0
    matched = 0
    for left, right in zip(bridge, earlier[: len(bridge)]):
        if left != right:
            break
        matched += 1
    return matched, matched / max(1, len(bridge))


def suppress_selected_attempt_before_failed_bridge_v2(
    selected: Iterable[DraftClip],
    discarded: Iterable[DraftClip],
    diagnostics: dict,
    *,
    maximum_gap_sec: float = 20.0,
) -> tuple[tuple[DraftClip, ...], tuple[DraftClip, ...], tuple[dict, ...]]:
    selected_list = list(selected)
    discarded_list = list(discarded)
    failures = base._semantic_failures(diagnostics)
    removed_ids: set[str] = set()
    audit: list[dict] = []

    for index, earlier in enumerate(selected_list):
        if earlier.clip_id in removed_ids:
            continue
        for later in selected_list[index + 1 :]:
            if later.clip_id in removed_ids or later.source_asset_id != earlier.source_asset_id:
                continue
            gap = float(later.start) - float(earlier.end)
            if gap < 0:
                continue
            if gap > maximum_gap_sec:
                break

            shared_pair, _, later_cov = base._coverage(earlier.text, later.text)
            if shared_pair < 3 or later_cov < 0.55:
                continue

            candidates = []
            for bridge in discarded_list:
                if bridge.source_asset_id != earlier.source_asset_id:
                    continue
                if float(bridge.start) < float(earlier.end) or float(bridge.end) > float(later.start):
                    continue
                failure_conf = failures.get(bridge.clip_id, 0.0)
                if failure_conf < 0.80:
                    continue
                shared_bridge, bridge_content_cov, _ = base._coverage(bridge.text, earlier.text)
                opening_words, opening_cov = _opening_repeat_coverage(bridge.text, earlier.text)
                lexical_retry = bool(
                    (shared_bridge >= 2 and bridge_content_cov >= 0.80)
                    or (opening_words >= 4 and opening_cov >= 0.80)
                )
                if not lexical_retry:
                    continue
                candidates.append((opening_cov, bridge_content_cov, failure_conf, shared_bridge, opening_words, bridge))
            if not candidates:
                continue

            opening_cov, bridge_content_cov, bridge_failure, bridge_shared, opening_words, bridge = max(
                candidates, key=lambda item: item[:5]
            )
            reset_count, reset_conf = base._reset_count_between(earlier, later, diagnostics)
            if reset_count < 2 and not (
                bridge_failure >= 0.90
                and max(opening_cov, bridge_content_cov) >= 0.80
            ):
                continue

            removed_ids.add(earlier.clip_id)
            audit.append({
                "reason": "selected_attempt_yields_across_failed_repeated_opening_bridge",
                "removed_clip_id": earlier.clip_id,
                "failed_bridge_clip_id": bridge.clip_id,
                "winner_clip_id": later.clip_id,
                "bridge_failure_confidence": round(bridge_failure, 4),
                "opening_repeat_words": opening_words,
                "opening_repeat_coverage": round(opening_cov, 4),
                "bridge_content_coverage": round(bridge_content_cov, 4),
                "later_coverage": round(later_cov, 4),
                "reset_event_count": reset_count,
                "best_reset_confidence": round(reset_conf, 4),
                "removed_text": earlier.text,
                "bridge_text": bridge.text,
                "winner_text": later.text,
            })
            break

    if not removed_ids:
        return tuple(selected_list), tuple(discarded_list), ()
    removed = [clip for clip in selected_list if clip.clip_id in removed_ids]
    survivors = tuple(clip for clip in selected_list if clip.clip_id not in removed_ids)
    existing = {clip.clip_id for clip in discarded_list}
    discarded_out = tuple(discarded_list + [
        replace(clip, selected=False)
        for clip in removed
        if clip.clip_id not in existing
    ])
    return survivors, discarded_out, tuple(audit)


def install_selected_failed_bridge_integrity() -> None:
    current = base.suppress_selected_attempt_before_failed_bridge
    if getattr(current, "_cutsell_failed_bridge_v2", False):
        return
    suppress_selected_attempt_before_failed_bridge_v2._cutsell_failed_bridge_v2 = True
    base.suppress_selected_attempt_before_failed_bridge = suppress_selected_attempt_before_failed_bridge_v2
