"""Round 11 final semantic retry cleanup.

A selected take can survive deterministic/Hybrid cleanup even when Hybrid itself marked
that take failed in one overlapping window and a later take is a high-confidence winner.
This pass removes only an *open/incomplete* selected attempt when the later winner covers
substantial content from it. It is intentionally conservative and runs at final draft
level, where all selected/discarded evidence is visible.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .contracts import DraftClip
from . import final_draft_retry_integrity as retry_base


def _semantic_winners(diagnostics: dict) -> dict[str, float]:
    winners: dict[str, float] = {}
    for chunk in diagnostics.get("hybrid_editorial_chunks") or ():
        if not isinstance(chunk, dict):
            continue
        for item in chunk.get("decisions") or ():
            if not isinstance(item, dict):
                continue
            if str(item.get("label") or "").strip().lower() != "winner":
                continue
            cid = str(item.get("clip_id") or "")
            if cid:
                winners[cid] = max(winners.get(cid, 0.0), float(item.get("confidence") or 0.0))
    return winners


def suppress_failed_open_attempt_before_later_winner(
    selected: Iterable[DraftClip],
    discarded: Iterable[DraftClip],
    diagnostics: dict,
    *,
    maximum_gap_sec: float = 30.0,
) -> tuple[tuple[DraftClip, ...], tuple[DraftClip, ...], tuple[dict, ...]]:
    selected_list = list(selected)
    discarded_list = list(discarded)
    failures = retry_base._semantic_failures(diagnostics)
    winners = _semantic_winners(diagnostics)
    removed: set[str] = set()
    audit: list[dict] = []

    for index, earlier in enumerate(selected_list):
        failed_conf = failures.get(earlier.clip_id, 0.0)
        if failed_conf < 0.85 or not retry_base._is_open_text(earlier.text):
            continue
        for later in selected_list[index + 1 :]:
            if later.source_asset_id != earlier.source_asset_id:
                continue
            gap = float(later.start) - float(earlier.end)
            if gap < 0:
                continue
            if gap > maximum_gap_sec:
                break
            winner_conf = winners.get(later.clip_id, 0.0)
            if winner_conf < 0.90:
                continue
            shared, earlier_cov, later_cov = retry_base._coverage(earlier.text, later.text)
            # Require the failed open attempt to be substantially covered by the later
            # winner. The winner may be longer because it finishes the thought.
            if shared < 4 or earlier_cov < 0.45 or later_cov < 0.20:
                continue
            # Any explicit numeric/negation facts in the failed attempt must not be lost.
            if not retry_base._critical(earlier.text).issubset(retry_base._critical(later.text)):
                continue
            removed.add(earlier.clip_id)
            audit.append({
                "reason": "failed_open_attempt_superseded_by_later_semantic_winner",
                "removed_clip_id": earlier.clip_id,
                "winner_clip_id": later.clip_id,
                "failed_confidence": round(failed_conf, 4),
                "winner_confidence": round(winner_conf, 4),
                "shared_content_tokens": shared,
                "failed_coverage": round(earlier_cov, 4),
                "winner_coverage": round(later_cov, 4),
                "gap_sec": round(gap, 3),
                "removed_text": earlier.text,
                "winner_text": later.text,
            })
            break

    if not removed:
        return tuple(selected_list), tuple(discarded_list), ()
    moved = [replace(clip, selected=False) for clip in selected_list if clip.clip_id in removed]
    existing = {clip.clip_id for clip in discarded_list}
    discarded_out = tuple(discarded_list + [clip for clip in moved if clip.clip_id not in existing])
    selected_out = tuple(clip for clip in selected_list if clip.clip_id not in removed)
    return selected_out, discarded_out, tuple(audit)


def install_round11_semantic_retry_cleanup() -> None:
    from . import pipeline

    original = pipeline.build_flow_b_draft
    if getattr(original, "_cutsell_round11_semantic_retry_cleanup", False):
        return

    def build_with_round11_semantic_retry_cleanup(*args, **kwargs):
        result = original(*args, **kwargs)
        draft = result.draft
        diagnostics = dict(draft.diagnostics or {})
        selected, discarded, audit = suppress_failed_open_attempt_before_later_winner(
            draft.selected,
            draft.discarded,
            diagnostics,
        )
        if not audit:
            return result
        diagnostics["round11_semantic_retry_cleanup"] = list(audit)
        repaired = replace(draft, selected=selected, discarded=discarded, diagnostics=diagnostics)
        return replace(result, draft=repaired)

    build_with_round11_semantic_retry_cleanup._cutsell_round11_semantic_retry_cleanup = True
    pipeline.build_flow_b_draft = build_with_round11_semantic_retry_cleanup
