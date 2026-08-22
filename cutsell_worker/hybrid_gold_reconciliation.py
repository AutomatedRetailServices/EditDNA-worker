"""Final conservative reconciliation for Hybrid decisions exposed by human Gold.

Two failure modes can survive earlier guards:
1) a clean complete retake is deleted as a short alternate "covered by neighbors" while
   the preceding retry attempt has local failure evidence and remains selected;
2) an incomplete alternate is correctly deleted beside a winner, but its immediate
   continuation survives without a semantic label and renders the same losing delivery.

This pass only repairs those already-proven structures. It does not create new retries
from topic similarity alone.
"""
from __future__ import annotations

import re
from typing import Iterable

from .contracts import CandidateTake

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_STOP = frozenset({
    "a","al","and","are","as","at","be","but","by","como","con","de","del","el","en",
    "es","esta","este","for","from","in","is","it","la","las","lo","los","me","mi","mis",
    "of","on","or","para","pero","por","porque","que","se","so","su","sus","that","the",
    "this","to","un","una","was","we","with","y","yo",
})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) >= 3 and token not in _STOP}


def _numbers(text: str) -> set[str]:
    return {token for token in _tokens(text) if any(ch.isdigit() for ch in token)}


def _coverage(source: CandidateTake, peers: Iterable[CandidateTake]) -> tuple[int, float]:
    own = _content(source.text)
    if not own:
        return 0, 0.0
    covered: set[str] = set()
    for peer in peers:
        covered.update(_content(peer.text))
    shared = len(own & covered)
    return shared, shared / max(1, len(own))


def _gap(left: CandidateTake, right: CandidateTake) -> float:
    if left.source_asset_id != right.source_asset_id:
        return float("inf")
    if left.end <= right.start:
        return max(0.0, right.start - left.end)
    if right.end <= left.start:
        return max(0.0, left.start - right.end)
    return 0.0


def _local_failure(take: CandidateTake, context) -> bool:
    from .hybrid_session_cleanup import _failed_local_evidence
    failed, _ = _failed_local_evidence(take, context)
    return bool(failed)


def _deleted_short_alternate_ids(diagnostics) -> set[str]:
    out: set[str] = set()
    def visit(value):
        if isinstance(value, dict):
            if value.get("reason") == "semantic_short_alternate_covered_by_neighbors" and value.get("clip_id"):
                out.add(str(value["clip_id"]))
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
    visit(diagnostics)
    return out


def _deleted_incomplete_alternates(diagnostics) -> list[dict]:
    out: list[dict] = []
    def visit(value):
        if isinstance(value, dict):
            if value.get("reason") == "semantic_alternate_incomplete_retry_after_winner" and value.get("clip_id") and value.get("winner_clip_id"):
                out.append(dict(value))
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
    visit(diagnostics)
    return out


def reconcile_human_gold_hybrid(result, source_takes, context=None):
    if not result.kept:
        return result
    source_tuple = tuple(sorted(source_takes, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    by_id = {take.clip_id: take for take in source_tuple}
    semantic = {str(cid): (str(label), float(conf)) for cid, label, conf in result.semantic_decisions}
    kept = {take.clip_id: take for take in result.kept}
    deleted = {take.clip_id: take for take in result.deleted}
    diagnostics: list[dict] = []

    # Repair 1: restore a complete retake wrongly deleted as short alternate debris,
    # and remove the preceding locally-failed retry attempt it supersedes.
    for cid in _deleted_short_alternate_ids(result.diagnostics):
        candidate = deleted.get(cid)
        if candidate is None or not candidate.complete_idea:
            continue
        label, conf = semantic.get(cid, ("", 0.0))
        if label not in {"alternate", "winner", "keep"} or conf < 0.80:
            continue
        prior_options = []
        for previous in kept.values():
            if previous.source_asset_id != candidate.source_asset_id or previous.end > candidate.start:
                continue
            gap = _gap(previous, candidate)
            if gap > 15.0 or not _local_failure(previous, context):
                continue
            shared, previous_cov = _coverage(previous, (candidate,))
            if shared < 3 or previous_cov < 0.55:
                continue
            prior_options.append((previous_cov, shared, -gap, previous))
        if not prior_options:
            continue
        _, shared, _, previous = max(prior_options, key=lambda item: item[:3])
        kept.pop(previous.clip_id, None)
        deleted[previous.clip_id] = previous
        deleted.pop(candidate.clip_id, None)
        kept[candidate.clip_id] = candidate
        diagnostics.append({
            "reason": "restore_clean_retake_remove_failed_previous",
            "restored_clip_id": candidate.clip_id,
            "removed_clip_id": previous.clip_id,
            "shared_content_tokens": shared,
        })

    # Repair 2: when an incomplete alternate after a winner was already removed,
    # remove its immediate unlabeled continuation if the combined losing delivery is
    # substantially covered by that same winner and numeric facts stay compatible.
    for item in _deleted_incomplete_alternates(result.diagnostics):
        prefix = by_id.get(str(item["clip_id"]))
        winner = kept.get(str(item["winner_clip_id"])) or by_id.get(str(item["winner_clip_id"]))
        if prefix is None or winner is None:
            continue
        continuations = [
            take for take in kept.values()
            if take.source_asset_id == prefix.source_asset_id
            and take.start >= prefix.end
            and 0.0 <= take.start - prefix.end <= 3.0
        ]
        if len(continuations) != 1:
            continue
        continuation = continuations[0]
        cont_label, cont_conf = semantic.get(continuation.clip_id, ("", 0.0))
        if cont_label in {"winner", "keep"} and cont_conf >= 0.88:
            continue
        combined = CandidateTake(
            clip_id=f"{prefix.clip_id}+{continuation.clip_id}",
            source_asset_id=prefix.source_asset_id,
            source_order=prefix.source_order,
            start=prefix.start,
            end=continuation.end,
            text=f"{prefix.text} {continuation.text}",
            words=tuple(prefix.words) + tuple(continuation.words),
            signals=prefix.signals,
            complete_idea=continuation.complete_idea,
        )
        shared, combined_cov = _coverage(combined, (winner,))
        winner_shared, winner_cov = _coverage(winner, (combined,))
        nums_combined = _numbers(combined.text)
        nums_winner = _numbers(winner.text)
        numbers_ok = not nums_combined or not nums_winner or nums_combined == nums_winner
        if not (numbers_ok and shared >= 7 and combined_cov >= 0.35 and winner_shared >= 7 and winner_cov >= 0.35):
            continue
        kept.pop(continuation.clip_id, None)
        deleted[continuation.clip_id] = continuation
        diagnostics.append({
            "reason": "remove_orphan_continuation_of_deleted_incomplete_alternate",
            "prefix_clip_id": prefix.clip_id,
            "continuation_clip_id": continuation.clip_id,
            "winner_clip_id": winner.clip_id,
            "combined_coverage": round(combined_cov, 4),
            "winner_coverage": round(winner_cov, 4),
        })

    if not diagnostics:
        return result
    ordered_kept = tuple(take for take in source_tuple if take.clip_id in kept)
    ordered_deleted = tuple(take for take in source_tuple if take.clip_id in deleted)
    return type(result)(
        kept=ordered_kept,
        deleted=ordered_deleted,
        requested_chunk_count=result.requested_chunk_count,
        available_chunk_count=result.available_chunk_count,
        diagnostics=tuple(result.diagnostics) + ({"hybrid_gold_reconciliation": diagnostics},),
        semantic_decisions=result.semantic_decisions,
    )


def install_hybrid_gold_reconciliation() -> None:
    from . import hybrid_session_cleanup
    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_hybrid_gold_reconciliation", False):
        return

    def apply_with_gold_reconciliation(*args, **kwargs):
        source_takes = tuple(args[0]) if args else tuple(kwargs.get("takes") or ())
        context = args[1] if len(args) > 1 else kwargs.get("context")
        result = original(*args, **kwargs)
        return reconcile_human_gold_hybrid(result, source_takes, context)

    apply_with_gold_reconciliation._cutsell_hybrid_gold_reconciliation = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_gold_reconciliation
