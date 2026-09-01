"""Repair split retries whose failure evidence is distributed across adjacent fragments.

Two conservative structures are handled:

1) failed prefix + immediate continuation: both fragments belong to one losing retry and
   are removed when a nearby authoritative delivery covers the combined message;
2) selected prefix + immediate failed suffix: the prefix may look clean in isolation and
   even receive a Hybrid ``winner`` label, but the creator continues the same delivery,
   fumbles, and abandons it. When an earlier complete delivery already covers that whole
   later attempt with the same critical number/negation facts, the selected prefix must
   yield as part of the failed attempt rather than survive as a Frankenstein clean ending.

The second structure is the exact Video 00 Round 5 Gold failure. This module does not infer
retries from topic similarity alone: it requires immediate temporal adjacency, strong
failed semantics on the suffix, strong lexical continuity between prefix and suffix, and
substantial bidirectional coverage by one nearby complete earlier delivery.
"""
from __future__ import annotations

import re
from typing import Iterable

from .contracts import CandidateTake

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by", "for", "from",
    "has", "have", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "so",
    "that", "the", "this", "to", "was", "we", "were", "what", "with", "you", "your",
    "al", "como", "con", "cuando", "de", "del", "el", "en", "ella", "ellas", "ellos",
    "es", "esta", "este", "la", "las", "le", "les", "lo", "los", "mi", "mis", "o",
    "para", "pero", "por", "porque", "que", "se", "si", "su", "sus", "un", "una",
    "unos", "unas", "y", "yo",
})
_NEGATION = frozenset({
    "no", "not", "never", "nunca", "sin", "without", "nadie", "ningun", "ningún",
    "ninguna", "ninguno", "nobody", "none", "neither",
})
_NEGATION_CANONICAL = "__negation__"


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _content(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) >= 3 and token not in _STOP}


def _critical(text: str) -> set[str]:
    raw = str(text or "")
    out: set[str] = {f"num:{number}" for number in _NUMBER_RE.findall(raw)}
    if any(token in _NEGATION for token in _tokens(raw)):
        out.add(_NEGATION_CANONICAL)
    return out


def _combined_coverage(left: CandidateTake, right: CandidateTake, winner: CandidateTake) -> tuple[float, int, bool]:
    failed_content = _content(left.text) | _content(right.text)
    winner_content = _content(winner.text)
    shared = len(failed_content & winner_content)
    coverage = shared / max(1, len(failed_content))
    failed_critical = _critical(left.text) | _critical(right.text)
    critical_preserved = failed_critical.issubset(_critical(winner.text))
    return coverage, shared, critical_preserved


def _coverage_both(source_text: str, peer_text: str) -> tuple[int, float, float]:
    source = _content(source_text)
    peer = _content(peer_text)
    if not source or not peer:
        return 0, 0.0, 0.0
    shared = len(source & peer)
    return shared, shared / max(1, len(source)), shared / max(1, len(peer))


def collapse_failed_split_retry_continuations(
    kept: Iterable[CandidateTake],
    semantic_decisions: Iterable[tuple[str, str, float]],
    *,
    maximum_continuation_gap_sec: float = 3.0,
    maximum_winner_gap_sec: float = 45.0,
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    kept_tuple = tuple(sorted(kept, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    semantic = {
        str(clip_id): (str(label), float(confidence))
        for clip_id, label, confidence in semantic_decisions
    }
    removed_ids: set[str] = set()
    diagnostics: list[dict] = []

    for index, failed in enumerate(kept_tuple):
        label, confidence = semantic.get(failed.clip_id, ("", 0.0))
        if label != "failed" or confidence < 0.80 or failed.complete_idea:
            continue

        continuation = None
        for other in kept_tuple[index + 1 :]:
            if other.source_asset_id != failed.source_asset_id:
                continue
            gap = other.start - failed.end
            if gap < 0:
                continue
            if gap > maximum_continuation_gap_sec:
                break
            other_label, other_confidence = semantic.get(other.clip_id, ("", 0.0))
            if other_label in {"winner", "keep"} and other_confidence >= 0.85:
                continue
            continuation = other
            break
        if continuation is None:
            continue

        winners = []
        for other in kept_tuple:
            if other.clip_id in {failed.clip_id, continuation.clip_id}:
                continue
            if other.source_asset_id != failed.source_asset_id:
                continue
            winner_label, winner_confidence = semantic.get(other.clip_id, ("", 0.0))
            if winner_label not in {"winner", "keep"} or winner_confidence < 0.90:
                continue
            if other.end <= failed.start:
                gap = failed.start - other.end
            elif continuation.end <= other.start:
                gap = other.start - continuation.end
            else:
                gap = 0.0
            if gap <= maximum_winner_gap_sec:
                winners.append(other)

        best = None
        for winner in winners:
            coverage, shared, critical_preserved = _combined_coverage(failed, continuation, winner)
            candidate = (coverage, shared, critical_preserved, winner)
            if best is None or (coverage, shared) > (best[0], best[1]):
                best = candidate
        if best is None:
            continue

        coverage, shared, critical_preserved, winner = best
        if shared < 6 or coverage < 0.50 or not critical_preserved:
            continue

        removed_ids.update({failed.clip_id, continuation.clip_id})
        diagnostics.append({
            "failed_clip_id": failed.clip_id,
            "continuation_clip_id": continuation.clip_id,
            "winner_clip_id": winner.clip_id,
            "reason": "failed_split_retry_covered_by_authoritative_winner",
            "failed_confidence": round(confidence, 4),
            "combined_coverage": round(coverage, 4),
            "shared_content_tokens": shared,
            "critical_preserved": True,
            "failed_text": failed.text,
            "continuation_text": continuation.text,
            "winner_text": winner.text,
        })

    survivors = tuple(take for take in kept_tuple if take.clip_id not in removed_ids)
    removed = tuple(take for take in kept_tuple if take.clip_id in removed_ids)
    return survivors, removed, tuple(diagnostics)


def suppress_selected_prefixes_with_failed_suffixes(
    kept: Iterable[CandidateTake],
    deleted: Iterable[CandidateTake],
    semantic_decisions: Iterable[tuple[str, str, float]],
    *,
    maximum_suffix_gap_sec: float = 3.0,
    maximum_suffix_chain_sec: float = 12.0,
    maximum_prior_delivery_gap_sec: float = 45.0,
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    """Drop a seemingly clean selected prefix when its immediate same-delivery suffix fails.

    A later retry can be split by segmentation exactly at the point where the creator has
    not yet visibly failed. Judging only that prefix can make it look like a winner. If
    immediate deleted failed fragments continue/repeat that same delivery and an earlier
    complete selected delivery covers the whole chain, the later prefix is not a valid
    standalone winner; it is the clean-looking beginning of a failed attempt.
    """
    kept_tuple = tuple(sorted(kept, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    deleted_tuple = tuple(sorted(deleted, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    semantic = {
        str(clip_id): (str(label), float(confidence))
        for clip_id, label, confidence in semantic_decisions
    }
    removed_ids: set[str] = set()
    diagnostics: list[dict] = []

    for candidate in kept_tuple:
        candidate_label, candidate_conf = semantic.get(candidate.clip_id, ("", 0.0))
        if candidate_label not in {"winner", "keep"} or candidate_conf < 0.85:
            continue

        # Build the immediate failed suffix chain. Each fragment must already be proven
        # failed/BTS semantically; no unlabeled speech is swept into the delete.
        chain = []
        cursor = float(candidate.end)
        for fragment in deleted_tuple:
            if fragment.source_asset_id != candidate.source_asset_id or fragment.start < candidate.end:
                continue
            gap = float(fragment.start) - cursor
            if gap > maximum_suffix_gap_sec:
                if not chain:
                    continue
                break
            if float(fragment.end) - float(candidate.end) > maximum_suffix_chain_sec:
                break
            label, confidence = semantic.get(fragment.clip_id, ("", 0.0))
            if label not in {"failed", "bts"} or confidence < 0.80:
                if chain:
                    break
                continue
            chain.append(fragment)
            cursor = float(fragment.end)
            if len(chain) >= 3:
                break
        if not chain:
            continue

        suffix_text = " ".join(fragment.text for fragment in chain)
        suffix_shared, suffix_cov, _ = _coverage_both(suffix_text, candidate.text)
        # The failed suffix must clearly continue/repeat the selected prefix rather than
        # merely happen to follow it in time.
        if suffix_shared < 4 or suffix_cov < 0.60:
            continue

        combined_text = " ".join([candidate.text, suffix_text])
        combined_critical = _critical(combined_text)

        prior_options = []
        for prior in kept_tuple:
            if prior.clip_id == candidate.clip_id or prior.clip_id in removed_ids:
                continue
            if prior.source_asset_id != candidate.source_asset_id:
                continue
            if prior.end > candidate.start:
                continue
            gap = float(candidate.start) - float(prior.end)
            if gap > maximum_prior_delivery_gap_sec:
                continue
            if not prior.complete_idea:
                continue
            prior_label, prior_conf = semantic.get(prior.clip_id, ("", 0.0))
            if prior_label not in {"winner", "keep", "alternate"} or prior_conf < 0.70:
                continue
            shared, combined_cov, prior_cov = _coverage_both(combined_text, prior.text)
            critical_preserved = combined_critical.issubset(_critical(prior.text))
            if shared < 8 or combined_cov < 0.45 or prior_cov < 0.35 or not critical_preserved:
                continue
            prior_options.append((combined_cov, prior_cov, shared, prior_conf, -gap, prior))

        if not prior_options:
            continue
        combined_cov, prior_cov, shared, prior_conf, _, prior = max(prior_options, key=lambda item: item[:5])
        removed_ids.add(candidate.clip_id)
        diagnostics.append({
            "removed_clip_id": candidate.clip_id,
            "prior_winner_clip_id": prior.clip_id,
            "failed_suffix_clip_ids": [fragment.clip_id for fragment in chain],
            "reason": "selected_prefix_yields_when_immediate_suffix_fails_same_retry",
            "selected_prefix_label": candidate_label,
            "selected_prefix_confidence": round(candidate_conf, 4),
            "prior_label": semantic.get(prior.clip_id, ("", 0.0))[0],
            "prior_confidence": round(prior_conf, 4),
            "suffix_shared_content_tokens": suffix_shared,
            "combined_shared_content_tokens": shared,
            "combined_coverage": round(combined_cov, 4),
            "prior_coverage": round(prior_cov, 4),
            "critical_preserved": True,
            "removed_text": candidate.text,
            "failed_suffix_text": suffix_text,
            "prior_text": prior.text,
        })

    survivors = tuple(take for take in kept_tuple if take.clip_id not in removed_ids)
    removed = tuple(take for take in kept_tuple if take.clip_id in removed_ids)
    return survivors, removed, tuple(diagnostics)


def install_hybrid_failed_continuation_integrity() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_hybrid_failed_continuation_integrity", False):
        return

    def apply_with_failed_continuation_integrity(*args, **kwargs):
        if args:
            source_takes = tuple(args[0])
            result = original(source_takes, *args[1:], **kwargs)
        else:
            source_takes = tuple(kwargs.get("takes") or ())
            call_kwargs = dict(kwargs)
            call_kwargs["takes"] = source_takes
            result = original(**call_kwargs)

        if not result.kept or not result.semantic_decisions:
            return result

        kept, first_removed, first_diagnostics = collapse_failed_split_retry_continuations(
            result.kept,
            result.semantic_decisions,
        )
        deleted_pool_ids = {take.clip_id for take in result.deleted}
        deleted_pool_ids.update(take.clip_id for take in first_removed)
        deleted_pool = tuple(take for take in source_takes if take.clip_id in deleted_pool_ids)

        kept, second_removed, second_diagnostics = suppress_selected_prefixes_with_failed_suffixes(
            kept,
            deleted_pool,
            result.semantic_decisions,
        )
        if not first_diagnostics and not second_diagnostics:
            return result

        deleted_ids = set(deleted_pool_ids)
        deleted_ids.update(take.clip_id for take in second_removed)
        deleted = tuple(take for take in source_takes if take.clip_id in deleted_ids)
        diagnostics = tuple(result.diagnostics) + ({
            "hybrid_failed_continuation_integrity": [
                *list(first_diagnostics),
                *list(second_diagnostics),
            ],
            "deleted_ids": sorted(take.clip_id for take in (*first_removed, *second_removed)),
        },)
        return type(result)(
            kept=kept,
            deleted=deleted,
            requested_chunk_count=result.requested_chunk_count,
            available_chunk_count=result.available_chunk_count,
            diagnostics=diagnostics,
            semantic_decisions=result.semantic_decisions,
        )

    apply_with_failed_continuation_integrity._cutsell_hybrid_failed_continuation_integrity = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_failed_continuation_integrity
