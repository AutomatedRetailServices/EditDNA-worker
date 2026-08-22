"""Final conservative delivery-integrity pass for Clean Cut.

This pass runs after the existing Hybrid guards and repairs three structures that can
survive when segmentation/provider labels are locally plausible but globally wrong:

1) two selected deliveries of the same idea separated by a strong retry_setup event;
   the earlier attempt yields only when watch/listen ranking or explicit failure evidence
   proves the later delivery is cleaner;
2) a selected open-ended retry prefix followed by already-failed suffix fragments;
   when one earlier complete delivery covers the whole later attempt, the open prefix
   cannot survive as a Frankenstein ending;
3) a selected incomplete delivery followed immediately by a tiny discarded completion
   fragment; the fragment is restored only when it closes the same source delivery and
   no retry boundary separates them.

The rules are provider-neutral, never fabricate words, and fail open when evidence is
ambiguous.
"""
from __future__ import annotations

from dataclasses import replace
import re
import unicodedata
from typing import Iterable

from .contracts import CandidateTake
from .take_judge import rank_takes
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]*\s*$")
_OPEN_TAIL = frozenset({
    "a", "al", "and", "as", "at", "because", "but", "by", "con", "como", "cuando",
    "de", "del", "el", "en", "for", "from", "if", "in", "into", "la", "las", "los",
    "o", "of", "on", "or", "para", "pero", "por", "porque", "que", "si", "sin", "so",
    "than", "that", "the", "to", "un", "una", "unos", "unas", "when", "which", "while",
    "who", "with", "without", "y",
})
_STOP = frozenset({
    "a", "al", "and", "are", "as", "at", "be", "but", "by", "como", "con", "cuando",
    "de", "del", "el", "en", "es", "esta", "este", "for", "from", "fue", "in", "is",
    "it", "la", "las", "lo", "los", "me", "mi", "mis", "of", "on", "or", "para",
    "pero", "por", "porque", "que", "se", "si", "so", "su", "sus", "that", "the",
    "this", "to", "un", "una", "was", "we", "with", "y", "yo",
})
_NEGATION = frozenset({
    "no", "not", "never", "nunca", "sin", "without", "nadie", "ningun", "ningún",
    "ninguna", "ninguno", "nobody", "none", "neither",
})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _concept_token(token: str) -> str:
    value = "".join(
        char for char in unicodedata.normalize("NFKD", str(token or "").casefold())
        if not unicodedata.combining(char)
    )
    if len(value) >= 7 and value.endswith("es"):
        value = value[:-2]
    elif len(value) >= 6 and value.endswith("s"):
        value = value[:-1]
    return value


def _content(text: str) -> set[str]:
    out = set()
    for token in _tokens(text):
        if len(token) < 3 or token in _STOP:
            continue
        concept = _concept_token(token)
        if concept:
            out.add(concept)
    return out


def _critical(text: str) -> set[str]:
    out = {f"num:{token}" for token in _tokens(text) if any(ch.isdigit() for ch in token)}
    if any(token in _NEGATION for token in _tokens(text)):
        out.add("__negation__")
    return out


def _coverage(left_text: str, right_text: str) -> tuple[int, float, float]:
    left = _content(left_text)
    right = _content(right_text)
    if not left or not right:
        return 0, 0.0, 0.0
    shared = len(left & right)
    return shared, shared / max(1, len(left)), shared / max(1, len(right))


def _retry_setup_between(
    earlier: CandidateTake,
    later: CandidateTake,
    context: WholeVideoContext | None,
) -> float:
    if context is None or earlier.source_asset_id != later.source_asset_id:
        return 0.0
    start = max(float(earlier.start), float(earlier.end) - 0.50)
    end = min(float(later.start), float(earlier.end) + 3.0)
    if end < start:
        return 0.0
    best = 0.0
    for source in context.sources:
        if source.source_asset_id != earlier.source_asset_id:
            continue
        for event in source.events:
            kind = str(event.kind or "").strip().lower().replace("-", "_").replace(" ", "_")
            if kind != "retry_setup":
                continue
            if float(event.end) < start or float(event.start) > end:
                continue
            best = max(best, float(event.confidence))
        break
    return best


def _semantic_map(decisions: Iterable[tuple[str, str, float]]) -> dict[str, tuple[str, float]]:
    return {
        str(clip_id): (str(label), float(confidence))
        for clip_id, label, confidence in decisions
    }


def _semantic_priority(label: str) -> int:
    return {"failed": 0, "bts": 0, "uncertain": 1, "alternate": 2, "keep": 3, "winner": 4}.get(str(label), 1)


def _visual_prefers_later(earlier: CandidateTake, later: CandidateTake) -> bool:
    left = earlier.signals
    right = later.signals
    if left is None or right is None:
        return False
    if float(left.visual_fumble) >= float(right.visual_fumble) + 0.10:
        return True
    if float(left.distraction_risk) >= float(right.distraction_risk) + 0.18:
        return True
    if (
        float(left.expression_naturalness) + 0.18 <= float(right.expression_naturalness)
        and float(left.gesture_naturalness) + 0.12 <= float(right.gesture_naturalness)
    ):
        return True
    return False


def collapse_proven_retry_transitions(
    kept: Iterable[CandidateTake],
    semantic_decisions: Iterable[tuple[str, str, float]],
    context: WholeVideoContext | None,
    *,
    maximum_gap_sec: float = 20.0,
    minimum_retry_confidence: float = 0.84,
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    """Remove an earlier selected attempt when a proven retry transition supersedes it."""
    ordered = tuple(sorted(kept, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    semantic = _semantic_map(semantic_decisions)
    removed_ids: set[str] = set()
    diagnostics: list[dict] = []

    for index, earlier in enumerate(ordered):
        if earlier.clip_id in removed_ids:
            continue
        for later in ordered[index + 1 :]:
            if later.clip_id in removed_ids or later.source_asset_id != earlier.source_asset_id:
                continue
            gap = float(later.start) - float(earlier.end)
            if gap < 0:
                continue
            if gap > maximum_gap_sec:
                break

            retry_conf = _retry_setup_between(earlier, later, context)
            if retry_conf < minimum_retry_confidence:
                continue

            shared, earlier_cov, later_cov = _coverage(earlier.text, later.text)
            if shared < 3 or max(earlier_cov, later_cov) < 0.60:
                continue

            later_label, later_conf = semantic.get(later.clip_id, ("", 0.0))
            earlier_label, earlier_conf = semantic.get(earlier.clip_id, ("", 0.0))
            if later_label in {"failed", "bts"} and later_conf >= 0.82:
                continue

            ranked = rank_takes((earlier, later))
            rank_by_id = {item.clip_id: item for item in ranked}
            left_rank = rank_by_id.get(earlier.clip_id)
            right_rank = rank_by_id.get(later.clip_id)
            rank_prefers_later = bool(
                left_rank is not None
                and right_rank is not None
                and float(right_rank.score) >= float(left_rank.score) + 0.015
            )
            semantic_prefers_later = bool(
                _semantic_priority(later_label) > _semantic_priority(earlier_label)
                and later_conf >= 0.70
            )
            completeness_prefers_later = bool(not earlier.complete_idea and later.complete_idea)
            visual_prefers_later = _visual_prefers_later(earlier, later)

            # A retry_setup proves relation, not quality. Require a second independent
            # quality/completeness signal before deleting the earlier selected delivery.
            if not (
                rank_prefers_later
                or semantic_prefers_later
                or completeness_prefers_later
                or visual_prefers_later
            ):
                continue

            # Do not overturn a high-confidence earlier winner unless Watch+Listen or
            # deterministic ranking explicitly prefers the later attempt.
            if (
                earlier_label == "winner"
                and earlier_conf >= 0.90
                and not (rank_prefers_later or visual_prefers_later or completeness_prefers_later)
            ):
                continue

            removed_ids.add(earlier.clip_id)
            diagnostics.append({
                "reason": "proven_retry_transition_later_clean_delivery_wins",
                "removed_clip_id": earlier.clip_id,
                "winner_clip_id": later.clip_id,
                "retry_setup_confidence": round(retry_conf, 4),
                "gap_sec": round(gap, 3),
                "shared_content_tokens": shared,
                "earlier_coverage": round(earlier_cov, 4),
                "later_coverage": round(later_cov, 4),
                "earlier_label": earlier_label,
                "earlier_confidence": round(earlier_conf, 4),
                "later_label": later_label,
                "later_confidence": round(later_conf, 4),
                "rank_prefers_later": rank_prefers_later,
                "visual_prefers_later": visual_prefers_later,
                "completeness_prefers_later": completeness_prefers_later,
                "removed_text": earlier.text,
                "winner_text": later.text,
            })
            break

    survivors = tuple(take for take in ordered if take.clip_id not in removed_ids)
    removed = tuple(take for take in ordered if take.clip_id in removed_ids)
    return survivors, removed, tuple(diagnostics)


def _is_open_ended(take: CandidateTake) -> bool:
    tokens = _tokens(take.text)
    if not tokens:
        return False
    if tokens[-1] in _OPEN_TAIL:
        return True
    text = str(take.text or "").strip()
    return bool(text.endswith((",", ":", ";", "-", "–", "—")))


def suppress_open_retry_prefixes(
    kept: Iterable[CandidateTake],
    deleted: Iterable[CandidateTake],
    semantic_decisions: Iterable[tuple[str, str, float]],
    *,
    maximum_suffix_gap_sec: float = 5.0,
    maximum_suffix_chain_sec: float = 14.0,
    maximum_prior_gap_sec: float = 45.0,
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    """Remove an open selected retry prefix when its immediate continuation already failed."""
    kept_tuple = tuple(sorted(kept, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    deleted_tuple = tuple(sorted(deleted, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    semantic = _semantic_map(semantic_decisions)
    removed_ids: set[str] = set()
    diagnostics: list[dict] = []

    for candidate in kept_tuple:
        open_candidate = _is_open_ended(candidate) or not candidate.complete_idea
        if not open_candidate:
            continue

        chain = []
        cursor = float(candidate.end)
        for fragment in deleted_tuple:
            if fragment.source_asset_id != candidate.source_asset_id or fragment.start < candidate.end:
                continue
            gap = float(fragment.start) - cursor
            if gap > maximum_suffix_gap_sec:
                if chain:
                    break
                continue
            if float(fragment.end) - float(candidate.end) > maximum_suffix_chain_sec:
                break
            label, confidence = semantic.get(fragment.clip_id, ("", 0.0))
            if label not in {"failed", "bts"} or confidence < 0.74:
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
        suffix_shared, suffix_cov, _ = _coverage(suffix_text, candidate.text)
        if suffix_shared < 2 or suffix_cov < 0.25:
            continue

        combined_text = f"{candidate.text} {suffix_text}".strip()
        combined_critical = _critical(combined_text)
        options = []
        for prior in kept_tuple:
            if prior.clip_id == candidate.clip_id or prior.clip_id in removed_ids:
                continue
            if prior.source_asset_id != candidate.source_asset_id or prior.end > candidate.start:
                continue
            gap = float(candidate.start) - float(prior.end)
            if gap > maximum_prior_gap_sec or not prior.complete_idea:
                continue
            shared, combined_cov, prior_cov = _coverage(combined_text, prior.text)
            if shared < 7 or combined_cov < 0.35 or prior_cov < 0.32:
                continue
            if not combined_critical.issubset(_critical(prior.text)):
                continue
            options.append((combined_cov, prior_cov, shared, -gap, prior))
        if not options:
            continue

        combined_cov, prior_cov, shared, _, prior = max(options, key=lambda item: item[:4])
        removed_ids.add(candidate.clip_id)
        diagnostics.append({
            "reason": "open_selected_retry_prefix_yields_to_complete_prior_delivery",
            "removed_clip_id": candidate.clip_id,
            "prior_winner_clip_id": prior.clip_id,
            "failed_suffix_clip_ids": [fragment.clip_id for fragment in chain],
            "suffix_shared_content_tokens": suffix_shared,
            "combined_shared_content_tokens": shared,
            "combined_coverage": round(combined_cov, 4),
            "prior_coverage": round(prior_cov, 4),
            "removed_text": candidate.text,
            "failed_suffix_text": suffix_text,
            "prior_text": prior.text,
        })

    survivors = tuple(take for take in kept_tuple if take.clip_id not in removed_ids)
    removed = tuple(take for take in kept_tuple if take.clip_id in removed_ids)
    return survivors, removed, tuple(diagnostics)


def restore_immediate_completion_fragments(
    kept: Iterable[CandidateTake],
    deleted: Iterable[CandidateTake],
    semantic_decisions: Iterable[tuple[str, str, float]],
    context: WholeVideoContext | None,
    *,
    maximum_gap_sec: float = 1.80,
    maximum_fragment_sec: float = 2.50,
    maximum_fragment_tokens: int = 4,
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    """Restore a tiny deleted suffix that completes an otherwise incomplete kept delivery."""
    kept_tuple = tuple(sorted(kept, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    deleted_tuple = tuple(sorted(deleted, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    semantic = _semantic_map(semantic_decisions)
    restored_ids: set[str] = set()
    replacement: dict[str, CandidateTake] = {}
    diagnostics: list[dict] = []

    for candidate in kept_tuple:
        if candidate.complete_idea or _SENTENCE_END_RE.search(str(candidate.text or "").strip()):
            continue
        candidate_tokens = _tokens(candidate.text)
        if len(candidate_tokens) < 4:
            continue

        options = []
        for fragment in deleted_tuple:
            if fragment.clip_id in restored_ids or fragment.source_asset_id != candidate.source_asset_id:
                continue
            if fragment.start < candidate.end:
                continue
            gap = float(fragment.start) - float(candidate.end)
            if gap > maximum_gap_sec:
                break
            fragment_tokens = _tokens(fragment.text)
            if not fragment_tokens or len(fragment_tokens) > maximum_fragment_tokens:
                continue
            if fragment.duration_sec > maximum_fragment_sec:
                continue
            if not (fragment.complete_idea or _SENTENCE_END_RE.search(str(fragment.text or "").strip())):
                continue
            label, confidence = semantic.get(fragment.clip_id, ("", 0.0))
            if label == "bts" and confidence >= 0.84:
                continue
            if label == "failed" and confidence >= 0.94:
                continue
            retry_conf = _retry_setup_between(candidate, fragment, context)
            if retry_conf >= 0.84:
                continue
            options.append((gap, confidence, fragment, label))

        if not options:
            continue
        gap, confidence, fragment, label = min(options, key=lambda item: (item[0], item[1], item[2].start))
        merged_words = tuple(candidate.words) + tuple(fragment.words)
        merged_text = f"{str(candidate.text or '').strip()} {str(fragment.text or '').strip()}".strip()
        merged_signals = (
            replace(candidate.signals, end=float(fragment.end))
            if candidate.signals is not None else None
        )
        child = replace(
            candidate,
            end=float(fragment.end),
            text=merged_text,
            words=merged_words,
            signals=merged_signals,
            complete_idea=True,
        )
        replacement[candidate.clip_id] = child
        restored_ids.add(fragment.clip_id)
        diagnostics.append({
            "reason": "restore_immediate_deleted_completion_fragment",
            "clip_id": candidate.clip_id,
            "restored_fragment_id": fragment.clip_id,
            "gap_sec": round(gap, 3),
            "fragment_label": label,
            "fragment_confidence": round(confidence, 4),
            "original_text": candidate.text,
            "restored_text": merged_text,
        })

    if not diagnostics:
        return kept_tuple, deleted_tuple, ()
    output = tuple(replacement.get(take.clip_id, take) for take in kept_tuple)
    remaining_deleted = tuple(take for take in deleted_tuple if take.clip_id not in restored_ids)
    return output, remaining_deleted, tuple(diagnostics)


def install_final_delivery_integrity() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_final_delivery_integrity", False):
        return

    def apply_with_final_delivery_integrity(*args, **kwargs):
        source_takes = tuple(args[0]) if args else tuple(kwargs.get("takes") or ())
        context = args[1] if len(args) > 1 else kwargs.get("context")
        result = original(*args, **kwargs)
        if not result.kept or not result.semantic_decisions:
            return result

        kept, retry_removed, retry_diag = collapse_proven_retry_transitions(
            result.kept,
            result.semantic_decisions,
            context,
        )
        deleted_ids = {take.clip_id for take in result.deleted}
        deleted_ids.update(take.clip_id for take in retry_removed)
        deleted_pool = tuple(take for take in source_takes if take.clip_id in deleted_ids)

        kept, open_removed, open_diag = suppress_open_retry_prefixes(
            kept,
            deleted_pool,
            result.semantic_decisions,
        )
        deleted_ids.update(take.clip_id for take in open_removed)
        deleted_pool = tuple(take for take in source_takes if take.clip_id in deleted_ids)

        kept, deleted_pool, rescue_diag = restore_immediate_completion_fragments(
            kept,
            deleted_pool,
            result.semantic_decisions,
            context,
        )
        remaining_deleted_ids = {take.clip_id for take in deleted_pool}
        deleted = tuple(take for take in source_takes if take.clip_id in remaining_deleted_ids)

        if not retry_diag and not open_diag and not rescue_diag:
            return result
        diagnostics = tuple(result.diagnostics) + ({
            "final_delivery_integrity": [
                *list(retry_diag),
                *list(open_diag),
                *list(rescue_diag),
            ],
        },)
        return type(result)(
            kept=kept,
            deleted=deleted,
            requested_chunk_count=result.requested_chunk_count,
            available_chunk_count=result.available_chunk_count,
            diagnostics=diagnostics,
            semantic_decisions=result.semantic_decisions,
        )

    apply_with_final_delivery_integrity._cutsell_final_delivery_integrity = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_final_delivery_integrity
