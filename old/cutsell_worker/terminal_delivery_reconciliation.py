"""Final terminal-delivery reconciliation after Hybrid selection.

This pass fixes two boundary/attempt structures without inventing speech:
- an open selected retry prefix (for example ending in a function phrase such as
  ``de los``) must yield when its immediate deleted continuation is already proven
  failed and an earlier complete delivery covers the same message;
- an incomplete selected delivery may reclaim one tiny immediate deleted fragment when
  that fragment closes the sentence and no retry boundary separates the two pieces.

The pass uses source words/timestamps only. It never fabricates text or timestamps.
"""
from __future__ import annotations

from dataclasses import replace
import re
import unicodedata
from typing import Iterable

from .contracts import CandidateTake
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
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


def _concept(token: str) -> str:
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
    return {
        _concept(token)
        for token in _tokens(text)
        if len(token) >= 3 and token not in _STOP and _concept(token)
    }


def _coverage(left_text: str, right_text: str) -> tuple[int, float, float]:
    left = _content(left_text)
    right = _content(right_text)
    if not left or not right:
        return 0, 0.0, 0.0
    shared = len(left & right)
    return shared, shared / max(1, len(left)), shared / max(1, len(right))


def _critical(text: str) -> set[str]:
    raw = str(text or "")
    out = {f"num:{value}" for value in _NUMBER_RE.findall(raw)}
    if any(token in _NEGATION for token in _tokens(raw)):
        out.add("__negation__")
    return out


def _semantic_map(decisions: Iterable[tuple[str, str, float]]) -> dict[str, tuple[str, float]]:
    return {
        str(clip_id): (str(label), float(confidence))
        for clip_id, label, confidence in decisions
    }


def _is_open(take: CandidateTake) -> bool:
    tokens = _tokens(take.text)
    if not tokens:
        return False
    if tokens[-1] in _OPEN_TAIL:
        return True
    return str(take.text or "").strip().endswith((",", ":", ";", "-", "–", "—"))


def _retry_setup_between(
    left: CandidateTake,
    right: CandidateTake,
    context: WholeVideoContext | None,
) -> float:
    if context is None or left.source_asset_id != right.source_asset_id:
        return 0.0
    start = max(float(left.start), float(left.end) - 0.35)
    end = min(float(right.start), float(left.end) + 2.5)
    best = 0.0
    for source in context.sources:
        if source.source_asset_id != left.source_asset_id:
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


def suppress_open_failed_retry_prefixes(
    kept: Iterable[CandidateTake],
    deleted: Iterable[CandidateTake],
    semantic_decisions: Iterable[tuple[str, str, float]],
    *,
    maximum_suffix_gap_sec: float = 5.0,
    maximum_chain_sec: float = 14.0,
    maximum_prior_gap_sec: float = 45.0,
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    kept_tuple = tuple(sorted(kept, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    deleted_tuple = tuple(sorted(deleted, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    semantic = _semantic_map(semantic_decisions)
    removed_ids: set[str] = set()
    diagnostics: list[dict] = []

    for candidate in kept_tuple:
        if not (_is_open(candidate) or not candidate.complete_idea):
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
            if float(fragment.end) - float(candidate.end) > maximum_chain_sec:
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
            "reason": "open_failed_retry_prefix_yields_to_complete_prior_delivery",
            "removed_clip_id": candidate.clip_id,
            "prior_winner_clip_id": prior.clip_id,
            "failed_suffix_clip_ids": [fragment.clip_id for fragment in chain],
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


def restore_tiny_completion_suffixes(
    kept: Iterable[CandidateTake],
    deleted: Iterable[CandidateTake],
    semantic_decisions: Iterable[tuple[str, str, float]],
    context: WholeVideoContext | None,
    *,
    maximum_gap_sec: float = 1.8,
    maximum_fragment_sec: float = 2.5,
    maximum_fragment_tokens: int = 4,
) -> tuple[tuple[CandidateTake, ...], tuple[CandidateTake, ...], tuple[dict, ...]]:
    kept_tuple = tuple(sorted(kept, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    deleted_tuple = tuple(sorted(deleted, key=lambda t: (t.source_order, t.start, t.end, t.clip_id)))
    semantic = _semantic_map(semantic_decisions)
    restored_takes: dict[str, CandidateTake] = {}
    restored_ids: set[str] = set()
    diagnostics: list[dict] = []

    for candidate in kept_tuple:
        if candidate.complete_idea or _SENTENCE_END_RE.search(str(candidate.text or "").strip()):
            continue
        if len(_tokens(candidate.text)) < 4:
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
            tokens = _tokens(fragment.text)
            if not tokens or len(tokens) > maximum_fragment_tokens or fragment.duration_sec > maximum_fragment_sec:
                continue
            if not (fragment.complete_idea or _SENTENCE_END_RE.search(str(fragment.text or "").strip())):
                continue
            label, confidence = semantic.get(fragment.clip_id, ("", 0.0))
            if label == "bts" and confidence >= 0.84:
                continue
            if label == "failed" and confidence >= 0.94:
                continue
            if _retry_setup_between(candidate, fragment, context) >= 0.84:
                continue
            options.append((gap, confidence, fragment, label))

        if not options:
            continue
        gap, confidence, fragment, label = min(options, key=lambda item: (item[0], item[1], item[2].start))
        merged_text = f"{str(candidate.text or '').strip()} {str(fragment.text or '').strip()}".strip()
        # D-094.2: restored as its OWN kept take (identity preserved), never
        # merged into the candidate by text mutation -- see the same-named
        # comment in final_delivery_integrity.restore_immediate_completion_
        # fragments for the live evidence (run 33983880111) and why a
        # merged child cannot survive the CompositeResolver chain.
        restored_takes[fragment.clip_id] = fragment
        restored_ids.add(fragment.clip_id)
        diagnostics.append({
            "reason": "restore_tiny_completion_suffix",
            "clip_id": candidate.clip_id,
            "restored_fragment_id": fragment.clip_id,
            "restored_as": "separate_take",
            "gap_sec": round(gap, 3),
            "fragment_label": label,
            "fragment_confidence": round(confidence, 4),
            "restored_text": merged_text,
        })

    if not diagnostics:
        return kept_tuple, deleted_tuple, ()
    output = tuple(sorted(
        (*kept_tuple, *restored_takes.values()),
        key=lambda t: (t.source_order, t.start, t.end, t.clip_id),
    ))
    remaining_deleted = tuple(take for take in deleted_tuple if take.clip_id not in restored_ids)
    return output, remaining_deleted, tuple(diagnostics)


def install_terminal_delivery_reconciliation() -> None:
    from . import hybrid_session_cleanup

    original = hybrid_session_cleanup.apply_hybrid_session_cleanup
    if getattr(original, "_cutsell_terminal_delivery_reconciliation", False):
        return

    def apply_with_terminal_delivery_reconciliation(*args, **kwargs):
        source_takes = tuple(args[0]) if args else tuple(kwargs.get("takes") or ())
        context = args[1] if len(args) > 1 else kwargs.get("context")
        result = original(*args, **kwargs)
        if not result.kept or not result.semantic_decisions:
            return result

        kept, removed, suppress_diag = suppress_open_failed_retry_prefixes(
            result.kept,
            result.deleted,
            result.semantic_decisions,
        )
        deleted_ids = {take.clip_id for take in result.deleted}
        deleted_ids.update(take.clip_id for take in removed)
        deleted_pool = tuple(take for take in source_takes if take.clip_id in deleted_ids)

        kept, deleted_pool, restore_diag = restore_tiny_completion_suffixes(
            kept,
            deleted_pool,
            result.semantic_decisions,
            context,
        )
        if not suppress_diag and not restore_diag:
            return result

        remaining_deleted_ids = {take.clip_id for take in deleted_pool}
        deleted = tuple(take for take in source_takes if take.clip_id in remaining_deleted_ids)
        diagnostics = tuple(result.diagnostics) + ({
            "terminal_delivery_reconciliation": [*list(suppress_diag), *list(restore_diag)],
        },)
        return type(result)(
            kept=kept,
            deleted=deleted,
            requested_chunk_count=result.requested_chunk_count,
            available_chunk_count=result.available_chunk_count,
            diagnostics=diagnostics,
            semantic_decisions=result.semantic_decisions,
        )

    apply_with_terminal_delivery_reconciliation._cutsell_terminal_delivery_reconciliation = True
    hybrid_session_cleanup.apply_hybrid_session_cleanup = apply_with_terminal_delivery_reconciliation
