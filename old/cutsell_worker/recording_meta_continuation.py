"""Remove continuations of already-proven recording-process windows.

ASR can split behind-the-scenes thoughts across candidate boundaries. This module only
removes a survivor when neighboring speech already proves recording-process intent and
the continuation is tightly coupled. It deliberately fails open around viewer-facing
CTAs and product speech.

Detection may use the original take set as evidence because earlier cleaners can remove
anchors before this wrapper executes. Destructive action remains limited to survivors.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .recording_process_context import _is_direct_recording_meta

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_STRONG_PROCESS_RE = re.compile(
    r"\bi\s+(?:do\s+not|don't|dont)\s+know\s+how\s+to\s+end\b|"
    r"\bcall\s+to\s+action\b|"
    r"\b(?:whole|full)\s+sentence\b|"
    r"\b(?:start\s+over|redo\s+that|do\s+that\s+again)\b|"
    r"\bi\s+hate\s+(?:saying|being)\b",
    re.IGNORECASE,
)
_PROCESS_TERMS = frozenset({
    "end", "ending", "say", "saying", "said", "stop", "stopping", "script",
    "line", "take", "recording", "video", "videos", "cta",
})
_SAY_TERMS = frozenset({"say", "saying", "said"})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold().replace("'", "") for token in _TOKEN_RE.findall(str(text or "")))


def _process_term_count(text: str) -> int:
    return len(set(_tokens(text)) & _PROCESS_TERMS)


def _discarded_chain_before(
    take: CandidateTake,
    discarded: tuple[CandidateTake, ...],
    *,
    maximum_gap_sec: float = 0.20,
    maximum_span_sec: float = 16.0,
) -> tuple[CandidateTake, ...]:
    candidates = sorted(
        (
            item for item in discarded
            if item.source_asset_id == take.source_asset_id and item.end <= take.start + 0.02
        ),
        key=lambda item: (item.end, item.start),
        reverse=True,
    )
    if not candidates:
        return ()
    nearest = candidates[0]
    if not -0.02 <= take.start - nearest.end <= maximum_gap_sec:
        return ()

    chain = [nearest]
    current = nearest
    for item in candidates[1:]:
        gap = current.start - item.end
        if gap < -0.02 or gap > maximum_gap_sec:
            break
        if take.start - item.start > maximum_span_sec:
            break
        chain.append(item)
        current = item
    return tuple(reversed(chain))


def _nearest_prior_direct_meta(
    take: CandidateTake,
    evidence: tuple[CandidateTake, ...],
    *,
    maximum_gap_sec: float,
) -> CandidateTake | None:
    prior = [
        item for item in evidence
        if item.clip_id != take.clip_id
        and item.source_asset_id == take.source_asset_id
        and item.end <= take.start + 0.02
        and -0.02 <= take.start - item.end <= maximum_gap_sec
        and _is_direct_recording_meta(item)
    ]
    return max(prior, key=lambda item: (item.end, item.start)) if prior else None


def _legacy_tiny_continuation(take: CandidateTake) -> bool:
    """D-171: the original `_direct_meta_short_tail` length/duration
    heuristic, extracted unchanged -- "is this take short enough (<=5
    words, <=2.2s) to plausibly be a trailing fragment." Kept as its own
    function so `_direct_meta_short_tail` below can fold it against
    canonical Language Spine evidence without duplicating it."""
    tokens = _tokens(take.text)
    return bool(tokens) and len(tokens) <= 5 and take.duration_sec <= 2.2


def _direct_meta_short_tail(
    take: CandidateTake,
    evidence: tuple[CandidateTake, ...],
    *,
    maximum_gap_sec: float = 1.10,
) -> CandidateTake | None:
    """Return direct-meta anchor for a tiny syntactic continuation, if any.

    D-171 Language Spine Phase D, TARGET B: the "is this a tiny
    continuation" verdict now consults canonical Language Spine evidence
    (`LanguageAttempt.meaning_completion`, built from this take's own ASR
    word timing) alongside the legacy word-count/duration heuristic, via
    `language_spine_consumer_migration.continuation_migration`. Per that
    module's fail-open contract, the returned verdict is ALWAYS either the
    legacy value, or a value proven identical to it for this take -- this
    call site's editorial behavior is provably unchanged by D-171; only the
    evidence source backing an agreeing verdict differs. When Spine
    evidence disagrees or is unavailable (e.g. no word timing on `take`),
    the pre-D-171 legacy verdict is used as-is.
    """
    legacy_is_tiny = _legacy_tiny_continuation(take)
    from . import language_spine_consumer_migration
    is_tiny, _trace = language_spine_consumer_migration.continuation_migration(
        "recording_meta_continuation._direct_meta_short_tail",
        take.source_asset_id, take.words, legacy_is_tiny,
    )
    if not is_tiny:
        return None
    return _nearest_prior_direct_meta(take, evidence, maximum_gap_sec=maximum_gap_sec)


def _process_heavy_continuation_after_direct_meta(
    take: CandidateTake,
    evidence: tuple[CandidateTake, ...],
    *,
    maximum_gap_sec: float = 6.0,
) -> CandidateTake | None:
    """Detect BTS continuation after a proven direct recording-meta anchor.

    Require at least two process terms plus explicit self-direction language about what
    is being said. This protects viewer-facing copy that happens to mention words like
    ``video``, ``stop`` or ``end``.
    """
    tokens = set(_tokens(take.text))
    if take.duration_sec > 6.5 or _process_term_count(take.text) < 2:
        return None
    if not tokens.intersection(_SAY_TERMS):
        return None
    return _nearest_prior_direct_meta(take, evidence, maximum_gap_sec=maximum_gap_sec)


def apply_recording_meta_continuation_cleanup(
    kept: Iterable[CandidateTake],
    discarded: Iterable[CandidateTake],
    *,
    evidence_takes: Iterable[CandidateTake] | None = None,
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    discarded_tuple = tuple(discarded)
    evidence_tuple = tuple(evidence_takes) if evidence_takes is not None else tuple(kept_tuple + discarded_tuple)
    survivors, removed, diagnostics = [], [], []

    for take in kept_tuple:
        direct_anchor = _direct_meta_short_tail(take, evidence_tuple)
        if direct_anchor is not None:
            removed.append(take)
            diagnostics.append({
                "clip_id": take.clip_id,
                "reason": "short_continuation_after_direct_recording_meta",
                "text": take.text,
                "anchor_clip_ids": [direct_anchor.clip_id],
            })
            continue

        chain = _discarded_chain_before(take, discarded_tuple)
        strong_anchor = any(_STRONG_PROCESS_RE.search(str(item.text or "")) for item in chain)
        should_remove = (
            take.duration_sec <= 6.0
            and len(chain) >= 2
            and strong_anchor
            and _process_term_count(take.text) >= 2
        )
        if should_remove:
            removed.append(take)
            diagnostics.append({
                "clip_id": take.clip_id,
                "reason": "recording_process_continuation_after_discarded_meta_chain",
                "text": take.text,
                "anchor_clip_ids": [item.clip_id for item in chain],
            })
            continue

        process_anchor = _process_heavy_continuation_after_direct_meta(take, evidence_tuple)
        if process_anchor is not None:
            removed.append(take)
            diagnostics.append({
                "clip_id": take.clip_id,
                "reason": "process_heavy_continuation_after_direct_recording_meta",
                "text": take.text,
                "anchor_clip_ids": [process_anchor.clip_id],
            })
            continue

        survivors.append(take)

    return tuple(survivors), tuple(removed), tuple(diagnostics)


def install_recording_meta_continuation_cleanup() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_recording_meta_continuation", False):
        return

    def apply_with_recording_meta_continuation(takes, context=None):
        take_tuple = tuple(takes)
        kept, discarded, decisions = original(take_tuple, context)
        kept, extra_discarded, diagnostics = apply_recording_meta_continuation_cleanup(
            kept,
            discarded,
            evidence_takes=take_tuple,
        )
        if not extra_discarded:
            return kept, discarded, decisions
        reason_by_id = {item["clip_id"]: item["reason"] for item in diagnostics}
        extra = tuple(
            CleanCutDecision(take.clip_id, False, reason_by_id[take.clip_id], 0.97)
            for take in extra_discarded
        )
        return kept, tuple(discarded) + tuple(extra_discarded), tuple(decisions) + extra

    apply_with_recording_meta_continuation._cutsell_recording_meta_continuation = True
    clean_cut.apply_clean_cut = apply_with_recording_meta_continuation
