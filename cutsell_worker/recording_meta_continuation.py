"""Remove a contiguous continuation of an already-proven recording-process window.

ASR may split one behind-the-scenes thought across several candidates. Earlier cleanup
can correctly discard the first pieces while a final continuation survives because it
contains no standalone anchor. This rule only removes that continuation when it is
contiguous with a discarded chain, the chain contains a strong recording-process
anchor, and the survivor itself contains multiple recording-process verbs/terms.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision

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


def apply_recording_meta_continuation_cleanup(
    kept: Iterable[CandidateTake],
    discarded: Iterable[CandidateTake],
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    discarded_tuple = tuple(discarded)
    survivors, removed, diagnostics = [], [], []

    for take in kept_tuple:
        chain = _discarded_chain_before(take, discarded_tuple)
        strong_anchor = any(_STRONG_PROCESS_RE.search(str(item.text or "")) for item in chain)
        should_remove = (
            take.duration_sec <= 6.0
            and len(chain) >= 2
            and strong_anchor
            and _process_term_count(take.text) >= 2
        )
        if not should_remove:
            survivors.append(take)
            continue
        removed.append(take)
        diagnostics.append({
            "clip_id": take.clip_id,
            "reason": "recording_process_continuation_after_discarded_meta_chain",
            "text": take.text,
            "anchor_clip_ids": [item.clip_id for item in chain],
        })

    return tuple(survivors), tuple(removed), tuple(diagnostics)


def install_recording_meta_continuation_cleanup() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_recording_meta_continuation", False):
        return

    def apply_with_recording_meta_continuation(takes, context=None):
        kept, discarded, decisions = original(takes, context)
        kept, extra_discarded, diagnostics = apply_recording_meta_continuation_cleanup(kept, discarded)
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
