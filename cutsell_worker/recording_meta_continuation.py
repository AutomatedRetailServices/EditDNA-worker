"""Remove continuations of already-proven recording-process windows.

ASR can split behind-the-scenes thoughts across candidate boundaries. This module only
removes a survivor when the preceding discarded speech already proves recording-process
intent and the continuation is tightly coupled. It deliberately fails open around
viewer-facing CTAs and product speech.
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


def _direct_meta_short_tail(
    take: CandidateTake,
    discarded: tuple[CandidateTake, ...],
    *,
    maximum_gap_sec: float = 1.10,
) -> CandidateTake | None:
    """Return the direct-meta anchor for a tiny syntactic continuation, if any.

    A two- or three-word ASR tail such as ``with kids`` may be split from an explicit
    statement about making the video. We only remove it when the immediately preceding
    discarded candidate is itself an unambiguous direct recording-meta utterance. A
    longer sentence, a larger gap, or ordinary discarded speech always fails open.
    """
    tokens = _tokens(take.text)
    if not tokens or len(tokens) > 4 or take.duration_sec > 2.2:
        return None
    prior = [
        item for item in discarded
        if item.source_asset_id == take.source_asset_id
        and item.end <= take.start + 0.02
        and -0.02 <= take.start - item.end <= maximum_gap_sec
    ]
    if not prior:
        return None
    nearest = max(prior, key=lambda item: (item.end, item.start))
    return nearest if _is_direct_recording_meta(nearest) else None


def apply_recording_meta_continuation_cleanup(
    kept: Iterable[CandidateTake],
    discarded: Iterable[CandidateTake],
) -> tuple[Tuple[CandidateTake, ...], Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    kept_tuple = tuple(kept)
    discarded_tuple = tuple(discarded)
    survivors, removed, diagnostics = [], [], []

    for take in kept_tuple:
        direct_anchor = _direct_meta_short_tail(take, discarded_tuple)
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
