"""Prefer a clean internal retake over an earlier broken delivery inside one candidate.

Best-Take can only compare separate candidates. In raw talking-head recordings ASR may
merge a failed delivery, a visible retry/reset, and the clean retake into one CandidateTake.
This pass detects that structure conservatively and keeps the later retake.

The pass normally requires a strong recording-process retry/failure event plus lexical
coverage. Round 5 exposed a second real structure: the visual/performance analyzer can
prove a fumble on the merged take while failing to emit one authoritative retry event at
the exact restart boundary. When the creator then repeats a long opening phrase verbatim,
that repeated opening itself is strong retry structure. In that narrow case we allow the
lexical restart plus high visual-fumble evidence to define the split.

It never cuts through a spoken word and fails open when evidence is ambiguous.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable, Tuple

from .contracts import CandidateTake, CleanCutDecision
from .whole_video_analysis import WholeVideoContext

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)
_STOP = frozenset({
    "a", "al", "and", "are", "as", "at", "be", "but", "by", "como", "con", "de", "del",
    "el", "en", "es", "esta", "este", "for", "from", "in", "is", "it", "la", "las", "lo",
    "los", "me", "mi", "mis", "of", "on", "or", "para", "pero", "por", "que", "se", "so",
    "su", "sus", "that", "the", "this", "to", "un", "una", "was", "we", "with", "y", "yo",
})
_RETRY_EVENT_KINDS = frozenset({
    "retry_setup",
    "false_start",
    "wrong_take",
    "searching_for_words",
    "breaking_character",
    "unintentional_dead_air",
})
_RESET_EVENT_KINDS = frozenset({
    "body_reset_candidate",
    "camera_disengagement_candidate",
    "facial_expression_shift_candidate",
})


def _kind(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _norm(value: str) -> str:
    found = _TOKEN_RE.findall(str(value or "").casefold())
    return found[0] if found else ""


def _content_words(words) -> set[str]:
    out = set()
    for word in words:
        token = _norm(getattr(word, "text", ""))
        if len(token) >= 3 and token not in _STOP:
            out.add(token)
    return out


def _source_events(context: WholeVideoContext | None, source_asset_id: str):
    if context is None:
        return ()
    for source in context.sources:
        if source.source_asset_id == source_asset_id:
            return tuple(source.events)
    return ()


def _retry_boundaries(take: CandidateTake, context: WholeVideoContext | None):
    events = []
    for event in _source_events(context, take.source_asset_id):
        if event.end < take.start or event.start > take.end:
            continue
        kind = _kind(event.kind)
        confidence = float(event.confidence)
        if kind in _RETRY_EVENT_KINDS and confidence >= 0.78:
            events.append((event, "authoritative"))
        elif kind in _RESET_EVENT_KINDS and confidence >= (
            0.90 if kind == "body_reset_candidate" else 0.80
        ):
            events.append((event, "reset"))
    return tuple(events)


def _split_index_for_event(words, event) -> int | None:
    """Find the first word that begins after the retry/reset event midpoint."""
    if not words:
        return None
    pivot = (float(event.start) + float(event.end)) / 2.0
    for index, word in enumerate(words):
        if float(word.start) >= pivot:
            return index
    return None


def _overlap(left_words, right_words) -> tuple[int, float, float]:
    left = _content_words(left_words)
    right = _content_words(right_words)
    shared = len(left & right)
    left_cov = shared / max(1, len(left))
    right_cov = shared / max(1, len(right))
    return shared, left_cov, right_cov


def _repeated_opening_split(
    words,
    *,
    minimum_ngram_words: int = 5,
    maximum_ngram_words: int = 8,
    minimum_attempt_gap_words: int = 3,
) -> tuple[int, int, tuple[str, ...]] | None:
    """Find a later verbatim opening that restarts an earlier delivery.

    This is deliberately not generic repetition removal. We require a long contiguous
    phrase (5-8 words) to occur twice with enough material between the occurrences. The
    caller separately requires strong visual-fumble evidence before using this boundary.
    """
    normalized = tuple(_norm(getattr(word, "text", "")) for word in words)
    if len(normalized) < minimum_ngram_words * 2 + minimum_attempt_gap_words:
        return None

    for size in range(maximum_ngram_words, minimum_ngram_words - 1, -1):
        if len(normalized) < size * 2 + minimum_attempt_gap_words:
            continue
        first_seen: dict[tuple[str, ...], int] = {}
        for index in range(0, len(normalized) - size + 1):
            phrase = normalized[index : index + size]
            if any(not token for token in phrase):
                continue
            previous = first_seen.get(phrase)
            if previous is None:
                first_seen[phrase] = index
                continue
            if index - (previous + size) < minimum_attempt_gap_words:
                continue
            if len(normalized) - index < size + 3:
                continue
            return index, previous, phrase
    return None


def _lexical_visual_retry_split(take: CandidateTake, words) -> tuple[int, dict] | None:
    signals = take.signals
    if signals is None or float(signals.visual_fumble) < 0.65:
        return None
    found = _repeated_opening_split(words)
    if found is None:
        return None
    split, previous, phrase = found
    if split < 4 or len(words) - split < 5:
        return None

    # The repeated opening itself is the structural retry evidence. Do not require a
    # second nearby-window overlap check: Round 5 showed that such a window can start
    # after the first opening and therefore discard the very evidence that located the
    # retry. Instead require the verbatim opening to contain at least three meaningful
    # content words and verify that those words are present at the start of the retained
    # right-side delivery. Combined with visual_fumble >= 0.65 and the long contiguous
    # repetition above, this remains substantially narrower than generic repetition.
    phrase_content = {
        token for token in phrase
        if len(token) >= 3 and token not in _STOP
    }
    if len(phrase_content) < 3:
        return None
    right_content = _content_words(words[split : min(len(words), split + 18)])
    shared = len(phrase_content & right_content)
    phrase_coverage = shared / max(1, len(phrase_content))
    right_coverage = shared / max(1, len(right_content))
    if shared < 3 or phrase_coverage < 0.80:
        return None

    return split, {
        "evidence_type": "repeated_opening_plus_visual_fumble",
        "visual_fumble": round(float(signals.visual_fumble), 4),
        "repeated_phrase": " ".join(phrase),
        "first_phrase_index": previous,
        "shared_content_tokens": shared,
        "left_coverage": round(phrase_coverage, 4),
        "right_coverage": round(right_coverage, 4),
    }


def prefer_internal_clean_retakes(
    kept: Iterable[CandidateTake],
    context: WholeVideoContext | None,
    *,
    minimum_right_words: int = 5,
    minimum_shared_content: int = 3,
    minimum_directional_coverage: float = 0.45,
) -> tuple[Tuple[CandidateTake, ...], Tuple[dict, ...]]:
    output = []
    diagnostics = []

    for take in kept:
        words = tuple(sorted(take.words, key=lambda word: (float(word.start), float(word.end))))
        if len(words) < 10:
            output.append(take)
            continue

        candidates = []
        for event, evidence_type in _retry_boundaries(take, context):
            split = _split_index_for_event(words, event)
            if split is None or split < 4 or len(words) - split < minimum_right_words:
                continue

            left_start = max(0, split - 12)
            right_end = min(len(words), split + 14)
            left_window = words[left_start:split]
            right_window = words[split:right_end]
            shared, left_cov, right_cov = _overlap(left_window, right_window)
            if shared < minimum_shared_content:
                continue
            if max(left_cov, right_cov) < minimum_directional_coverage:
                continue

            priority = 2 if evidence_type == "authoritative" else 1
            score = (priority, shared, max(left_cov, right_cov), float(event.confidence))
            candidates.append((score, split, event, shared, left_cov, right_cov, evidence_type))

        if candidates:
            _, split, event, shared, left_cov, right_cov, evidence_type = max(candidates, key=lambda item: item[0])
            lexical_detail = None
        else:
            lexical = _lexical_visual_retry_split(take, words)
            if lexical is None:
                output.append(take)
                continue
            split, lexical_detail = lexical
            event = None
            shared = int(lexical_detail["shared_content_tokens"])
            left_cov = float(lexical_detail["left_coverage"])
            right_cov = float(lexical_detail["right_coverage"])
            evidence_type = str(lexical_detail["evidence_type"])

        right_words = words[split:]
        new_start = float(right_words[0].start)
        if new_start <= take.start + 0.25 or take.end - new_start < 1.0:
            output.append(take)
            continue

        text = " ".join(str(word.text or "").strip() for word in right_words).strip()
        if not text:
            output.append(take)
            continue

        child = replace(
            take,
            start=new_start,
            text=text,
            words=right_words,
            signals=(replace(take.signals, start=new_start) if take.signals is not None else None),
        )
        output.append(child)
        item = {
            "clip_id": take.clip_id,
            "reason": "internal_broken_attempt_yields_to_clean_retake",
            "evidence_type": evidence_type,
            "original_start": round(float(take.start), 3),
            "retake_start": round(new_start, 3),
            "shared_content_tokens": shared,
            "left_coverage": round(left_cov, 4),
            "right_coverage": round(right_cov, 4),
            "kept_text": text,
        }
        if event is not None:
            item.update({
                "event_kind": _kind(event.kind),
                "event_confidence": round(float(event.confidence), 4),
            })
        elif lexical_detail is not None:
            item.update({
                "event_kind": "lexical_restart",
                "event_confidence": float(lexical_detail["visual_fumble"]),
                "repeated_phrase": lexical_detail["repeated_phrase"],
            })
        diagnostics.append(item)

    return tuple(output), tuple(diagnostics)


def install_internal_retake_winner() -> None:
    from . import clean_cut

    original = clean_cut.apply_clean_cut
    if getattr(original, "_cutsell_internal_retake_winner", False):
        return

    def apply_with_internal_retake_winner(takes, context=None):
        take_tuple = tuple(takes)
        kept, discarded, decisions = original(take_tuple, context)
        resolved, diagnostics = prefer_internal_clean_retakes(kept, context)
        if not diagnostics:
            return kept, discarded, decisions
        extra = tuple(
            CleanCutDecision(
                clip_id=str(item["clip_id"]),
                keep=True,
                reason=str(item["reason"]),
                confidence=0.99,
            )
            for item in diagnostics
        )
        return resolved, discarded, tuple(decisions) + extra

    apply_with_internal_retake_winner._cutsell_internal_retake_winner = True
    clean_cut.apply_clean_cut = apply_with_internal_retake_winner
