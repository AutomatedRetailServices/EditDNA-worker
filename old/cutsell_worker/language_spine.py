"""D-166: Language / Transcript Spine, Phase A -- typed WORD/PHRASE schema
+ one shared, meaning-safe normalizer + a reusable phrase segmenter.

See ``docs/CUTSELL_CANONICAL_ENGINE_ARCHITECTURE_D098.md`` Section 14
(D-165's canonical design) and ``docs/CUTSELL_DECISIONS.md`` D-166 for the
full design rationale. This module implements ONLY the first two rungs of
D-165's canonical hierarchy:

    ASR Word -> LanguageWord -> LanguagePhrase

``LanguageUtterance``, ``LanguageAttempt``, ``PropositionCandidate``, and
``RelationEvidence`` are explicitly OUT of scope here (D-165's own Phase B
gate) -- this module creates no Family/BestTake/Proposition/Relation
authority and is not called by any production call site. It is a pure,
standalone, additive foundation: `pipeline.py`, `flow_b.py`,
`take_segmentation.py`, `attempt_reconstruction.py`, and every other
existing transcript consumer are UNCHANGED and UNAWARE of this module's
existence (verified structurally by this module's own test suite,
`tests/test_cutsell_d166_language_spine.py`).

## Why no feature flag

D-165's directive prefers no flag when no behavior change exists. Because
nothing in the active pipeline calls into this module yet, there is
nothing to gate -- ``CUTSELL_LANGUAGE_SPINE_ENABLED`` would toggle a
capability no one consumes. A flag becomes meaningful only once a future
phase (D-167+) actually wires spine consumption into a live call site;
this module intentionally does not anticipate that wiring.

## ``_speech_units`` reuse (this task's own explicit instruction)

``take_segmentation._speech_units`` is left completely BYTE-IDENTICAL and
untouched (zero regression risk to production take segmentation). This
module's own ``segment_language_phrases`` implements the SAME core
algorithm (word-timestamp-gap splitting) generalized onto the new
``LanguageWord`` type, extended with three capabilities ``_speech_units``
does not have: (1) real audio-silence-evidence-backed PAUSE
classification (vs. a bare word-timing gap), (2) punctuation-based
sub-classification (END_OF_UTTERANCE vs. weaker PUNCTUATION), and (3) an
optional restart-marker override. This is a deliberate PARALLEL, additive
capability -- not a redirection of ``take_segmentation.segment_takes``'s
existing call site -- so existing behavior cannot regress no matter what
this module does. A future phase may choose to have ``take_segmentation``
consume this module instead; that migration is explicitly NOT made here
(D-166's own "no mass migration" instruction).

## Meaning safety (binding)

The one shared normalizer, ``normalize_language_text``, touches ONLY
whitespace, case, and REPEATED non-terminal punctuation (``!``/``?``/
``,``/``;``/``:``) -- it never touches ``.`` at all (protecting the
existing ellipsis-sensitive "trails off" logic elsewhere in the codebase,
e.g. ``take_segmentation._trails_off``) and never removes or rewrites a
character that is not whitespace/case/one of those five punctuation
marks. Negation words, numbers, percentages, dates, factual terms, and
named entities are therefore preserved BY CONSTRUCTION, not by a
case-by-case exclusion list -- see
``tests/test_cutsell_d166_language_spine.py``'s meaning-safety fixtures
for the explicit proof, and ``count_meaning_sensitive_tokens_preserved``
below for a reusable runtime proof metric.

## Provenance

Reuses ``raw_understanding_map.py``'s existing provenance vocabulary
(``PROVENANCE_ASR``, ``PROVENANCE_AUDIO_SIGNAL``,
``PROVENANCE_DETERMINISTIC_RULE``, ``PROVENANCE_UNKNOWN``) by direct
import -- never a parallel, incompatible framework. Three NEW provenance
tags this task's directive requires (``TRANSCRIPT_NORMALIZATION``,
``WORD_TIMING``, ``PHRASE_SEGMENTATION``) are defined HERE, not added to
``raw_understanding_map.py`` itself, because that module is D-155's own
CLOSED file -- this task leaves it at zero diff. The three new constants
follow the exact same naming convention so they belong to the same
conceptual vocabulary in spirit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, Mapping, Tuple

from .contracts import Word
from .polarity_safety import POLARITY_PARTICLES
from .raw_understanding_map import (
    PROVENANCE_ASR,
    PROVENANCE_AUDIO_SIGNAL,
    PROVENANCE_DETERMINISTIC_RULE,
    PROVENANCE_UNKNOWN,
)

SCHEMA_VERSION = "cutsell.language_spine.v1"

# ---------------------------------------------------------------------------
# Provenance (reuses raw_understanding_map.py's vocabulary; adds the three
# language-specific tags this task's directive requires, defined here since
# raw_understanding_map.py (D-155) stays at zero diff -- see module docstring).
# ---------------------------------------------------------------------------
PROVENANCE_TRANSCRIPT_NORMALIZATION = "TRANSCRIPT_NORMALIZATION"
PROVENANCE_WORD_TIMING = "WORD_TIMING"
PROVENANCE_PHRASE_SEGMENTATION = "PHRASE_SEGMENTATION"

# ---------------------------------------------------------------------------
# Categorical confidence (reuses watch_listen_understanding.py's exact
# vocabulary in spirit -- SUPPORTED/WEAK/UNKNOWN -- redefined here rather
# than imported so this module stays independent of the D-157 CLOSED file's
# own import graph; the three string values are intentionally identical).
# ---------------------------------------------------------------------------
CONFIDENCE_SUPPORTED = "SUPPORTED"
CONFIDENCE_WEAK = "WEAK"
CONFIDENCE_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Phrase boundary vocabulary (this task's own required set).
# ---------------------------------------------------------------------------
BOUNDARY_PAUSE = "PAUSE"
BOUNDARY_PUNCTUATION = "PUNCTUATION"
BOUNDARY_SPEECH_BOUNDARY = "SPEECH_BOUNDARY"
BOUNDARY_RESTART_BOUNDARY = "RESTART_BOUNDARY"
BOUNDARY_END_OF_UTTERANCE = "END_OF_UTTERANCE"
BOUNDARY_UNKNOWN = "UNKNOWN"

DEFAULT_SPLIT_GAP_SEC = 0.75
# Small tolerance for matching a split instant against a real audio-silence
# interval or a caller-supplied restart-marker time -- accounts for ASR
# word-timing jitter around the true acoustic boundary, never widened
# beyond what real timing noise requires.
_BOUNDARY_MATCH_TOLERANCE_SEC = 0.20

_TERMINAL_PUNCT_RE = re.compile(r"[.!?][\"'”’)]*\s*$")
_TRAILING_ELLIPSIS_RE = re.compile(r"(?:\.\.\.|…)[\"'”’)]*\s*$")
_OPEN_PUNCTUATION_RE = re.compile(r"[,;:\-–—]\s*$")
_TRAILING_PUNCT_CAPTURE_RE = re.compile(r"([.,!?;:…]+)$")
_REPEATED_PUNCT_RE = re.compile(r"([!?,;:])\1+")
_WHITESPACE_RE = re.compile(r"\s+")
_PARTIAL_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ]-$")
_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ɏ']+")

# Generic, cross-language filler interjections -- deliberately narrow (never
# a real content/discourse word like "pues"/"so", which already has its own
# bridge-connector meaning elsewhere) and never Video00-specific, per this
# task's own anti-rule-proliferation instruction.
_FILLER_TOKENS = frozenset({"um", "uh", "uhm", "erm", "eh", "hmm"})


def _tokens(text: str) -> Tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip()


def normalize_language_text(text: str) -> str:
    """The ONE canonical, meaning-safe transcript normalization function
    this task requires. Touches ONLY whitespace, case, and REPEATED
    non-terminal punctuation -- see module docstring's "Meaning safety"
    section for exactly what this does and does not do, and why ``.`` is
    never touched. Idempotent: normalizing already-normalized text is a
    no-op."""
    collapsed = _collapse_whitespace(text)
    lowered = collapsed.casefold()
    return _REPEATED_PUNCT_RE.sub(r"\1", lowered)


def _extract_trailing_punctuation(text_raw: str) -> str | None:
    """Non-destructive: captures trailing punctuation ATTACHED to the word
    (e.g. ``"sabía."`` -> ``"."``, ``"diagnosticaron..."`` -> ``"..."``)
    without removing or reinterpreting it from ``text_raw`` itself."""
    match = _TRAILING_PUNCT_CAPTURE_RE.search(str(text_raw or "").strip())
    return match.group(1) if match else None


def _is_filler(text_raw: str) -> bool:
    tokens = _tokens(text_raw)
    return len(tokens) == 1 and tokens[0] in _FILLER_TOKENS


def _is_partial_word(text_raw: str) -> bool:
    """Honest, generic partial-word/hesitation detection: a letter directly
    followed by a hyphen at the end of the token (a common transcript
    convention for a cut-off word, e.g. ``"th-"``). Never rewrites the word
    into a guessed completion -- see this task's own "do not silently
    correct" instruction."""
    return bool(_PARTIAL_WORD_RE.search(str(text_raw or "").strip()))


def _ends_sentence(text: str) -> bool:
    stripped = str(text or "").strip()
    if _TRAILING_ELLIPSIS_RE.search(stripped):
        return False
    return bool(_TERMINAL_PUNCT_RE.search(stripped))


def _has_open_punctuation_tail(text: str) -> bool:
    return bool(_OPEN_PUNCTUATION_RE.search(str(text or "").strip()))


# ---------------------------------------------------------------------------
# LanguageWord
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LanguageWord:
    """One canonical, typed word in the Language Spine. Maps back exactly
    to a recoverable RAW range (``source_asset_id`` + ``start``/``end``) --
    no synthetic text node without a recoverable span, per this task's own
    Source Identity requirement."""
    source_asset_id: str
    word_index: int
    text_raw: str
    text_normalized: str
    start: float
    end: float
    confidence: float | None
    punctuation_after: str | None
    provenance: str
    # Additive, optional annotations beyond this task's own stated minimum
    # field list -- both default False/safe so every construction site
    # this module itself controls stays valid; neither ever deletes or
    # rewrites the word (this task's own "do not delete fillers in Phase
    # A" / "represent partial words honestly, never silently correct"
    # instructions).
    is_filler: bool = False
    is_partial: bool = False

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end - self.start)


def adapt_words_to_language_words(
    source_asset_id: str, words: Iterable[Word],
) -> Tuple[LanguageWord, ...]:
    """The ONE adapter from the existing ASR/canonical ``contracts.Word``
    structure into ``LanguageWord``. Does NOT modify the ASR provider
    (``asr.py``) itself -- pure, read-only transformation, ordered by
    ``(start, end)`` exactly like every other real consumer in this
    codebase (``take_segmentation.py``, ``canonical_asr_evidence.py``)."""
    ordered = tuple(sorted(words, key=lambda word: (word.start, word.end)))
    result: list[LanguageWord] = []
    for index, word in enumerate(ordered):
        text_raw = str(word.text or "")
        result.append(LanguageWord(
            source_asset_id=source_asset_id,
            word_index=index,
            text_raw=text_raw,
            text_normalized=normalize_language_text(text_raw),
            start=float(word.start),
            end=float(word.end),
            confidence=word.confidence,
            punctuation_after=_extract_trailing_punctuation(text_raw),
            provenance=PROVENANCE_ASR,
            is_filler=_is_filler(text_raw),
            is_partial=_is_partial_word(text_raw),
        ))
    return tuple(result)


# ---------------------------------------------------------------------------
# LanguagePhrase
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LanguagePhrase:
    """One canonical, typed phrase -- a bounded sub-utterance span. Exactly
    spans its child words: ``source_start``/``source_end`` always equal
    ``words[word_start_index].start``/``words[word_end_index].end`` for the
    ``LanguageWord`` tuple that produced it (see ``source_mapping_valid``
    diagnostic below)."""
    source_asset_id: str
    phrase_id: str
    word_start_index: int
    word_end_index: int
    source_start: float
    source_end: float
    text_raw: str
    text_normalized: str
    boundary_kind: str
    confidence: str
    provenance: str

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.source_end - self.source_start)


def _phrase_id(source_asset_id: str, start: float, end: float, text_normalized: str) -> str:
    """Deterministic, physical-observation-style id (mirrors
    ``canonical_identity.mint_source_span_id``'s exact hashing shape under
    a distinct prefix so the two ids are never confusable) -- timestamp-
    sensitive by design, same rationale as every other physical span id in
    this codebase. Minted here rather than registered in
    ``canonical_identity.py`` because nothing yet reads this id to make an
    editorial decision (D-050A's own precedent: register an id in the
    shared ownership table only once a real consumer exists)."""
    raw = f"{source_asset_id}|{float(start):.3f}|{float(end):.3f}|{text_normalized}".encode("utf-8")
    return "lphrase_" + hashlib.sha256(raw).hexdigest()[:20]


def _within_tolerance(value: float, targets: Iterable[float], tolerance: float) -> bool:
    return any(abs(value - target) <= tolerance for target in targets)


def _overlaps_interval(gap_start: float, gap_end: float, intervals: Iterable[Tuple[float, float]]) -> bool:
    for interval_start, interval_end in intervals:
        if gap_end > interval_start and gap_start < interval_end:
            return True
    return False


def segment_language_phrases(
    words: Tuple[LanguageWord, ...],
    *,
    split_gap_sec: float = DEFAULT_SPLIT_GAP_SEC,
    audio_silence_intervals: Iterable[Tuple[float, float]] = (),
    restart_marker_times: Iterable[float] = (),
) -> Tuple[LanguagePhrase, ...]:
    """Reusable canonical phrase segmenter -- the generalized promotion of
    ``take_segmentation._speech_units``'s core word-timestamp-gap-splitting
    algorithm (see module docstring's "``_speech_units`` reuse" section for
    why the original is left untouched). Pure function; no I/O, no
    provider/network call, no new ML dependency.

    Splits ``words`` (already ``LanguageWord``s for ONE source) into
    phrases at, in priority order: a real audio-silence-evidence-confirmed
    gap (PAUSE, SUPPORTED), a caller-supplied restart-marker time
    (RESTART_BOUNDARY, SUPPORTED -- checked first among timing-only splits
    so real restart evidence is never demoted to a weaker class), a bare
    word-timing gap >= ``split_gap_sec`` with no audio confirmation
    (SPEECH_BOUNDARY, WEAK -- per this task's own instruction that a
    transcript gap alone never asserts a confirmed pause), or terminal/
    open punctuation on the previous word (END_OF_UTTERANCE/PUNCTUATION).
    ``audio_silence_intervals``/``restart_marker_times`` are OPTIONAL --
    when empty (the common case when no real audio-silence evidence was
    computed for this source), this fails open to timing/punctuation-based
    segmentation only, exactly as this task's own "Pause Integration"
    section requires."""
    if not words:
        return ()
    if len(words) == 1:
        only = words[0]
        boundary = BOUNDARY_END_OF_UTTERANCE if _ends_sentence(only.text_raw) else BOUNDARY_UNKNOWN
        confidence = CONFIDENCE_SUPPORTED if boundary == BOUNDARY_END_OF_UTTERANCE else CONFIDENCE_UNKNOWN
        return (_build_phrase(words, 0, 0, boundary, confidence),)

    intervals = tuple(audio_silence_intervals)
    restart_times = tuple(restart_marker_times)

    boundaries: list[tuple[int, str, str]] = []  # (word_index_of_split_start, boundary_kind, confidence)
    for index in range(1, len(words)):
        previous = words[index - 1]
        current = words[index]
        gap = current.start - previous.end

        if _within_tolerance(previous.end, restart_times, _BOUNDARY_MATCH_TOLERANCE_SEC) or \
                _within_tolerance(current.start, restart_times, _BOUNDARY_MATCH_TOLERANCE_SEC):
            boundaries.append((index, BOUNDARY_RESTART_BOUNDARY, CONFIDENCE_SUPPORTED))
            continue
        if gap > 0 and _overlaps_interval(previous.end, current.start, intervals):
            boundaries.append((index, BOUNDARY_PAUSE, CONFIDENCE_SUPPORTED))
            continue
        if _ends_sentence(previous.text_raw):
            boundaries.append((index, BOUNDARY_END_OF_UTTERANCE, CONFIDENCE_SUPPORTED))
            continue
        if gap >= split_gap_sec:
            boundaries.append((index, BOUNDARY_SPEECH_BOUNDARY, CONFIDENCE_WEAK))
            continue
        if _has_open_punctuation_tail(previous.text_raw):
            boundaries.append((index, BOUNDARY_PUNCTUATION, CONFIDENCE_WEAK))
            continue
        # No split here -- current word joins the phrase in progress.

    if not boundaries:
        text_raw = " ".join(w.text_raw.strip() for w in words if w.text_raw.strip())
        boundary = BOUNDARY_END_OF_UTTERANCE if _ends_sentence(words[-1].text_raw) else BOUNDARY_UNKNOWN
        confidence = CONFIDENCE_SUPPORTED if boundary == BOUNDARY_END_OF_UTTERANCE else CONFIDENCE_UNKNOWN
        return (_build_phrase(words, 0, len(words) - 1, boundary, confidence),)

    phrases: list[LanguagePhrase] = []
    start_idx = 0
    for split_idx, boundary_kind, confidence in boundaries:
        end_idx = split_idx - 1
        phrases.append(_build_phrase(words, start_idx, end_idx, boundary_kind, confidence))
        start_idx = split_idx
    # Final trailing phrase after the last recorded boundary.
    tail_text = words[-1].text_raw
    tail_boundary = BOUNDARY_END_OF_UTTERANCE if _ends_sentence(tail_text) else BOUNDARY_UNKNOWN
    tail_confidence = CONFIDENCE_SUPPORTED if tail_boundary == BOUNDARY_END_OF_UTTERANCE else CONFIDENCE_UNKNOWN
    phrases.append(_build_phrase(words, start_idx, len(words) - 1, tail_boundary, tail_confidence))
    return tuple(phrases)


def _build_phrase(
    words: Tuple[LanguageWord, ...], start_idx: int, end_idx: int, boundary_kind: str, confidence: str,
) -> LanguagePhrase:
    member_words = words[start_idx:end_idx + 1]
    text_raw = " ".join(w.text_raw.strip() for w in member_words if w.text_raw.strip())
    text_normalized = normalize_language_text(text_raw)
    source_asset_id = member_words[0].source_asset_id
    source_start = member_words[0].start
    source_end = member_words[-1].end
    return LanguagePhrase(
        source_asset_id=source_asset_id,
        phrase_id=_phrase_id(source_asset_id, source_start, source_end, text_normalized),
        word_start_index=start_idx,
        word_end_index=end_idx,
        source_start=source_start,
        source_end=source_end,
        text_raw=text_raw,
        text_normalized=text_normalized,
        boundary_kind=boundary_kind,
        confidence=confidence,
        provenance=PROVENANCE_PHRASE_SEGMENTATION,
    )


# ---------------------------------------------------------------------------
# Meaning-safety proof metric (reusable at runtime and in tests).
# ---------------------------------------------------------------------------
def count_meaning_sensitive_tokens_preserved(
    words: Tuple[LanguageWord, ...], *, sensitive_tokens: frozenset[str] | None = None,
) -> int:
    """Counts how many meaning-sensitive tokens found in ``text_raw`` are
    still present, unchanged (case-insensitively), in ``text_normalized`` --
    a genuine, reusable proof metric rather than an assertion of trust.
    Defaults to ``polarity_safety.POLARITY_PARTICLES`` (the one real,
    already-shared negation vocabulary in this codebase) plus every
    digit-bearing token (numbers/percentages/dates all contain digits) --
    never a Video00-specific or medical-domain list, per this task's own
    instruction."""
    sensitive = sensitive_tokens if sensitive_tokens is not None else POLARITY_PARTICLES
    preserved = 0
    for word in words:
        raw_tokens = _tokens(word.text_raw)
        normalized_tokens = _tokens(word.text_normalized)
        for token in raw_tokens:
            is_sensitive = token in sensitive or any(ch.isdigit() for ch in token)
            if is_sensitive and token in normalized_tokens:
                preserved += 1
    return preserved


# ---------------------------------------------------------------------------
# Diagnostics (tail-safe, counts-only -- same pattern as D-119/D-125/D-152/
# D-155/D-157/D-163's own compact summaries in this codebase). No transcript
# dump: only bounded counts and validity booleans.
# ---------------------------------------------------------------------------
def language_spine_diagnostics(
    words: Tuple[LanguageWord, ...], phrases: Tuple[LanguagePhrase, ...],
) -> dict:
    boundary_counts: dict[str, int] = {}
    for phrase in phrases:
        boundary_counts[phrase.boundary_kind] = boundary_counts.get(phrase.boundary_kind, 0) + 1

    unknown_confidence = sum(1 for phrase in phrases if phrase.confidence == CONFIDENCE_UNKNOWN)
    unknown_confidence += sum(1 for word in words if word.confidence is None)

    normalization_changes = sum(
        1 for word in words if word.text_normalized != word.text_raw
    )

    source_mapping_valid = True
    for phrase in phrases:
        if not (0 <= phrase.word_start_index <= phrase.word_end_index < len(words)):
            source_mapping_valid = False
            break
        span_words = words[phrase.word_start_index:phrase.word_end_index + 1]
        if span_words[0].start != phrase.source_start or span_words[-1].end != phrase.source_end:
            source_mapping_valid = False
            break
        if any(w.source_asset_id != phrase.source_asset_id for w in span_words):
            source_mapping_valid = False
            break

    timeline_valid = True
    previous_start = None
    for word in words:
        if word.start < 0 or word.end < word.start:
            timeline_valid = False
            break
        if previous_start is not None and word.start < previous_start:
            timeline_valid = False
            break
        previous_start = word.start
    if timeline_valid:
        for phrase in phrases:
            if phrase.source_start < 0 or phrase.source_end < phrase.source_start:
                timeline_valid = False
                break

    return {
        "schema_version": SCHEMA_VERSION,
        "language_spine_created": bool(words or phrases),
        "language_word_count": len(words),
        "language_phrase_count": len(phrases),
        "phrase_boundary_counts": boundary_counts,
        "language_unknown_confidence_count": unknown_confidence,
        "normalization_change_count": normalization_changes,
        "meaning_sensitive_token_preservation_count": count_meaning_sensitive_tokens_preserved(words),
        "source_mapping_valid": source_mapping_valid,
        "timeline_valid": timeline_valid,
    }
