"""Conservative retry grouping for valid takes.

The creator may repeat the same idea with small wording changes. Grouping is fuzzy
enough to recognize those retries, but deliberately refuses to cluster short or
weakly-overlapping phrases just because they share a commercial role.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, Mapping, Tuple

from .contracts import CandidateTake
from .semantic_atom_importance import _clause_has_any

_CONTRACTION_SUFFIXES = frozenset({"m", "re", "ve", "ll", "d", "s", "t"})


def semantic_key(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9áéíóúñü]+", " ", text.casefold())
    tokens = [token for token in normalized.split() if token]
    return " ".join(tokens[:18])


def _natural_tokens(text: str) -> tuple[str, ...]:
    """Preserve semantic keys while counting common contractions as one word."""
    raw = semantic_key(text).split()
    output: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if index + 1 < len(raw) and raw[index + 1] in _CONTRACTION_SUFFIXES:
            output.append(token + "'" + raw[index + 1])
            index += 2
            continue
        output.append(token)
        index += 1
    return tuple(output)


def retry_similarity(left: str, right: str) -> float:
    a = semantic_key(left)
    b = semantic_key(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    tokens_a = a.split()
    tokens_b = b.split()
    # Short phrases are too semantically dense for fuzzy grouping. Count common
    # contractions as one natural word so punctuation normalization cannot make a
    # three-word phrase accidentally enter the fuzzy path.
    if min(len(_natural_tokens(left)), len(_natural_tokens(right))) <= 3:
        return 0.0
    set_a, set_b = set(tokens_a), set(tokens_b)
    containment = len(set_a & set_b) / max(1, min(len(set_a), len(set_b)))
    sequence = SequenceMatcher(None, a, b).ratio()
    if containment < 0.60:
        return 0.0
    return round(0.55 * sequence + 0.45 * containment, 4)


def _gap_between(left: CandidateTake, right: CandidateTake) -> float:
    if left.end <= right.start:
        return right.start - left.end
    if right.end <= left.start:
        return left.start - right.end
    return 0.0


def _safe_short_prefix_retry(
    left: CandidateTake,
    right: CandidateTake,
    *,
    maximum_gap_sec: float = 12.0,
) -> bool:
    """Join only a 2-3 word exact-prefix false start to a nearby longer retry."""
    if left.source_asset_id != right.source_asset_id:
        return False
    if _gap_between(left, right) > maximum_gap_sec:
        return False
    left_tokens = _natural_tokens(left.text)
    right_tokens = _natural_tokens(right.text)
    if not left_tokens or not right_tokens:
        return False
    short, long = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    if not 2 <= len(short) <= 3 or len(long) < 5:
        return False
    return long[:len(short)] == short


# D-097.A (retry-family completeness): SAME-OPENING RESTART evidence.
#
# Run 34008386434 (D-096 collision C-1): a failed take, its abandoned
# restart and the clean retry all began with the same four words within a
# few seconds of each other, yet `retry_similarity` scored the failed take
# vs the clean retry 0.0 -- its 0.60 word-containment floor is defeated by
# a self-correction that rewrites the middle of the sentence ("hablé con"
# -> "cambié de", "todos los test" -> "un test de todo") -- and the
# semantic arbiter, asked about the pair, declined. A creator restarting a
# sentence from its first words within seconds IS recording-process
# evidence of a retry, independent of what any semantic judge thinks about
# the factual detail that changed: the two deliveries must compete (the
# corrected one should win) rather than both play. Two deterministic shapes,
# both bounded by temporal adjacency and shared content beyond the opening
# so a shared discourse connector ("y después de eso ...") never qualifies:
# (1) RESTART -- both takes long enough to carry content after the opening,
#     and their post-opening content overlaps materially;
# (2) ABANDONED START -- the shorter take is at most half the longer one,
#     shares the opening and at least one content word beyond it (the
#     "Al terminar mi contrato le pedía a mi ginecóloga" shape, too short
#     for (1) and not an exact prefix for `_safe_short_prefix_retry`).
# No Video00 wording is referenced; thresholds are structural.
_RESTART_OPENING_TOKENS = 4
_RESTART_MAXIMUM_GAP_SEC = 8.0
_RESTART_MINIMUM_REMAINDER_OVERLAP = 0.40
_RESTART_MINIMUM_SHARED_CONTENT = 2
_RESTART_STOP = frozenset({
    # es
    "que", "con", "por", "para", "una", "uno", "unos", "unas", "los", "las", "del", "les",
    "como", "pero", "muy", "más", "mas", "sin", "sobre", "entre", "hasta", "desde", "ella",
    "ellos", "ellas", "eso", "esa", "ese", "esto", "esta", "este", "aquí", "ahí", "allí",
    "era", "fue", "son", "está", "esta", "hay", "había", "tenía", "tener", "sea", "ser",
    # en
    "the", "and", "that", "this", "with", "for", "was", "were", "are", "have", "has", "had",
    "you", "your", "our", "they", "them", "there", "here", "then", "than", "but", "not",
    "just", "very", "also", "into", "from", "what", "when", "which", "who",
})


def _restart_content(tokens: tuple[str, ...]) -> set[str]:
    return {token for token in tokens if len(token) >= 3 and token not in _RESTART_STOP}


# D-156 (RAW #119 audit): an incomplete take can be completed by a retry
# that re-conjugates or re-inflects the SAME word rather than repeating it
# verbatim -- "me mando." (he/she sent me, singular) completed by "me
# mandaron a hacer sonografia..." (they sent me, plural) shares zero exact
# tokens on "mando"/"mandaron" even though it is the same verb, so the
# shared-content-word floors below (which exist to require real, non-
# coincidental topical overlap beyond a shared opener) never fire on the
# most literal shape of self-correction: fixing subject/verb agreement or
# number while continuing the same idea. Exact-token equality is too
# strict for this; a bare fixed-length prefix match is too loose (an
# unrelated pair like "contest"/"context" shares 5 of "contest"'s 7
# characters purely by coincidence). The conservative middle: two content
# words of real length (>= 5 chars, so a short/common word can never
# supply this alone) whose shared PREFIX covers at least 80% of the
# shorter word's length are treated as the same underlying word --
# "mando"/"mandaron" share 4 of "mando"'s 5 characters (80%, the boundary
# itself); "contest"/"context" share only 5 of 7 (71%, below it). This is
# a general inflection-tolerance rule, not Video00 vocabulary -- it fires
# identically on any language/topic whose retries fix agreement, tense or
# number on an otherwise-unchanged word.
_STEM_MATCH_MIN_TOKEN_LEN = 5
_STEM_MATCH_PREFIX_RATIO = 0.80


def _content_words_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if len(left) < _STEM_MATCH_MIN_TOKEN_LEN or len(right) < _STEM_MATCH_MIN_TOKEN_LEN:
        return False
    shorter = min(len(left), len(right))
    prefix_len = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix_len += 1
    return (prefix_len / shorter) >= _STEM_MATCH_PREFIX_RATIO


def _shared_content_count(left: set[str], right: set[str]) -> int:
    """Stem-aware shared-word count between two content-word sets (see
    `_content_words_match`). Each right-side word is credited at most once,
    even if it stem-matches more than one left-side word."""
    matched_right: set[str] = set()
    count = 0
    for word in left:
        for other in right:
            if other in matched_right:
                continue
            if _content_words_match(word, other):
                matched_right.add(other)
                count += 1
                break
    return count


def same_opening_restart(
    left: CandidateTake,
    right: CandidateTake,
    *,
    maximum_gap_sec: float = _RESTART_MAXIMUM_GAP_SEC,
    opening_tokens: int = _RESTART_OPENING_TOKENS,
    minimum_remainder_overlap: float = _RESTART_MINIMUM_REMAINDER_OVERLAP,
    minimum_shared_content: int = _RESTART_MINIMUM_SHARED_CONTENT,
) -> str | None:
    """Return the deterministic restart evidence kind ("same_opening_restart"
    or "same_opening_abandoned_start") joining two takes, or None. See the
    D-097.A module comment above for the rationale and bounds."""
    if left.source_asset_id != right.source_asset_id:
        return None
    if _gap_between(left, right) > maximum_gap_sec:
        return None
    left_tokens = _natural_tokens(left.text)
    right_tokens = _natural_tokens(right.text)
    if min(len(left_tokens), len(right_tokens)) < opening_tokens + 2:
        return None
    if left_tokens[:opening_tokens] != right_tokens[:opening_tokens]:
        return None
    left_rest = _restart_content(left_tokens[opening_tokens:])
    right_rest = _restart_content(right_tokens[opening_tokens:])
    if not left_rest or not right_rest:
        return None
    shared_count = _shared_content_count(left_rest, right_rest)
    smaller = min(len(left_rest), len(right_rest))
    if shared_count >= minimum_shared_content and shared_count / smaller >= minimum_remainder_overlap:
        return "same_opening_restart"
    short, long = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    if shared_count and len(short) * 2 <= len(long):
        return "same_opening_abandoned_start"
    return None


# D-097.12 (stomach family, bounded encargo): an ABANDONED ATTEMPT completed
# by a later retry the creator self-corrects with different wording all the
# way through (not just the first four words `same_opening_restart` requires).
#
# RAW 34043967265/34045158712/34047064840/34048444463 (D-097.9-.11): "Tuve
# problemas estomacales ... y me diagnosticaron con..." (grammatically open
# tail, R15) and "Tuve problemas de digestión ... dijeron que tenía
# gastritis." share only a 2-word opening -- the creator rewrote "estomacales"
# to "de digestión" and "diagnosticaron con" to "dijeron que tenía gastritis"
# -- so `same_opening_restart`'s 4-token exact-prefix requirement never
# fires, and the semantic arbiter answered this exact pair SAME 0.95 once and
# NOT-same 0.85/0.90 twice on "incomplete vs complete", the one reason the
# module comment above already rules must never gate a retry family. An
# EARLIER take that is provably incomplete (open tail / trailing off, not a
# label or position alone) is recording-process evidence independent of the
# arbiter's answer -- but a 2-word opening is common enough ("Tuve
# problemas...") that it must never join on the opening alone: the SAME
# gastritis pair vs an unrelated "Tuve problemas de estómago ... no hay que
# preguntar." aside shares the identical 2-word opening and ZERO further
# content, and must not merge (an independent sentence sharing only a topic
# opener, D-020). The gate is therefore the SAME shared-content-beyond-the-
# opening test `same_opening_restart` already uses, just at a shorter
# opening and a wider gap (the creator's aside intervenes between the
# abandoned attempt and its completion in the source recording).
_INCOMPLETE_RETRY_OPENING_TOKENS = 2
_INCOMPLETE_RETRY_MAXIMUM_GAP_SEC = 20.0
_INCOMPLETE_RETRY_MINIMUM_TOKENS = 5
_INCOMPLETE_RETRY_MINIMUM_SHARED_CONTENT = 2


def incomplete_attempt_completed_by_retry(
    left: CandidateTake,
    right: CandidateTake,
    *,
    maximum_gap_sec: float = _INCOMPLETE_RETRY_MAXIMUM_GAP_SEC,
    opening_tokens: int = _INCOMPLETE_RETRY_OPENING_TOKENS,
    minimum_tokens: int = _INCOMPLETE_RETRY_MINIMUM_TOKENS,
    minimum_shared_content: int = _INCOMPLETE_RETRY_MINIMUM_SHARED_CONTENT,
) -> str | None:
    """Return `"incomplete_attempt_completed_by_retry"` when an earlier,
    grammatically incomplete take is completed by a later delivery sharing
    its opening and real content beyond it, or None. Order-independent:
    the chronologically earlier take must be the incomplete one. See the
    D-097.12 module comment above."""
    if left.source_asset_id != right.source_asset_id:
        return None
    if _gap_between(left, right) > maximum_gap_sec:
        return None
    earlier, later = (left, right) if left.start <= right.start else (right, left)
    if earlier.complete_idea or not later.complete_idea:
        return None
    earlier_tokens = _natural_tokens(earlier.text)
    later_tokens = _natural_tokens(later.text)
    if min(len(earlier_tokens), len(later_tokens)) < minimum_tokens:
        return None
    if earlier_tokens[:opening_tokens] != later_tokens[:opening_tokens]:
        return None
    earlier_rest = _restart_content(earlier_tokens[opening_tokens:])
    later_rest = _restart_content(later_tokens[opening_tokens:])
    if _shared_content_count(earlier_rest, later_rest) >= minimum_shared_content:
        return "incomplete_attempt_completed_by_retry"
    return None


# D-287 (RAW #120 audit): a real, code-verified gap `incomplete_attempt_
# completed_by_retry` structurally cannot close either. Its own precondition
# (`earlier.complete_idea` must be False) relies entirely on `take_
# segmentation._looks_complete_idea`, which grades completeness from
# PUNCTUATION alone (`_ends_sentence`): any text ending in '.'/'!'/'?' is
# `complete_idea=True`, full stop, with no assessment of whether its CONTENT
# is specific enough to stand as a finished editorial idea. "Ahí fue cuando
# me mandó." is grammatically terminated (so `complete_idea=True`) but
# editorially VAGUE -- it never says what the creator was sent to do -- and
# is immediately completed by "Ahí fue cuando me mandaron a hacer sonografía
# de tiroides y otras sonografías." sharing the same opening. Both clips
# graded complete_idea=True, so `incomplete_attempt_completed_by_retry`'s
# own gate declines the pair before its (already-fixed, D-156) shared-
# content step is ever reached -- confirmed on the real RAW #120 JSON: the
# two clips carry different `take_group_id`s and never even compete for
# Best Take.
#
# This is deliberately NOT a change to `_looks_complete_idea` itself (used
# pervasively across the whole pipeline; loosening it here would be a
# blast-radius change far outside this defect). Instead, a narrowly-scoped
# sibling rule for exactly this shape, with a SAFETY BAR STRICTER than
# `incomplete_attempt_completed_by_retry`'s own `minimum_shared_content>=2`
# floor: FULL content coverage of the vague side, not just two shared
# words. This is the direct answer to the D-156 audit concern (a lexical/
# stem coincidence must never fuse two COMPLETE realizations with UNEQUAL
# coverage, e.g. a long distinct conclusion folded into an unrelated short
# microclip that merely opens similarly) -- every real content word the
# vague delivery makes must be present in the longer one, and the longer
# one must be materially longer, not just superficially similar.
_VAGUE_RETRY_OPENING_TOKENS = 2
_VAGUE_RETRY_MAXIMUM_GAP_SEC = 20.0
_VAGUE_RETRY_MINIMUM_LENGTH_RATIO = 2.0


def vague_retry_completed_by_detailed_retry(
    left: CandidateTake,
    right: CandidateTake,
    *,
    maximum_gap_sec: float = _VAGUE_RETRY_MAXIMUM_GAP_SEC,
    opening_tokens: int = _VAGUE_RETRY_OPENING_TOKENS,
    minimum_length_ratio: float = _VAGUE_RETRY_MINIMUM_LENGTH_RATIO,
) -> str | None:
    """Return `"vague_retry_completed_by_detailed_retry"` when an earlier,
    punctuation-complete but editorially VAGUE take is completed by a later,
    materially more detailed delivery sharing its opening and covering
    EVERY real content word the vague take makes, or None. Order-
    independent: the chronologically earlier take must be the vague one.
    See the D-287 module comment above."""
    if left.source_asset_id != right.source_asset_id:
        return None
    if _gap_between(left, right) > maximum_gap_sec:
        return None
    earlier, later = (left, right) if left.start <= right.start else (right, left)
    if not earlier.complete_idea or not later.complete_idea:
        return None
    earlier_tokens = _natural_tokens(earlier.text)
    later_tokens = _natural_tokens(later.text)
    if min(len(earlier_tokens), len(later_tokens)) < opening_tokens + 1:
        return None
    if earlier_tokens[:opening_tokens] != later_tokens[:opening_tokens]:
        return None
    earlier_content = _restart_content(earlier_tokens[opening_tokens:])
    later_content = _restart_content(later_tokens[opening_tokens:])
    if not earlier_content:
        return None  # nothing beyond the shared opener to prove real overlap (D-020)
    if len(later_content) < minimum_length_ratio * len(earlier_content):
        return None
    if _shared_content_count(earlier_content, later_content) < len(earlier_content):
        return None  # FULL coverage required -- a partial/coincidental overlap declines
    return "vague_retry_completed_by_detailed_retry"


# D-150 (Gate 6 correction, real RAW #118 audit): a real, code-verified gap
# `incomplete_attempt_completed_by_retry` structurally cannot close. A real
# abandoned opening can share its first few words with its later completion
# and then diverge into COMPLETELY DIFFERENT VOCABULARY THE REST OF THE WAY
# -- the creator restarted with a different noun/phrasing choice throughout,
# not just at the opening. `incomplete_attempt_completed_by_retry`'s own
# `minimum_shared_content` gate (content overlap BEYOND the opening) then
# never fires, by construction, no matter how its `minimum_tokens` floor is
# tuned -- there is no lexical overlap left to find. This is exactly the
# risk D-097.12's own module comment already named and refused to solve
# lexically: "the SAME gastritis pair vs an unrelated 'Tuve problemas de
# estómago ... no hay que preguntar.' aside shares the identical 2-word
# opening and ZERO further content, and must not merge" -- an opening-only
# match is NOT enough evidence on its own, full stop.
#
# The general, non-lexical evidence that tells the two cases apart: a real
# abandoned-then-completed attempt has NOTHING ELSE said in between -- the
# creator paused, then continued. An unrelated aside sharing only a generic
# opener instead has ITS OWN speech content occupying that time. Already-
# measured source silence (`audio_silence.py`, real ffmpeg evidence, the
# same signal `attempt_reconstruction.py`'s own `_measured_pause_at_
# transition` and `perceptual_watch_listen.py`'s `_measured_pause_near`
# already trust) that covers MOST of the gap between the two takes is
# structural proof of "paused, then continued" -- proof an unrelated aside
# with real content in between could never produce, since its own words
# would occupy that time instead of silence. Deliberately conservative:
# still requires the SAME 2-word opening match (not run on totally
# unrelated pairs), completeness asymmetry, and a high silence-coverage
# floor over the WHOLE gap, not just a moment somewhere inside it.
_PAUSE_BRIDGED_RETRY_OPENING_TOKENS = 2
_PAUSE_BRIDGED_RETRY_MAXIMUM_GAP_SEC = 20.0
_PAUSE_BRIDGED_RETRY_MIN_SILENCE_COVERAGE = 0.60


def measured_pause_bridged_retry(
    left: CandidateTake,
    right: CandidateTake,
    silence_intervals: Mapping[str, Tuple[Tuple[float, float], ...]] | None,
    *,
    opening_tokens: int = _PAUSE_BRIDGED_RETRY_OPENING_TOKENS,
    maximum_gap_sec: float = _PAUSE_BRIDGED_RETRY_MAXIMUM_GAP_SEC,
    min_silence_coverage: float = _PAUSE_BRIDGED_RETRY_MIN_SILENCE_COVERAGE,
) -> str | None:
    """Return `"measured_pause_bridged_retry"` when an earlier, incomplete
    take shares its opening with a later, complete take AND the gap between
    them is almost entirely measured silence, or `None`. See the module
    comment above for the three-gate rationale. `silence_intervals` empty
    or `None` returns `None` immediately -- zero behavior change when no
    measured-silence evidence is supplied."""
    if not silence_intervals:
        return None
    if left.source_asset_id != right.source_asset_id:
        return None
    events = silence_intervals.get(left.source_asset_id)
    if not events:
        return None
    earlier, later = (left, right) if left.start <= right.start else (right, left)
    if earlier.complete_idea or not later.complete_idea:
        return None  # completeness asymmetry -- same safe shape every rule in this family requires
    gap = max(0.0, float(later.start) - float(earlier.end))
    if gap <= 0.0 or gap > maximum_gap_sec:
        return None
    earlier_tokens = _natural_tokens(earlier.text)
    later_tokens = _natural_tokens(later.text)
    if min(len(earlier_tokens), len(later_tokens)) < opening_tokens:
        return None
    if earlier_tokens[:opening_tokens] != later_tokens[:opening_tokens]:
        return None  # still requires the same-opener signal -- never runs on unrelated pairs
    silence_covered = 0.0
    for s_start, s_end in events:
        overlap = min(float(s_end), float(later.start)) - max(float(s_start), float(earlier.end))
        if overlap > 0.0:
            silence_covered += overlap
    if (silence_covered / gap) < min_silence_coverage:
        return None
    return "measured_pause_bridged_retry"


# ---------------------------------------------------------------------------
# D-289.1: SENTENCE CONTINUATION -- one delivery split by the ASR/attempt
# boundary into a head that stops on a dangling function word and a tail
# that finishes the sentence.
# ---------------------------------------------------------------------------
#
# Real audited shape (RAW #122 conclusion cluster, D-288 audit item 5): the
# creator restates a claim, pauses for several seconds mid-sentence after a
# preposition+article ("... de los"), then finishes it ("... son
# hereditarios."). The AttemptReconstructor's continuation ceiling (1.20 s)
# correctly calls a multi-second pause a `real_speech_pause`, so the two
# halves reach grouping as two candidates: an INCOMPLETE head (no terminal
# punctuation, dangling ending) and a punctuation-complete 3-word tail whose
# meaning, read alone, is the OPPOSITE of the sentence it belongs to (the
# quantifier lives in the head). Judged separately, the tail is a
# "complete" fragment the ranker may keep on its own -- a predicate without
# its quantifier -- and the head's number is judged without its predicate.
#
# This relation is DETERMINISTIC recording-process evidence that two
# candidates are ONE realization, never two competitors: (1) same source and
# chronological, the tail starting within the same retry-adjacency bound the
# restart rules already use (`_RESTART_MAXIMUM_GAP_SEC`); (2) the head is
# grammatically incomplete: not `complete_idea`, no terminal punctuation,
# and its LAST natural token is a function word that cannot end a sentence
# (`_DANGLING_FUNCTION_WORDS`: articles, prepositions, conjunctions, the
# relative "que"/"that"); (3) the tail begins in lower case -- the ASR's own
# sentence-case signal that no new sentence started -- and carries at least
# one natural token. The CALLER must also establish adjacency (no other
# candidate of the same source between them); this function judges the pair
# alone. A same-opening restart, a marker-introduced new point, a capitalised
# new sentence or a punctuation-complete head are never a continuation. No
# Video00 phrase, id or timestamp is read here.
# D-039/D-048's distinct-addition markers (moved here from
# take_grouping_provider.py, which re-exports them unchanged, so this module
# can consult them without an import cycle -- D-289.2).
_DISTINCT_ADDITION_MARKERS = (
    "otro sintoma", "otro síntoma", "otra cosa", "otro problema", "otro punto",
    "otra situacion", "otra situación", "otro detalle", "otro aspecto",
    "another symptom", "another issue", "another problem", "another thing",
    "a different issue", "a different problem", "an additional", "one more thing",
    "on top of that",
)


def _has_distinct_addition_marker(text: str) -> bool:
    return _clause_has_any(text, _DISTINCT_ADDITION_MARKERS)


_FULL_TEXT_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+", re.IGNORECASE)


def _full_text_tokens(text: str) -> tuple[str, ...]:
    """Every token of the WHOLE text (casefolded). `semantic_key`/`_natural_
    tokens` deliberately keep only the first 18 tokens for retry
    similarity; a last-word test must never read a truncated middle word
    (D-289.2)."""
    return tuple(_FULL_TEXT_TOKEN_RE.findall(str(text or "").casefold()))


_DANGLING_FUNCTION_WORDS = frozenset({
    # Spanish articles / prepositions / conjunctions / relatives
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a", "en",
    "con", "por", "para", "sin", "sobre", "entre", "hacia", "hasta", "desde", "que", "y",
    "e", "o", "u", "ni", "como", "cuando", "donde", "mi", "mis", "tu", "tus", "su", "sus",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "lo", "le", "les", "se",
    # English articles / prepositions / conjunctions / relatives
    "the", "an", "of", "to", "in", "on", "at", "for", "with", "from", "by", "into", "onto",
    "and", "or", "nor", "but", "that", "which", "who", "whose", "than", "as", "my", "your",
    "our", "their", "his", "her", "its", "this", "these", "those",
})
_CONTINUATION_TERMINAL_PUNCT_RE = re.compile(r"[.!?…][\"'”’)\]]*\s*$")
_CONTINUATION_MAXIMUM_GAP_SEC = _RESTART_MAXIMUM_GAP_SEC


def sentence_continuation(left: CandidateTake, right: CandidateTake) -> bool:
    """True when `right` deterministically finishes the sentence `left`
    stopped in the middle of -- see the module comment above. Adjacency
    (no same-source candidate between the two) is the caller's check."""
    if left.source_asset_id != right.source_asset_id:
        return False
    if right.start < left.end - 0.03:
        return False
    if right.start - left.end > _CONTINUATION_MAXIMUM_GAP_SEC:
        return False
    if left.complete_idea:
        return False
    left_text = str(left.text or "").rstrip()
    if not left_text or _CONTINUATION_TERMINAL_PUNCT_RE.search(left_text):
        return False
    # The LAST word of the WHOLE head text (never `_natural_tokens`' 18-token
    # prefix, which would test a middle word of a long head -- D-289.2).
    left_tokens = _full_text_tokens(left_text)
    if not left_tokens or left_tokens[-1] not in _DANGLING_FUNCTION_WORDS:
        return False
    right_text = str(right.text or "").lstrip()
    first_alpha = next((ch for ch in right_text if ch.isalpha()), None)
    if first_alpha is None or not first_alpha.islower():
        return False
    right_tokens = _full_text_tokens(right_text)
    if not right_tokens:
        return False
    # D-289.2 guards: a tail that RESTARTS the head's own sentence is a retry
    # (D-097.A's `same_opening_restart` / the reconstructor's own two-token
    # restart evidence), never the head's continuation -- the incomplete
    # attempt and its complete repetition must compete, not fuse; and a tail
    # that opens with a distinct-addition marker (D-039/D-048) announces a
    # different point, never the end of the head's sentence.
    if same_opening_restart(left, right) is not None:
        return False
    if len(left_tokens) >= 2 and len(right_tokens) >= 2 and left_tokens[:2] == right_tokens[:2]:
        return False
    if _has_distinct_addition_marker(right_text):
        return False
    return True


def continuation_pairs(takes: Iterable[CandidateTake]) -> frozenset[frozenset[str]]:
    """Every ADJACENT same-source candidate pair (consecutive in start order,
    nothing of that source between them) that `sentence_continuation`
    accepts, keyed order-insensitively. The one place adjacency is decided,
    so reconcile and the cohesion pass can never disagree about which pairs
    form a continuation chain."""
    by_source: dict[str, list[CandidateTake]] = {}
    for take in takes:
        by_source.setdefault(take.source_asset_id, []).append(take)
    pairs: set[frozenset[str]] = set()
    for members in by_source.values():
        ordered = sorted(members, key=lambda t: (t.start, t.end, t.clip_id))
        for left, right in zip(ordered, ordered[1:]):
            if sentence_continuation(left, right):
                pairs.add(frozenset((left.clip_id, right.clip_id)))
    return frozenset(pairs)


def group_takes(
    takes: Iterable[CandidateTake],
    *,
    similarity_threshold: float = 0.72,
) -> Dict[str, Tuple[CandidateTake, ...]]:
    ordered = sorted(takes, key=lambda take: (take.source_order, take.start, take.end, take.clip_id))
    clusters: list[list[CandidateTake]] = []

    for take in ordered:
        best_index = None
        best_score = 0.0
        for index, cluster in enumerate(clusters):
            representatives = (cluster[0], cluster[-1]) if len(cluster) > 1 else (cluster[0],)
            score = 0.0
            for item in representatives:
                candidate_score = retry_similarity(take.text, item.text)
                if _safe_short_prefix_retry(take, item) or same_opening_restart(take, item):
                    candidate_score = max(candidate_score, 1.0)
                score = max(score, candidate_score)
            if score > best_score:
                best_score, best_index = score, index
        if best_index is not None and best_score >= similarity_threshold:
            clusters[best_index].append(take)
        else:
            clusters.append([take])

    output: Dict[str, Tuple[CandidateTake, ...]] = {}
    for cluster in clusters:
        base = semantic_key(cluster[0].text) or f"__silent__:{cluster[0].clip_id}"
        key = base
        suffix = 1
        while key in output:
            suffix += 1
            key = f"{base} #{suffix}"
        output[key] = tuple(cluster)
    return output


# D-100 (D-099 Gap #1, bounded encargo): CONFIRMED MULTIMODAL
# RECORDING-BEHAVIOR CORROBORATION.
#
# D-099's dataflow trace found that `performance_confirmation.py` already
# promotes dense local visual candidates (MediaPipe reset/break evidence
# cross-checked against a nearby lexically-similar take) into CONFIRMED
# `wrong_take`/`retry_setup` events -- real evidence the creator visibly
# reset, stumbled, or interrupted a delivery -- but this module never
# received any of it: every restart-evidence rule above is lexical only.
# D-097.12's stomach fix needed a brand-new lexical rule (opening-token
# window + shared-content-beyond-opening) precisely because of this
# blindness. This rule closes that ONE plumbing gap by letting a
# CONFIRMED event corroborate a WEAKER lexical link than the rules above
# require -- never replacing them, never running when they already fired
# (see `reconcile_semantic_idea_equivalence`'s merge loop: this is tried
# only after `same_opening_restart`, `_safe_short_prefix_retry` and
# `incomplete_attempt_completed_by_retry` all return None).
#
# Deliberately NOT a standalone trigger -- three independent safety gates,
# each proven necessary by a specific negative control (see
# tests/test_cutsell_d100_multimodal_retry_corroboration.py):
#   1. REAL shared topical content beyond stopwords must exist between the
#      two takes (D-020: an independent sentence sharing nothing but a
#      confirmed nearby reset must never merge on recording behavior
#      alone -- confirmed evidence with no textual link is not evidence
#      of a retry, it is evidence of *something* happening nearby).
#   2. The two takes must be asymmetric in completeness (one incomplete,
#      one complete) -- the same incomplete-attempt-completed-by-retry
#      shape D-097.12 already established as the safe retry pattern. Two
#      genuinely complementary COMPLETE statements (the D-020 composite
#      case) are excluded by this gate even when they share real content
#      and a confirmed event happens to sit between them.
#   3. The confirmed event must sit at the physical BOUNDARY between the
#      two takes, not merely somewhere in the source -- a reset far from
#      this specific transition is not evidence about THIS pair.
_MULTIMODAL_CORROBORATION_KINDS = frozenset({"wrong_take", "retry_setup"})
_MULTIMODAL_CORROBORATION_MAXIMUM_GAP_SEC = 10.0
# D-144 (Gate 6, Gap A): a false start is short by definition -- the
# creator caught themselves and stopped -- so a floor higher than the
# three DOCUMENTED safety gates above actually need excludes exactly the
# shortest, most classic false-start shape (a 3-token abandoned opening)
# from this authority's own bounded backstop before any of those three
# gates gets a chance to run. The real protection against a tiny/noise
# fragment is already gate 1 below (a real shared CONTENT token, length
# >=3, ratio >=0.25) plus gate 2 (completeness asymmetry) plus gate 3 (a
# CONFIRMED event at this exact boundary) -- three independent, unrelated
# coincidences a stray short fragment is very unlikely to satisfy by
# accident. Lowered from 4 to 3 (not further): 3 is the minimum that can
# still carry one qualifying `_restart_content` token (>=3 chars) plus at
# least one more word of grammatical scaffolding, so a bare 1-2 token
# interjection ("uh", "no wait") still never reaches this rule.
_MULTIMODAL_CORROBORATION_MINIMUM_TOKENS = 3
# D-147 (Gate 6 correction, real RAW #118 audit): raised 1 -> 2. A SINGLE
# shared content word carries no distinguishing power at all once the
# eligibility/token floors were generalized to short false starts (D-144):
# a genuinely unrelated short fragment that happens to share exactly one
# incidental word with a long, topically distant later clip ("I had
# meetings" / "the schedule had many meetings planned for next quarter")
# scores IDENTICALLY on both `minimum_shared_content` and
# `minimum_overlap_ratio` (ratio is computed against the SMALLER side, so
# one match out of one possible word is always ratio=1.0) to a genuine
# short false start that shares its one meaningful word with its real
# completion -- there is no lexical signal that tells the two apart, and a
# confirmed event at the boundary alone is not enough (that is gate 3's
# job, not gate 1's). Requiring at least TWO independent shared content
# words makes a coincidental match require two independent coincidences,
# not one -- consistent with WHEN UNCERTAIN, KEEP (do not merge on
# ambiguous lexical evidence, even with multimodal corroboration).
_MULTIMODAL_CORROBORATION_MINIMUM_SHARED_CONTENT = 2
_MULTIMODAL_CORROBORATION_MINIMUM_OVERLAP_RATIO = 0.25
_MULTIMODAL_CORROBORATION_EVENT_BEFORE_SEC = 1.0
_MULTIMODAL_CORROBORATION_EVENT_AFTER_SEC = 2.0

# One confirmed-event record: (kind, start, end). Plain tuples, not
# `TemporalEvent` -- this module stays a pure lexical/text module with no
# dependency on `whole_video_analysis.py`; the caller (`take_grouping_
# provider.reconcile_semantic_idea_equivalence`) is responsible for
# narrowing `WholeVideoContext` down to this shape (see `whole_video_
# analysis.confirmed_recording_behavior_events`).
ConfirmedEvent = Tuple[str, float, float]


def _confirmed_event_at_boundary(
    events: Iterable[ConfirmedEvent],
    *,
    boundary_start: float,
    boundary_end: float,
) -> ConfirmedEvent | None:
    for event in events:
        kind, start, end = event
        if kind not in _MULTIMODAL_CORROBORATION_KINDS:
            continue
        if end >= boundary_start and start <= boundary_end:
            return event
    return None


def multimodal_corroborated_retry(
    left: CandidateTake,
    right: CandidateTake,
    confirmed_events: Mapping[str, Tuple[ConfirmedEvent, ...]] | None,
    *,
    maximum_gap_sec: float = _MULTIMODAL_CORROBORATION_MAXIMUM_GAP_SEC,
    minimum_tokens: int = _MULTIMODAL_CORROBORATION_MINIMUM_TOKENS,
    minimum_shared_content: int = _MULTIMODAL_CORROBORATION_MINIMUM_SHARED_CONTENT,
    minimum_overlap_ratio: float = _MULTIMODAL_CORROBORATION_MINIMUM_OVERLAP_RATIO,
) -> Tuple[str, ConfirmedEvent] | None:
    """Return `("multimodal_corroborated_retry", confirmed_event)` when a
    CONFIRMED recording-behavior event (`wrong_take`/`retry_setup`, from
    `performance_confirmation.py`) sits at the physical boundary between
    `left` and `right` AND they share real (weaker-than-the-lexical-rules-
    above) topical content with an incomplete/complete asymmetry, or
    `None`. See the module comment above for why each gate exists and
    which negative control it protects. `confirmed_events` empty or
    `None` returns `None` immediately -- zero behavior change when no
    multimodal evidence is supplied (backward compatibility)."""
    if not confirmed_events:
        return None
    if left.source_asset_id != right.source_asset_id:
        return None
    if _gap_between(left, right) > maximum_gap_sec:
        return None
    events = confirmed_events.get(left.source_asset_id)
    if not events:
        return None
    earlier, later = (left, right) if left.start <= right.start else (right, left)
    if earlier.complete_idea and later.complete_idea:
        return None  # gate 2: no asymmetry -- protects complementary-COMPLETE pairs
    earlier_tokens = _natural_tokens(earlier.text)
    later_tokens = _natural_tokens(later.text)
    if min(len(earlier_tokens), len(later_tokens)) < minimum_tokens:
        return None
    earlier_content = _restart_content(earlier_tokens)
    later_content = _restart_content(later_tokens)
    if not earlier_content or not later_content:
        return None
    shared_count = _shared_content_count(earlier_content, later_content)
    smaller = min(len(earlier_content), len(later_content))
    if shared_count < minimum_shared_content or shared_count / smaller < minimum_overlap_ratio:
        return None  # gate 1: no real shared topic -- protects unrelated pairs
    found = _confirmed_event_at_boundary(
        events,
        boundary_start=earlier.end - _MULTIMODAL_CORROBORATION_EVENT_BEFORE_SEC,
        boundary_end=later.start + _MULTIMODAL_CORROBORATION_EVENT_AFTER_SEC,
    )
    if found is None:
        return None  # gate 3: no confirmed event at THIS boundary
    return "multimodal_corroborated_retry", found
