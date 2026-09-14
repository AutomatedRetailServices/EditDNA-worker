"""D-235P: SHARED ATTEMPT/PROPOSITION IDENTITY SEAM -- CANONICAL
WORD-MEMBERSHIP IMPLEMENTATION, OFFLINE ONLY.

See ``docs/CUTSELL_DECISIONS.md`` D-235J-D-235O for the full forensic
chain this task implements the recommended seam for. D-235O's own
verdict C ("exact canonical word-membership identity is the smallest
safe seam") is implemented HERE, bounded exactly to that recommendation:
a pure, additive provenance module bridging a reconstructed attempt
(``contracts.CandidateTake``, post ``attempt_reconstruction.py``) to a
canonical ``language_utterance_attempt.LanguageAttempt`` via EXACT
canonical word-index-set membership -- never fuzzy text, never a
timestamp-overlap heuristic.

## What this module is NOT

This module mints NO new semantic id. It does not replace, alias, or
touch ``clip_id``/``attempt_id`` (``canonical_identity.mint_attempt_id``)/
``LanguageAttempt.attempt_id``/``PropositionCandidate.proposition_
candidate_id``/any P1 id. It is not called by ``pipeline.py``,
``take_segmentation.py``, ``attempt_reconstruction.py``,
``editorial_moment_sequence_integration.py``,
``language_spine_live_integration.py``, ``take_grouping*.py``,
``hybrid_session_cleanup.py``, ``semantic_idea_equivalence.py``,
``deterministic_best_take_authority.py``, ``take_judge.py``,
``boundary_engine_pass.py``, ``dialogue_pacing_transition.py``,
``repair_loop.py``, or any Freeze/materiality/resolver authority
(module-leaf grep tests, ``tests/test_cutsell_d235p_shared_attempt_
word_identity.py``). D-235N's own forensic (maximum-time-overlap
bridging is non-authoritative) is NOT superseded in the live path by
this task -- ``language_spine_live_integration.language_attempts_by_
span_id_for_source`` remains completely UNCHANGED, byte-identical, and
still the only bridge any live P1 call site actually consults. This
module offers a SEPARATE, more precise evidence source a future,
separately-authorized task (D-235Q+) may choose to wire in; it does
not do that wiring itself.

It also does NOT solve atom-level ownership: when a reconstructed
attempt's exact word membership maps to MULTIPLE ``PropositionCandidate``
ids (a real, honestly-represented shape -- see "1->N support" below),
this module returns the whole exact SET and explicitly never guesses
which individual proposition owns a specific lost semantic atom. That
narrower question is out of scope here by the task's own instruction.

## Canonical word identity representation (the task's own required proof)

The task requires proving whether a reconstructed attempt's word
membership is ALWAYS CONTIGUOUS in canonical source-word order before
choosing a start/end range vs. an explicit index tuple. Traced
mechanically through the real code (not assumed):

  * ``take_segmentation.py::_speech_units`` sorts one ASR segment's own
    words by ``(start, end)`` and slices them into gap-based chunks --
    each chunk IS a contiguous run of that segment's own sorted words.
  * ``take_segmentation.py::_repair_boundary_fragments`` joins two
    ADJACENT (in the time-sorted ``ordered`` list) candidates via
    ``_join_takes``'s ``words=tuple(left.words) + tuple(right.words)`` --
    but this join is explicitly permitted across a SMALL NEGATIVE gap
    (``-0.02 <= gap``, i.e. ``left`` may end fractionally AFTER
    ``right`` starts) at several call sites (the polarity rejoin, the
    strict/bridge contiguous joins). A small allowed overlap is direct,
    literal evidence against an unconditional "always non-overlapping,
    therefore always contiguous in the GLOBAL per-source sort" claim --
    nothing in this codebase enforces that no OTHER candidate's words
    could fall inside such an overlap window in a real (if rare)
    adversarial or degenerate timing case.
  * ``attempt_reconstruction.py::_merge_attempt`` concatenates
    ``members``' words in ``(source_order, start, end, clip_id)``
    order -- again a concatenation of ALREADY-CONSTRUCTED per-candidate
    word tuples, never a fresh global re-sort/re-validation against
    every OTHER candidate's words for the same source.
  * Both real, live global-order authorities in this codebase
    (``language_spine.py::adapt_words_to_language_words`` and
    ``canonical_asr_evidence.py``) explicitly RE-SORT every word by
    ``(start, end)`` rather than trusting segment/candidate order --
    itself evidence that the codebase's own authors did not treat
    "segment order already equals canonical global order" as a safe
    assumption to skip.

Conclusion: reconstructed-attempt word membership is contiguous in the
OVERWHELMING common case (ordinary, non-overlapping ASR timing, which
is what every real fixture and Video00 RAW to date has produced), but
is NOT PROVABLY ALWAYS CONTIGUOUS given the code's own tolerance for a
small adjacent-candidate overlap. Per the task's own "do not assume"
instruction, this module therefore uses the explicit, immutable index
SET representation (``word_indices: Tuple[int, ...]``, sorted/deduped)
for the reconstructed-attempt side, never a bare start/end range that
would silently misrepresent a hole or a foreign word.

The ``LanguageAttempt`` side is the opposite, and PROVABLY so: an
utterance's own ``phrase_start_index``/``phrase_end_index`` is built by
direct slicing (``_build_phrase``'s ``words[start_idx:end_idx + 1]``),
an attempt's own ``utterance_ids`` are always CONSECUTIVE utterances
(``build_language_attempts`` only ever ``buffer.append``s the NEXT
utterance in sequence, never skips one) -- so a ``LanguageAttempt``'s
word-index range is a genuine, provably contiguous composition of
already-contiguous ranges over the SAME canonical ordinal space
(``language_spine.LanguageWord.word_index``). This module still
materializes it as the same ``word_indices`` tuple shape (cheap for
real attempt sizes) so both sides compare uniformly through one set of
functions -- never two parallel comparison code paths.

## Word identity source (binding)

Every canonical word index used here is
``language_spine.LanguageWord.word_index`` -- the SAME numbering
``language_spine_live_integration.build_live_language_spine_for_source``
already assigns from ``RawUnderstandingMap.word_timings`` (itself built,
per D-235O's own forensic, from the exact same ``contracts.Word``
objects ``CandidateTake.words``/``TranscriptSegment.words`` already
carry -- no second ASR pass, ever). This module mints NO independent
second ordinal system; it MATCHES a reconstructed attempt's own
``Word`` values against the already-assigned ``LanguageWord.word_index``
values by exact ``(start, end, text)`` value equality (never Python
object identity -- ``id(word)`` -- so this survives serialization/
reconstruction), consuming canonical entries in non-decreasing order
(a stable positional multiset match, never a text-similarity search).
Word identity is always ``(source_asset_id, canonical_word_index)`` --
never compared across two different ``source_asset_id`` values.

## No fuzzy text, no timestamp authority, no numeric threshold

This module's own comparison logic (``classify_word_membership_
relationship``) operates ENTIRELY on ``frozenset`` operations over
integer word-index sets: equality, subset, intersection. It contains
no ``SequenceMatcher``/``difflib``/text-similarity call, no percentage/
IoU/overlap-ratio threshold, and reads no ``.start``/``.end`` timestamp
value as decision authority anywhere (timestamps are used only, once,
upstream by the already-existing ``LanguageWord``/``Word`` matching
step to establish exact value equality -- never as a fuzzy proximity
score here). Verified structurally by this module's own test suite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

from .contracts import CandidateTake, Word
from .language_spine import LanguagePhrase, LanguageWord
from .language_utterance_attempt import LanguageAttempt, LanguageUtterance

SCHEMA_VERSION = "cutsell.shared_attempt_word_identity.v1"

# ---------------------------------------------------------------------------
# Word-identity status vocabulary.
# ---------------------------------------------------------------------------
WORD_IDENTITY_AVAILABLE = "AVAILABLE"
WORD_IDENTITY_PARTIAL = "PARTIAL"
WORD_IDENTITY_MISSING = "MISSING_WORD_IDENTITY"

# ---------------------------------------------------------------------------
# Relationship vocabulary (this task's own required 8-value set).
# ---------------------------------------------------------------------------
RELATIONSHIP_EXACT_SAME_MEMBERSHIP = "EXACT_SAME_MEMBERSHIP"
RELATIONSHIP_EXACT_RECONSTRUCTED_CONTAINS_LANGUAGE = "EXACT_RECONSTRUCTED_CONTAINS_LANGUAGE"
RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED = "EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED"
RELATIONSHIP_ONE_RECONSTRUCTED_TO_MULTIPLE_LANGUAGE_ATTEMPTS_EXACT_PARTITION = (
    "ONE_RECONSTRUCTED_TO_MULTIPLE_LANGUAGE_ATTEMPTS_EXACT_PARTITION"
)
RELATIONSHIP_EXACT_PARTIAL_WORD_MEMBERSHIP_OVERLAP = "EXACT_PARTIAL_WORD_MEMBERSHIP_OVERLAP"
RELATIONSHIP_DISJOINT = "DISJOINT"
RELATIONSHIP_SOURCE_MISMATCH = "SOURCE_MISMATCH"
RELATIONSHIP_MISSING_WORD_IDENTITY = "MISSING_WORD_IDENTITY"
RELATIONSHIP_AMBIGUOUS = "AMBIGUOUS"

ALLOWED_RELATIONSHIP_STATUSES: frozenset[str] = frozenset({
    RELATIONSHIP_EXACT_SAME_MEMBERSHIP,
    RELATIONSHIP_EXACT_RECONSTRUCTED_CONTAINS_LANGUAGE,
    RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED,
    RELATIONSHIP_ONE_RECONSTRUCTED_TO_MULTIPLE_LANGUAGE_ATTEMPTS_EXACT_PARTITION,
    RELATIONSHIP_EXACT_PARTIAL_WORD_MEMBERSHIP_OVERLAP,
    RELATIONSHIP_DISJOINT,
    RELATIONSHIP_SOURCE_MISMATCH,
    RELATIONSHIP_MISSING_WORD_IDENTITY,
    RELATIONSHIP_AMBIGUOUS,
})

# Per this task's own "Correspondence Authority" section: only these two
# categories may produce an AUTHORITATIVE attempt<->proposition linkage.
# Containment is reported (evidence), never auto-promoted; partial overlap
# is NEVER promoted to identity.
AUTHORITATIVE_RELATIONSHIP_STATUSES: frozenset[str] = frozenset({
    RELATIONSHIP_EXACT_SAME_MEMBERSHIP,
    RELATIONSHIP_ONE_RECONSTRUCTED_TO_MULTIPLE_LANGUAGE_ATTEMPTS_EXACT_PARTITION,
})

# ---------------------------------------------------------------------------
# Identity-source vocabulary for the P1 adapter (Part 4) -- the existing
# heuristic fallback must always be labelled honestly, never silently
# reported as exact.
# ---------------------------------------------------------------------------
IDENTITY_SOURCE_EXACT_WORD_MEMBERSHIP = "EXACT_WORD_MEMBERSHIP"
IDENTITY_SOURCE_HEURISTIC_OVERLAP = "HEURISTIC_OVERLAP"
IDENTITY_SOURCE_MISSING = "MISSING"

PROVENANCE_CANONICAL_WORD_MEMBERSHIP = "CANONICAL_WORD_MEMBERSHIP"


# ---------------------------------------------------------------------------
# WordMembership
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WordMembership:
    """One entity's (a reconstructed attempt OR a ``LanguageAttempt``)
    exact canonical word-index membership set, source-scoped. Never
    compared across two different ``source_asset_id`` values by any
    function in this module."""
    source_asset_id: str
    entity_id: str
    word_indices: Tuple[int, ...]
    identity_status: str  # AVAILABLE / PARTIAL / MISSING_WORD_IDENTITY


def _canonical_words_for_source(
    canonical_words: Iterable[LanguageWord], source_asset_id: str,
) -> Tuple[LanguageWord, ...]:
    same_source = tuple(w for w in canonical_words if w.source_asset_id == source_asset_id)
    return tuple(sorted(same_source, key=lambda w: w.word_index))


def _match_word_indices(
    member_words: Tuple[Word, ...], ordered_canonical: Tuple[LanguageWord, ...],
) -> Tuple[list[int], int]:
    """Deterministic, monotonic positional match: for each of
    ``member_words`` (assumed already time-ordered by construction, per
    this module's own docstring -- defensively re-sorted here anyway),
    finds the first UNCONSUMED canonical entry with an EXACT
    ``(start, end, text)`` value match, never Python object identity.
    Returns ``(indices_or_none_per_word, matched_count)`` -- a ``None``
    entry means that member word could not be matched (fail-open, never
    a guessed nearest match)."""
    ordered_members = tuple(sorted(member_words, key=lambda w: (w.start, w.end)))
    n = len(ordered_canonical)
    canon_ptr = 0
    indices: list[int | None] = []
    matched = 0
    for word in ordered_members:
        found: int | None = None
        probe = canon_ptr
        target = (float(word.start), float(word.end), str(word.text or ""))
        while probe < n:
            cw = ordered_canonical[probe]
            candidate_key = (float(cw.start), float(cw.end), str(cw.text_raw or ""))
            if candidate_key == target:
                found = probe
                break
            if (float(cw.start), float(cw.end)) > (target[0], target[1]):
                break  # canonical list has moved strictly past this word's timing
            probe += 1
        if found is not None:
            indices.append(found)
            canon_ptr = found + 1
            matched += 1
        else:
            indices.append(None)
    return indices, matched


def build_reconstructed_attempt_word_membership(
    candidate: CandidateTake, canonical_words: Iterable[LanguageWord],
) -> WordMembership:
    """Pure builder for the reconstructed-attempt (``CandidateTake``, post
    ``attempt_reconstruction.py``) side. Never reads/writes ``candidate.
    word_indices`` (the additive-only, unpopulated-by-any-live-call-site
    schema field on ``contracts.CandidateTake``) -- computes fresh, every
    time, from ``candidate.words`` against the supplied canonical word
    list, per this module's own OFFLINE, no-live-wiring scope."""
    entity_id = str(candidate.attempt_id or candidate.source_span_id or candidate.clip_id)
    member_words = tuple(candidate.words or ())
    if not member_words:
        return WordMembership(
            source_asset_id=candidate.source_asset_id, entity_id=entity_id,
            word_indices=(), identity_status=WORD_IDENTITY_MISSING,
        )
    ordered_canonical = _canonical_words_for_source(canonical_words, candidate.source_asset_id)
    if not ordered_canonical:
        return WordMembership(
            source_asset_id=candidate.source_asset_id, entity_id=entity_id,
            word_indices=(), identity_status=WORD_IDENTITY_MISSING,
        )
    indices, matched = _match_word_indices(member_words, ordered_canonical)
    resolved = tuple(sorted({i for i in indices if i is not None}))
    if matched == 0:
        status = WORD_IDENTITY_MISSING
    elif matched < len(member_words):
        status = WORD_IDENTITY_PARTIAL
    else:
        status = WORD_IDENTITY_AVAILABLE
    return WordMembership(
        source_asset_id=candidate.source_asset_id, entity_id=entity_id,
        word_indices=resolved, identity_status=status,
    )


def build_language_attempt_word_membership(
    attempt: LanguageAttempt,
    utterances_by_id: Mapping[str, LanguageUtterance],
    phrases: Tuple[LanguagePhrase, ...],
) -> WordMembership:
    """Pure builder for the ``LanguageAttempt`` side -- derives the exact
    (provably contiguous, per this module's own docstring proof) word-
    index range via the EXISTING hierarchy (``utterance_ids`` ->
    ``phrase_start_index``/``phrase_end_index`` -> ``word_start_index``/
    ``word_end_index``), never a fresh re-segmentation. Materialized as
    the same ``word_indices`` tuple shape the reconstructed side uses, so
    both sides compare through one set of functions."""
    resolved: set[int] = set()
    total_members = len(attempt.utterance_ids)
    matched_members = 0
    for utterance_id in attempt.utterance_ids:
        utterance = utterances_by_id.get(utterance_id)
        if utterance is None:
            continue
        if not (0 <= utterance.phrase_start_index <= utterance.phrase_end_index < len(phrases)):
            continue
        start_phrase = phrases[utterance.phrase_start_index]
        end_phrase = phrases[utterance.phrase_end_index]
        resolved.update(range(start_phrase.word_start_index, end_phrase.word_end_index + 1))
        matched_members += 1
    if total_members == 0 or matched_members == 0:
        status = WORD_IDENTITY_MISSING
    elif matched_members < total_members:
        status = WORD_IDENTITY_PARTIAL
    else:
        status = WORD_IDENTITY_AVAILABLE
    return WordMembership(
        source_asset_id=attempt.source_asset_id, entity_id=attempt.attempt_id,
        word_indices=tuple(sorted(resolved)), identity_status=status,
    )


# ---------------------------------------------------------------------------
# Pairwise + N-way exact bridge.
# ---------------------------------------------------------------------------
def classify_word_membership_relationship(
    reconstructed: WordMembership, language: WordMembership,
) -> str:
    """The one canonical PAIRWISE relationship classifier this task
    requires. Pure set arithmetic only -- see module docstring's "No
    fuzzy text, no timestamp authority" section."""
    if reconstructed.source_asset_id != language.source_asset_id:
        return RELATIONSHIP_SOURCE_MISMATCH
    if reconstructed.identity_status == WORD_IDENTITY_MISSING or language.identity_status == WORD_IDENTITY_MISSING:
        return RELATIONSHIP_MISSING_WORD_IDENTITY
    r = frozenset(reconstructed.word_indices)
    l = frozenset(language.word_indices)
    if not r or not l:
        return RELATIONSHIP_MISSING_WORD_IDENTITY
    if reconstructed.identity_status == WORD_IDENTITY_PARTIAL or language.identity_status == WORD_IDENTITY_PARTIAL:
        return RELATIONSHIP_AMBIGUOUS
    if r == l:
        return RELATIONSHIP_EXACT_SAME_MEMBERSHIP
    if l < r:
        return RELATIONSHIP_EXACT_RECONSTRUCTED_CONTAINS_LANGUAGE
    if r < l:
        return RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED
    if r & l:
        return RELATIONSHIP_EXACT_PARTIAL_WORD_MEMBERSHIP_OVERLAP
    return RELATIONSHIP_DISJOINT


@dataclass(frozen=True)
class AttemptLanguageIdentityMatch:
    """The exact bridge result for ONE reconstructed attempt against the
    full set of candidate ``LanguageAttempt``s for its source. No score --
    a categorical ``relationship_status`` only, per this task's own
    "no numeric threshold" instruction."""
    reconstructed_attempt_id: str
    language_attempt_ids: Tuple[str, ...]
    source_asset_id: str
    reconstructed_word_membership: WordMembership
    language_word_memberships: Tuple[WordMembership, ...]
    relationship_status: str
    exact_shared_word_count: int
    reconstructed_word_count: int
    language_word_count: int
    provenance: Tuple[str, ...]


def _is_partition(reconstructed_set: frozenset, memberships: Tuple[WordMembership, ...]) -> bool:
    """True iff ``memberships`` (each already known to intersect
    ``reconstructed_set``) are pairwise disjoint AND their union exactly
    equals ``reconstructed_set`` -- "no foreign words, no missing
    canonical words, no duplicate ownership" per this task's own
    Partition Contract, checked with exact set arithmetic only."""
    if len(memberships) < 2:
        return False
    total_len = 0
    union: set[int] = set()
    for m in memberships:
        s = frozenset(m.word_indices)
        if s & union:
            return False  # duplicate ownership -- not a clean partition
        union |= s
        total_len += len(s)
    return union == reconstructed_set and total_len == len(union)


def match_reconstructed_attempt_against_language_attempts(
    reconstructed: WordMembership, language_memberships: Tuple[WordMembership, ...],
) -> AttemptLanguageIdentityMatch:
    """The one canonical N-way matcher this task requires. Supports 1->N
    (this reconstructed attempt subdivided into several LanguageAttempts,
    an exact partition) and is symmetric under the caller iterating the
    reverse direction for N->1 (see module diagnostics helpers below) --
    never forces a false 1:1."""
    same_source = tuple(m for m in language_memberships if m.source_asset_id == reconstructed.source_asset_id)

    if reconstructed.identity_status == WORD_IDENTITY_MISSING or not reconstructed.word_indices:
        status = RELATIONSHIP_MISSING_WORD_IDENTITY
        return AttemptLanguageIdentityMatch(
            reconstructed_attempt_id=reconstructed.entity_id, language_attempt_ids=(),
            source_asset_id=reconstructed.source_asset_id,
            reconstructed_word_membership=reconstructed, language_word_memberships=(),
            relationship_status=status, exact_shared_word_count=0,
            reconstructed_word_count=len(reconstructed.word_indices), language_word_count=0,
            provenance=(PROVENANCE_CANONICAL_WORD_MEMBERSHIP,),
        )

    if not same_source:
        status = RELATIONSHIP_SOURCE_MISMATCH if language_memberships else RELATIONSHIP_MISSING_WORD_IDENTITY
        return AttemptLanguageIdentityMatch(
            reconstructed_attempt_id=reconstructed.entity_id, language_attempt_ids=(),
            source_asset_id=reconstructed.source_asset_id,
            reconstructed_word_membership=reconstructed, language_word_memberships=(),
            relationship_status=status, exact_shared_word_count=0,
            reconstructed_word_count=len(reconstructed.word_indices), language_word_count=0,
            provenance=(PROVENANCE_CANONICAL_WORD_MEMBERSHIP,),
        )

    if reconstructed.identity_status == WORD_IDENTITY_PARTIAL:
        return AttemptLanguageIdentityMatch(
            reconstructed_attempt_id=reconstructed.entity_id,
            language_attempt_ids=tuple(sorted(m.entity_id for m in same_source)),
            source_asset_id=reconstructed.source_asset_id,
            reconstructed_word_membership=reconstructed, language_word_memberships=same_source,
            relationship_status=RELATIONSHIP_AMBIGUOUS, exact_shared_word_count=0,
            reconstructed_word_count=len(reconstructed.word_indices),
            language_word_count=len(set().union(*(frozenset(m.word_indices) for m in same_source)) if same_source else ()),
            provenance=(PROVENANCE_CANONICAL_WORD_MEMBERSHIP,),
        )

    r = frozenset(reconstructed.word_indices)

    # Exact 1:1 same-membership takes precedence over every other shape.
    exact_matches = tuple(
        m for m in same_source
        if m.identity_status == WORD_IDENTITY_AVAILABLE and frozenset(m.word_indices) == r
    )
    if len(exact_matches) == 1:
        m = exact_matches[0]
        return _build_match(reconstructed, (m,), RELATIONSHIP_EXACT_SAME_MEMBERSHIP, r, r)

    # Overlapping candidates only (ignore anything wholly disjoint) --
    # a partition/containment/partial verdict can only involve these.
    overlapping = tuple(
        m for m in same_source
        if m.identity_status == WORD_IDENTITY_AVAILABLE and (frozenset(m.word_indices) & r)
    )
    ambiguous_overlap = any(
        m for m in same_source
        if m.identity_status == WORD_IDENTITY_PARTIAL and (frozenset(m.word_indices) & r)
    )

    if not overlapping:
        if ambiguous_overlap:
            return _build_match(reconstructed, tuple(
                m for m in same_source if m.identity_status == WORD_IDENTITY_PARTIAL and (frozenset(m.word_indices) & r)
            ), RELATIONSHIP_AMBIGUOUS, r, frozenset())
        return _build_match(reconstructed, (), RELATIONSHIP_DISJOINT, r, frozenset())

    if _is_partition(r, overlapping):
        union = frozenset().union(*(frozenset(m.word_indices) for m in overlapping))
        return _build_match(
            reconstructed, overlapping,
            RELATIONSHIP_ONE_RECONSTRUCTED_TO_MULTIPLE_LANGUAGE_ATTEMPTS_EXACT_PARTITION, r, union,
        )

    if len(overlapping) == 1:
        rel = classify_word_membership_relationship(reconstructed, overlapping[0])
        if rel in (
            RELATIONSHIP_EXACT_RECONSTRUCTED_CONTAINS_LANGUAGE,
            RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED,
            RELATIONSHIP_EXACT_PARTIAL_WORD_MEMBERSHIP_OVERLAP,
        ):
            union = frozenset(overlapping[0].word_indices)
            return _build_match(reconstructed, overlapping, rel, r, union)

    union = frozenset().union(*(frozenset(m.word_indices) for m in overlapping))
    return _build_match(
        reconstructed, overlapping, RELATIONSHIP_EXACT_PARTIAL_WORD_MEMBERSHIP_OVERLAP, r, union,
    )


def _build_match(
    reconstructed: WordMembership, language_memberships: Tuple[WordMembership, ...],
    status: str, reconstructed_set: frozenset, language_union: frozenset,
) -> AttemptLanguageIdentityMatch:
    return AttemptLanguageIdentityMatch(
        reconstructed_attempt_id=reconstructed.entity_id,
        language_attempt_ids=tuple(sorted(m.entity_id for m in language_memberships)),
        source_asset_id=reconstructed.source_asset_id,
        reconstructed_word_membership=reconstructed,
        language_word_memberships=language_memberships,
        relationship_status=status,
        exact_shared_word_count=len(reconstructed_set & language_union),
        reconstructed_word_count=len(reconstructed_set),
        language_word_count=len(language_union),
        provenance=(PROVENANCE_CANONICAL_WORD_MEMBERSHIP,),
    )


def build_attempt_language_identity_matches_for_source(
    *,
    reconstructed_attempts: Tuple[CandidateTake, ...],
    canonical_words: Tuple[LanguageWord, ...],
    language_attempts: Tuple[LanguageAttempt, ...],
    utterances_by_id: Mapping[str, LanguageUtterance],
    phrases: Tuple[LanguagePhrase, ...],
) -> Tuple[AttemptLanguageIdentityMatch, ...]:
    """Batch, per-source form: one ``AttemptLanguageIdentityMatch`` per
    ``reconstructed_attempts`` entry, in input order. Pure; no I/O."""
    language_memberships = tuple(
        build_language_attempt_word_membership(a, utterances_by_id, phrases) for a in language_attempts
    )
    return tuple(
        match_reconstructed_attempt_against_language_attempts(
            build_reconstructed_attempt_word_membership(c, canonical_words), language_memberships,
        )
        for c in reconstructed_attempts
    )


def exact_proposition_candidate_ids_for_match(
    match: AttemptLanguageIdentityMatch,
    proposition_candidate_ids_by_attempt_id: Mapping[str, Tuple[str, ...]],
) -> Tuple[str, ...]:
    """Part 3.5 -- once an AUTHORITATIVE relationship is established
    (``AUTHORITATIVE_RELATIONSHIP_STATUSES`` only; containment and
    partial overlap are evidence, never auto-promoted, per this task's
    own "Correspondence Authority" section), returns the exact, bounded
    SET of proposition-candidate ids for the matched ``LanguageAttempt``
    set -- a set, deliberately, since a 1->N partition may span more
    than one proposition and this module NEVER guesses which individual
    proposition owns a specific lost atom (out of scope, per this task's
    own "Do Not Solve Atom Ownership Yet" section)."""
    if match.relationship_status not in AUTHORITATIVE_RELATIONSHIP_STATUSES:
        return ()
    ids: set[str] = set()
    for language_attempt_id in match.language_attempt_ids:
        ids.update(proposition_candidate_ids_by_attempt_id.get(language_attempt_id, ()))
    return tuple(sorted(ids))


# ---------------------------------------------------------------------------
# Part 4 -- P1 exact identity adapter (bounded helper; not wired into any
# live P1 call site by this task -- see module docstring).
# ---------------------------------------------------------------------------
def p1_identity_provenance_for_clip(
    *,
    exact_match: AttemptLanguageIdentityMatch | None,
    heuristic_attempt_id: str | None,
    proposition_candidate_ids_by_attempt_id: Mapping[str, Tuple[str, ...]],
) -> dict:
    """Bounded provenance-only projection P1 (unchanged, byte-identical)
    could later consume: exact word-membership identity wins whenever it
    is AUTHORITATIVE; the existing D-199 maximum-overlap bridge is
    honestly labelled ``HEURISTIC_OVERLAP`` (never silently reported as
    exact) when exact identity is unavailable; ``MISSING`` when neither
    exists. Never invoked from any live P1 call site by this task."""
    if exact_match is not None and exact_match.relationship_status in AUTHORITATIVE_RELATIONSHIP_STATUSES:
        return {
            "identity_source": IDENTITY_SOURCE_EXACT_WORD_MEMBERSHIP,
            "language_attempt_ids": exact_match.language_attempt_ids,
            "exact_proposition_candidate_ids": exact_proposition_candidate_ids_for_match(
                exact_match, proposition_candidate_ids_by_attempt_id,
            ),
            "relationship_status": exact_match.relationship_status,
        }
    if heuristic_attempt_id is not None:
        return {
            "identity_source": IDENTITY_SOURCE_HEURISTIC_OVERLAP,
            "language_attempt_ids": (heuristic_attempt_id,),
            "exact_proposition_candidate_ids": (),
            "relationship_status": exact_match.relationship_status if exact_match is not None else None,
        }
    return {
        "identity_source": IDENTITY_SOURCE_MISSING,
        "language_attempt_ids": (),
        "exact_proposition_candidate_ids": (),
        "relationship_status": exact_match.relationship_status if exact_match is not None else None,
    }


# ---------------------------------------------------------------------------
# Diagnostics (tail-safe, no transcript dump -- same pattern as every
# other D-19x/D-235x compact summary in this codebase).
# ---------------------------------------------------------------------------
def attempt_language_identity_match_diagnostics(
    match: AttemptLanguageIdentityMatch,
    proposition_candidate_ids_by_attempt_id: Mapping[str, Tuple[str, ...]] | None = None,
) -> dict:
    ids_map = proposition_candidate_ids_by_attempt_id or {}
    exact_ids = exact_proposition_candidate_ids_for_match(match, ids_map)
    return {
        "exact_identity_available": match.relationship_status in AUTHORITATIVE_RELATIONSHIP_STATUSES,
        "identity_source": (
            IDENTITY_SOURCE_EXACT_WORD_MEMBERSHIP
            if match.relationship_status in AUTHORITATIVE_RELATIONSHIP_STATUSES
            else IDENTITY_SOURCE_MISSING
        ),
        "reconstructed_attempt_id": match.reconstructed_attempt_id,
        "language_attempt_ids": list(match.language_attempt_ids),
        "exact_proposition_candidate_ids": list(exact_ids),
        "relationship_status": match.relationship_status,
        "canonical_word_count_reconstructed": match.reconstructed_word_count,
        "canonical_word_count_language_union": match.language_word_count,
        "missing_identity_reason": (
            match.relationship_status
            if match.relationship_status in (
                RELATIONSHIP_MISSING_WORD_IDENTITY, RELATIONSHIP_SOURCE_MISMATCH, RELATIONSHIP_AMBIGUOUS,
            ) else None
        ),
    }


def shared_attempt_word_identity_diagnostics(matches: Iterable[AttemptLanguageIdentityMatch]) -> dict:
    """Batch, counts-only CI summary."""
    matches = tuple(matches)
    status_counts: dict[str, int] = {}
    for m in matches:
        status_counts[m.relationship_status] = status_counts.get(m.relationship_status, 0) + 1
    exact_count = sum(1 for m in matches if m.relationship_status in AUTHORITATIVE_RELATIONSHIP_STATUSES)
    return {
        "schema_version": SCHEMA_VERSION,
        "match_count": len(matches),
        "exact_identity_available_count": exact_count,
        "relationship_status_counts": status_counts,
        "missing_identity_count": sum(1 for m in matches if m.relationship_status == RELATIONSHIP_MISSING_WORD_IDENTITY),
    }
