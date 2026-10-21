"""D-168: Language / Transcript Spine, Phase B -- typed LANGUAGEUTTERANCE +
LANGUAGEATTEMPT, built deterministically from D-166's LanguageWord/
LanguagePhrase plus existing segmentation/attempt evidence.

See ``docs/CUTSELL_CANONICAL_ENGINE_ARCHITECTURE_D098.md`` Section 14
(D-165's canonical design, 14.21 Phase B) and ``docs/CUTSELL_DECISIONS.md``
D-168 for the full design rationale. This module implements ONLY the third
and fourth rungs of D-165's canonical hierarchy:

    LanguageWord -> LanguagePhrase -> LanguageUtterance -> LanguageAttempt

``PropositionCandidate`` and ``RelationEvidence`` (D-165's own Phase C gate)
are explicitly OUT of scope here -- this module mints no final
``proposition_id``/``retry_family_id`` and creates no Family/BestTake/
Proposition/final-Relation authority. It is not called by any production
call site: `pipeline.py`, `flow_b.py`, `take_segmentation.py`,
`attempt_reconstruction.py`, `take_grouping*.py`,
`hybrid_session_cleanup.py`, `semantic_idea_equivalence.py`,
`deterministic_best_take_authority.py`, `take_judge.py`,
`watch_listen_besttake_evidence.py`, `watch_listen_zone_usability_v2.py`,
`boundary_engine_pass.py`, `dialogue_pacing_transition.py`, and
`semantic_authority_observability.py` are all confirmed unaware of this
module's existence (module-leaf grep tests,
`tests/test_cutsell_d168_language_utterance_attempt.py`).

## No mass migration (this task's own explicit instruction)

`take_segmentation.py` and `attempt_reconstruction.py` are left completely
BYTE-IDENTICAL and untouched. This module REUSES their already-vetted pure
helpers directly by import -- `take_segmentation._looks_complete_idea`/
`_ends_sentence`/`_grammatically_open_tail`/`_trails_off` for meaning
completion, `attempt_reconstruction._restart_evidence` for lexical restart
detection -- rather than writing a competing completion/restart engine, per
this task's own "Do NOT create a competing new completion engine" and
"Do not require exact lexical duplication" instructions. This is the SAME
reuse-by-direct-import precedent D-157's `watch_listen_understanding.py`
and D-163's `case_b_performance_evidence.py` already established for a
sibling module's private helpers.

## Two-pass construction (why)

Utterance BOUNDARIES (where a phrase run is cut) are decided purely from
structural evidence (`LanguagePhrase.boundary_kind` -- a real audio-pause,
a restart marker, or terminal punctuation). Meaning completion is decided
SEPARATELY, by asking `take_segmentation._looks_complete_idea` about the
accumulated text -- this is exactly what implements this task's own "Do
not equate punctuation with utterance completion" requirement: a phrase
ending in a period is only a completion CANDIDATE, never an automatic
completion verdict.

Utterance-to-utterance RELATION (false start vs. abandoned, correction vs.
plain retry, continuation vs. a new audience beat) needs to look at a pair
of already-built utterances, so it runs as an explicit second pass over the
Pass 1 output -- the same sequential, pairwise, "look at the immediate
predecessor" architecture `watch_listen_understanding.py`'s own
`_relation_for_pair` already uses (a proven shape in this codebase, not a
new design).

## False start vs. abandoned attempt (no new threshold zoo)

Per this task's own "Do not create a new threshold zoo" instruction, the
FALSE_START/ABANDONED_ATTEMPT split reuses the EXACT SAME two numbers
`take_segmentation._looks_complete_idea` already uses for its own
short-fallback heuristic (>=6 words OR >=3.0s duration "looks complete
enough" even without terminal punctuation) -- an abandoned utterance that
is BOTH under 6 words AND under 3.0s is classified FALSE_START (a quick,
immediate rephrase); anything longer is ABANDONED_ATTEMPT (a fuller
delivery that was dropped). No new number is introduced.

## Recording process (no invented phrase lists)

`recording_process_evidence` is populated ONLY from an explicit, caller-
supplied index set (`recording_process_utterance_indices`) -- this module
invents no meta-speech/self-instruction/discard-commentary phrase
vocabulary of its own (CLAUDE.md's binding "Never hardcode ... phrases"
rule, and this task's own "Do not create person-specific phrases"
instruction). A future, separately-authorized task may correlate this
against `raw_understanding_map.BEHAVIOR_RECORDING_PROCESS` (D-155,
unchanged) evidence for the same source span; that correlation is not
performed here.

## Confidence and provenance

Categorical only (`SUPPORTED`/`WEAK`/`MIXED`/`UNKNOWN`) -- `SUPPORTED`/
`WEAK`/`UNKNOWN` are the exact string values `language_spine.py` already
uses (imported, not redefined with different spellings); `MIXED` is added
here (this module's own directive requires it) using the identical spelling
`watch_listen_understanding.py` already uses for the same concept.
Provenance reuses `language_spine.py`'s vocabulary by import and adds two
new tags this task's directive requires (`UTTERANCE_SEGMENTATION`,
`ATTEMPT_RECONSTRUCTION`), defined HERE rather than added to
`language_spine.py` itself (D-166, CLOSED, stays at zero diff).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence, Tuple

from .attempt_reconstruction import _restart_evidence
from .language_spine import (
    BOUNDARY_END_OF_UTTERANCE,
    BOUNDARY_PAUSE,
    BOUNDARY_PUNCTUATION,
    BOUNDARY_RESTART_BOUNDARY,
    BOUNDARY_SPEECH_BOUNDARY,
    BOUNDARY_UNKNOWN,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
    LanguagePhrase,
    normalize_language_text,
)
from .take_segmentation import _looks_complete_idea

SCHEMA_VERSION = "cutsell.language_utterance_attempt.v1"

# ---------------------------------------------------------------------------
# Provenance (reuses language_spine.py's vocabulary; adds the two new tags
# this task's directive requires, defined here since language_spine.py
# (D-166) stays at zero diff -- see module docstring).
# ---------------------------------------------------------------------------
PROVENANCE_UTTERANCE_SEGMENTATION = "UTTERANCE_SEGMENTATION"
PROVENANCE_ATTEMPT_RECONSTRUCTION = "ATTEMPT_RECONSTRUCTION"

# ---------------------------------------------------------------------------
# Confidence vocabulary (SUPPORTED/WEAK/UNKNOWN reused verbatim from
# language_spine.py; MIXED added here per this task's own directive, same
# spelling watch_listen_understanding.py already uses for the concept).
# ---------------------------------------------------------------------------
CONFIDENCE_MIXED = "MIXED"
_CONFIDENCE_RANK: Mapping[str, int] = {
    CONFIDENCE_SUPPORTED: 2, CONFIDENCE_WEAK: 1, CONFIDENCE_UNKNOWN: 0,
}

# ---------------------------------------------------------------------------
# Meaning-completion vocabulary (shared string values between an utterance's
# own completion status and the coarser utterance_state -- an utterance with
# no restart/correction context IS simply its completion state).
# ---------------------------------------------------------------------------
MEANING_COMPLETE = "COMPLETE"
MEANING_INCOMPLETE = "INCOMPLETE"
MEANING_UNCERTAIN = "UNCERTAIN"

# ---------------------------------------------------------------------------
# Utterance state vocabulary (this task's own required 6-value set).
# ---------------------------------------------------------------------------
UTTERANCE_COMPLETE = MEANING_COMPLETE
UTTERANCE_INCOMPLETE = MEANING_INCOMPLETE
UTTERANCE_ABANDONED = "ABANDONED"
UTTERANCE_RESTARTED = "RESTARTED"
UTTERANCE_CORRECTED = "CORRECTED"
UTTERANCE_UNCERTAIN = MEANING_UNCERTAIN
ALLOWED_UTTERANCE_STATES: frozenset[str] = frozenset({
    UTTERANCE_COMPLETE, UTTERANCE_INCOMPLETE, UTTERANCE_ABANDONED,
    UTTERANCE_RESTARTED, UTTERANCE_CORRECTED, UTTERANCE_UNCERTAIN,
})

# ---------------------------------------------------------------------------
# Attempt state vocabulary (this task's own required 7-value set).
# ---------------------------------------------------------------------------
ATTEMPT_CLEAN = "CLEAN_ATTEMPT"
ATTEMPT_FALSE_START = "FALSE_START"
ATTEMPT_ABANDONED = "ABANDONED_ATTEMPT"
ATTEMPT_CORRECTION = "CORRECTION"
ATTEMPT_CONTINUATION = "CONTINUATION"
ATTEMPT_RECORDING_PROCESS = "RECORDING_PROCESS"
ATTEMPT_UNCERTAIN = "UNCERTAIN"
ALLOWED_ATTEMPT_STATES: frozenset[str] = frozenset({
    ATTEMPT_CLEAN, ATTEMPT_FALSE_START, ATTEMPT_ABANDONED, ATTEMPT_CORRECTION,
    ATTEMPT_CONTINUATION, ATTEMPT_RECORDING_PROCESS, ATTEMPT_UNCERTAIN,
})

# Mirrors attempt_reconstruction.reconstruct_delivery_attempts's own
# already-vetted default (1.20s) -- the SAME production constant, not a new
# number invented for this module (same precedent watch_listen_
# understanding.py's own _DEFAULT_MAX_CONTINUATION_GAP_SEC already set).
_DEFAULT_MAX_CONTINUATION_GAP_SEC = 1.20

# Reuses take_segmentation._looks_complete_idea's OWN short-fallback
# thresholds (>=6 words OR >=3.0s "looks complete enough") to distinguish a
# brief FALSE_START from a fuller ABANDONED_ATTEMPT -- see module docstring
# "False start vs. abandoned attempt". Not a new number.
_BRIEF_WORD_CEILING = 6
_BRIEF_DURATION_CEILING_SEC = 3.0

_STRONG_BOUNDARY_KINDS = frozenset({BOUNDARY_RESTART_BOUNDARY, BOUNDARY_PAUSE, BOUNDARY_END_OF_UTTERANCE})
_WEAK_BOUNDARY_KINDS = frozenset({BOUNDARY_PUNCTUATION, BOUNDARY_SPEECH_BOUNDARY})


def _word_count(text: str) -> int:
    return len([tok for tok in str(text or "").split() if tok.strip()])


def _utterance_id(source_asset_id: str, start: float, end: float, text_normalized: str) -> str:
    """Deterministic, physical-observation-style id -- mirrors
    ``canonical_identity.mint_source_span_id``'s exact hashing shape and
    ``language_spine._phrase_id``'s own precedent (a distinct prefix so this
    id is never confusable with either), minted here rather than registered
    in ``canonical_identity.py`` because nothing yet reads this id to make
    an editorial decision (D-050A's own precedent)."""
    raw = f"{source_asset_id}|{float(start):.3f}|{float(end):.3f}|{text_normalized}".encode("utf-8")
    return "lutt_" + hashlib.sha256(raw).hexdigest()[:20]


def _attempt_id(member_utterance_ids: Sequence[str]) -> str:
    """Deterministic, MEMBERSHIP-anchored id -- mirrors
    ``canonical_identity.mint_attempt_id``'s exact shape (sorted member id
    set, never timestamp-anchored) under a distinct prefix, minted here for
    the same D-050A "no consumer yet" reason as ``_utterance_id`` above."""
    raw = "|".join(sorted(str(v) for v in member_utterance_ids if v)).encode("utf-8")
    return "latt_" + hashlib.sha256(raw).hexdigest()[:20]


# ---------------------------------------------------------------------------
# LanguageUtterance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LanguageUtterance:
    """One canonical, typed, bounded editorially-meaningful spoken unit --
    NOT necessarily a grammatical sentence (this task's own definition).
    Maps back exactly to its child ``LanguagePhrase`` range: ``source_start``/
    ``source_end`` always equal ``phrases[phrase_start_index].source_start``/
    ``phrases[phrase_end_index].source_end`` for the phrase tuple this
    utterance was segmented from (see ``source_mapping_valid`` diagnostic)."""
    source_asset_id: str
    utterance_id: str
    phrase_start_index: int
    phrase_end_index: int
    source_start: float
    source_end: float
    text_raw: str
    text_normalized: str
    utterance_state: str
    meaning_completion: str
    boundary_start_kind: str
    boundary_end_kind: str
    confidence: str
    provenance: str

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.source_end - self.source_start)


def _classify_meaning(text_raw: str, duration_sec: float) -> str:
    if not str(text_raw or "").strip():
        return MEANING_UNCERTAIN
    return MEANING_COMPLETE if _looks_complete_idea(text_raw, duration_sec) else MEANING_INCOMPLETE


def _utterance_confidence(boundary_end_kind: str, meaning: str) -> str:
    if meaning == MEANING_UNCERTAIN:
        return CONFIDENCE_UNKNOWN
    if boundary_end_kind in _STRONG_BOUNDARY_KINDS:
        return CONFIDENCE_SUPPORTED
    if boundary_end_kind in _WEAK_BOUNDARY_KINDS:
        return CONFIDENCE_WEAK
    return CONFIDENCE_UNKNOWN


def _pass1_raw_utterances(phrases: Tuple[LanguagePhrase, ...]) -> list[dict]:
    """Pass 1: cut phrases into utterance runs purely on structural boundary
    evidence (strong: RESTART_BOUNDARY/PAUSE/END_OF_UTTERANCE; weak signals
    never force a cut on their own), then classify each run's OWN meaning
    completion independently -- see module docstring's "Two-pass
    construction" section. Returns plain dicts (not yet the frozen
    dataclass, not yet pairwise-classified) for pass 2 to mutate/finalize."""
    if not phrases:
        return []

    raw: list[dict] = []
    run_start = 0
    previous_boundary_kind = BOUNDARY_UNKNOWN
    for index, phrase in enumerate(phrases):
        is_last = index == len(phrases) - 1
        if phrase.boundary_kind in _STRONG_BOUNDARY_KINDS or is_last:
            run = phrases[run_start:index + 1]
            text_raw = " ".join(p.text_raw.strip() for p in run if p.text_raw.strip())
            text_normalized = normalize_language_text(text_raw)
            source_start = run[0].source_start
            source_end = run[-1].source_end
            meaning = _classify_meaning(text_raw, max(0.0, source_end - source_start))
            boundary_end_kind = phrase.boundary_kind
            raw.append({
                "phrase_start_index": run_start,
                "phrase_end_index": index,
                "source_start": source_start,
                "source_end": source_end,
                "text_raw": text_raw,
                "text_normalized": text_normalized,
                "meaning_completion": meaning,
                "boundary_start_kind": previous_boundary_kind,
                "boundary_end_kind": boundary_end_kind,
                "source_asset_id": run[0].source_asset_id,
            })
            previous_boundary_kind = phrase.boundary_kind
            run_start = index + 1
    return raw


def segment_language_utterances(phrases: Tuple[LanguagePhrase, ...]) -> Tuple[LanguageUtterance, ...]:
    """The one canonical Phrase -> Utterance segmenter this task requires.
    Pure function; no I/O, no provider/network call. See module docstring
    for the two-pass design (structural cut, then pairwise ABANDONED/
    RESTARTED/CORRECTED relation classification via ``_restart_evidence``,
    reused verbatim from ``attempt_reconstruction.py``)."""
    raw = _pass1_raw_utterances(phrases)
    if not raw:
        return ()

    states: list[str] = [row["meaning_completion"] for row in raw]  # baseline: state == own completion
    for i in range(len(raw) - 1):
        left, right = raw[i], raw[i + 1]
        if not _restart_evidence(left["text_raw"], right["text_raw"]):
            continue
        if left["meaning_completion"] == MEANING_INCOMPLETE:
            states[i] = UTTERANCE_ABANDONED
            # `right` keeps its own baseline completion-derived state --
            # it is the fresh clean retry, not itself flagged.
        elif left["meaning_completion"] == MEANING_COMPLETE:
            states[i] = UTTERANCE_RESTARTED
            states[i + 1] = UTTERANCE_CORRECTED
        # left["meaning_completion"] == UNCERTAIN: no confident relation to
        # assert -- both sides keep their own baseline state ("WHEN
        # UNCERTAIN, KEEP" -- fail toward NOT asserting a relation).

    utterances: list[LanguageUtterance] = []
    for row, state in zip(raw, states):
        confidence = _utterance_confidence(row["boundary_end_kind"], row["meaning_completion"])
        utterances.append(LanguageUtterance(
            source_asset_id=row["source_asset_id"],
            utterance_id=_utterance_id(row["source_asset_id"], row["source_start"], row["source_end"], row["text_normalized"]),
            phrase_start_index=row["phrase_start_index"],
            phrase_end_index=row["phrase_end_index"],
            source_start=row["source_start"],
            source_end=row["source_end"],
            text_raw=row["text_raw"],
            text_normalized=row["text_normalized"],
            utterance_state=state,
            meaning_completion=row["meaning_completion"],
            boundary_start_kind=row["boundary_start_kind"],
            boundary_end_kind=row["boundary_end_kind"],
            confidence=confidence,
            provenance=PROVENANCE_UTTERANCE_SEGMENTATION,
        ))
    return tuple(utterances)


# ---------------------------------------------------------------------------
# LanguageAttempt
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LanguageAttempt:
    """One attempt to communicate one editorial unit or bounded portion of
    one -- one clean utterance, or several utterances connected by restart/
    correction/continuation evidence (this task's own definition). No final
    ``proposition_id``/``retry_family_id`` -- see module docstring."""
    source_asset_id: str
    attempt_id: str
    utterance_ids: Tuple[str, ...]
    source_start: float
    source_end: float
    text_raw: str
    text_normalized: str
    attempt_state: str
    meaning_completion: str
    restart_evidence: bool
    correction_evidence: bool
    continuation_evidence: bool
    recording_process_evidence: bool
    confidence: str
    provenance: str

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.source_end - self.source_start)


def _abandoned_attempt_state(utterance: LanguageUtterance) -> str:
    brief = _word_count(utterance.text_raw) < _BRIEF_WORD_CEILING and utterance.duration_sec < _BRIEF_DURATION_CEILING_SEC
    return ATTEMPT_FALSE_START if brief else ATTEMPT_ABANDONED


def _weakest_confidence(confidences: Sequence[str]) -> str:
    if not confidences:
        return CONFIDENCE_UNKNOWN
    return min(confidences, key=lambda c: _CONFIDENCE_RANK.get(c, 0))


def _finalize_attempt(
    members: list[LanguageUtterance],
    *,
    state: str,
    restart_evidence: bool,
    correction_evidence: bool,
    continuation_evidence: bool,
    recording_process_evidence: bool,
) -> LanguageAttempt:
    text_raw = " ".join(u.text_raw.strip() for u in members if u.text_raw.strip())
    text_normalized = normalize_language_text(text_raw)
    source_start = members[0].source_start
    source_end = members[-1].source_end
    utterance_ids = tuple(u.utterance_id for u in members)
    meaning = members[-1].meaning_completion if state != ATTEMPT_UNCERTAIN else MEANING_UNCERTAIN
    if state in (ATTEMPT_FALSE_START, ATTEMPT_ABANDONED):
        meaning = MEANING_INCOMPLETE
    base_confidence = _weakest_confidence([u.confidence for u in members])
    confidence = CONFIDENCE_MIXED if (restart_evidence and continuation_evidence) else base_confidence
    return LanguageAttempt(
        source_asset_id=members[0].source_asset_id,
        attempt_id=_attempt_id(utterance_ids),
        utterance_ids=utterance_ids,
        source_start=source_start,
        source_end=source_end,
        text_raw=text_raw,
        text_normalized=text_normalized,
        attempt_state=state,
        meaning_completion=meaning,
        restart_evidence=restart_evidence,
        correction_evidence=correction_evidence,
        continuation_evidence=continuation_evidence,
        recording_process_evidence=recording_process_evidence,
        confidence=confidence,
        provenance=PROVENANCE_ATTEMPT_RECONSTRUCTION,
    )


def build_language_attempts(
    utterances: Tuple[LanguageUtterance, ...],
    *,
    max_continuation_gap_sec: float = _DEFAULT_MAX_CONTINUATION_GAP_SEC,
    recording_process_utterance_indices: frozenset[int] = frozenset(),
) -> Tuple[LanguageAttempt, ...]:
    """The one canonical Utterance -> Attempt grouper this task requires.
    Deterministic, sequential, pairwise -- see module docstring. Mints no
    ``proposition_id``/``retry_family_id`` (D-165's own Phase C gate,
    untouched). ``recording_process_utterance_indices`` is the ONLY source
    of RECORDING_PROCESS evidence (indices into ``utterances``, 0-based) --
    this module invents no meta-speech phrase vocabulary of its own (module
    docstring "Recording process" section)."""
    if not utterances:
        return ()

    attempts: list[LanguageAttempt] = []
    buffer: list[LanguageUtterance] = [utterances[0]]
    buf_restart = False
    buf_correction = utterances[0].utterance_state == UTTERANCE_CORRECTED
    buf_continuation = False
    buf_recording = 0 in recording_process_utterance_indices

    def _close(buf: list[LanguageUtterance], restart: bool, correction: bool, continuation: bool, recording: bool) -> LanguageAttempt:
        if recording:
            state = ATTEMPT_RECORDING_PROCESS
        elif buf[-1].utterance_state in (UTTERANCE_ABANDONED,) and len(buf) == 1:
            state = _abandoned_attempt_state(buf[0])
        elif correction:
            state = ATTEMPT_CORRECTION
        elif continuation:
            state = ATTEMPT_CONTINUATION
        elif buf[0].utterance_state == UTTERANCE_UNCERTAIN and len(buf) == 1:
            state = ATTEMPT_UNCERTAIN
        else:
            state = ATTEMPT_CLEAN
        return _finalize_attempt(
            buf, state=state, restart_evidence=restart, correction_evidence=correction,
            continuation_evidence=continuation, recording_process_evidence=recording,
        )

    for index in range(1, len(utterances)):
        left = buffer[-1]
        right = utterances[index]
        right_recording = index in recording_process_utterance_indices
        gap = max(0.0, right.source_start - left.source_end)

        if left.utterance_state == UTTERANCE_ABANDONED:
            attempts.append(_close(buffer, buf_restart, buf_correction, buf_continuation, buf_recording))
            buffer = [right]
            buf_restart = True  # this new attempt begins right after an abandoned/false-start attempt
            buf_correction = right.utterance_state == UTTERANCE_CORRECTED
            buf_continuation = False
            buf_recording = right_recording
            continue

        if right.utterance_state == UTTERANCE_CORRECTED:
            buffer.append(right)
            buf_correction = True
            buf_recording = buf_recording or right_recording
            continue

        if left.utterance_state == UTTERANCE_INCOMPLETE:
            # Plain continuation: an open utterance followed directly by its
            # completing continuation, joined into ONE attempt (never
            # physically merged utterances -- see module docstring).
            buffer.append(right)
            buf_continuation = True
            buf_recording = buf_recording or right_recording
            continue

        if (
            left.utterance_state == UTTERANCE_COMPLETE
            and right.utterance_state == UTTERANCE_COMPLETE
            and left.boundary_end_kind != BOUNDARY_PAUSE
            and gap <= max_continuation_gap_sec
        ):
            # Multi-sentence clean attempt: no real measured pause, no
            # restart -- an uninterrupted continuous delivery.
            buffer.append(right)
            buf_recording = buf_recording or right_recording
            continue

        # Otherwise: a real pause between two complete utterances (a new
        # audience beat) or any other unmerged case -- close and start new.
        attempts.append(_close(buffer, buf_restart, buf_correction, buf_continuation, buf_recording))
        buffer = [right]
        buf_restart = False
        buf_correction = right.utterance_state == UTTERANCE_CORRECTED
        buf_continuation = False
        buf_recording = right_recording

    attempts.append(_close(buffer, buf_restart, buf_correction, buf_continuation, buf_recording))
    return tuple(attempts)


# ---------------------------------------------------------------------------
# RAW Understanding / Watch+Listen compatibility adapters (additive only --
# this task's own "RAW-Understanding Compatibility"/"Watch+Listen
# Compatibility" requirements). Neither raw_understanding_map.py (D-155) nor
# watch_listen_understanding.py (D-157) is modified; both stay at zero diff.
# No production call site constructs or consumes these rows.
# ---------------------------------------------------------------------------
def raw_understanding_compatibility_reference(
    span_id: str, *, utterance_id: str | None = None, attempt_id: str | None = None,
) -> dict:
    """Bounded, JSON-safe reference row associating an existing
    ``RawUnderstandingSpan.span_id`` (D-155, unchanged) with this task's own
    ``utterance_id``/``attempt_id`` -- WITHOUT modifying ``raw_understanding_
    map.py`` at all. A future, separately-authorized task may attach this as
    an optional annotation; nothing does so here."""
    return {"span_id": span_id, "language_utterance_id": utterance_id, "language_attempt_id": attempt_id}


def watch_listen_compatibility_reference(
    understanding_span_id: str, *, attempt_id: str | None = None,
) -> dict:
    """Same shape as ``raw_understanding_compatibility_reference`` for
    ``UnderstandingSpan.span_id`` (D-157, unchanged) -- a future,
    separately-authorized task may let ``WatchListenUnderstanding``
    corroborate/contradict a ``LanguageAttempt`` via this reference; not
    performed here."""
    return {"understanding_span_id": understanding_span_id, "language_attempt_id": attempt_id}


# ---------------------------------------------------------------------------
# Diagnostics (tail-safe, counts-only -- same pattern as D-119/D-125/D-152/
# D-155/D-157/D-163/D-166/D-167's own compact summaries). No transcript
# dump. Per this task's own required field list -- RESTARTED/CORRECTED
# utterance-state counts are not separately enumerated here (not in the
# directive's own required list) but remain fully inspectable on the
# returned LanguageUtterance tuples themselves.
# ---------------------------------------------------------------------------
def language_utterance_attempt_diagnostics(
    utterances: Tuple[LanguageUtterance, ...], attempts: Tuple[LanguageAttempt, ...],
) -> dict:
    source_mapping_valid = True
    timeline_valid = True
    previous_end = None
    for utterance in utterances:
        if utterance.source_start < 0 or utterance.source_end < utterance.source_start:
            timeline_valid = False
        if previous_end is not None and utterance.source_start < previous_end - 1e-6:
            timeline_valid = False
        previous_end = utterance.source_end
        if not (0 <= utterance.phrase_start_index <= utterance.phrase_end_index):
            source_mapping_valid = False

    utterance_ids_seen = {u.utterance_id: u for u in utterances}
    for attempt in attempts:
        if attempt.source_start < 0 or attempt.source_end < attempt.source_start:
            timeline_valid = False
        if not attempt.utterance_ids:
            source_mapping_valid = False
            continue
        members = [utterance_ids_seen.get(uid) for uid in attempt.utterance_ids]
        if any(m is None for m in members):
            source_mapping_valid = False
            continue
        if members[0].source_start != attempt.source_start or members[-1].source_end != attempt.source_end:
            source_mapping_valid = False

    return {
        "schema_version": SCHEMA_VERSION,
        "language_utterance_count": len(utterances),
        "language_attempt_count": len(attempts),
        "utterance_complete_count": sum(1 for u in utterances if u.utterance_state == UTTERANCE_COMPLETE),
        "utterance_incomplete_count": sum(1 for u in utterances if u.utterance_state == UTTERANCE_INCOMPLETE),
        "utterance_abandoned_count": sum(1 for u in utterances if u.utterance_state == UTTERANCE_ABANDONED),
        "utterance_uncertain_count": sum(1 for u in utterances if u.utterance_state == UTTERANCE_UNCERTAIN),
        "attempt_clean_count": sum(1 for a in attempts if a.attempt_state == ATTEMPT_CLEAN),
        "attempt_false_start_count": sum(1 for a in attempts if a.attempt_state == ATTEMPT_FALSE_START),
        "attempt_abandoned_count": sum(1 for a in attempts if a.attempt_state == ATTEMPT_ABANDONED),
        "attempt_correction_count": sum(1 for a in attempts if a.attempt_state == ATTEMPT_CORRECTION),
        "attempt_continuation_count": sum(1 for a in attempts if a.attempt_state == ATTEMPT_CONTINUATION),
        "attempt_recording_process_count": sum(1 for a in attempts if a.attempt_state == ATTEMPT_RECORDING_PROCESS),
        "attempt_uncertain_count": sum(1 for a in attempts if a.attempt_state == ATTEMPT_UNCERTAIN),
        "source_mapping_valid": source_mapping_valid,
        "timeline_valid": timeline_valid,
    }
