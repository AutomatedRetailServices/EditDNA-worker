"""D-171: Language / Transcript Spine, Phase D -- incremental, bounded
consumer migration onto the canonical Language Spine (D-165/D-166/D-168/
D-169).

See ``docs/CUTSELL_CANONICAL_ENGINE_ARCHITECTURE_D098.md`` Section 14 and
``docs/CUTSELL_DECISIONS.md`` D-171 for the full design rationale. This
module migrates exactly TWO duplicated transcript-consumer clusters (this
task's own "no more than 3 consumer clusters" cap; a third, conclusion/CTA
advisory, was evaluated and SKIPPED -- see D-171's own entry: ``realization_
resolver.py`` already consumes the shared, canonical ``semantic_claims.
classify_claim`` vocabulary directly, the SAME primitive ``language_
proposition_relation.build_claim_signature``'s own advisory slot evidence
reuses, so no independent duplicate parser exists there to migrate):

    TARGET A (proposition / retry divergence evidence):
        ``take_grouping_provider._marked_side_diverges_in_content`` /
        ``_within_group_arbiter_confirmation_diverges`` (D-048/D-083's own
        marker-gated content-overlap guards).
    TARGET B (continuation evidence):
        ``recording_meta_continuation._direct_meta_short_tail`` (the tiny-
        continuation word-count/duration heuristic).

## Design contract (this task's own binding instruction, enforced
## structurally, not merely tested)

Every migrated call site NEVER changes the boolean/decision its caller
acts on. The Spine-computed value is substituted for the legacy
computation ONLY in the branch where an explicit, per-call comparison
proves the two are IDENTICAL for that exact input -- so substituting is
provably a no-op on behavior, by construction. Whenever the Spine's
independent verdict DISAGREES with legacy, or Spine evidence cannot be
computed at all (e.g. no ASR word-level timing on a ``CandidateTake``),
the caller's existing pre-D-171 legacy path is exactly what gets used --
fail-open, per this task's own "Language Spine available -> use canonical
evidence. Language Spine absent -> existing pre-D-171 path remains
available" directive, and per its CONFLICT POLICY ("do not silently pick
the new one... No forced migration result").

No Family/Proposition-final/Attempt-Relationship/DeliveryScorer/BestTake/
Boundary/Pacing authority is read or written here -- this module only ever
consumes ``PropositionCandidate``/``ClaimSignature`` (D-169) and
``LanguageAttempt``/``meaning_completion`` (D-168) as EVIDENCE, folded
against a caller-supplied legacy verdict, never decided on its own. No id
migration (D-169's own id namespace is read for evidence/diagnostics only,
never written back onto any serialized ``DraftClip``/``CandidateTake``
field -- this task's own "NO ID MIGRATION YET" instruction).

## Migration states

    LEGACY_ONLY                     -- Spine evidence could not be computed
                                        for this input (no words, or a
                                        build_claim_signature/attempt-
                                        construction exception); legacy
                                        verdict used as-is.
    SPINE_AVAILABLE_LEGACY_FALLBACK -- Spine computed a verdict but it
                                        DISAGREES with legacy; legacy
                                        verdict used as-is.
    SPINE_CONSUMED                  -- Spine computed a verdict that AGREES
                                        with legacy; the returned verdict is
                                        explicitly SOURCED from the Spine
                                        computation (provably identical to
                                        legacy for this input, by the
                                        agreement check itself).
    SPINE_CONFLICT_FALLBACK         -- like SPINE_AVAILABLE_LEGACY_FALLBACK,
                                        but the disagreement is corroborated
                                        by a concrete negation/number
                                        CONFLICT (``claim_signatures_
                                        conflict``, Target A only) -- a
                                        materially stronger, distinctly
                                        reported disagreement; behaviorally
                                        identical to the ordinary fallback
                                        state (legacy verdict used as-is).

## Provider role

No provider/network call anywhere in this module -- both targets reuse
only already-vetted, deterministic, offline Language Spine evidence
(D-166/D-168/D-169), never a new NLP/LLM engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .contracts import Word
from .language_proposition_relation import (
    build_claim_signature,
    claim_signatures_conflict,
    signatures_describe_same_proposition,
)
from .language_spine import adapt_words_to_language_words, segment_language_phrases
from .language_utterance_attempt import (
    MEANING_INCOMPLETE,
    build_language_attempts,
    segment_language_utterances,
)

SCHEMA_VERSION = "cutsell.language_spine_consumer_migration.v1"

LEGACY_ONLY = "LEGACY_ONLY"
SPINE_AVAILABLE_LEGACY_FALLBACK = "SPINE_AVAILABLE_LEGACY_FALLBACK"
SPINE_CONSUMED = "SPINE_CONSUMED"
SPINE_CONFLICT_FALLBACK = "SPINE_CONFLICT_FALLBACK"

_MIGRATION_STATES = frozenset({
    LEGACY_ONLY, SPINE_AVAILABLE_LEGACY_FALLBACK, SPINE_CONSUMED, SPINE_CONFLICT_FALLBACK,
})


@dataclass(frozen=True)
class ConsumerMigrationTrace:
    """Compact, deterministic, no-transcript-dump migration diagnostic for
    ONE evaluated pair/candidate, per this task's own required shape."""
    consumer_name: str
    spine_available: bool
    spine_used: bool
    legacy_fallback_used: bool
    conflict: bool
    result_source: str


def _trace(consumer_name: str, state: str, *, conflict: bool = False) -> ConsumerMigrationTrace:
    assert state in _MIGRATION_STATES
    return ConsumerMigrationTrace(
        consumer_name=consumer_name,
        spine_available=state != LEGACY_ONLY,
        spine_used=state == SPINE_CONSUMED,
        legacy_fallback_used=state in (LEGACY_ONLY, SPINE_AVAILABLE_LEGACY_FALLBACK, SPINE_CONFLICT_FALLBACK),
        conflict=conflict,
        result_source=state,
    )


# ---------------------------------------------------------------------------
# TARGET A -- proposition / retry divergence evidence
# ---------------------------------------------------------------------------
def proposition_divergence_migration(
    consumer_name: str,
    left_id: str,
    left_text: str,
    right_id: str,
    right_text: str,
    legacy_diverges: bool,
) -> Tuple[bool, ConsumerMigrationTrace]:
    """Fail-open Spine consumption for ``take_grouping_provider``'s D-048/
    D-083 marker-divergence guards. ``left_id``/``right_id`` are evidence/
    diagnostic tags only (fed to ``build_claim_signature``'s own
    ``source_id`` parameter) -- never a family/proposition identity read or
    minted here. Returns ``(verdict, trace)``; ``verdict`` is ALWAYS either
    ``legacy_diverges``, or a value proven identical to it this call -- see
    module docstring."""
    try:
        left_sig = build_claim_signature(left_id, left_text)
        right_sig = build_claim_signature(right_id, right_text)
    except Exception:
        return legacy_diverges, _trace(consumer_name, LEGACY_ONLY)

    # Spine's own divergence read, same semantic direction as the legacy
    # guard's own True/False meaning ("the two sides should be treated as
    # distinct"): either the signatures do not even describe the same
    # proposition, or they do but materially conflict (negation/number).
    same_proposition = signatures_describe_same_proposition(left_sig, right_sig)
    conflicts = claim_signatures_conflict(left_sig, right_sig)
    spine_diverges = (not same_proposition) or conflicts

    if spine_diverges == legacy_diverges:
        return spine_diverges, _trace(consumer_name, SPINE_CONSUMED)
    if conflicts:
        return legacy_diverges, _trace(consumer_name, SPINE_CONFLICT_FALLBACK, conflict=True)
    return legacy_diverges, _trace(consumer_name, SPINE_AVAILABLE_LEGACY_FALLBACK)


# ---------------------------------------------------------------------------
# TARGET B -- continuation evidence
# ---------------------------------------------------------------------------
def continuation_migration(
    consumer_name: str,
    source_asset_id: str,
    words: Sequence[Word],
    legacy_is_continuation: bool,
) -> Tuple[bool, ConsumerMigrationTrace]:
    """Fail-open Spine consumption for ``recording_meta_continuation``'s
    tiny-continuation word-count/duration heuristic. Builds a real
    LanguageWord -> LanguagePhrase -> LanguageUtterance -> LanguageAttempt
    hierarchy (D-166/D-168, unmodified) from the candidate's own ASR word
    timing (``CandidateTake.words``) and reads the resulting attempt's
    ``meaning_completion`` as the canonical continuation signal
    (``MEANING_INCOMPLETE`` == a plausible trailing continuation fragment,
    the same semantic direction as the legacy heuristic's own verdict).
    Returns ``(verdict, trace)``; ``verdict`` is ALWAYS either
    ``legacy_is_continuation``, or a value proven identical to it this call
    -- see module docstring."""
    word_tuple = tuple(words)
    if not word_tuple:
        return legacy_is_continuation, _trace(consumer_name, LEGACY_ONLY)

    try:
        language_words = adapt_words_to_language_words(source_asset_id, word_tuple)
        phrases = segment_language_phrases(language_words)
        utterances = segment_language_utterances(phrases)
        attempts = build_language_attempts(utterances)
    except Exception:
        return legacy_is_continuation, _trace(consumer_name, LEGACY_ONLY)

    if not attempts:
        return legacy_is_continuation, _trace(consumer_name, LEGACY_ONLY)

    # A tiny trailing continuation is, by definition, ONE short attempt
    # spanning this whole candidate -- the first (and for a short candidate,
    # normally only) attempt's own meaning_completion is the canonical read.
    spine_is_continuation = attempts[0].meaning_completion == MEANING_INCOMPLETE

    if spine_is_continuation == legacy_is_continuation:
        return spine_is_continuation, _trace(consumer_name, SPINE_CONSUMED)
    return legacy_is_continuation, _trace(consumer_name, SPINE_AVAILABLE_LEGACY_FALLBACK)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def language_spine_consumer_migration_diagnostics(traces: Sequence[ConsumerMigrationTrace]) -> dict:
    """Compact, CI-safe migration summary -- no transcript dump, per this
    task's own instruction. Global counters plus one row per trace."""
    trace_tuple = tuple(traces)
    evaluated = len(trace_tuple)
    used = sum(1 for t in trace_tuple if t.spine_used)
    legacy_fallback = sum(1 for t in trace_tuple if t.legacy_fallback_used)
    conflict = sum(1 for t in trace_tuple if t.conflict)
    return {
        "schema_version": SCHEMA_VERSION,
        "language_spine_consumer_evaluated_count": evaluated,
        "language_spine_consumer_used_count": used,
        "language_spine_consumer_legacy_fallback_count": legacy_fallback,
        "language_spine_consumer_conflict_count": conflict,
        "traces": tuple(
            {
                "consumer_name": t.consumer_name,
                "spine_available": t.spine_available,
                "spine_used": t.spine_used,
                "legacy_fallback_used": t.legacy_fallback_used,
                "conflict": t.conflict,
                "result_source": t.result_source,
            }
            for t in trace_tuple
        ),
    }
