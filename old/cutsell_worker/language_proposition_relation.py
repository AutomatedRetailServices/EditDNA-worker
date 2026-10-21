"""D-169: Language / Transcript Spine, Phase C -- typed PROPOSITIONCANDIDATE
+ RELATIONEVIDENCE, and the explicit separation of PROPOSITION identity
from RETRY-FAMILY identity.

See ``docs/CUTSELL_CANONICAL_ENGINE_ARCHITECTURE_D098.md`` Section 14 and
``docs/CUTSELL_DECISIONS.md`` D-169 for the full design rationale and the
conflation audit this module's own docstring restates below. This module
implements ONLY the fifth and sixth rungs of D-165's canonical hierarchy:

    LanguageAttempt -> PropositionCandidate -> RelationEvidence

It mints NO `retry_family_id`, creates no Family/BestTake/final-Relation
authority, and is not called by any production call site: `pipeline.py`,
`flow_b.py`, `take_grouping.py`, `take_grouping_provider.py`,
`hybrid_session_cleanup.py`, `semantic_idea_equivalence.py`,
`attempt_relationship_authority.py`, `deterministic_best_take_authority.py`,
`take_judge.py`, `watch_listen_besttake_evidence.py`,
`watch_listen_zone_usability_v2.py`, `boundary_engine_pass.py`,
`dialogue_pacing_transition.py`, and `semantic_authority_observability.py`
are all confirmed unaware of this module's existence (module-leaf grep
tests, `tests/test_cutsell_d169_language_proposition_relation.py`).

## CORE PRINCIPLE (this task's own instruction, enforced structurally)

    PROPOSITION IDENTITY PRECEDES RETRY IDENTITY.

A `PropositionCandidate` answers WHAT editorial claim/informational job is
being communicated. A retry family answers WHICH alternative realizations
compete to express that SAME proposition. These are different objects and
never share identity here merely for implementation convenience -- see the
"Conflation forensic" section below for the concrete, already-existing
place in this codebase where they currently DO (deliberately, by a prior
task's own explicit design note) share identity, and why this module's own
new id namespace does not repeat that choice.

## Conflation forensic (this task's own required audit, factual, not
## guessed -- read directly from the real code)

| LOCATION | CURRENT FIELD | SEMANTIC MEANING | ACTUAL BEHAVIOR | CONFLATION RISK |
|---|---|---|---|---|
| `canonical_identity.mint_retry_family_id` | `retry_family_id` | "which alternative realizations compete for one slot" | `return mint_semantic_idea_id(group_key)` -- literally the SAME hash of the SAME input as `semantic_idea_id`. The function's own docstring states this outright: "D-050A intentionally mints this identically to `mint_semantic_idea_id`". | **CONFIRMED, DOCUMENTED, MAXIMAL.** For every `DraftClip` this codebase has ever produced, `semantic_idea_id == retry_family_id`, always, by construction -- not a coincidental collision, a deliberate placeholder decision. |
| `pipeline._draft_clip` | `semantic_idea_id`, `retry_family_id` | "the idea this clip belongs to" / "the retry family this clip competes in" | Both minted from the SAME `group_id` (the post-semantic-equivalence take-group key) in the SAME two-line block: `semantic_idea_id = mint_semantic_idea_id(group_id)`; `retry_family_id = mint_retry_family_id(group_id)`. This is the ONE minting owner for both fields (`canonical_identity.py`'s own ID OWNERSHIP table). | Same as above -- this is the single site both fields are ever produced, and it produces one value under two names. |
| `contracts.DraftClip` | `take_group_id`, `semantic_idea_id`, `retry_family_id` | `take_group_id` = the REAL, pre-existing, load-bearing grouping key (Family Formation's own live identity, used everywhere); `semantic_idea_id`/`retry_family_id` = D-050A's additive SHADOW metadata layered on top, both literally aliasing `take_group_id` via the hash above | `take_group_id` is genuinely live (BestTake, Boundary, render all key off it, unmodified by D-050A); `semantic_idea_id`/`retry_family_id` are BOTH dormant shadow fields nothing in the active pipeline reads to make an editorial decision (D-050A's own module docstring: "Additive-only, shadow-metadata scope... Nothing in the current pipeline reads these ids to make a KEEP/DISCARD, winner, coverage, or freeze decision yet"). | Low runtime risk TODAY (neither field is consumed for a decision), but HIGH design risk for any future consumer: reading `semantic_idea_id` today gives you `retry_family_id`'s value and vice versa -- a future author cannot distinguish "propositions with the same idea" from "realizations competing in the same family" using these two fields, because they are the same field wearing two names. |
| `semantic_ledger.py` (`RealizationRecord`/`IdeaLedgerEntry`) | `semantic_idea_id`, `retry_family_ids` (plural, tuple) | An idea entry CAN already carry MULTIPLE `retry_family_id`s (`assign_retry_family(semantic_idea_id, retry_family_id)` appends to a tuple) -- the Ledger's own schema already anticipates one idea having more than one retry family over time, which is directly compatible with this task's target separation. | Never exercised with more than one family today because `retry_family_id` is always minted identically to `semantic_idea_id` at the one call site above -- the Ledger's own multiplicity is currently theoretical, not reachable. | The Ledger schema is NOT the conflation site -- it is already correctly shaped for the separation this task performs. The conflation is entirely upstream, at the single minting call in `pipeline._draft_clip` / `canonical_identity.mint_retry_family_id`. |
| `contracts.CandidateTake` / `take_grouping_provider.reconcile_semantic_idea_equivalence` | `take_group_id` (via `group_id`/group index), no `semantic_idea_id`/`retry_family_id` field at all pre-`_draft_clip` | The ACTUAL, live retry-family-formation decision (union-find over group indices via lexical/semantic-equivalence evidence) | Operates entirely on `take_group_id`/group indices; never touches `semantic_idea_id`/`retry_family_id` (those do not exist yet at this stage -- they are minted downstream, once, in `_draft_clip`). | None directly -- this module is Family Formation's real authority and is correctly untouched by the conflation; it is simply the thing `take_group_id` (and hence, one hop later, both shadow fields) is minted FROM. |
| `attempt_relationship_authority.py` (D-158) | `FinalAttemptRelationship.relation`/`family_membership_action` | The real, structured 5+2-way relation vocabulary (RETRY/CORRECTION/CONTINUATION/COMPLEMENTARY/NEW_AUDIENCE_BEAT/DISTINCT_PROPOSITION/UNCERTAIN) this task's own `RelationEvidence.relation_candidate` vocabulary is required to mirror | Decides `would_merge` (family action) from Watch+Listen + pre-existing evidence -- never reads or writes `semantic_idea_id`/`retry_family_id` at all. | None -- D-158 already correctly treats "relation between two candidates" and "family membership action" as related but distinct concepts (its own `_FAMILY_ACTION_FOR_RELATION` table), which is exactly the same posture this task's `RelationEvidence` (evidence) vs. a future Family consumer (decision) takes. |

**Conclusion (D-050/D-165's own documented gap, confirmed by direct code
read, not guessed):** the conflation is a single, narrow, well-understood
point -- one function (`mint_retry_family_id`) that is a literal alias of
another (`mint_semantic_idea_id`), called from one site
(`pipeline._draft_clip`). Nothing about `DraftClip`'s schema, the Semantic
Ledger's schema, or Family Formation's real logic forces this; it was a
deliberate D-050A placeholder ("Kept as its own function... so a future
D-050B/C separation... only has to change this one function's body"). This
module does NOT change that function (Family Formation stays byte-
identical, per this task's own scope) -- it instead builds the SEPARATE
`proposition_candidate_id` namespace `mint_retry_family_id`'s own docstring
already anticipated, entirely outside the conflated pair, so a future,
separately-authorized migration task can retarget consumers at the correct
namespace one at a time (D-170+) without any schema break to the
currently-serialized `semantic_idea_id`/`retry_family_id` fields.

## Meaning safety (binding, reused verbatim)

The one claim signature this module builds, `ClaimSignature`, is composed
ENTIRELY from `semantic_claims.extract_claims`'s own already-vetted,
deterministic, marker-based claim extraction (negation, numbers, claim
type, negation role) -- never a new NLP/LLM engine, per this task's own
"Do NOT create a new LLM" instruction. Negation, numbers, and claim type
are therefore preserved by construction (the same guarantee D-038/D-065/
D-066 already proved for `semantic_claims.py` itself), not by a new,
parallel heuristic.

## Editorial slot evidence (advisory, generic, no hardcoded phrases)

`SLOT_CTA`/`SLOT_CONCLUSION` reactivate D-165's own finding (`contracts.
SemanticRole.CTA` exists but is dormant) as ADVISORY evidence only, using
exclusively GENERIC, already-existing signals this codebase already
computes: `semantic_claims.classify_claim`'s own `UNIQUE_CONCLUSION`/
`ACTION_EVENT` claim types (general marker vocabulary, no Video00 phrase --
see `semantic_claims.py`'s own module docstring) plus ordinary source
position (first/last proposition in a source). No selection/CTA authority
is touched; `contracts.SemanticRole` itself is untouched (still 8 values,
still dormant on the active path) -- this module defines its OWN, wider,
purely-advisory 8-value slot vocabulary here (adding `SETUP`/`CONCLUSION`,
the two values D-165 found missing) rather than modifying the CLOSED
`contracts.py` enum.

## Provider role

No provider call anywhere in this module. `semantic_support` is an
OPTIONAL, caller-supplied `SUPPORT`/`CONFLICT`/`UNKNOWN` value per pair
(defaults to `UNKNOWN`) representing an EXTERNALLY-computed arbiter
decision (e.g. `semantic_idea_equivalence.py`'s own arbiter, called
elsewhere, never here) -- this module treats it purely as evidence
provenance, never as ontology, per this task's own instruction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping, Sequence, Tuple

from .language_utterance_attempt import (
    CONFIDENCE_MIXED,
    CONFIDENCE_SUPPORTED,
    CONFIDENCE_UNKNOWN,
    CONFIDENCE_WEAK,
    LanguageAttempt,
    MEANING_COMPLETE,
    MEANING_INCOMPLETE,
    MEANING_UNCERTAIN,
    _DEFAULT_MAX_CONTINUATION_GAP_SEC,
)
from .final_sibling_grouping import _numbers
from .semantic_claims import (
    ACTION_EVENT,
    CRITICAL,
    NEGATION,
    UNIQUE_CONCLUSION,
    extract_claims,
)

SCHEMA_VERSION = "cutsell.language_proposition_relation.v1"

# ---------------------------------------------------------------------------
# Provenance (new tags for this phase; language_utterance_attempt.py (D-168)
# stays at zero diff, same convention as D-166/D-168's own new tags).
# ---------------------------------------------------------------------------
PROVENANCE_LANGUAGE_ATTEMPT = "LANGUAGE_ATTEMPT"
PROVENANCE_CLAIM_SIGNATURE = "CLAIM_SIGNATURE"
PROVENANCE_SLOT_POSITION = "SLOT_POSITION"
PROVENANCE_RELATION_EVIDENCE_FUSION = "RELATION_EVIDENCE_FUSION"
PROVENANCE_SEMANTIC_PROVIDER = "SEMANTIC_PROVIDER"  # caller-supplied only, never called here
PROVENANCE_WATCH_LISTEN = "WATCH_LISTEN"

# ---------------------------------------------------------------------------
# Support/conflict vocabulary -- SUPPORT/CONFLICT/UNKNOWN style reasoning,
# never a weighted master score (this task's own instruction).
# ---------------------------------------------------------------------------
SUPPORT = "SUPPORT"
CONFLICT = "CONFLICT"
UNKNOWN = "UNKNOWN"
ALLOWED_SUPPORT_STATUSES: frozenset[str] = frozenset({SUPPORT, CONFLICT, UNKNOWN})

# ---------------------------------------------------------------------------
# Relation vocabulary -- mirrors D-157/D-158's own labels verbatim.
# ---------------------------------------------------------------------------
RELATION_RETRY = "RETRY"
RELATION_CORRECTION = "CORRECTION"
RELATION_CONTINUATION = "CONTINUATION"
RELATION_COMPLEMENTARY = "COMPLEMENTARY"
RELATION_NEW_AUDIENCE_BEAT = "NEW_AUDIENCE_BEAT"
RELATION_DISTINCT_PROPOSITION = "DISTINCT_PROPOSITION"
RELATION_UNCERTAIN = "UNCERTAIN"
ALLOWED_RELATIONS: frozenset[str] = frozenset({
    RELATION_RETRY, RELATION_CORRECTION, RELATION_CONTINUATION,
    RELATION_COMPLEMENTARY, RELATION_NEW_AUDIENCE_BEAT,
    RELATION_DISTINCT_PROPOSITION, RELATION_UNCERTAIN,
})

# ---------------------------------------------------------------------------
# Advisory editorial slot vocabulary (this module's OWN, wider vocabulary --
# contracts.SemanticRole is untouched). Advisory only; never a selection
# authority.
# ---------------------------------------------------------------------------
SLOT_HOOK = "HOOK"
SLOT_SETUP = "SETUP"
SLOT_PROBLEM = "PROBLEM"
SLOT_FEATURE = "FEATURE"
SLOT_PROOF = "PROOF"
SLOT_CONCLUSION = "CONCLUSION"
SLOT_CTA = "CTA"
SLOT_OTHER = "OTHER"
ALLOWED_SLOTS: frozenset[str] = frozenset({
    SLOT_HOOK, SLOT_SETUP, SLOT_PROBLEM, SLOT_FEATURE, SLOT_PROOF,
    SLOT_CONCLUSION, SLOT_CTA, SLOT_OTHER,
})

# Minimum shared content-token count/coverage for two claim signatures to be
# considered "about the same proposition" at all (below this, a conflict
# check would be comparing unrelated content -- mirrors semantic_claims.
# claim_coverage's own "relevant sentence" relevance floor in spirit, a
# small, explicit, reused-shape constant, not a new arbitrary number).
_MIN_SHARED_FOR_SAME_PROPOSITION = 2
_MIN_COVERAGE_FOR_SAME_PROPOSITION = 0.5


# ---------------------------------------------------------------------------
# ClaimSignature
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClaimSignature:
    """A conservative, reusable claim signature built ENTIRELY from
    ``semantic_claims.extract_claims``'s own already-vetted output -- never
    a new engine. Preserves negation/numbers/claim-type by construction."""
    content_tokens: frozenset[str]
    negation_present: bool
    numbers: frozenset[str]
    claim_type: str
    negation_role: str
    signature_hash: str


def build_claim_signature(source_id: str, text_normalized: str) -> ClaimSignature:
    """Pure function; no I/O, no provider call. Reuses ``extract_claims``
    verbatim (D-038/D-040/D-065/D-066, unchanged) -- this is the ONLY place
    this module reads meaning out of text, and it delegates entirely."""
    claims = extract_claims(source_id, text_normalized)
    content_tokens = frozenset(token for claim in claims for token in claim.content_tokens)
    negation_present = any(claim.claim_type == NEGATION for claim in claims)
    # D-169: `_numbers` is read directly off the text, not filtered through
    # `content_tokens` -- `extract_claims`'s own clause tokenizer
    # (`final_sibling_grouping._content`) requires >=3 characters, which
    # silently drops short numerals ("50") that `_numbers` itself would
    # still find. Reused verbatim (the SAME helper `semantic_claims.py`
    # itself imports), never a second number-detection heuristic.
    numbers = frozenset(_numbers(text_normalized))
    negation_roles = {claim.negation_role for claim in claims if claim.negation_role}
    negation_role = (
        "FACTUAL_NEGATION" if "FACTUAL_NEGATION" in negation_roles
        else "CONTRASTIVE_HINDSIGHT_NEGATION" if "CONTRASTIVE_HINDSIGHT_NEGATION" in negation_roles
        else ""
    )
    critical = [claim for claim in claims if claim.importance == CRITICAL]
    dominant_claim_type = critical[0].claim_type if critical else (claims[0].claim_type if claims else "NONE")
    signature_raw = "|".join((
        dominant_claim_type,
        "1" if negation_present else "0",
        ",".join(sorted(numbers)),
        ",".join(sorted(content_tokens)),
    )).encode("utf-8")
    signature_hash = hashlib.sha256(signature_raw).hexdigest()[:20]
    return ClaimSignature(
        content_tokens=content_tokens,
        negation_present=negation_present,
        numbers=numbers,
        claim_type=dominant_claim_type,
        negation_role=negation_role,
        signature_hash=signature_hash,
    )


def _content_overlap(left: ClaimSignature, right: ClaimSignature) -> Tuple[int, float]:
    shared = left.content_tokens & right.content_tokens
    shorter = min(len(left.content_tokens), len(right.content_tokens)) or 1
    return len(shared), len(shared) / shorter


def signatures_describe_same_proposition(left: ClaimSignature, right: ClaimSignature) -> bool:
    """Conservative "these two look like the same editorial claim" check --
    content overlap only, never a conflict verdict on its own. Used both as
    a relation-classification input and directly by tests to prove "same
    proposition, two attempts" without asserting identical ids (this task's
    own bounded V1 choice -- see module docstring's id-minting note)."""
    shared_count, coverage = _content_overlap(left, right)
    return shared_count >= _MIN_SHARED_FOR_SAME_PROPOSITION and coverage >= _MIN_COVERAGE_FOR_SAME_PROPOSITION


def claim_signatures_conflict(left: ClaimSignature, right: ClaimSignature) -> bool:
    """True when two signatures address the SAME proposition (sufficient
    content overlap) but materially disagree -- negation polarity flips, or
    both sides state a number and the numbers differ. Never a conflict
    verdict for signatures that are not even about the same thing (no
    shared content -> not comparable, not a conflict -- exactly D-059's own
    "no fallback to the whole blob" posture, reused in spirit)."""
    if not signatures_describe_same_proposition(left, right):
        return False
    if left.negation_present != right.negation_present:
        return True
    if left.numbers and right.numbers and left.numbers != right.numbers:
        return True
    return False


# ---------------------------------------------------------------------------
# PropositionCandidate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PropositionCandidate:
    """One bounded informational claim / editorial job -- V1 is built ONE
    PER `LanguageAttempt` (bounded, simple, per this task's own "keep V1
    bounded and simple" instruction); ``attempt_ids`` is a tuple (not a
    single id) so a future phase MAY widen this to a multi-attempt
    proposition without a schema break, but nothing does so here. No
    ``retry_family_id`` field -- see module docstring's conflation
    forensic."""
    source_asset_id: str
    proposition_candidate_id: str
    attempt_ids: Tuple[str, ...]
    source_start: float
    source_end: float
    text_raw: str
    text_normalized: str
    claim_signature: ClaimSignature
    meaning_completion: str
    editorial_slot_evidence: str
    confidence: str
    provenance: str
    conflict_flags: Tuple[str, ...] = ()

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.source_end - self.source_start)


def _proposition_id(source_asset_id: str, signature_hash: str, start: float, end: float) -> str:
    """Deterministic: same (source_asset_id, claim_signature, timing) always
    mints the same id -- this task's own "same frozen proposition input ->
    same id" requirement, satisfied literally. A SEPARATE namespace/prefix
    from every existing id (``prop_``, never confusable with
    ``source_span_id``/``attempt_id``/``lutt_``/``latt_``/``idea_``/the
    conflated ``retry_family_id``) -- minted here, not registered in
    ``canonical_identity.py`` (D-050A's own "no consumer yet" precedent)."""
    raw = f"{source_asset_id}|{signature_hash}|{float(start):.3f}|{float(end):.3f}".encode("utf-8")
    return "prop_" + hashlib.sha256(raw).hexdigest()[:20]


def _slot_evidence(*, index: int, count: int, meaning_completion: str, claim_type: str) -> Tuple[str, str]:
    """Advisory-only, generic, positional + reused-claim-type evidence --
    see module docstring's "Editorial slot evidence" section for why no
    phrase is hardcoded. Returns (slot, confidence)."""
    if claim_type == UNIQUE_CONCLUSION:
        return SLOT_CONCLUSION, CONFIDENCE_WEAK
    if index == count - 1 and meaning_completion == MEANING_COMPLETE:
        if claim_type == ACTION_EVENT:
            return SLOT_CTA, CONFIDENCE_WEAK
        return SLOT_CONCLUSION, CONFIDENCE_WEAK
    if index == 0:
        return SLOT_HOOK, CONFIDENCE_WEAK
    return SLOT_OTHER, CONFIDENCE_UNKNOWN


def build_proposition_candidates(attempts: Tuple[LanguageAttempt, ...]) -> Tuple[PropositionCandidate, ...]:
    """The one canonical LanguageAttempt -> PropositionCandidate builder
    this task requires. Pure function; no I/O, no provider call. Mints NO
    `retry_family_id`/final family membership. Deterministic: same input
    attempts (in any order) always produce the same output, sorted by
    source position."""
    ordered = tuple(sorted(attempts, key=lambda a: (a.source_asset_id, a.source_start, a.source_end, a.attempt_id)))
    # Slot position is computed per-source (the "first"/"last" proposition
    # in THIS source, never across unrelated sources).
    by_source: dict[str, list[LanguageAttempt]] = {}
    for attempt in ordered:
        by_source.setdefault(attempt.source_asset_id, []).append(attempt)

    candidates: list[PropositionCandidate] = []
    for attempt in ordered:
        siblings = by_source[attempt.source_asset_id]
        index = siblings.index(attempt)
        count = len(siblings)
        signature = build_claim_signature(attempt.source_asset_id, attempt.text_normalized)
        slot, slot_confidence = _slot_evidence(
            index=index, count=count, meaning_completion=attempt.meaning_completion, claim_type=signature.claim_type,
        )
        conflict_flags: list[str] = []
        if attempt.meaning_completion == MEANING_UNCERTAIN:
            conflict_flags.append("UNCERTAIN_ATTEMPT_BASIS")
        confidence = attempt.confidence if attempt.confidence != CONFIDENCE_MIXED else CONFIDENCE_MIXED
        if slot_confidence == CONFIDENCE_UNKNOWN and confidence == CONFIDENCE_SUPPORTED:
            confidence = CONFIDENCE_SUPPORTED  # slot evidence never downgrades attempt-level confidence
        candidates.append(PropositionCandidate(
            source_asset_id=attempt.source_asset_id,
            proposition_candidate_id=_proposition_id(
                attempt.source_asset_id, signature.signature_hash, attempt.source_start, attempt.source_end,
            ),
            attempt_ids=(attempt.attempt_id,),
            source_start=attempt.source_start,
            source_end=attempt.source_end,
            text_raw=attempt.text_raw,
            text_normalized=attempt.text_normalized,
            claim_signature=signature,
            meaning_completion=attempt.meaning_completion,
            editorial_slot_evidence=slot,
            confidence=confidence,
            provenance=PROVENANCE_LANGUAGE_ATTEMPT,
            conflict_flags=tuple(conflict_flags),
        ))
    return tuple(candidates)


# ---------------------------------------------------------------------------
# RelationEvidence
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RelationEvidence:
    """Structured EVIDENCE about a proposition pair -- never a final merge
    action. D-158's own structured authority still decides final relation
    when this evidence is consumed later (not performed here)."""
    left_proposition_candidate_id: str
    right_proposition_candidate_id: str
    relation_candidate: str
    support_status: str
    confidence: str
    semantic_support: str
    language_support: str
    watch_listen_support: str
    meaning_conflict: bool
    proposition_conflict: bool
    provenance: Tuple[str, ...]


def _language_support(
    left: PropositionCandidate, right: PropositionCandidate, right_attempt: LanguageAttempt | None,
) -> Tuple[str, bool, bool]:
    """Returns (language_support, meaning_conflict, proposition_conflict).
    Deterministic, reuses only claim-signature overlap plus the LanguageAttempt
    restart/correction/continuation flags D-168 already computed -- no new
    heuristic beyond composing existing evidence."""
    same_proposition = signatures_describe_same_proposition(left.claim_signature, right.claim_signature)
    conflict = claim_signatures_conflict(left.claim_signature, right.claim_signature)
    if not same_proposition:
        return UNKNOWN, False, False
    if conflict:
        return CONFLICT, True, True
    restart = bool(right_attempt.restart_evidence) if right_attempt is not None else False
    correction = bool(right_attempt.correction_evidence) if right_attempt is not None else False
    if restart or correction:
        return SUPPORT, False, False
    return SUPPORT, False, False


def _fuse_support(*statuses: str) -> str:
    """SUPPORT/CONFLICT/UNKNOWN style fusion -- never a weighted score. Any
    CONFLICT dominates (meaning disagreement is never smoothed over by an
    agreeing source); SUPPORT requires at least one real SUPPORT and no
    CONFLICT; otherwise UNKNOWN."""
    values = tuple(statuses)
    if CONFLICT in values:
        return CONFLICT
    if SUPPORT in values:
        return SUPPORT
    return UNKNOWN


def _confidence_for_relation(
    language_support: str, semantic_support: str, watch_listen_support: str,
) -> str:
    supports = (language_support, semantic_support, watch_listen_support)
    non_unknown = [s for s in supports if s != UNKNOWN]
    if not non_unknown:
        return CONFIDENCE_UNKNOWN
    if len(set(non_unknown)) > 1:
        return CONFIDENCE_MIXED
    if language_support in (SUPPORT, CONFLICT):
        return CONFIDENCE_SUPPORTED
    return CONFIDENCE_WEAK


def classify_relation_candidate(
    left: PropositionCandidate,
    right: PropositionCandidate,
    *,
    right_attempt: LanguageAttempt | None = None,
    semantic_support: str = UNKNOWN,
    watch_listen_support: str = UNKNOWN,
) -> RelationEvidence:
    """The one canonical relation-classification function this task
    requires. Pure; no provider call (``semantic_support``/``watch_listen_
    support`` are caller-supplied evidence, never computed here). Never a
    final merge action -- see module docstring."""
    language_support, meaning_conflict, proposition_conflict = _language_support(left, right, right_attempt)
    fused = _fuse_support(language_support, semantic_support, watch_listen_support)

    restart = bool(right_attempt.restart_evidence) if right_attempt is not None else False
    correction = bool(right_attempt.correction_evidence) if right_attempt is not None else False
    same_proposition = signatures_describe_same_proposition(left.claim_signature, right.claim_signature)

    if fused == CONFLICT and (restart or correction):
        relation = RELATION_CORRECTION
    elif correction:
        relation = RELATION_CORRECTION
    elif left.meaning_completion == MEANING_INCOMPLETE and not restart and not meaning_conflict:
        relation = RELATION_CONTINUATION
    elif restart and same_proposition and not meaning_conflict:
        relation = RELATION_RETRY
    elif meaning_conflict and same_proposition and not restart:
        relation = RELATION_DISTINCT_PROPOSITION
    elif not same_proposition:
        gap = max(0.0, right.source_start - left.source_end)
        if left.editorial_slot_evidence == right.editorial_slot_evidence:
            relation = RELATION_COMPLEMENTARY
        elif (
            left.meaning_completion == MEANING_COMPLETE
            and right.meaning_completion == MEANING_COMPLETE
            and gap > _DEFAULT_MAX_CONTINUATION_GAP_SEC
        ):
            # Mirrors watch_listen_understanding._relation_for_pair's own
            # NEW_AUDIENCE_BEAT gate (D-157): both sides cleanly complete,
            # no restart, AND a real gap beyond the continuation ceiling --
            # never merely "not the same proposition" (this task's own
            # "same topic/product is insufficient" instruction).
            relation = RELATION_NEW_AUDIENCE_BEAT
        else:
            relation = RELATION_DISTINCT_PROPOSITION
    else:
        relation = RELATION_UNCERTAIN

    # Conflicting evidence sources (e.g. language says SUPPORT/retry,
    # semantic/watch-listen says CONFLICT, or vice versa) never get forced
    # toward either extreme -- per this task's own UNCERTAIN section.
    statuses_present = {language_support, semantic_support, watch_listen_support} - {UNKNOWN}
    if SUPPORT in statuses_present and CONFLICT in statuses_present:
        relation = RELATION_UNCERTAIN

    # A proposition with no confident basis at all (empty/uncertain source
    # text) has no basis to assert even DISTINCT_PROPOSITION confidently --
    # "WHEN UNCERTAIN, KEEP [the finding UNCERTAIN]", never a confident
    # DISTINCT/RETRY guess from an absent proposition.
    if left.meaning_completion == MEANING_UNCERTAIN or right.meaning_completion == MEANING_UNCERTAIN:
        relation = RELATION_UNCERTAIN

    confidence = _confidence_for_relation(language_support, semantic_support, watch_listen_support)
    provenance = (PROVENANCE_CLAIM_SIGNATURE,)
    if semantic_support != UNKNOWN:
        provenance += (PROVENANCE_SEMANTIC_PROVIDER,)
    if watch_listen_support != UNKNOWN:
        provenance += (PROVENANCE_WATCH_LISTEN,)
    if len(provenance) > 1:
        provenance = (PROVENANCE_RELATION_EVIDENCE_FUSION,) + provenance

    return RelationEvidence(
        left_proposition_candidate_id=left.proposition_candidate_id,
        right_proposition_candidate_id=right.proposition_candidate_id,
        relation_candidate=relation,
        support_status=fused,
        confidence=confidence,
        semantic_support=semantic_support,
        language_support=language_support,
        watch_listen_support=watch_listen_support,
        meaning_conflict=meaning_conflict,
        proposition_conflict=proposition_conflict,
        provenance=provenance,
    )


def build_relation_evidence(
    propositions: Tuple[PropositionCandidate, ...],
    attempts_by_id: Mapping[str, LanguageAttempt],
    *,
    semantic_support_by_pair: Mapping[Tuple[str, str], str] | None = None,
    watch_listen_support_by_pair: Mapping[Tuple[str, str], str] | None = None,
) -> Tuple[RelationEvidence, ...]:
    """Pairwise, sequential (same-source immediate neighbors only) --
    mirrors D-157/D-168's own architecture. ``semantic_support_by_pair``/
    ``watch_listen_support_by_pair`` are OPTIONAL, caller-supplied evidence
    keyed by ``(left_proposition_candidate_id, right_proposition_candidate_
    id)``; absent entries default to UNKNOWN (fail-open -- this module never
    calls a provider or Watch+Listen itself)."""
    semantic_map = semantic_support_by_pair or {}
    watch_listen_map = watch_listen_support_by_pair or {}
    ordered = tuple(sorted(propositions, key=lambda p: (p.source_asset_id, p.source_start, p.source_end, p.proposition_candidate_id)))
    evidence: list[RelationEvidence] = []
    by_source: dict[str, list[PropositionCandidate]] = {}
    for proposition in ordered:
        by_source.setdefault(proposition.source_asset_id, []).append(proposition)
    for siblings in by_source.values():
        for left, right in zip(siblings, siblings[1:]):
            key = (left.proposition_candidate_id, right.proposition_candidate_id)
            right_attempt = attempts_by_id.get(right.attempt_ids[0]) if right.attempt_ids else None
            evidence.append(classify_relation_candidate(
                left, right,
                right_attempt=right_attempt,
                semantic_support=semantic_map.get(key, UNKNOWN),
                watch_listen_support=watch_listen_map.get(key, UNKNOWN),
            ))
    return tuple(evidence)


# ---------------------------------------------------------------------------
# RAW Understanding / Watch+Listen compatibility adapters (additive only --
# same pattern as D-168's own two adapters). No production call site
# constructs or consumes these rows.
# ---------------------------------------------------------------------------
def raw_understanding_proposition_reference(
    span_id: str, *, proposition_candidate_id: str | None = None,
) -> dict:
    """Bounded, JSON-safe reference row associating an existing
    ``RawUnderstandingSpan.span_id`` (D-155, unchanged) with this task's own
    ``proposition_candidate_id`` -- WITHOUT modifying that CLOSED module."""
    return {"span_id": span_id, "proposition_candidate_id": proposition_candidate_id}


def watch_listen_proposition_reference(
    understanding_span_id: str, *, proposition_candidate_id: str | None = None,
) -> dict:
    """Same shape for ``UnderstandingSpan.span_id`` (D-157, unchanged)."""
    return {"understanding_span_id": understanding_span_id, "proposition_candidate_id": proposition_candidate_id}


# ---------------------------------------------------------------------------
# Diagnostics (tail-safe, counts-only -- same pattern as every prior phase).
# ---------------------------------------------------------------------------
def language_proposition_relation_diagnostics(
    propositions: Tuple[PropositionCandidate, ...], relations: Tuple[RelationEvidence, ...],
) -> dict:
    relation_counts = {
        "retry_evidence_count": 0, "continuation_evidence_count": 0,
        "correction_evidence_count": 0, "complementary_evidence_count": 0,
        "new_beat_evidence_count": 0, "distinct_proposition_evidence_count": 0,
        "uncertain_relation_evidence_count": 0,
    }
    label_to_key = {
        RELATION_RETRY: "retry_evidence_count", RELATION_CONTINUATION: "continuation_evidence_count",
        RELATION_CORRECTION: "correction_evidence_count", RELATION_COMPLEMENTARY: "complementary_evidence_count",
        RELATION_NEW_AUDIENCE_BEAT: "new_beat_evidence_count",
        RELATION_DISTINCT_PROPOSITION: "distinct_proposition_evidence_count",
        RELATION_UNCERTAIN: "uncertain_relation_evidence_count",
    }
    for relation in relations:
        relation_counts[label_to_key.get(relation.relation_candidate, "uncertain_relation_evidence_count")] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "language_proposition_candidate_count": len(propositions),
        "language_relation_evidence_count": len(relations),
        **relation_counts,
        "cta_slot_candidate_count": sum(1 for p in propositions if p.editorial_slot_evidence == SLOT_CTA),
        "conclusion_slot_candidate_count": sum(1 for p in propositions if p.editorial_slot_evidence == SLOT_CONCLUSION),
        "proposition_conflict_count": sum(1 for r in relations if r.proposition_conflict),
        "meaning_conflict_count": sum(1 for r in relations if r.meaning_conflict),
    }
