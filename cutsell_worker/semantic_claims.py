"""General per-Idea semantic claim extraction and coverage -- D-038.

RAW 33423953391 exposed the real architectural gap this module closes:
CoverageLedger's existing checks are either scoped to numbers/negations
(`semantic_atom_importance.py`, D-031) or compare a discarded clip's
vocabulary against the ENTIRE final KEEP timeline's bag of words
(`final_story_coherence_validation._lost_semantic_atoms`). The second check
can be fooled: an idea's specific composed proposition ("the biopsy
confirmed the papillary cancer diagnosis") can be judged "covered" merely
because unrelated words from it ("cancer", "thyroid", "biopsy") happen to
also appear in a DIFFERENT, unrelated selected clip elsewhere in the video
(one about a routine screening, say) -- whole-video word presence is not
the same thing as this Idea's own claim surviving in its own place.

    WHOLE-VIDEO WORD PRESENCE  !=  PER-IDEA CLAIM PRESERVATION

This module treats a sentence-level PROPOSITION -- not a loose vocabulary
token -- as the unit CoverageLedger and BestTake compete/protect on. It is
purely deterministic pattern-matching over general linguistic markers
(reporting/result verbs, identification copulas, cause/effect connectors,
temporal connectors, correction language, generalizing-statistic language)
in English and Spanish -- no Video00 fact, phrase, disease, or product name
is hardcoded anywhere below; the same rules fire identically on any
talking-head subject matter. A bounded `ClaimEquivalenceArbiter` (fails
open toward "not confirmed", i.e. still counts as lost -- same "WHEN
UNCERTAIN, KEEP" posture as `semantic_atom_importance.py`) is the only
place a real semantic-paraphrase judgment would be made; no implementation
is wired in this codebase (same honest-gap pattern as `CausalOrderArbiter`
and `SemanticAtomImportanceArbiter`).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Protocol

from .canonical_identity import mint_canonical_claim_id
from .final_sibling_grouping import _content, _negations, _numbers
from .semantic_atom_importance import (
    _CORRECTION_MARKERS,
    _CURRENCY_MARKERS,
    _DOSE_MARKERS,
    _MEASUREMENT_UNIT_MARKERS,
    _PERCENT_MARKERS,
    _clause_has_any,
)

# --- Claim types ------------------------------------------------------------
ENTITY_RELATION = "ENTITY_RELATION"
STATE_RESULT = "STATE_RESULT"
DIAGNOSIS_IDENTIFICATION = "DIAGNOSIS_IDENTIFICATION"
CAUSE_EFFECT = "CAUSE_EFFECT"
ACTION_EVENT = "ACTION_EVENT"
MEASUREMENT_QUANTITY = "MEASUREMENT_QUANTITY"
NEGATION = "NEGATION"
CORRECTION = "CORRECTION"
TEMPORAL_RELATION = "TEMPORAL_RELATION"
UNIQUE_CONCLUSION = "UNIQUE_CONCLUSION"

_VALID_CLAIM_TYPES = frozenset({
    ENTITY_RELATION, STATE_RESULT, DIAGNOSIS_IDENTIFICATION, CAUSE_EFFECT,
    ACTION_EVENT, MEASUREMENT_QUANTITY, NEGATION, CORRECTION,
    TEMPORAL_RELATION, UNIQUE_CONCLUSION,
})

# --- Importance (reuses D-031's vocabulary, extended with SUPPORTING/REDUNDANT) ---
CRITICAL = "CRITICAL"
SUPPORTING = "SUPPORTING"
CONTEXTUAL = "CONTEXTUAL"
REDUNDANT = "REDUNDANT"

_VALID_IMPORTANCE = frozenset({CRITICAL, SUPPORTING, CONTEXTUAL, REDUNDANT})

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# General (English + Spanish) linguistic-marker vocabulary. Matched as a
# plain substring of the sentence (casefolded, padded) -- the same coarse,
# whole-clause-presence style semantic_atom_importance.py already uses. No
# Video00-specific fact, disease, product, or phrase appears below.
_RESULT_REPORTING_MARKERS = (
    "confirmo", "confirmó", "confirmaron", "confirma", "revelo", "reveló",
    "revelaron", "mostro", "mostró", "indico", "indicó", "determino", "determinó",
    "resulto", "resultó", "salio", "salió", "dio como resultado", "arrojo", "arrojó",
    "confirmed", "revealed", "showed", "indicated", "determined", "turned out",
    "came back", "resulted in", "found that", "encontraron que", "descubrieron que",
)

# Unambiguous identification/diagnosis language -- safe to treat as
# identity evidence on its own, standalone (no other marker needed).
_STRONG_IDENTIFICATION_MARKERS = (
    "se trataba de", "resulto ser", "resultó ser", "turned out to be",
    "diagnosticado con", "diagnosticada con", "diagnosed with",
    "el diagnostico fue", "el diagnóstico fue", "diagnosis was",
    "se llamaba", "fue diagnosticado", "fue diagnosticada",
)
# A bare copula ("it WAS A tumor") is too generic to be identity evidence
# by itself -- "it was a good day", "it was a bit strange" would false-
# positive just as easily. Trailing space ("was a ", not "was a") also
# guards `_clause_has_any`'s plain substring match against firing as a
# false-positive PREFIX of an unrelated word -- "it was ALREADY late", "it
# was ALSO nice", "it was ANOTHER story", "era ÚNICO" are none of them an
# identification copula. Only counted as identity evidence when the same
# sentence ALSO carries explicit result-reporting language (see
# classify_claim) -- e.g. "the biopsy confirmed it was a tumor".
_WEAK_COPULA_MARKERS = ("era un ", "era una ", "fue un ", "fue una ", "was a ", "was an ")
_CAUSE_EFFECT_MARKERS = (
    "porque", "por eso", "asi que", "así que", "debido a", "ya que", "por lo que",
    "because", "so that", "therefore", "due to", "as a result", "which is why",
)
_TEMPORAL_MARKERS = (
    "despues de", "después de", "antes de", "luego", "entonces", "cuando",
    "after", "before", "then", "when", "once",
)
_UNIQUE_CONCLUSION_MARKERS = (
    "solo un", "solo una", "unicamente", "únicamente", "la mayoria", "la mayoría",
    "en general", "por lo general", "unico", "único", "unica", "única",
    "only", "solely", "the majority", "in general", "generally",
)
_STATE_RESULT_MARKERS = (
    "dio positivo", "dio negativo", "salio positivo", "salió positivo",
    "salio negativo", "salió negativo", "resultado fue", "came back positive",
    "came back negative", "tested positive", "tested negative", "result was",
)
# Contrastive connectors -- like _CAUSE_EFFECT_MARKERS/_TEMPORAL_MARKERS,
# these introduce a clause that qualifies rather than restates the one
# before it. Used only for clause splitting (below), not by classify_claim
# itself, since a bare contrast on its own says nothing about claim type.
_CONTRASTIVE_MARKERS = (
    "pero", "aunque", "sin embargo", "but", "although", "however", "though",
)
# "lo que"/"lo cual" ("which") are specific enough to safely mark a
# relative-clause addition; bare "que" is deliberately excluded from clause
# splitting -- it is Spanish's all-purpose subordinator ("que tenía", "que
# salían", ...) and splitting on it would shatter ordinary sentence
# structure rather than separate genuine propositions.
_RELATIVE_ADDITION_MARKERS = ("lo que", "lo cual", "which")

COVERAGE_THRESHOLD = 0.6
# D-058 Phase 3 (docs/CUTSELL_DECISIONS.md D-057's 5-10% forensic): lowered
# from 0.3. The live false positive this fixes had raw token overlap 0.15 --
# a genuine paraphrase of the same hereditary-percentage claim ("estoy
# convencida y la ciencia lo avala que solo un 5-10% de los" vs "esta
# comprobado cientificamente ... solo un 5-10% son de caracter
# hereditario") whose surrounding scaffolding words differ enough that raw
# content-token overlap alone undersells it, while the number itself
# ("5-10%") and the core claim survive verbatim. 0.3 was too high a floor to
# ever let a real case like this reach the arbiter for a paraphrase
# judgment at all. The floor still exists -- below it a claim is confidently
# lost regardless of arbiter availability, too little overlap for a
# paraphrase judgment to plausibly apply -- it is only calibrated lower.
# `_DEFINITIVE_MISMATCH_COVERAGE_CAP` (below) is a separate, fixed value
# for confidently-mismatched claims (negation flip, number change) so
# lowering this floor can never let one of those genuine mismatches drift
# into the ambiguous band merely because the floor moved.
AMBIGUOUS_COVERAGE_FLOOR = 0.10
# A genuine negation flip or number change is confidently NOT the same
# claim, never escalated to the arbiter regardless of how low
# `AMBIGUOUS_COVERAGE_FLOOR` is calibrated -- see `claim_coverage`'s own
# negation-flip and number-mismatch guards below, both of which cap
# coverage at this fixed value rather than at a fraction of the (now
# lower, tunable) ambiguous floor.
_DEFINITIVE_MISMATCH_COVERAGE_CAP = 0.05

_DEDUPE_SIMILARITY_THRESHOLD = 0.7


def _split_sentences(text: str) -> tuple[str, ...]:
    text = str(text or "").strip()
    if not text:
        return ()
    parts = _SENTENCE_SPLIT_RE.split(text)
    return tuple(part.strip() for part in parts if part.strip())


# D-040 (claim granularity): a multi-clause sentence can bundle a CORE
# proposition with a merely-SUPPORTING reason/context clause -- "Nunca se
# nos ocurrio hacer un chequeo de tiroides, pues porque cada ano me hacia
# minimo dos examenes" is a NEGATION core ("we never thought to check the
# thyroid") plus a supporting elaboration ("because ... two exams a year"),
# not one indivisible proposition. Scoring the WHOLE sentence as one claim
# means a winning realization that keeps the core but drops the supporting
# detail scores as if it dropped the core too -- a real false positive an
# offline audit of RAW 33448261223 traced to exactly this (the Human Gold
# reference itself keeps only the core; the CRITICAL_CLAIM_LOST finding was
# wrong, not the selection).
#
# Splitting reuses the SAME connector vocabulary classify_claim's own
# CAUSE_EFFECT/TEMPORAL_RELATION rules already draw on -- a sentence that
# opens with a real reporting/negation/correction marker BEFORE any
# connector keeps that marker's own critical weight on its own clause; the
# connector-introduced remainder is classified independently by the exact
# same deterministic rules, so it only ever comes out CRITICAL if it
# contains its OWN critical marker (a second core fact in one sentence,
# not a demotion of the first). No Video00 phrase is hardcoded.
_CLAUSE_SPLIT_MARKERS = (
    _CAUSE_EFFECT_MARKERS + _TEMPORAL_MARKERS + _CONTRASTIVE_MARKERS + _RELATIVE_ADDITION_MARKERS
)
_CLAUSE_SPLIT_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in sorted(_CLAUSE_SPLIT_MARKERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
# A split is only accepted when BOTH resulting sides clear the same
# >=2-content-token floor extract_claims already uses for a whole sentence
# -- otherwise a connector near the start/end of a short sentence ("it was
# fine after") would carve off a degenerate, meaningless fragment.
_CLAUSE_MIN_CONTENT_TOKENS = 2


def _split_into_clauses(text: str, *, _search_from: int = 0) -> tuple[str, ...]:
    """Split one sentence into an ordered tuple of clauses at the first
    connector that produces two substantive sides, recursing into the
    remainder so a chain of connectors ("core, porque X, pero Y") yields
    every piece rather than only the first. Returns `(text,)` unchanged
    when no genuine split point exists -- the overwhelmingly common case
    (most sentences are already one clause)."""
    match = _CLAUSE_SPLIT_RE.search(text, _search_from)
    if not match:
        return (text,)
    left = text[:match.start()].strip()
    right = text[match.start():].strip()
    if len(_content(left)) < _CLAUSE_MIN_CONTENT_TOKENS or len(_content(right)) < _CLAUSE_MIN_CONTENT_TOKENS:
        # This particular connector doesn't produce a valid split (e.g. it
        # is the sentence's own first word, or nothing substantial follows
        # it) -- keep looking later in the string for a real one instead of
        # giving up on splitting the sentence entirely.
        return _split_into_clauses(text, _search_from=match.end())
    connector_len = match.end() - match.start()
    further = _split_into_clauses(right[connector_len:])
    if len(further) == 1:
        return (left, right)
    return (left, (right[:connector_len] + further[0]).strip()) + further[1:]


def classify_claim(sentence: str) -> tuple[str, str, str]:
    """Deterministic (claim_type, importance, evidence) for one sentence.
    General marker-based rules only -- see module docstring."""
    negations = _negations(sentence)
    numbers = _numbers(sentence)

    if negations:
        return NEGATION, CRITICAL, "negation_present"
    if _clause_has_any(sentence, _CORRECTION_MARKERS):
        return CORRECTION, CRITICAL, "correction_language_present"
    if numbers and _clause_has_any(
        sentence, _PERCENT_MARKERS + _CURRENCY_MARKERS + _MEASUREMENT_UNIT_MARKERS + _DOSE_MARKERS
    ):
        return MEASUREMENT_QUANTITY, CRITICAL, "quantity_with_unit_marker"
    if numbers and _clause_has_any(sentence, _UNIQUE_CONCLUSION_MARKERS):
        return UNIQUE_CONCLUSION, CRITICAL, "generalizing_statistic_language"
    if _clause_has_any(sentence, _STATE_RESULT_MARKERS):
        return STATE_RESULT, CRITICAL, "result_state_language"

    has_reporting = _clause_has_any(sentence, _RESULT_REPORTING_MARKERS)
    has_strong_identity = _clause_has_any(sentence, _STRONG_IDENTIFICATION_MARKERS)
    has_weak_copula = _clause_has_any(sentence, _WEAK_COPULA_MARKERS)
    # A weak, generic copula ("was a X") only counts as identity evidence
    # alongside explicit reporting language -- alone it is too common a
    # sentence shape to safely mark CRITICAL (see _WEAK_COPULA_MARKERS).
    has_identity = has_strong_identity or (has_weak_copula and has_reporting)
    if has_reporting and has_identity:
        return DIAGNOSIS_IDENTIFICATION, CRITICAL, "result_reporting_plus_identification_language"
    if has_strong_identity:
        return DIAGNOSIS_IDENTIFICATION, CRITICAL, "identification_language"
    if has_reporting:
        return ENTITY_RELATION, CRITICAL, "result_reporting_language"

    if _clause_has_any(sentence, _CAUSE_EFFECT_MARKERS):
        return CAUSE_EFFECT, SUPPORTING, "cause_effect_connector"
    if _clause_has_any(sentence, _TEMPORAL_MARKERS):
        return TEMPORAL_RELATION, CONTEXTUAL, "temporal_connector"
    if numbers:
        return MEASUREMENT_QUANTITY, SUPPORTING, "bare_number_no_unit_marker"
    return ACTION_EVENT, SUPPORTING, "general_statement"


@dataclass(frozen=True)
class Claim:
    claim_id: str
    source_clip_id: str
    claim_type: str
    text: str
    importance: str
    evidence: str
    content_tokens: frozenset
    # D-050A: canonical, cross-source claim identity -- see
    # canonical_identity.py's module docstring (ID OWNERSHIP: "Claim
    # canonicalization"). Minted from (claim_type, content_tokens) only,
    # deliberately NOT from source_clip_id/exact text, so two near-
    # duplicate restatements of the same fact from different sibling
    # realizations of one merged idea share this id today, in metadata
    # only -- observability, not a decision input. Optional/defaulted so
    # every existing construction site (this module's own extract_claims,
    # and every test building a Claim by keyword) stays valid unchanged;
    # nothing in claim_coverage_best_take.py reads this field yet.
    canonical_claim_id: str = ""


def _claim_id(source_clip_id: str, text: str) -> str:
    digest = hashlib.sha256(f"{source_clip_id}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"claim_{digest}"


def extract_claims(
    source_clip_id: str, text: str, *, clause_role_arbiter: "ClauseRoleArbiter | None" = None,
) -> tuple[Claim, ...]:
    """Split `text` into clause-level claims: each sentence first splits on
    genuine connectors (`_split_into_clauses` -- D-040) into a CORE clause
    plus zero or more SUPPORTING/CONTEXTUAL clauses, and each clause is then
    classified independently via `classify_claim`. A clause with fewer than
    two content tokens is too thin to be a standalone audience-facing
    proposition (a bare reaction/filler, e.g. "Okay." or "Sí.") and is
    skipped, mirroring `_lost_semantic_atoms`'s own short-clip floor. This
    is claim-LOCAL by construction: a critical fact bundled with a merely
    supporting reason in one sentence produces two separate claims, so
    coverage is checked (and can legitimately pass) per clause rather than
    penalizing a preserved core claim for a dropped supporting one.

    `clause_role_arbiter` (D-040) is consulted only for a clause
    `classify_claim` could not confidently place at all (its weakest,
    marker-less fallback) -- see `resolve_ambiguous_clause_role`'s own
    docstring for exactly when and how it can change the result; defaults
    to `None` (unwired), the same honest-gap pattern as every other bounded
    arbiter in this module."""
    claims: list[Claim] = []
    for sentence in _split_sentences(text):
        for clause in _split_into_clauses(sentence):
            tokens = _content(clause)
            if len(tokens) < 2:
                continue
            claim_type, importance, evidence = classify_claim(clause)
            importance = resolve_ambiguous_clause_role(
                clause, sentence,
                deterministic_importance=importance, evidence=evidence,
                arbiter=clause_role_arbiter,
            )
            content_tokens = frozenset(tokens)
            claims.append(Claim(
                claim_id=_claim_id(source_clip_id, clause),
                source_clip_id=source_clip_id,
                claim_type=claim_type,
                text=clause,
                importance=importance,
                evidence=evidence,
                content_tokens=content_tokens,
                canonical_claim_id=mint_canonical_claim_id(claim_type, content_tokens),
            ))
    return tuple(claims)


def _jaccard(left: frozenset, right: frozenset) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def dedupe_claims(claims: tuple[Claim, ...], *, similarity_threshold: float = _DEDUPE_SIMILARITY_THRESHOLD) -> tuple[Claim, ...]:
    """Collapse near-duplicate claims across sibling retry attempts (the
    same proposition restated with minor wording variance) to one
    representative -- so a retry family's aggregate claim set reflects
    distinct propositions, not one proposition counted once per attempt
    that happened to say it."""
    kept: list[Claim] = []
    for claim in claims:
        is_duplicate = False
        for existing in kept:
            if claim.claim_type != existing.claim_type:
                continue
            if _jaccard(claim.content_tokens, existing.content_tokens) >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(claim)
    return tuple(kept)


def claim_coverage(claim: Claim, candidate_text: str) -> float:
    """Content-token overlap ratio between `claim` and `candidate_text` --
    the fraction of the claim's own content tokens present in the
    candidate. 1.0 for a degenerate claim with no content tokens at all
    (never produced by `extract_claims`'s own floor, but safe either way).

    Negation-flip guard: plain token overlap cannot tell "the biopsy
    confirmed it was benign" from "the biopsy did NOT confirm it was
    benign" -- same nouns, opposite proposition. When exactly one of
    claim/candidate carries a negation marker (`_negations`, general
    English+Spanish set, no Video00 vocabulary), the two assert different
    things regardless of noun overlap, so coverage is capped at
    `_DEFINITIVE_MISMATCH_COVERAGE_CAP` -- confidently NOT covered, same as
    too-low overlap, never escalated to the arbiter for what is actually a
    clear negation mismatch. Number-mismatch guard immediately below this
    docstring is the identical shape, for a changed number/percentage
    instead of a changed polarity.

    Both checks are scoped to the SENTENCE(S) of `candidate_text` that
    actually share content tokens with the claim, not the whole,
    possibly multi-sentence or multi-clip (`_lost_critical_claims`/
    `claim_coverage_best_take` both pass a joined "winning realization"
    covering several clips) candidate blob. Found via real-chain testing:
    a candidate's OTHER, unrelated sentence can carry a negation ("no creo
    ... son hereditarios" a few sentences before an unrelated "solo un
    5-10% son de hereditario" claim in the same clip) that has nothing to
    do with the claim actually being checked -- checking negation over the
    whole blob falsely capped coverage for a claim whose own sentence was
    present, uncontradicted, verbatim. Scoping to the overlapping
    sentence(s) fixes that false positive while still catching a genuine
    same-sentence contradiction (the negated sentence then itself shares
    the claim's own content tokens, so it IS included in scope)."""
    if not claim.content_tokens:
        return 1.0
    candidate_tokens = _content(candidate_text)
    shared = len(claim.content_tokens & candidate_tokens)
    overlap = shared / len(claim.content_tokens)
    if not shared:
        return overlap
    # A sentence counts as "relevant" (in scope for the negation check)
    # only once it shares a SUBSTANTIVE portion of the claim's own tokens,
    # not merely one -- a single very common word (e.g. Spanish "son",
    # "they/you-all are") can coincidentally appear in an entirely
    # unrelated sentence and must not pull that sentence's own, unrelated
    # negation into scope. Requires the lesser of 2 tokens or the claim's
    # own full token count, so a thin (2-token) claim still needs a full
    # match to be considered the same sentence.
    min_shared_for_relevance = min(2, len(claim.content_tokens))
    candidate_sentences = _split_sentences(candidate_text) or (candidate_text,)
    relevant_sentences = [
        s for s in candidate_sentences
        if len(claim.content_tokens & _content(s)) >= min_shared_for_relevance
    ]
    relevant_scope = " ".join(relevant_sentences) if relevant_sentences else candidate_text
    if bool(_negations(claim.text)) != bool(_negations(relevant_scope)):
        return min(overlap, _DEFINITIVE_MISMATCH_COVERAGE_CAP)
    # D-058 Phase 3 (docs/CUTSELL_DECISIONS.md D-057/D-058): number-mismatch
    # guard, the same shape as the negation-flip guard immediately above --
    # "only 5% are hereditary" and "only 10% are hereditary" share every
    # other content token and must never be treated as a covered paraphrase
    # of each other just because a lower ambiguous floor (below) now lets
    # thinner-overlap PARAPHRASES reach the arbiter. Scoped to the same
    # relevant-sentence(s) as the negation check, for the identical reason:
    # a candidate's OTHER, unrelated sentence can carry its own unrelated
    # number. Only fires when BOTH sides actually state a number at all --
    # a claim with no number of its own, or a candidate that never restates
    # any number, is not a number disagreement, just missing evidence
    # (already reflected in `overlap`).
    claim_numbers = _numbers(claim.text)
    candidate_numbers = _numbers(relevant_scope)
    if claim_numbers and candidate_numbers and claim_numbers != candidate_numbers:
        return min(overlap, _DEFINITIVE_MISMATCH_COVERAGE_CAP)
    # D-058 Phase 3: causal-inversion guard, scoped to the same connector
    # vocabulary `classify_claim` already uses to detect a CAUSE_EFFECT
    # claim at all (`_CAUSE_EFFECT_MARKERS`) -- never a new marker list.
    # Bag-of-words overlap is blind to WHICH side of the connector each
    # entity sits on ("stress happens because of the flare-ups" and "the
    # flare-ups happen because of stress" share every content token), so a
    # reversed cause/effect pair can otherwise score as a near-perfect
    # paraphrase. Splits both the claim and its relevant candidate
    # sentence(s) on the first marker found; when both sides have real
    # content on both halves, an inversion is evidence when the SWAPPED
    # pairing (claim's cause vs candidate's effect, and vice versa) shares
    # strictly more than the SAME-side pairing -- the entities are still
    # all there, just on the wrong side of the connector. A general,
    # non-connector causal reversal (a bare "X triggers Y" vs "Y triggers
    # X" with no connector at all) is not detectable from bag-of-words
    # tokens alone without real parsing -- an honest, documented gap, the
    # same class `contradiction_signal.py`'s own module docstring already
    # declares out of scope for that primitive.
    claim_split = _split_on_cause_effect_marker(claim.text)
    if claim_split is not None:
        candidate_split = _split_on_cause_effect_marker(relevant_scope)
        if candidate_split is not None:
            claim_before, claim_after = _content(claim_split[0]), _content(claim_split[1])
            candidate_before, candidate_after = _content(candidate_split[0]), _content(candidate_split[1])
            if claim_before and claim_after and candidate_before and candidate_after:
                same_side = len(claim_before & candidate_before) + len(claim_after & candidate_after)
                swapped_side = len(claim_before & candidate_after) + len(claim_after & candidate_before)
                if swapped_side > same_side:
                    return min(overlap, _DEFINITIVE_MISMATCH_COVERAGE_CAP)
    return overlap


def _split_on_cause_effect_marker(text: str) -> tuple[str, str] | None:
    """First `_CAUSE_EFFECT_MARKERS` connector found in `text`, splitting it
    into (before, after). `None` when no connector is present -- the
    causal-inversion guard above only ever activates when BOTH the claim
    and the candidate's relevant scope contain one."""
    lowered = text.casefold()
    for marker in _CAUSE_EFFECT_MARKERS:
        index = lowered.find(marker)
        if index != -1:
            return text[:index], text[index + len(marker):]
    return None


def claim_is_covered(claim: Claim, candidate_text: str, *, threshold: float = COVERAGE_THRESHOLD) -> bool:
    return claim_coverage(claim, candidate_text) >= threshold


class ClaimEquivalenceArbiter(Protocol):
    """Bounded arbiter for exactly one narrow question per claim: "Does the
    proposed winning realization preserve this audience-facing claim of the
    Idea, even if paraphrased?" Given the claim's own text and the winning
    realization's text as minimal context -- mirrors this codebase's other
    bounded arbiters (text only, no clip identity, no whole-video context).
    Returns (covered: bool, confidence 0..1, short general reason)."""

    def claim_covered(self, claim_text: str, winning_realization_text: str) -> tuple[bool, float, str]: ...


class ClauseRoleArbiter(Protocol):
    """Bounded arbiter for exactly one narrow question per clause (D-040):
    "Would removing this clause materially change the audience-facing
    factual meaning of this Idea?" Given the clause's own text and its
    parent sentence as minimal context -- mirrors `ClaimEquivalenceArbiter`
    (text only, no clip identity, no whole-video context). Returns
    (role, confidence 0..1, short general reason), role one of
    CORE_CRITICAL/SUPPORTING/CONTEXTUAL/UNCERTAIN."""

    def clause_role(self, clause_text: str, parent_sentence_text: str) -> tuple[str, float, str]: ...


_CLAUSE_ROLE_CORE_CRITICAL = "CORE_CRITICAL"
_CLAUSE_ROLE_UNCERTAIN = "UNCERTAIN"
_CLAUSE_ROLE_TO_IMPORTANCE = {
    _CLAUSE_ROLE_CORE_CRITICAL: CRITICAL,
    "SUPPORTING": SUPPORTING,
    "CONTEXTUAL": CONTEXTUAL,
}


def resolve_ambiguous_clause_role(
    clause_text: str,
    parent_sentence_text: str,
    *,
    deterministic_importance: str,
    evidence: str,
    arbiter: ClauseRoleArbiter | None,
) -> str:
    """The clause's own final importance. `classify_claim`'s deterministic
    rules are confident for every marker-based evidence value (a real
    negation/reporting/cause-effect/temporal/etc. signal) -- those are left
    untouched here, matching resolve_ambiguous_coverage's own posture of
    only ever intervening in the genuinely uncertain band. The one
    genuinely ambiguous case is `evidence == "general_statement"`: no
    marker fired at all, so `deterministic_importance` is only ever
    classify_claim's own weakest fallback (SUPPORTING) -- never a confirmed
    judgment. With no arbiter (the default everywhere in this pipeline
    today, same honest-gap pattern as every other bounded arbiter here),
    that fallback is left exactly as classify_claim decided -- forcing a
    blanket escalation for every marker-less clause in a video would
    reintroduce the exact over-blocking this module exists to avoid. Once a
    real arbiter IS wired: a confirmed CORE_CRITICAL upgrades the clause;
    an explicit UNCERTAIN verdict (a real semantic check that still
    couldn't decide) or an arbiter exception also upgrades to CRITICAL --
    "WHEN UNCERTAIN, KEEP" -- rather than silently trusting the weak
    fallback; SUPPORTING/CONTEXTUAL verdicts are applied directly."""
    if evidence != "general_statement" or arbiter is None:
        return deterministic_importance
    try:
        role, _confidence, _reason = arbiter.clause_role(clause_text, parent_sentence_text)
    except Exception:
        return CRITICAL
    if role == _CLAUSE_ROLE_UNCERTAIN:
        return CRITICAL
    return _CLAUSE_ROLE_TO_IMPORTANCE.get(role, deterministic_importance)


def resolve_ambiguous_coverage(
    claim: Claim,
    winning_realization_text: str,
    *,
    coverage: float,
    arbiter: ClaimEquivalenceArbiter | None,
) -> bool:
    """True iff `claim` should be treated as covered. A coverage at or
    above `COVERAGE_THRESHOLD` is confidently covered; below
    `AMBIGUOUS_COVERAGE_FLOOR` is confidently lost (too little overlap for
    a paraphrase judgment to plausibly apply, arbiter not consulted). The
    ambiguous band in between is escalated to the bounded arbiter when one
    is available; no arbiter, an arbiter exception, or a verdict that is
    not explicitly `True` all fail open toward LOST -- the same "WHEN
    UNCERTAIN, KEEP [the finding]" posture `semantic_atom_importance.py`
    already uses."""
    if coverage >= COVERAGE_THRESHOLD:
        return True
    if coverage < AMBIGUOUS_COVERAGE_FLOOR or arbiter is None:
        return False
    try:
        covered, _confidence, _reason = arbiter.claim_covered(claim.text, winning_realization_text)
    except Exception:
        return False
    return bool(covered) is True
