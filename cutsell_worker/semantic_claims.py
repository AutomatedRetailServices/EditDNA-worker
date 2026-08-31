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

COVERAGE_THRESHOLD = 0.6
# Below this, a claim is confidently lost regardless of arbiter availability
# -- too little overlap for a paraphrase judgment to plausibly apply.
AMBIGUOUS_COVERAGE_FLOOR = 0.3

_DEDUPE_SIMILARITY_THRESHOLD = 0.7


def _split_sentences(text: str) -> tuple[str, ...]:
    text = str(text or "").strip()
    if not text:
        return ()
    parts = _SENTENCE_SPLIT_RE.split(text)
    return tuple(part.strip() for part in parts if part.strip())


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


def _claim_id(source_clip_id: str, text: str) -> str:
    digest = hashlib.sha256(f"{source_clip_id}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"claim_{digest}"


def extract_claims(source_clip_id: str, text: str) -> tuple[Claim, ...]:
    """Split `text` into sentence-level claims. A sentence with fewer than
    two content tokens is too thin to be a standalone audience-facing
    proposition (a bare reaction/filler, e.g. "Okay." or "Sí.") and is
    skipped, mirroring `_lost_semantic_atoms`'s own short-clip floor."""
    claims: list[Claim] = []
    for sentence in _split_sentences(text):
        tokens = _content(sentence)
        if len(tokens) < 2:
            continue
        claim_type, importance, evidence = classify_claim(sentence)
        claims.append(Claim(
            claim_id=_claim_id(source_clip_id, sentence),
            source_clip_id=source_clip_id,
            claim_type=claim_type,
            text=sentence,
            importance=importance,
            evidence=evidence,
            content_tokens=frozenset(tokens),
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
    things regardless of noun overlap, so coverage is capped below
    `AMBIGUOUS_COVERAGE_FLOOR` -- confidently NOT covered, same as too-low
    overlap, never escalated to the arbiter for what is actually a clear
    negation mismatch."""
    if not claim.content_tokens:
        return 1.0
    candidate_tokens = _content(candidate_text)
    shared = len(claim.content_tokens & candidate_tokens)
    overlap = shared / len(claim.content_tokens)
    if bool(_negations(claim.text)) != bool(_negations(candidate_text)):
        return min(overlap, AMBIGUOUS_COVERAGE_FLOOR / 2)
    return overlap


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
