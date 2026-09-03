"""D-059 -- CLAIM-COVERAGE PROPOSITION SCOPE FIX.

Root defect (docs/CUTSELL_DECISIONS.md D-058 canary's own residual
finding): `semantic_claims.claim_coverage`'s relevance-scoping required a
candidate sentence to share >= min(2, claim_token_count) content tokens (or,
pre-existing, nothing at all) before treating it as "the sentence this claim
is actually about." When a claim's own surrounding scaffolding was
paraphrased heavily enough that NO sentence cleared that bar, the function
fell back to the WHOLE candidate text for its negation/number/causal
mismatch guards -- so an unrelated clause's own negation could poison a
claim it was never actually about (the exact D-058 canary live shape: an
earlier "I do not believe..." clause capped coverage for a completely
unrelated, later, correctly-restated quantitative claim).

Fixed two ways: (1) the relevance test also accepts a shared NUMBER between
claim and candidate sentence as decisive on its own -- the same proposition-
scoping anchor `contradiction_signal._clauses_address_same_proposition`
already established for D-056.5, reused verbatim; (2) when truly no sentence
is relevant even under that broadened test, the mismatch guards are SKIPPED
entirely (never fall back to the whole text) -- the plain overlap ratio
(already whole-text scoped, unrelated to this per-sentence check) stands on
its own, so a genuine low-overlap paraphrase still reaches the ambiguous
band and gets a real chance at the bounded arbiter, and a claim with no
credible matching content anywhere is honestly reported not covered via low
raw overlap, never via a fabricated mismatch cap.

Entirely generic -- no Video00 clip ids or phrases.
"""
from cutsell_worker.semantic_claims import (
    AMBIGUOUS_COVERAGE_FLOOR,
    COVERAGE_THRESHOLD,
    claim_coverage,
    extract_claims,
    resolve_ambiguous_coverage,
)


class _AlwaysCoveredArbiter:
    def claim_covered(self, claim_text, winning_realization_text):
        return True, 0.9, "paraphrase confirmed"


def _claim(text):
    claims = extract_claims("clip_a", text)
    assert claims, f"expected extract_claims to produce at least one claim for {text!r}"
    return claims[0]


# --- 1/2. Unrelated negation before/after a valid paraphrased claim --------
#          must never poison coverage -----------------------------------

def test_unrelated_negation_in_previous_clause_does_not_poison_coverage():
    candidate = (
        "I do not think the weather affected anything at all that week. "
        "The doctor said only 5 to 10 percent of these cases are hereditary."
    )
    claim = _claim("Only 5 to 10 percent of these cases are hereditary in nature.")
    coverage = claim_coverage(claim, candidate)
    assert coverage >= COVERAGE_THRESHOLD
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=None) is True


def test_unrelated_negation_in_following_clause_does_not_poison_coverage():
    candidate = (
        "The doctor said only 5 to 10 percent of these cases are hereditary. "
        "I do not think the weather affected anything at all that week."
    )
    claim = _claim("Only 5 to 10 percent of these cases are hereditary in nature.")
    coverage = claim_coverage(claim, candidate)
    assert coverage >= COVERAGE_THRESHOLD
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=None) is True


# --- 3. Same-proposition negation mismatch -> not covered -------------------

def test_same_proposition_negation_mismatch_still_not_covered():
    claim = _claim("The biopsy confirmed the tumor was hereditary in nature.")
    candidate = "The biopsy confirmed the tumor was not hereditary in nature."
    coverage = claim_coverage(claim, candidate)
    assert coverage < AMBIGUOUS_COVERAGE_FLOOR
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=_AlwaysCoveredArbiter()) is False


# --- 4. Same-proposition number mismatch -> not covered ---------------------

def test_same_proposition_number_mismatch_still_not_covered():
    claim = _claim("Only 5 percent of these cases are hereditary in nature.")
    candidate = "About 10 percent of cancers are hereditary, according to her doctor."
    coverage = claim_coverage(claim, candidate)
    assert coverage < AMBIGUOUS_COVERAGE_FLOOR
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=_AlwaysCoveredArbiter()) is False


# --- 5. Same-proposition causal inversion -> not covered --------------------

def test_same_proposition_causal_inversion_still_not_covered():
    claim = _claim("The flare-ups happen because of stress.")
    candidate = "Stress happens because of the flare-ups."
    coverage = claim_coverage(claim, candidate)
    assert coverage < AMBIGUOUS_COVERAGE_FLOOR
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=_AlwaysCoveredArbiter()) is False


# --- 6. Low lexical overlap but clear semantic paraphrase -> covered via ----
#        the existing bounded arbiter path -----------------------------------

def test_low_overlap_clear_paraphrase_reaches_arbiter():
    claim = _claim("The endoscopy showed she had gastritis, nothing severe.")
    candidate = "Further testing confirmed a mild case of gastritis was responsible for her symptoms."
    coverage = claim_coverage(claim, candidate)
    assert AMBIGUOUS_COVERAGE_FLOOR <= coverage < COVERAGE_THRESHOLD
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=_AlwaysCoveredArbiter()) is True
    # No safety weakening: without an arbiter, still fails open to LOST.
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=None) is False


# --- 7. No credible matching proposition -> honestly not covered via low ---
#        raw overlap, never via the mismatch cap (no fabricated mismatch) ---

def test_no_credible_matching_proposition_is_not_a_fabricated_mismatch():
    claim = _claim("Only 5 to 10 percent of these cases are hereditary in nature.")
    candidate = "The weather that week had been unusually mild for the season."
    coverage = claim_coverage(claim, candidate)
    assert coverage < AMBIGUOUS_COVERAGE_FLOOR
    from cutsell_worker.semantic_claims import _DEFINITIVE_MISMATCH_COVERAGE_CAP
    # Distinguishes "genuinely no evidence" from "confirmed mismatch": the
    # low score here must NOT be the fixed mismatch cap -- no guard fired,
    # there was simply nothing to find.
    assert coverage != _DEFINITIVE_MISMATCH_COVERAGE_CAP
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=_AlwaysCoveredArbiter()) is False


# --- 8. Multi-sentence clip with only one relevant clause -------------------

def test_multi_sentence_clip_with_only_one_relevant_clause():
    candidate = (
        "This is my experience with the whole process. "
        "I do not think the weather affected anything at all that week. "
        "Only 5 to 10 percent of these cases are hereditary in nature, according to her doctor. "
        "Mostly it comes down to lifestyle choices, so take care of yourself."
    )
    claim = _claim("Only 5 to 10 percent of these cases are hereditary in nature.")
    coverage = claim_coverage(claim, candidate)
    assert coverage >= COVERAGE_THRESHOLD
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=None) is True


# --- 9. D-058 canary shape, reproduced generically end to end --------------

def test_d058_canary_shape_reproduced_generically_no_longer_false_positive():
    """Generic paraphrase of the exact D-058 canary structure: a complete
    realization rhetorically negates a BROADER, unrelated claim in an
    earlier clause about a different subject, then restates the shared
    figure in a later clause without negation; the claim under test comes
    from a DIFFERENT, incomplete retry of that same later clause, so its
    own surrounding scaffolding words are paraphrased heavily (near-zero
    shared vocabulary) and only the shared number anchors it."""
    claim_text = (
        "I am convinced, and the science backs it up, that only about 10 percent of"
    )
    candidate = (
        "This is my experience. I am the only one in my family with this diagnosis. "
        "That is why I do not believe, and science backs this up, that these conditions "
        "are broadly hereditary. Rather, only about 10 percent are hereditary in nature. "
        "Mostly it comes down to lifestyle, so take care of yourself."
    )
    claim = _claim(claim_text)
    coverage = claim_coverage(claim, candidate)
    # Not poisoned by the unrelated negation clause: coverage must clear
    # the ambiguous floor at minimum (it may land anywhere from "ambiguous,
    # needs the arbiter" up to "confidently covered" depending on exactly
    # how much of the claim's own scaffolding happens to overlap -- either
    # outcome proves the fix; only the old whole-text-negation-poisoned
    # cap (well below the floor) would prove it broken).
    assert coverage >= AMBIGUOUS_COVERAGE_FLOOR
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=_AlwaysCoveredArbiter()) is True
