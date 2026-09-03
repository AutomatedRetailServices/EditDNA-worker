"""D-058 Phase 3 -- VALIDATOR CLAIM-PARAPHRASE ALIGNMENT.

Root defect (docs/CUTSELL_DECISIONS.md D-057's 5-10% forensic): this is NOT
a Selection failure -- the winning realization already preserves the same
hereditary-percentage fact under different wording (live coverage: 0.15).
`resolve_ambiguous_coverage`'s `AMBIGUOUS_COVERAGE_FLOOR` (0.3) was too high
for a real case like this to ever reach the bounded claim-equivalence
arbiter for a paraphrase judgment at all -- it was declared "confidently
lost" by raw token overlap alone.

Fixed by lowering `AMBIGUOUS_COVERAGE_FLOOR` to 0.10 so a genuine low-
token-overlap paraphrase gets a chance at arbiter confirmation, while three
explicit, deterministic "definitively mismatched" guards inside
`claim_coverage` (negation flip, number change, causal-connector inversion)
cap coverage at a fixed, floor-independent `_DEFINITIVE_MISMATCH_COVERAGE_
CAP` (0.05) so none of those genuine mismatches can ever drift into the
now-wider ambiguous band merely because the floor moved. Number sensitivity,
negation sensitivity, and (for connector-based causal claims) causal
direction are preserved -- a bare, connector-less causal-verb inversion
("X triggers Y" vs "Y triggers X") is a documented, honest gap in bag-of-
words claim coverage, the same class `contradiction_signal.py`'s own module
docstring already declares out of scope for that primitive.

Entirely generic -- no Video00 clip ids or phrases.
"""
from cutsell_worker.semantic_claims import (
    AMBIGUOUS_COVERAGE_FLOOR,
    COVERAGE_THRESHOLD,
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


# --- 1. Paraphrased same quantitative fact -> covered -----------------------

def test_paraphrased_same_quantitative_fact_is_covered_via_arbiter():
    claim = _claim("Only 5 to 10 percent of these cases are hereditary in nature.")
    candidate = "About 5 to 10 percent of cancers are hereditary, according to her doctor."
    # Genuine low-overlap paraphrase of the SAME number/claim: below the old
    # 0.3 floor, at or above the new 0.10 floor -- must reach the arbiter.
    from cutsell_worker.semantic_claims import claim_coverage
    coverage = claim_coverage(claim, candidate)
    assert AMBIGUOUS_COVERAGE_FLOOR <= coverage < COVERAGE_THRESHOLD
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=_AlwaysCoveredArbiter()) is True


def test_paraphrased_same_quantitative_fact_fails_open_without_an_arbiter():
    claim = _claim("Only 5 to 10 percent of these cases are hereditary in nature.")
    candidate = "About 5 to 10 percent of cancers are hereditary, according to her doctor."
    from cutsell_worker.semantic_claims import claim_coverage
    coverage = claim_coverage(claim, candidate)
    # No safety weakening: with no arbiter available, the ambiguous band
    # still fails open toward LOST exactly as before D-058 -- only the
    # width of that band changed, never the fail-open default.
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=None) is False


# --- 2. Number change -> not covered, even with an arbiter that would ------
#        confirm anything -----------------------------------------------

def test_number_change_is_never_covered_even_with_an_always_confirming_arbiter():
    claim = _claim("Only 5 percent of these cases are hereditary in nature.")
    candidate = "About 10 percent of cancers are hereditary, according to her doctor."
    from cutsell_worker.semantic_claims import claim_coverage
    coverage = claim_coverage(claim, candidate)
    assert coverage < AMBIGUOUS_COVERAGE_FLOOR
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=_AlwaysCoveredArbiter()) is False


# --- 3. Negation change -> not covered, even with an always-confirming ----
#        arbiter -------------------------------------------------------------

def test_negation_change_is_never_covered_even_with_an_always_confirming_arbiter():
    claim = _claim("The biopsy confirmed the tumor was hereditary in nature.")
    candidate = "The biopsy confirmed the tumor was not hereditary in nature."
    from cutsell_worker.semantic_claims import claim_coverage
    coverage = claim_coverage(claim, candidate)
    assert coverage < AMBIGUOUS_COVERAGE_FLOOR
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=_AlwaysCoveredArbiter()) is False


# --- 4. Entity/diagnosis change -> not covered (naturally low overlap, ------
#        stays below the floor without any special-case guard) -------------

def test_entity_diagnosis_change_stays_below_the_ambiguous_floor():
    claim = _claim("The doctor diagnosed her with gastritis after the endoscopy.")
    candidate = "It turned out to be an ulcer instead, following further tests months later."
    from cutsell_worker.semantic_claims import claim_coverage
    coverage = claim_coverage(claim, candidate)
    assert coverage < AMBIGUOUS_COVERAGE_FLOOR
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=_AlwaysCoveredArbiter()) is False


# --- 5. Causal inversion (connector-based) -> not covered even with an ------
#        always-confirming arbiter ------------------------------------------

def test_causal_connector_inversion_is_never_covered_even_with_an_always_confirming_arbiter():
    claim = _claim("The flare-ups happen because of stress.")
    candidate = "Stress happens because of the flare-ups."
    from cutsell_worker.semantic_claims import claim_coverage
    coverage = claim_coverage(claim, candidate)
    assert coverage < AMBIGUOUS_COVERAGE_FLOOR
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=_AlwaysCoveredArbiter()) is False


# --- 6. Same fact, different wording that is NOT a number/negation/causal --
#        connector claim still benefits from the lower floor -----------------

def test_generic_low_overlap_paraphrase_without_special_markers_reaches_arbiter():
    claim = _claim("The endoscopy showed she had gastritis, nothing severe.")
    candidate = "Further testing confirmed a mild case of gastritis was responsible for her symptoms."
    from cutsell_worker.semantic_claims import claim_coverage
    coverage = claim_coverage(claim, candidate)
    assert AMBIGUOUS_COVERAGE_FLOOR <= coverage < COVERAGE_THRESHOLD
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=_AlwaysCoveredArbiter()) is True
