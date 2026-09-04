"""D-065/D-066: negation semantic role classification -- semantic_claims.py.

Generic (English + Spanish) fixtures only -- no Video00-specific fact,
disease, or product name. Mirrors the D-064 forensic's own reasoning and
the D-065 design doc's adversarial/positive-paraphrase matrices verbatim.
"""
from cutsell_worker.semantic_claims import (
    CONTRASTIVE_HINDSIGHT_NEGATION,
    FACTUAL_NEGATION,
    NEGATION,
    extract_claims,
)


def _negation_claims(text: str):
    return [c for c in extract_claims("x", text) if c.claim_type == NEGATION]


# --- Section 9: adversarial safety suite -- every one of these must NEVER
#     produce CONTRASTIVE_HINDSIGHT_NEGATION -----------------------------

def test_true_factual_negation_stays_factual():
    claims = _negation_claims("I did not have gastritis.")
    assert len(claims) == 1
    assert claims[0].negation_role == FACTUAL_NEGATION


def test_diagnosis_negation_stays_factual():
    claims = _negation_claims(
        "The biopsy was not cancer, but now the doctor realizes it was benign."
    )
    assert claims
    assert claims[0].negation_role == FACTUAL_NEGATION


def test_number_mismatch_negation_stays_factual():
    claims = _negation_claims("It was not 5 percent, it was 10 percent.")
    assert claims
    assert claims[0].negation_role == FACTUAL_NEGATION


def test_price_negation_stays_factual():
    claims = _negation_claims("It does not cost 49 dollars, but now I realize the value.")
    assert claims
    assert claims[0].negation_role == FACTUAL_NEGATION


def test_entity_substitution_negation_stays_factual():
    claims = _negation_claims("It was not the cream, but now I realize it was the serum.")
    assert claims
    assert claims[0].negation_role == FACTUAL_NEGATION


def test_causal_direction_negation_stays_factual():
    claims = _negation_claims(
        "It did not cause the breakout, but now I realize the breakout caused me to stop."
    )
    assert claims
    assert claims[0].negation_role == FACTUAL_NEGATION


def test_incomplete_sentence_negation_stays_factual():
    # No after-state at all -- fails closed.
    claims = _negation_claims("I did not think it was serious...")
    assert claims
    assert claims[0].negation_role == FACTUAL_NEGATION


def test_double_negation_stays_factual():
    claims = _negation_claims(
        "It is not that I did not notice, I just did not think it mattered."
    )
    assert claims
    assert all(c.negation_role == FACTUAL_NEGATION for c in claims)


def test_sarcasm_like_phrasing_stays_factual():
    # No belief/perception verb ("was not going to work" asserts a
    # definite prediction, not a subjective impression) -- fails closed.
    claims = _negation_claims(
        "Oh sure, it definitely was not going to work, but now I realize it did."
    )
    assert claims
    assert claims[0].negation_role == FACTUAL_NEGATION


def test_temporal_reversal_stays_factual():
    # The "realization" precedes the negation in this sentence -- no LATER
    # clause follows the negation clause, so no after-state is found.
    claims = _negation_claims("I realize now it was fine, but I did not think so before.")
    assert claims
    assert claims[-1].negation_role == FACTUAL_NEGATION


def test_both_sides_negated_stays_factual():
    # The would-be after-clause is itself negated -- a more complex
    # double-transition shape, deliberately not attempted in Phase 1.
    claims = _negation_claims("I did not think it would work, but it did not help either.")
    assert claims
    assert all(c.negation_role == FACTUAL_NEGATION for c in claims)


def test_no_contrast_marker_stays_factual():
    claims = _negation_claims("I did not think it was serious and now I realize it was.")
    assert claims
    assert claims[0].negation_role == FACTUAL_NEGATION


# --- Section 10: positive paraphrase suite ------------------------------

def test_positive_hindsight_realization_qualifies():
    claims = _negation_claims("I did not think it was serious, but now I realize it was.")
    assert claims
    assert claims[0].negation_role == CONTRASTIVE_HINDSIGHT_NEGATION


def test_positive_notice_recognize_qualifies():
    claims = _negation_claims(
        "I did not notice the signs at first, but later I recognized them."
    )
    assert claims
    assert claims[0].negation_role == CONTRASTIVE_HINDSIGHT_NEGATION


def test_d064_generic_spanish_shape_qualifies():
    # The exact fresh-ASR shape D-062.1/D-063/D-064 traced (generic
    # wording, not the literal Video00 clause).
    claims = _negation_claims(
        "Síntomas que no me parecían sospechosos pero que ahora que lo analizo si eran sospechosos."
    )
    assert claims
    assert claims[0].negation_role == CONTRASTIVE_HINDSIGHT_NEGATION


# --- Section 11: sales/UGC regression shapes ----------------------------

def test_beauty_positive_shape_qualifies():
    claims = _negation_claims(
        "I did not notice any difference in my skin at first, but after two weeks it looked clearer."
    )
    assert claims
    assert claims[0].negation_role == CONTRASTIVE_HINDSIGHT_NEGATION


def test_consumer_product_positive_shape_qualifies():
    claims = _negation_claims(
        "I did not think the blender was powerful enough, but it crushed the ice easily."
    )
    assert claims
    assert claims[0].negation_role == CONTRASTIVE_HINDSIGHT_NEGATION


def test_bloating_factual_negation_never_qualifies():
    # The mandatory counter-example: a quantified product-outcome negation
    # must never be treated as a hindsight paraphrase.
    claims = _negation_claims("It did not reduce bloating.")
    assert claims
    assert claims[0].negation_role == FACTUAL_NEGATION


def test_side_effects_factual_negation_never_qualifies():
    claims = _negation_claims("It did not cause any side effects, but now I realize it helped.")
    assert claims
    assert claims[0].negation_role == FACTUAL_NEGATION


# --- Known Phase-1 conservative boundaries (disclosed, not a safety gap) --
# These are genuine hindsight-shaped sentences that stay FACTUAL_NEGATION
# in Phase 1 because the after-clause is too short to form its own clause
# (D-040's pre-existing >=2-content-token split floor) -- a false
# negative (an eligible case not recognized), never a false positive (an
# unsafe merge). Documented here so a future phase's calibration has a
# concrete, disclosed target rather than silently regressing.

def test_known_boundary_short_after_clause_stays_factual():
    claims = _negation_claims("I did not think it would work, but it helped.")
    assert claims
    assert claims[0].negation_role == FACTUAL_NEGATION
