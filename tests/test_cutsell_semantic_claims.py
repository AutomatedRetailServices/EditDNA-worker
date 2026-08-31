"""Unit tests for semantic_claims.py -- D-038.

Covers deterministic claim classification across the general marker
vocabulary (no Video00-specific fact is referenced here -- these are
generic sentences exercising the same rule shapes), coverage scoring,
dedupe of near-duplicate restatements, and the bounded ambiguous-coverage
arbiter escalation contract (fails open toward LOST).
"""
from cutsell_worker.semantic_claims import (
    ACTION_EVENT,
    AMBIGUOUS_COVERAGE_FLOOR,
    CAUSE_EFFECT,
    CONTEXTUAL,
    CORRECTION,
    COVERAGE_THRESHOLD,
    CRITICAL,
    Claim,
    DIAGNOSIS_IDENTIFICATION,
    MEASUREMENT_QUANTITY,
    NEGATION,
    STATE_RESULT,
    SUPPORTING,
    TEMPORAL_RELATION,
    UNIQUE_CONCLUSION,
    classify_claim,
    claim_coverage,
    claim_is_covered,
    dedupe_claims,
    extract_claims,
    resolve_ambiguous_coverage,
)


# --- classify_claim: one representative sentence per type ------------------

def test_negation_is_critical():
    claim_type, importance, evidence = classify_claim("The test did not show any infection.")
    assert claim_type == NEGATION
    assert importance == CRITICAL
    assert evidence == "negation_present"


def test_correction_language_is_critical():
    claim_type, importance, _ = classify_claim("Actually, that was wrong, it was really a different condition.")
    assert claim_type == CORRECTION
    assert importance == CRITICAL


def test_quantity_with_unit_marker_is_critical():
    claim_type, importance, _ = classify_claim("The dose was 200 milligrams every day.")
    assert claim_type == MEASUREMENT_QUANTITY
    assert importance == CRITICAL


def test_generalizing_statistic_is_critical():
    # "percent" also matches the unit-marker rule, which is checked first --
    # still CRITICAL either way, so this asserts the actually-observed type
    # rather than over-specifying which marker rule fires first.
    claim_type, importance, _ = classify_claim("Only 5 percent of patients see this reaction.")
    assert importance == CRITICAL
    assert claim_type == MEASUREMENT_QUANTITY


def test_unique_conclusion_without_unit_marker_is_critical():
    claim_type, importance, _ = classify_claim("Only 5 of them ever saw a full recovery.")
    assert claim_type == UNIQUE_CONCLUSION
    assert importance == CRITICAL


def test_state_result_language_is_critical():
    claim_type, importance, _ = classify_claim("The test came back positive for the condition.")
    assert claim_type == STATE_RESULT
    assert importance == CRITICAL


def test_diagnosis_identification_is_critical():
    claim_type, importance, _ = classify_claim("The biopsy confirmed it was a benign tumor.")
    assert claim_type == DIAGNOSIS_IDENTIFICATION
    assert importance == CRITICAL


def test_identification_language_alone_is_critical():
    claim_type, importance, _ = classify_claim("She was diagnosed with a thyroid condition.")
    assert claim_type == DIAGNOSIS_IDENTIFICATION
    assert importance == CRITICAL


def test_result_reporting_alone_is_entity_relation():
    claim_type, importance, evidence = classify_claim("The doctor confirmed the results this morning.")
    assert claim_type == "ENTITY_RELATION"
    assert importance == CRITICAL
    assert evidence == "result_reporting_language"


def test_cause_effect_is_supporting():
    claim_type, importance, _ = classify_claim("I felt tired because I skipped breakfast.")
    assert claim_type == CAUSE_EFFECT
    assert importance == SUPPORTING


def test_temporal_relation_is_contextual():
    claim_type, importance, _ = classify_claim("After the appointment, I went home.")
    assert claim_type == TEMPORAL_RELATION
    assert importance == CONTEXTUAL


def test_bare_number_without_unit_is_supporting_measurement():
    claim_type, importance, _ = classify_claim("There were 12 people in the waiting room.")
    assert claim_type == MEASUREMENT_QUANTITY
    assert importance == SUPPORTING


def test_general_statement_falls_back_to_action_event_supporting():
    claim_type, importance, evidence = classify_claim("I walked into the clinic and sat down.")
    assert claim_type == ACTION_EVENT
    assert importance == SUPPORTING
    assert evidence == "general_statement"


def test_bare_weak_copula_alone_is_not_critical():
    # A generic "it was a X" copula with no reporting language is too
    # common a sentence shape to safely mark CRITICAL on its own -- "it was
    # a good experience" is not a diagnosis/identification claim.
    claim_type, importance, _ = classify_claim("It was a good experience overall, nothing special to add.")
    assert importance == SUPPORTING
    assert claim_type == ACTION_EVENT


def test_weak_copula_plus_reporting_language_is_still_critical():
    claim_type, importance, evidence = classify_claim("The biopsy confirmed it was a benign tumor.")
    assert claim_type == DIAGNOSIS_IDENTIFICATION
    assert importance == CRITICAL
    assert evidence == "result_reporting_plus_identification_language"


def test_weak_copula_substring_false_positive_prefix_words_are_not_matched():
    # "was already"/"was also"/"was another" must not be treated as the
    # "was a"/"was an" identity copula just because they share a prefix.
    for sentence in (
        "It was already late by the time we got there.",
        "It was also mentioned during the call.",
        "It was another story entirely.",
    ):
        claim_type, importance, _ = classify_claim(sentence)
        assert claim_type != DIAGNOSIS_IDENTIFICATION
        assert importance != CRITICAL


# --- extract_claims ----------------------------------------------------------

def test_extract_claims_splits_sentences_and_skips_thin_ones():
    text = "Okay. The biopsy confirmed it was a benign tumor. Sí."
    claims = extract_claims("clip_a", text)
    assert len(claims) == 1
    assert claims[0].claim_type == DIAGNOSIS_IDENTIFICATION
    assert claims[0].source_clip_id == "clip_a"


def test_extract_claims_empty_text_returns_nothing():
    assert extract_claims("clip_a", "") == ()
    assert extract_claims("clip_a", "   ") == ()


def test_extract_claims_claim_id_is_stable_for_same_clip_and_text():
    text = "The biopsy confirmed it was a benign tumor."
    a = extract_claims("clip_a", text)[0]
    b = extract_claims("clip_a", text)[0]
    assert a.claim_id == b.claim_id


def test_extract_claims_claim_id_differs_by_source_clip():
    text = "The biopsy confirmed it was a benign tumor."
    a = extract_claims("clip_a", text)[0]
    b = extract_claims("clip_b", text)[0]
    assert a.claim_id != b.claim_id


# --- dedupe_claims ------------------------------------------------------------

def test_dedupe_claims_collapses_near_identical_restatement():
    claims = extract_claims("clip_a", "The biopsy confirmed it was a benign tumor.") + \
        extract_claims("clip_b", "The biopsy confirmed it was a benign tumor in the end.")
    deduped = dedupe_claims(claims)
    assert len(deduped) == 1


def test_dedupe_claims_keeps_distinct_propositions():
    claims = extract_claims("clip_a", "The biopsy confirmed it was a benign tumor.") + \
        extract_claims("clip_b", "I felt tired because I skipped breakfast.")
    deduped = dedupe_claims(claims)
    assert len(deduped) == 2


def test_dedupe_claims_does_not_merge_across_different_claim_types():
    # Same tokens, but classified differently -- must not collapse.
    claims = (
        Claim(claim_id="c1", source_clip_id="a", claim_type=DIAGNOSIS_IDENTIFICATION,
              text="x", importance=CRITICAL, evidence="e", content_tokens=frozenset({"foo", "bar"})),
        Claim(claim_id="c2", source_clip_id="b", claim_type=ACTION_EVENT,
              text="y", importance=SUPPORTING, evidence="e", content_tokens=frozenset({"foo", "bar"})),
    )
    deduped = dedupe_claims(claims)
    assert len(deduped) == 2


# --- claim_coverage / claim_is_covered ---------------------------------------

def test_claim_coverage_full_overlap_is_one():
    claim = extract_claims("clip_a", "The biopsy confirmed it was a benign tumor.")[0]
    assert claim_coverage(claim, "The biopsy confirmed it was a benign tumor.") == 1.0


def test_claim_coverage_no_overlap_is_zero():
    claim = extract_claims("clip_a", "The biopsy confirmed it was a benign tumor.")[0]
    assert claim_coverage(claim, "completely unrelated sentence about something else entirely") == 0.0


def test_claim_coverage_partial_overlap_is_fractional():
    claim = extract_claims("clip_a", "The biopsy confirmed it was a benign tumor.")[0]
    coverage = claim_coverage(claim, "The biopsy confirmed something different.")
    assert 0.0 < coverage < 1.0


def test_claim_is_covered_respects_threshold():
    claim = extract_claims("clip_a", "The biopsy confirmed it was a benign tumor.")[0]
    assert claim_is_covered(claim, "The biopsy confirmed it was a benign tumor.")
    assert not claim_is_covered(claim, "unrelated text with nothing in common at all")


def test_claim_coverage_negation_flip_is_not_covered_despite_shared_nouns():
    # Same nouns as the claim, but negated -- an opposite proposition that
    # plain token overlap alone would otherwise score as near-total coverage.
    claim = extract_claims("clip_a", "The biopsy confirmed it was a benign tumor.")[0]
    coverage = claim_coverage(claim, "The biopsy did not confirm it was a benign tumor.")
    assert coverage < AMBIGUOUS_COVERAGE_FLOOR
    assert not claim_is_covered(claim, "The biopsy did not confirm it was a benign tumor.")


def test_claim_coverage_unrelated_negation_in_a_different_sentence_does_not_deflate_coverage():
    # RAW 33432104336: a multi-sentence clip's OTHER sentence carried a
    # negation ("no creo ... son hereditarios") with nothing to do with a
    # later, unrelated claim in the same clip ("Mas bien solo un 5-10% son
    # de hereditario."), which the whole-text negation check falsely
    # treated as a negation mismatch even though the claim's own sentence
    # was present verbatim and uncontradicted. The check must be scoped to
    # the sentence(s) that actually overlap the claim's own content.
    claim = extract_claims(
        "clip_a", "Mas bien solo un 5-10% son de hereditario.",
    )[0]
    candidate_text = (
        "Esta es mi experiencia. Soy la unica en mi familia que tiene este "
        "tipo de cancer. Por eso no creo y esta comprobado cientificamente "
        "que los canceres son hereditarios. Mas bien solo un 5-10% son de "
        "hereditario. Mayormente son nuestras elecciones de vida."
    )
    coverage = claim_coverage(claim, candidate_text)
    assert coverage >= COVERAGE_THRESHOLD
    assert claim_is_covered(claim, candidate_text)


def test_claim_coverage_negation_in_the_same_overlapping_sentence_still_caps_coverage():
    # The flip side of the fix above: when the negation is IN the sentence
    # that actually shares the claim's own content tokens (a genuine same-
    # proposition contradiction, not an unrelated aside elsewhere), the cap
    # must still apply.
    claim = extract_claims("clip_a", "El medico confirmo que era cancer.")[0]
    candidate_text = "Aqui hablo de otras cosas. El medico no confirmo que era cancer."
    coverage = claim_coverage(claim, candidate_text)
    assert coverage < AMBIGUOUS_COVERAGE_FLOOR


def test_claim_coverage_negation_claim_covered_only_by_also_negated_candidate():
    claim = extract_claims("clip_a", "The test did not show any infection.")[0]
    assert claim.claim_type == NEGATION
    assert claim_is_covered(claim, "The test did not show any infection at all.")
    # The affirmative restatement asserts the opposite -- must not be
    # scored as covering the negated claim just because the nouns match.
    assert not claim_is_covered(claim, "The test showed an infection.")


# --- resolve_ambiguous_coverage -----------------------------------------------

class _AlwaysCoveredArbiter:
    def claim_covered(self, claim_text, winning_realization_text):
        return True, 0.9, "paraphrase confirmed"


class _AlwaysLostArbiter:
    def claim_covered(self, claim_text, winning_realization_text):
        return False, 0.9, "not the same claim"


class _ExplodingArbiter:
    def claim_covered(self, claim_text, winning_realization_text):
        raise RuntimeError("boom")


def _claim():
    return extract_claims("clip_a", "The biopsy confirmed it was a benign tumor.")[0]


def test_resolve_ambiguous_coverage_confident_high_is_covered_without_arbiter():
    claim = _claim()
    assert resolve_ambiguous_coverage(claim, "text", coverage=COVERAGE_THRESHOLD, arbiter=None) is True


def test_resolve_ambiguous_coverage_confident_low_is_lost_without_consulting_arbiter():
    claim = _claim()
    arbiter = _AlwaysCoveredArbiter()
    result = resolve_ambiguous_coverage(
        claim, "text", coverage=AMBIGUOUS_COVERAGE_FLOOR - 0.01, arbiter=arbiter,
    )
    assert result is False


def test_resolve_ambiguous_coverage_ambiguous_band_no_arbiter_fails_open_to_lost():
    claim = _claim()
    mid = (AMBIGUOUS_COVERAGE_FLOOR + COVERAGE_THRESHOLD) / 2
    assert resolve_ambiguous_coverage(claim, "text", coverage=mid, arbiter=None) is False


def test_resolve_ambiguous_coverage_ambiguous_band_arbiter_confirms_covered():
    claim = _claim()
    mid = (AMBIGUOUS_COVERAGE_FLOOR + COVERAGE_THRESHOLD) / 2
    assert resolve_ambiguous_coverage(claim, "text", coverage=mid, arbiter=_AlwaysCoveredArbiter()) is True


def test_resolve_ambiguous_coverage_ambiguous_band_arbiter_denies_covered():
    claim = _claim()
    mid = (AMBIGUOUS_COVERAGE_FLOOR + COVERAGE_THRESHOLD) / 2
    assert resolve_ambiguous_coverage(claim, "text", coverage=mid, arbiter=_AlwaysLostArbiter()) is False


def test_resolve_ambiguous_coverage_arbiter_exception_fails_open_to_lost():
    claim = _claim()
    mid = (AMBIGUOUS_COVERAGE_FLOOR + COVERAGE_THRESHOLD) / 2
    assert resolve_ambiguous_coverage(claim, "text", coverage=mid, arbiter=_ExplodingArbiter()) is False
