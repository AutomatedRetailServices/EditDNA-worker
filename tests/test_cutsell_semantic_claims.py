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
    _split_into_clauses,
    classify_claim,
    claim_coverage,
    claim_is_covered,
    dedupe_claims,
    extract_claims,
    resolve_ambiguous_clause_role,
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
        extract_claims("clip_b", "I walked into the clinic and sat down.")
    deduped = dedupe_claims(claims)
    assert len(deduped) == 2


def test_dedupe_claims_keeps_distinct_clauses_from_the_same_sentence():
    # D-040: a core+supporting split from ONE sentence produces two
    # distinct propositions (different claim_type), not near-duplicates.
    claims = extract_claims("clip_a", "I felt tired because I skipped breakfast.")
    assert len(claims) == 2
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


# --- D-040: clause splitting (core vs. supporting/contextual clause) -------
#
# Offline audit of RAW 33448261223: a whole multi-clause sentence was
# treated as one atomic claim, so a winning realization that preserved the
# CORE proposition but dropped a merely-supporting reason clause scored as
# if the core itself were lost. `extract_claims` now splits on genuine
# connectors first, and each clause is classified independently by the
# same deterministic `classify_claim` rules -- no new importance axis, no
# Video00 phrase hardcoded. The 10 cases below are the directive's own
# false-positive-protection list.

def test_split_into_clauses_no_connector_is_unsplit():
    assert _split_into_clauses("I walked into the clinic and sat down.") == \
        ("I walked into the clinic and sat down.",)


def test_split_into_clauses_short_trailing_remainder_is_not_split():
    # "but it was benign" has only one content token ("benign") after
    # stopword filtering -- too thin to stand alone, so the connector is
    # skipped rather than carving off a degenerate fragment.
    assert _split_into_clauses("The tumor measured 3 centimeters, but it was benign.") == \
        ("The tumor measured 3 centimeters, but it was benign.",)


# 1. Core claim preserved, trailing supporting clause omitted -> PASS
def test_case_1_core_preserved_supporting_omitted_is_not_critical_loss():
    sentence = "Nunca se nos ocurrio hacer un chequeo de la tiroides, porque cada ano me hacia dos examenes normales."
    claims = extract_claims("clip_a", sentence)
    critical = [c for c in claims if c.importance == CRITICAL]
    supporting = [c for c in claims if c.importance != CRITICAL]
    assert len(critical) == 1
    assert critical[0].claim_type == NEGATION
    assert len(supporting) == 1
    # The winning realization keeps only the core, drops the reason clause.
    winner_text = "Nunca se nos ocurrio hacer un chequeo de la tiroides."
    assert claim_is_covered(critical[0], winner_text)


# 2. Core diagnosis dropped, supporting reflection preserved -> FAIL
def test_case_2_core_diagnosis_dropped_still_blocks_despite_supporting_survival():
    sentence = "The biopsy confirmed cancer, which explained symptoms I had noticed before."
    claims = extract_claims("clip_a", sentence)
    critical = [c for c in claims if c.importance == CRITICAL]
    assert len(critical) == 1
    core = critical[0]
    # Winner keeps only the supporting reflection, not the core diagnosis.
    winner_text = "It explained symptoms I had noticed before."
    assert not claim_is_covered(core, winner_text)


# 3. Cause/effect-introduced clause that is ITSELF critical -> dropping it FAILs
def test_case_3_critical_clause_introduced_by_a_cause_effect_connector_still_blocks():
    sentence = "I stopped the medication because it was not working."
    claims = extract_claims("clip_a", sentence)
    critical = [c for c in claims if c.importance == CRITICAL]
    assert len(critical) == 1
    assert critical[0].claim_type == NEGATION
    winner_text = "I stopped the medication."
    assert not claim_is_covered(critical[0], winner_text)


# 4. Redundant/explanatory clause omitted -> PASS (never even checked as critical)
def test_case_4_redundant_explanatory_clause_omission_is_not_tracked_as_critical():
    sentence = "The biopsy confirmed cancer, which explained symptoms I had noticed before."
    claims = extract_claims("clip_a", sentence)
    explanatory = [c for c in claims if "explained" in c.text]
    assert len(explanatory) == 1
    assert explanatory[0].importance != CRITICAL


# 5. Incidental date/temporal clause omitted -> WARN, never blocking
def test_case_5_incidental_temporal_clause_is_contextual_not_critical():
    sentence = "I felt worse when the weather changed."
    claims = extract_claims("clip_a", sentence)
    temporal = [c for c in claims if c.claim_type == TEMPORAL_RELATION]
    assert len(temporal) == 1
    assert temporal[0].importance == CONTEXTUAL


# 6. Numeric measurement critical -> BLOCK if lost, survives clause splitting
def test_case_6_critical_measurement_clause_still_blocks_when_dropped():
    sentence = "The tumor measured 3 centimeters, but the doctors said it was benign tissue."
    claims = extract_claims("clip_a", sentence)
    critical = [c for c in claims if c.importance == CRITICAL]
    assert len(critical) == 1
    assert critical[0].claim_type == MEASUREMENT_QUANTITY
    winner_text = "The doctors said it was benign tissue."
    assert not claim_is_covered(critical[0], winner_text)


# 7. One sentence with two independently critical claims -> both tracked, both must survive
def test_case_7_two_independently_critical_claims_in_one_sentence_both_tracked():
    sentence = "The test came back positive, but I also learned I did not have the gene."
    claims = extract_claims("clip_a", sentence)
    critical = [c for c in claims if c.importance == CRITICAL]
    assert len(critical) == 2
    assert {c.claim_type for c in critical} == {STATE_RESULT, NEGATION}
    # Each is independently checkable -- dropping either one is a loss even
    # though the other survives.
    only_first = "The test came back positive."
    only_second = "I did not have the gene."
    state_result_claim = next(c for c in critical if c.claim_type == STATE_RESULT)
    negation_claim = next(c for c in critical if c.claim_type == NEGATION)
    assert claim_is_covered(state_result_claim, only_first)
    assert not claim_is_covered(negation_claim, only_first)
    assert claim_is_covered(negation_claim, only_second)
    assert not claim_is_covered(state_result_claim, only_second)


# 8. A subordinate/connector-introduced clause that materially changes
# meaning (a correction) is still classified critical -- clause splitting
# never demotes real critical content just because it followed a connector.
def test_case_8_subordinate_correction_clause_is_still_critical():
    sentence = "The doctor said it was cancer, but actually it was not cancer at all."
    claims = extract_claims("clip_a", sentence)
    critical = [c for c in claims if c.importance == CRITICAL]
    assert len(critical) == 1
    assert critical[0].claim_type == NEGATION
    assert "actually" in critical[0].text.lower()


# 9. Paraphrased core claim still counts as covered
def test_case_9_paraphrased_core_claim_counts_as_covered():
    sentence = "Nunca se nos ocurrio hacer un chequeo de la tiroides, porque cada ano me hacia dos examenes normales."
    core = next(c for c in extract_claims("clip_a", sentence) if c.importance == CRITICAL)
    paraphrase = "Nunca se nos ocurrio hacer un chequeo de la tiroides porque siempre salia normal en mis examenes."
    assert claim_is_covered(core, paraphrase)


# 10. Same vocabulary, different proposition -> no false coverage (clause-level)
def test_case_10_same_vocabulary_different_proposition_no_false_coverage_at_clause_level():
    sentence = "The biopsy confirmed it was a benign tumor."
    core = extract_claims("clip_a", sentence)[0]
    negated_restatement = "The biopsy did not confirm it was a benign tumor."
    assert not claim_is_covered(core, negated_restatement)


# --- resolve_ambiguous_clause_role (bounded arbiter, D-040) -----------------

class _AlwaysCoreCriticalClauseArbiter:
    def __init__(self):
        self.calls = 0

    def clause_role(self, clause_text, parent_sentence_text):
        self.calls += 1
        return "CORE_CRITICAL", 0.9, "materially changes meaning"


class _AlwaysUncertainClauseArbiter:
    def clause_role(self, clause_text, parent_sentence_text):
        return "UNCERTAIN", 0.4, "cannot tell"


class _ExplodingClauseArbiter:
    def clause_role(self, clause_text, parent_sentence_text):
        raise RuntimeError("boom")


def test_resolve_ambiguous_clause_role_leaves_marker_based_evidence_untouched():
    # A real deterministic marker (negation_present) is never second-guessed
    # by the arbiter, confirmed or not.
    result = resolve_ambiguous_clause_role(
        "it was not working", "parent", deterministic_importance=CRITICAL,
        evidence="negation_present", arbiter=_AlwaysCoreCriticalClauseArbiter(),
    )
    assert result == CRITICAL


def test_resolve_ambiguous_clause_role_no_arbiter_keeps_deterministic_fallback():
    result = resolve_ambiguous_clause_role(
        "some marker-less clause", "parent", deterministic_importance=SUPPORTING,
        evidence="general_statement", arbiter=None,
    )
    assert result == SUPPORTING


def test_resolve_ambiguous_clause_role_arbiter_confirms_core_critical_upgrades():
    arbiter = _AlwaysCoreCriticalClauseArbiter()
    result = resolve_ambiguous_clause_role(
        "some marker-less clause", "parent", deterministic_importance=SUPPORTING,
        evidence="general_statement", arbiter=arbiter,
    )
    assert result == CRITICAL
    assert arbiter.calls == 1


def test_resolve_ambiguous_clause_role_arbiter_uncertain_fails_open_to_critical():
    result = resolve_ambiguous_clause_role(
        "some marker-less clause", "parent", deterministic_importance=SUPPORTING,
        evidence="general_statement", arbiter=_AlwaysUncertainClauseArbiter(),
    )
    assert result == CRITICAL


def test_resolve_ambiguous_clause_role_arbiter_exception_fails_open_to_critical():
    result = resolve_ambiguous_clause_role(
        "some marker-less clause", "parent", deterministic_importance=SUPPORTING,
        evidence="general_statement", arbiter=_ExplodingClauseArbiter(),
    )
    assert result == CRITICAL
