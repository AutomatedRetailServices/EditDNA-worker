"""D-289.7 -- proposition scope for the two negation/number authorities
(residual R-289.4a), on the isolated branch `fix/editorial-realization-
closure`.

Reproduction the Product Owner verified: in RAW #123's winning clip W,
replacing ONLY the comma before "más bien" with a period (identical words
and figures) changed `detect_text_contradiction(W, R+T)`: negation_conflict
True -> False, and `claim_coverage(R+T's claim, W)`: 0.05 -> 0.5556. Both
authorities scoped their negation/number checks to the text between two
terminal punctuation marks, so the verdict depended on punctuation the
speaker never uttered.

Fix (no threshold changed, no transcript rewritten, no Video00 exception):
`semantic_claims.proposition_units` / `proposition_scope_units` -- a
sentence further divided at the genuine clause connectors D-040's claim
extraction already recognises, now including the corrective-contrast
connectors ("más bien"/"sino"/"rather"/"instead"...) -- is the ONE
segmentation both `semantic_claims.claim_coverage` and
`contradiction_signal` use. A reference that itself relates two
propositions keeps sentence scope (both halves of a relation stay in view);
a sentence with no recognised connector stays one unit, so an ambiguous
negation keeps its whole-sentence scope and fails closed as before.

Also fixed on the traced path: a bridge with ONE realization unit on each
side fixed the LEFT unit as the "newcomer" regardless of chronology, so an
earlier complete realization was tested as a restatement of its own later
restatement and guard 8 refused the shape before any preservation proof.

RAW #123 texts here are QA fixtures imported from the D-289.4 file;
production code reads none of them. Every arbiter answer is SIMULATED and
labelled as such.
"""
from __future__ import annotations

import pytest

from cutsell_worker import contradiction_signal as cs
from cutsell_worker import semantic_claims as sc
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.contradiction_signal import detect_text_contradiction
from cutsell_worker.semantic_claims import (
    AMBIGUOUS_COVERAGE_FLOOR,
    COVERAGE_THRESHOLD,
    _CONTRASTIVE_MARKERS,
    _CORRECTIVE_CONTRAST_MARKERS,
    _DEFINITIVE_MISMATCH_COVERAGE_CAP,
    _split_into_clauses,
    claim_coverage,
    extract_claims,
    proposition_scope_units,
    proposition_units,
    resolve_ambiguous_coverage,
    spans_multiple_propositions,
)
from cutsell_worker.semantic_idea_equivalence import SemanticEquivalenceGatePolicy
from cutsell_worker.take_grouping_provider import (
    CONTAINED_RESTATEMENT_ACCEPTANCE,
    _PAIR_BUDGET_PER_GROUP_CAP,
    _RetryEdge,
    _bridge_aware_components,
)

from tests.test_cutsell_d289_4_raw123_pre_cleanup_causes import R123, T123, W123
from tests.test_cutsell_d289_contained_realization_closure import ClaimArbiter, RecordedAnswersArbiter

CAP = _DEFINITIVE_MISMATCH_COVERAGE_CAP
RT123 = R123 + " " + T123
W123_PERIOD = W123.replace("hereditarios, más bien,", "hereditarios. Más bien,")


def _take(cid, start, end, text, *, complete=True, source="src"):
    return CandidateTake(cid, source, 0, start, end, text, complete_idea=complete)


def _claim(text):
    claims = extract_claims("c", text)
    assert len(claims) == 1, [c.text for c in claims]
    return claims[0]


def _both(claim_text, candidate):
    """(contradiction verdict, coverage of `claim_text`'s one claim by `candidate`)."""
    return detect_text_contradiction(claim_text, candidate), claim_coverage(_claim(claim_text), candidate)


# =============================================================================
# A. Reproduction: the verdict no longer depends on the comma
# =============================================================================

def test_reproduction_comma_and_period_now_read_identically_on_both_authorities():
    assert W123_PERIOD != W123 and W123_PERIOD.split() != W123.split()  # only the punctuation differs
    assert [t.strip(".,") for t in W123_PERIOD.casefold().split()] == [t.strip(".,") for t in W123.casefold().split()]
    for winner in (W123, W123_PERIOD):
        verdict = detect_text_contradiction(winner, RT123)
        assert verdict.negation_conflict is False and verdict.number_conflict is False
        (claim,) = extract_claims("RT", RT123)
        assert claim.importance == "CRITICAL"
        assert claim_coverage(claim, winner) == pytest.approx(0.5556, abs=1e-4)
    # the figure's proposition and the rejected proposition are separate units of the comma-run text
    units = proposition_units(W123)
    assert any(u.startswith("por eso no creo") and "5" not in u for u in units)
    assert any(u.startswith("más bien, solo un 5 o 10 %") and "no " not in u for u in units)
    assert all(u in W123 for u in units)


def test_reproduction_the_winner_own_claims_still_cover_the_winner():
    claims = extract_claims("W", W123)
    assert [c.claim_type for c in claims if c.importance == "CRITICAL"] == ["NEGATION", "MEASUREMENT_QUANTITY"]
    assert all(claim_coverage(c, W123) >= COVERAGE_THRESHOLD for c in claims)


# =============================================================================
# B. Shared logic: one segmentation, imported by both authorities
# =============================================================================

def test_contradiction_signal_uses_semantic_claims_segmentation_and_keeps_no_private_sentence_splitter():
    assert cs.proposition_units is sc.proposition_units
    assert cs.proposition_scope_units is sc.proposition_scope_units
    assert not hasattr(cs, "_SENTENCE_SPLIT_RE") and not hasattr(cs, "_sentences")
    assert cs._propositions("El producto no es caro, pero el envío es lento.") == proposition_units(
        "El producto no es caro, pero el envío es lento."
    ) == ("El producto no es caro,", "pero el envío es lento.")


def test_proposition_units_are_sentences_then_clauses_and_always_substrings():
    text = "Nunca revisamos la tiroides, porque cada año me hacía dos exámenes. Después cambié de médico, pero no mejoró mi salud."
    units = proposition_units(text)
    assert units == ("Nunca revisamos la tiroides,", "porque cada año me hacía dos exámenes.",
                     "Después cambié de médico,", "pero no mejoró mi salud.")
    assert all(u in text for u in units)
    assert proposition_units("") == () and proposition_units("solo tres palabras") == ("solo tres palabras",)


def test_corrective_contrast_connectors_split_for_scope_but_do_not_widen_the_d065_negation_role_gate():
    for marker in ("más bien", "mas bien", "sino", "mejor dicho", "rather", "instead"):
        assert marker in _CORRECTIVE_CONTRAST_MARKERS
        assert marker not in _CONTRASTIVE_MARKERS  # D-065/D-066 eligibility vocabulary untouched
    assert _split_into_clauses("No es un producto barato para todos, más bien es una inversión para pocos.") == (
        "No es un producto barato para todos,", "más bien es una inversión para pocos.")
    assert _split_into_clauses("It is not a cheap product for everyone, rather it is an investment for a few.") == (
        "It is not a cheap product for everyone,", "rather it is an investment for a few.")


# =============================================================================
# C. Generic controls (Spanish + English) -- both authorities, same evidence
# =============================================================================

@pytest.mark.parametrize("claim_text, candidate", [
    ("El producto es caro.", "El producto es caro, pero no lo recomiendo para pieles sensibles."),
    ("The cream left a sticky residue.", "The cream did not irritate my skin, but it left a sticky residue."),
    ("La causa es ambiental en la mayoría.", "La causa no es genética en estos casos, sino ambiental en la mayoría."),
    ("Solo un 5 % de los casos de cáncer son hereditarios.", "No es un 30 % de los casos de cáncer, más bien solo un 5 % son hereditarios."),
    ("Only 5 percent of these cases are hereditary.", "It is not most of these cases, rather only 5 percent are hereditary."),
])
def test_contrast_or_correction_whose_negation_belongs_to_the_OTHER_proposition_is_covered(claim_text, candidate):
    claim = _claim(claim_text)
    coverage = claim_coverage(claim, candidate)
    assert coverage >= COVERAGE_THRESHOLD, coverage
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=None) is True


@pytest.mark.parametrize("claim_text, candidate", [
    ("El producto es caro.", "El producto es caro, pero no lo recomiendo para pieles sensibles."),
    ("The cream left a sticky residue.", "The cream did not irritate my skin, but it left a sticky residue."),
    ("La causa es ambiental en la mayoría.", "La causa no es genética en estos casos, sino ambiental en la mayoría."),
])
def test_contrast_whose_negation_belongs_to_the_OTHER_proposition_is_not_a_contradiction(claim_text, candidate):
    assert detect_text_contradiction(claim_text, candidate).negation_conflict is False


@pytest.mark.parametrize("claim_text, candidate", [
    ("El producto es caro.", "El producto no es caro, pero el envío es lento."),
    ("The cream irritated my skin badly.", "The cream did not irritate my skin badly, but it left a sticky residue."),
    ("La causa es genética en estos casos.", "La causa no es genética en estos casos, sino ambiental en la mayoría."),
    ("The medication worked well for her symptoms.", "The medication never worked well for her symptoms."),
    ("La familia tiene antecedentes de esta condición.", "La familia no tiene antecedentes de esta condición, aunque sí de otras similares."),
])
def test_negation_that_belongs_to_the_compared_proposition_is_preserved_by_both_authorities(claim_text, candidate):
    verdict, coverage = _both(claim_text, candidate)
    assert verdict.negation_conflict is True
    assert coverage <= CAP
    assert resolve_ambiguous_coverage(_claim(claim_text), candidate, coverage=coverage, arbiter=ClaimArbiter(True)) is False


@pytest.mark.parametrize("claim_text, candidate", [
    ("Solo un 30 % de los casos de cáncer son hereditarios.", "No es un 30 % de los casos de cáncer, más bien solo un 5 % son hereditarios."),
    ("Only 30 percent of these cases are hereditary.", "It is not 30 percent of these cases, rather only 5 percent are hereditary."),
    ("Only 5 percent of these cases are hereditary.", "Only 10 percent of these cases are hereditary."),
    ("El tratamiento dura 3 semanas en total.", "El tratamiento dura 6 semanas en total, aunque no siempre."),
])
def test_a_numeric_difference_is_still_a_definitive_mismatch_on_both_authorities(claim_text, candidate):
    verdict, coverage = _both(claim_text, candidate)
    assert verdict.number_conflict is True
    assert coverage <= CAP


def test_a_corrected_number_is_flagged_by_the_claim_it_corrects_not_by_its_neighbour():
    candidate = "No es un 30 % de los casos de cáncer, más bien solo un 5 % son hereditarios."
    assert claim_coverage(_claim("Solo un 30 % de los casos de cáncer son hereditarios."), candidate) <= CAP
    assert claim_coverage(_claim("Solo un 5 % de los casos de cáncer son hereditarios."), candidate) >= COVERAGE_THRESHOLD
    # the primitive keeps its whole-text number rule: a correction of the number is a conflict (D-056.5 matrix)
    assert detect_text_contradiction("Solo un 30 % de los casos de cáncer son hereditarios.", candidate).number_conflict is True


@pytest.mark.parametrize("claim_text, candidate", [
    ("The flare-ups happen because of stress.", "Stress happens because of the flare-ups."),
    ("Los brotes aparecen porque me estreso.", "Me estreso porque aparecen los brotes."),
])
def test_a_complete_causal_relation_keeps_sentence_scope_so_the_inversion_is_still_detected(claim_text, candidate):
    assert spans_multiple_propositions(claim_text) is True
    assert proposition_scope_units(claim_text, candidate) == (candidate,)  # sentence granularity for a relation
    assert claim_coverage(_claim(claim_text), candidate) <= CAP


def test_sentence_level_relational_claim_of_d289_6_still_detects_the_inversion():
    (whole,) = extract_claims("U", "Anxiety occurs because of the severe stress.", split_clauses=False)
    assert claim_coverage(whole, "Severe stress occurs because of anxiety.") <= CAP
    assert claim_coverage(whole, "Anxiety occurs because of severe stress at work, most days.") >= COVERAGE_THRESHOLD


# --- ambiguous cases that must stay closed -----------------------------------

@pytest.mark.parametrize("claim_text, candidate", [
    # no recognised connector: the segmenter cannot see the clause boundary -> whole-sentence scope, as before
    ("Solo un 5 o 10 % de los cánceres son hereditarios.", "No creo que los cánceres son hereditarios, solo un 5 o 10 % de los cánceres lo son."),
    ("Only 5 to 10 percent of cancers are hereditary.", "I do not think cancers are hereditary, only 5 to 10 percent of cancers are."),
    # a connector whose left side is too thin to be a clause (D-040's floor): no split, fails closed
    ("Solo un 5 % son hereditarios.", "No es así, más bien solo un 5 % son hereditarios."),
])
def test_ambiguous_negation_without_a_visible_clause_boundary_still_fails_closed(claim_text, candidate):
    assert len(proposition_units(candidate)) == 1
    verdict, coverage = _both(claim_text, candidate)
    assert verdict.negation_conflict is True
    assert coverage <= CAP


def test_a_terse_rejection_of_the_broader_claim_next_to_the_figure_still_vetoes_in_the_primitive():
    """The primitive's D-056.5 lexical bar is unchanged: a short negated
    clause that shares most of its content with the compared figure ("no
    creo que los cánceres son hereditarios" vs "... % de los cánceres son
    hereditarios") is still read as the same proposition, negated -> the
    veto stays. RAW #123's W passes that bar only because its rejected
    clause carries its own extra content ("está comprobado
    científicamente") -- recorded honestly, not tuned."""
    terse = "no creo que los cánceres son hereditarios, más bien solo un 5 o 10 % de los cánceres lo son."
    assert len(proposition_units(terse)) == 2
    assert detect_text_contradiction("solo un 5 o 10 % de los cánceres son hereditarios.", terse).negation_conflict is True
    # the coverage authority, judging the claim's own proposition, reads the figure as asserted
    assert claim_coverage(_claim("solo un 5 o 10 % de los cánceres son hereditarios."), terse) >= COVERAGE_THRESHOLD


# =============================================================================
# D. Granularity rule and unchanged behaviour elsewhere
# =============================================================================

def test_spans_multiple_propositions_ignores_a_leading_connector():
    assert spans_multiple_propositions("por eso no creo que los cánceres son hereditarios") is False
    assert spans_multiple_propositions("so I am convinced that only 5 to 10 percent are hereditary") is False
    assert spans_multiple_propositions("The device never failed because we tested it daily") is True
    assert spans_multiple_propositions("") is False


def test_proposition_scope_units_degrade_to_the_whole_candidate():
    assert proposition_scope_units("una cosa", "") == ("",)
    assert proposition_scope_units("The flare-ups happen because of stress.", "one clause only") == ("one clause only",)


def test_d059_scoping_tests_shape_unchanged_unrelated_sentence_negation_never_poisons_coverage():
    candidate = ("I do not think the weather affected anything at all that week. "
                 "The doctor said only 5 to 10 percent of these cases are hereditary.")
    assert claim_coverage(_claim("Only 5 to 10 percent of these cases are hereditary in nature."), candidate) >= COVERAGE_THRESHOLD


def test_default_extract_claims_unchanged_for_text_without_the_new_connectors():
    text = "Anxiety occurs because of the severe stress."
    assert [c.text for c in extract_claims("U", text)] == ["Anxiety occurs", "because of the severe stress."]
    # D-040's floor unchanged: a thin trailing clause is not split (the semantic_claims suite's own fixture)
    assert [c.text for c in extract_claims("U", "The tumor measured 3 centimeters, but it was benign.")] == [
        "The tumor measured 3 centimeters, but it was benign."]


# =============================================================================
# E. The traced path: one unit on each side of the bridge
# =============================================================================

def _one_vs_one(edge_left, edge_right, *, claim_arbiter):
    """A complete realization W, its later restatement chain R+T, one
    semantic edge between W and R oriented as given. The pairwise arbiter
    answers nothing (any component probe is a labelled decline)."""
    W = _take("W", 0.0, 8.0, "Only about 5 to 10 percent of the cancers are hereditary, the rest is lifestyle, so take care.")
    R = _take("R", 12.0, 16.0, "so I am convinced that only about 5 to 10 percent of the", complete=False)
    T = _take("T", 16.5, 18.0, "cancers are hereditary.")
    takes = (W, R, T)
    edges = [_RetryEdge("R", "T", "deterministic", 1.0, "sentence_continuation"),
             _RetryEdge(edge_left, edge_right, "semantic", 0.85, "SIMULATED same-idea confirmation")]
    trace = []
    comps = _bridge_aware_components(
        ("W", "R", "T"), edges, protected_ids=frozenset(), take_map={t.clip_id: t for t in takes},
        arbiter=RecordedAnswersArbiter({}, unlisted_reason="simulated_decline"), policy=SemanticEquivalenceGatePolicy(),
        edge_trace=trace, claim_equivalence_arbiter=claim_arbiter,
    )
    return comps, [r for r in trace if r.get("bridge_sensitive")]


@pytest.mark.parametrize("edge_left, edge_right", [("W", "R"), ("R", "W")])
def test_one_unit_each_side_the_later_unit_is_the_newcomer_whichever_side_of_the_edge(edge_left, edge_right):
    comps, rows = _one_vs_one(edge_left, edge_right, claim_arbiter=ClaimArbiter(True))
    assert rows and rows[0]["accepted"] is True and rows[0]["accepted_by"] == CONTAINED_RESTATEMENT_ACCEPTANCE
    assert rows[0]["restated_unit_member_ids"] == ["R", "T"]
    assert rows[0]["preservation_evidence"]["preserving_member_clip_id"] == "W"
    assert ("W", "R", "T") in comps


def test_one_unit_each_side_without_preservation_the_path_falls_through_to_the_probe():
    comps, rows = _one_vs_one("W", "R", claim_arbiter=ClaimArbiter(False))
    # the restatement's claim is deterministically covered here (same words), so the
    # declining claim arbiter is never needed; show the fall-through with a winner that
    # does NOT preserve the figure instead
    assert rows and rows[0]["accepted"] is True
    W = _take("W", 0.0, 8.0, "Most of these cancers come down to lifestyle, so take care of yourself every day.")
    R = _take("R", 12.0, 16.0, "so I am convinced that only about 5 to 10 percent of the", complete=False)
    T = _take("T", 16.5, 18.0, "cancers are hereditary.")
    trace = []
    comps = _bridge_aware_components(
        ("W", "R", "T"),
        [_RetryEdge("R", "T", "deterministic", 1.0, "sentence_continuation"), _RetryEdge("W", "R", "semantic", 0.85, "SIMULATED")],
        protected_ids=frozenset(), take_map={t.clip_id: t for t in (W, R, T)},
        arbiter=RecordedAnswersArbiter({}, unlisted_reason="simulated_decline"), policy=SemanticEquivalenceGatePolicy(),
        edge_trace=trace, claim_equivalence_arbiter=ClaimArbiter(True),
    )
    rows = [r for r in trace if r.get("bridge_sensitive")]
    assert rows and rows[0]["accepted"] is False and rows[0]["reason_rejected"] == "component_cohesion_declined"
    assert ("W",) in comps and ("R", "T") in comps  # digits the winner never states: guard 3 refused, probe declined


# =============================================================================
# F. Budget and per-group cap unchanged (anti-monopolisation stays in force)
# =============================================================================

def test_pair_budget_and_per_group_cap_are_unchanged():
    assert SemanticEquivalenceGatePolicy().max_pairs_per_request == 14
    assert _PAIR_BUDGET_PER_GROUP_CAP == 2
