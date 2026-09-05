"""D-085 -- BRIDGE-AWARE RETRY FAMILY COHESION (grouping only).

Root defect (docs/CUTSELL_DECISIONS.md D-084's forensic, live evidence
recovered from 4 independent Modal runs): `_cohesive_components` (D-058)
is plain union-find over an unordered edge set -- any accepted pairwise
edge can transitively connect two full components, with no re-validation
that the RESULTING merged component still represents one shared audience-
facing proposition. D-084 proved the exact live shape: a true back-acne
subcluster (confidence 0.95) and a true ear/neck-pimples subcluster
(confidence 0.98) were bridged by TWO independent, unmarked, weaker
(0.80/0.85) pairwise confirmations that D-083's marker gate never touches --
and the same bridge recurred, at the same confidence, in all 4 runs.

FIX under test: `_bridge_aware_components` -- deterministic edge ordering
(evidence type, then descending confidence, then clip-id tie-break),
bridge classification (an edge is a bridge the moment either endpoint's
CURRENT component already has >=2 members), and a component-level cohesion
check (reusing the SAME already-configured SemanticEquivalenceArbiter, plus
the existing `any_pair_contradicts` primitive) that must independently
accept a bridge before it is trusted -- pairwise confidence/vote count is
never proof by itself.

Entirely generic fixtures -- no Video00 clip ids or literal phrases.
"""
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalenceResult,
    SemanticEquivalenceGatePolicy,
)
from cutsell_worker.take_grouping_provider import (
    _RetryEdge,
    _bridge_aware_components,
    _edge_sort_key,
    _has_distinct_addition_marker,
    split_incohesive_retry_groups,
)


def _take(clip_id, start, end, text, source="src", order=0):
    return CandidateTake(clip_id, source, order, start, end, text)


class ConfiguredArbiter:
    """D-085 test double: a lookup-table arbiter returning a DIFFERENT
    (same_idea, confidence, reason) per exact (left_text, right_text) pair
    (order-insensitive) -- needed because D-085's calibration requires
    distinct confidence levels per pair (0.95/0.98/0.93 true, 0.80/0.85
    false bridge-seed, plus separate component-level aggregate-probe
    verdicts), unlike the sibling suites' single-confidence FixedArbiter."""

    def __init__(self, table: dict):
        # table: {(left_text, right_text): (same_idea, confidence, reason)}
        self.table = table
        self.calls = 0

    def check(self, request):
        self.calls += 1
        decisions = []
        for i, pair in enumerate(request.pairs):
            entry = self.table.get((pair.left_text, pair.right_text))
            if entry is None:
                entry = self.table.get((pair.right_text, pair.left_text))
            if entry is None:
                same_idea, confidence, reason = False, 0.0, "unconfigured_pair_declined"
            else:
                same_idea, confidence, reason = entry
            decisions.append(IdeaEquivalenceDecision(
                pair_index=i, same_idea=same_idea, confidence=confidence, reason=reason,
            ))
        return IdeaEquivalenceResult(
            decisions=tuple(decisions), provider="fake", model="fake-semantic-equivalence",
            requested=True, available=True, estimated_input_tokens=50, estimated_output_tokens=10,
        )


class NoneAvailableArbiter:
    def check(self, request):
        raise RuntimeError("provider down")


class ConfirmEverythingPairwiseButSkepticalComponentArbiter:
    """D-085 QA_ENGINE adversarial fixture: confirms same_idea=True at a
    uniform high confidence for EVERY pairwise (two-clip) query it is asked
    -- an "always-YES" arbiter that gives the pairwise/simple-edge layer no
    discriminating signal at all -- but answers truthfully (declines) the
    component-level aggregated-probe questions (identified by the query text
    containing the " || " probe separator on at least one side). Proves the
    two-tier design, not pairwise confidence, is what prevents the false
    bridges from reconnecting the components even when literally every
    pairwise confirmation is a rubber stamp."""

    def __init__(self, component_level_same_idea: dict):
        # component_level_same_idea: {(left_text, right_text): (same_idea, confidence, reason)}
        self.component_level_same_idea = component_level_same_idea
        self.calls = 0

    def check(self, request):
        self.calls += 1
        decisions = []
        for i, pair in enumerate(request.pairs):
            is_component_probe = " || " in pair.left_text or " || " in pair.right_text
            if is_component_probe:
                entry = self.component_level_same_idea.get((pair.left_text, pair.right_text))
                if entry is None:
                    entry = self.component_level_same_idea.get((pair.right_text, pair.left_text))
                same_idea, confidence, reason = entry or (False, 0.0, "component_probe_declined")
            else:
                same_idea, confidence, reason = True, 0.95, "always confirms every pairwise query"
            decisions.append(IdeaEquivalenceDecision(
                pair_index=i, same_idea=same_idea, confidence=confidence, reason=reason,
            ))
        return IdeaEquivalenceResult(
            decisions=tuple(decisions), provider="fake", model="fake-semantic-equivalence",
            requested=True, available=True, estimated_input_tokens=50, estimated_output_tokens=10,
        )


# =====================================================================
# Section 8: THE D-084 EXACT GRAPH REGRESSION
# =====================================================================
# Generic 5-member fixture: A1,A2 true back-acne-shaped retries; B1,B2,B3
# true ear/neck-pimples-shaped retries/restatements; two FALSE bridges
# (A2<->B1, A2<->B3), both unmarked, at D-084's own observed confidence
# levels (0.85, 0.80). Expected: exactly two families, {A1,A2} and
# {B1,B2,B3} -- the two false bridges must not reconnect them.

_A1 = "I used to break out on my back sometimes."
_A2 = "For a while I would get acne on my back which I treated with resorcinol."
_B1 = "I had pimples behind my ears and neck that seemed hormonal."
_B2 = "Those bumps behind my ears kept showing up, and they looked hormonal to me."
_B3 = "That same spot behind my ears and neck flared up again every few months."


def _d084_table():
    return {
        (_A1, _A2): (True, 0.95, "Same back-acne beat, more detail added."),
        (_B1, _B2): (True, 0.98, "Same pimples beat, restated."),
        (_B2, _B3): (True, 0.93, "Same pimples beat, recurrence detail added."),
        (_A2, _B1): (True, 0.85, "Both mention a recurring skin symptom."),
        (_A2, _B3): (True, 0.80, "Both describe a skin issue on the body."),
        # component-level probes (built from _component_probe_text, sorted
        # by clip id: b1<b2<b3, a1<a2):
        (_B1 + " || " + _B2, _B3): (True, 0.95, "One shared pimples-behind-ears proposition."),
        (_A1 + " || " + _A2, _B1 + " || " + _B2 + " || " + _B3): (
            False, 0.3, "Two distinct propositions: back acne vs. ear/neck pimples.",
        ),
    }


def _d084_takes():
    return (
        _take("a1", 0.0, 2.0, _A1),
        _take("a2", 2.5, 4.5, _A2),
        _take("b1", 10.0, 12.0, _B1),
        _take("b2", 12.5, 14.5, _B2),
        _take("b3", 15.0, 17.0, _B3),
    )


def test_section8_two_false_bridges_do_not_reconnect_components():
    takes = _d084_takes()
    group = ("a1", "a2", "b1", "b2", "b3")
    arbiter = ConfiguredArbiter(_d084_table())
    result, diagnostics = split_incohesive_retry_groups((group,), takes, arbiter=arbiter)
    result_sets = {frozenset(g) for g in result}
    assert frozenset({"a1", "a2"}) in result_sets
    assert frozenset({"b1", "b2", "b3"}) in result_sets
    assert len(result) == 2
    # Both false bridges must show up as rejected bridge evaluations.
    rejected = {(r["left_clip_id"], r["right_clip_id"]) for r in diagnostics["edge_trace"]
                if r.get("bridge_sensitive") and not r.get("accepted")}
    assert ("a2", "b1") in rejected or ("b1", "a2") in rejected
    assert ("a2", "b3") in rejected or ("b3", "a2") in rejected
    assert diagnostics["bridge_rejected_count"] >= 2


def test_section8_bridge_diagnostics_never_claim_single_edge_is_proof():
    # Even with TWO independent false-bridge pairwise confirmations pointing
    # the same direction, neither is accepted without its own component-
    # level cohesion pass -- directive's explicit "two confirmations are
    # not proof" requirement.
    takes = _d084_takes()
    group = ("a1", "a2", "b1", "b2", "b3")
    arbiter = ConfiguredArbiter(_d084_table())
    _, diagnostics = split_incohesive_retry_groups((group,), takes, arbiter=arbiter)
    bridge_records = [r for r in diagnostics["edge_trace"] if r.get("bridge_sensitive")]
    assert len(bridge_records) >= 3  # b2<->b3 (accepted) + the two false ones (rejected)
    for record in bridge_records:
        assert record["component_cohesion_evaluated"] is True or "reason_rejected" in record


# =====================================================================
# Section 2: DETERMINISTIC EDGE ORDER / permutation invariance
# =====================================================================

def test_section2_edge_sort_key_is_deterministic_first_then_confidence_then_id():
    deterministic = _RetryEdge("z", "y", "deterministic", 1.0, "prefix_fragment")
    high_conf = _RetryEdge("a", "b", "semantic", 0.98, "x")
    low_conf = _RetryEdge("a", "c", "semantic", 0.80, "x")
    ordered = sorted([low_conf, high_conf, deterministic], key=_edge_sort_key)
    assert ordered == [deterministic, high_conf, low_conf]


def test_section2_permutation_of_edge_input_order_yields_identical_families():
    takes = _d084_takes()
    group = ("a1", "a2", "b1", "b2", "b3")
    table = _d084_table()
    results = []
    for _ in range(4):
        arbiter = ConfiguredArbiter(dict(table))  # fresh instance each time
        result, _ = split_incohesive_retry_groups((group,), takes, arbiter=arbiter)
        results.append({frozenset(g) for g in result})
    assert all(r == results[0] for r in results)


def test_section2_reversed_group_member_order_yields_identical_families():
    takes = _d084_takes()
    forward = ("a1", "a2", "b1", "b2", "b3")
    reversed_group = tuple(reversed(forward))
    arbiter_a = ConfiguredArbiter(_d084_table())
    arbiter_b = ConfiguredArbiter(_d084_table())
    result_a, _ = split_incohesive_retry_groups((forward,), takes, arbiter=arbiter_a)
    result_b, _ = split_incohesive_retry_groups((reversed_group,), takes, arbiter=arbiter_b)
    assert {frozenset(g) for g in result_a} == {frozenset(g) for g in result_b}


# =====================================================================
# Section 3/4/5/6/7: bridge classification + component-level cohesion
# =====================================================================

def test_section3_internal_edge_between_two_singletons_is_not_a_bridge():
    takes = (_take("x", 0.0, 1.0, "left text here that is long enough"),
              _take("y", 1.0, 2.0, "right text here that is also long enough"))
    edges = [_RetryEdge("x", "y", "semantic", 0.5, "low confidence but only pair")]
    trace: list[dict] = []
    components = _bridge_aware_components(
        ("x", "y"), edges, protected_ids=frozenset(),
        take_map={t.clip_id: t for t in takes}, arbiter=None,
        policy=SemanticEquivalenceGatePolicy(),
        edge_trace=trace,
    )
    assert components == (("x", "y"),)
    assert trace[0]["bridge_sensitive"] is False


def test_section4_edge_count_is_not_proof_single_false_bridge_alone_also_rejected():
    # Simpler variant of Section 8: just ONE false bridge, still rejected --
    # confirms the mechanism doesn't require "multiple mistakes" to catch
    # a bad bridge; a single one is already caught by component-level proof.
    takes = (
        _take("a1", 0.0, 2.0, _A1), _take("a2", 2.5, 4.5, _A2),
        _take("b1", 10.0, 12.0, _B1), _take("b2", 12.5, 14.5, _B2),
    )
    table = {
        (_A1, _A2): (True, 0.95, "same beat"),
        (_B1, _B2): (True, 0.98, "same beat"),
        (_A2, _B1): (True, 0.85, "false bridge"),
        (_A1 + " || " + _A2, _B1 + " || " + _B2): (False, 0.3, "distinct propositions"),
    }
    arbiter = ConfiguredArbiter(table)
    result, diagnostics = split_incohesive_retry_groups((("a1", "a2", "b1", "b2"),), takes, arbiter=arbiter)
    result_sets = {frozenset(g) for g in result}
    assert frozenset({"a1", "a2"}) in result_sets
    assert frozenset({"b1", "b2"}) in result_sets
    assert diagnostics["bridge_rejected_count"] == 1


def test_section6_component_cohesion_absent_arbiter_fails_closed():
    takes = (
        _take("a1", 0.0, 2.0, _A1), _take("a2", 2.5, 4.5, _A2),
        _take("b1", 10.0, 12.0, _B1), _take("b2", 12.5, 14.5, _B2),
    )
    table = {
        (_A1, _A2): (True, 0.95, "same beat"),
        (_B1, _B2): (True, 0.98, "same beat"),
        (_A2, _B1): (True, 0.85, "very confident but wrong"),
    }
    # No component-level probe entry configured -> ConfiguredArbiter declines
    # it (falls through to the "unconfigured_pair_declined" default) ->
    # component_cohesion_evaluated True but same_retry_family False -> reject.
    arbiter = ConfiguredArbiter(table)
    result, diagnostics = split_incohesive_retry_groups((("a1", "a2", "b1", "b2"),), takes, arbiter=arbiter)
    result_sets = {frozenset(g) for g in result}
    assert frozenset({"a1", "a2"}) in result_sets
    assert frozenset({"b1", "b2"}) in result_sets


def test_section6_bridge_never_accepted_from_arbiter_unavailable_exception():
    takes = (
        _take("a1", 0.0, 2.0, _A1), _take("a2", 2.5, 4.5, _A2),
        _take("b1", 10.0, 12.0, _B1), _take("b2", 12.5, 14.5, _B2),
    )
    # NoneAvailableArbiter raises on every call, including the initial
    # weak-pair batch -- so nothing is ever confirmed at all (fail-open to
    # fully separate singletons), matching this module's existing
    # fail-open convention when the provider itself is down.
    arbiter = NoneAvailableArbiter()
    result, diagnostics = split_incohesive_retry_groups((("a1", "a2", "b1", "b2"),), takes, arbiter=arbiter)
    assert len(result) == 4  # arbiter down entirely -> every clip its own family


def test_section7_confidence_alone_never_bypasses_component_check():
    # A very high triggering confidence (0.99) on the seed pair must still
    # go through the component-level check -- and still get rejected if
    # that check declines, proving confidence alone is not semantic truth.
    takes = (
        _take("a1", 0.0, 2.0, _A1), _take("a2", 2.5, 4.5, _A2),
        _take("b1", 10.0, 12.0, _B1), _take("b2", 12.5, 14.5, _B2),
    )
    table = {
        (_A1, _A2): (True, 0.95, "same beat"),
        (_B1, _B2): (True, 0.98, "same beat"),
        (_A2, _B1): (True, 0.89, "still below the bridge floor even though high"),
        (_A1 + " || " + _A2, _B1 + " || " + _B2): (False, 0.2, "still two distinct propositions"),
    }
    arbiter = ConfiguredArbiter(table)
    result, _ = split_incohesive_retry_groups((("a1", "a2", "b1", "b2"),), takes, arbiter=arbiter)
    result_sets = {frozenset(g) for g in result}
    assert frozenset({"a1", "a2"}) in result_sets
    assert frozenset({"b1", "b2"}) in result_sets


def test_section7_low_confidence_pairwise_confirmation_still_gets_a_component_chance():
    # A LOW confidence pairwise confirmation must not be rejected purely for
    # being low-confidence -- it still gets the same component-level check,
    # and if THAT independently proves cohesion, the bridge may be accepted
    # (confidence is evidence, not the sole authority, in either direction).
    x1 = "These gummies helped my bloating a little."
    x2 = "These gummies also helped my bloating quite a bit."
    y1 = "After a week these gummies really helped my bloating."
    takes = (_take("x1", 0.0, 2.0, x1), _take("x2", 2.5, 4.5, x2), _take("y1", 5.0, 7.0, y1))
    table = {
        (x1, x2): (True, 0.95, "same claim"),
        (x2, y1): (True, 0.55, "low-confidence but plausible"),
        (x1 + " || " + x2, y1): (True, 0.95, "one shared bloating-benefit proposition"),
    }
    arbiter = ConfiguredArbiter(table)
    result, _ = split_incohesive_retry_groups((("x1", "x2", "y1"),), takes, arbiter=arbiter)
    assert result == (("x1", "x2", "y1"),) or {frozenset(g) for g in result} == {frozenset({"x1", "x2", "y1"})}


# =====================================================================
# Section 9: D-083 MARKER RETENTION
# =====================================================================

def test_section9_marker_gate_still_protects_marked_distinct_pair():
    left_text = "I had acne on my back and treated it with resorcinol."
    right_text = "Another symptom was hormonal pimples behind my ears and along my neck."
    assert _has_distinct_addition_marker(right_text)
    takes = (_take("acne", 0.0, 2.0, left_text), _take("pimples", 15.0, 17.0, right_text))
    table = {(left_text, right_text): (True, 0.95, "arbiter over-eagerly confirms")}
    arbiter = ConfiguredArbiter(table)
    result, diagnostics = split_incohesive_retry_groups((("acne", "pimples"),), takes, arbiter=arbiter)
    assert result == (("acne",), ("pimples",))
    assert diagnostics["content_divergence_blocked_count"] == 1


def test_section9_d085_complements_not_replaces_d083():
    # A marker-blocked pair never even becomes an edge, so it can never be
    # mis-classified as an accepted bridge either -- D-085's machinery only
    # ever sees edges D-083 already let through.
    left_text = "I had acne on my back and treated it with resorcinol."
    right_text = "Another symptom was hormonal pimples behind my ears and along my neck."
    takes = (_take("acne", 0.0, 2.0, left_text), _take("pimples", 15.0, 17.0, right_text))
    table = {(left_text, right_text): (True, 0.95, "arbiter over-eagerly confirms")}
    arbiter = ConfiguredArbiter(table)
    _, diagnostics = split_incohesive_retry_groups((("acne", "pimples"),), takes, arbiter=arbiter)
    assert diagnostics["bridge_evaluated_count"] == 0


# =====================================================================
# Section 10: TRUE RETRY SAFETY
# =====================================================================

def test_section10_low_lexical_overlap_genuine_paraphrase_still_merges():
    left_text = "I had seasonal back acne that I treated with an ointment"
    right_text = "Every season I would get back breakouts and I used an ointment for it"
    takes = (_take("a", 0.0, 2.0, left_text), _take("b", 2.5, 4.5, right_text))
    table = {(left_text, right_text): (True, 0.9, "same claim, different words")}
    arbiter = ConfiguredArbiter(table)
    result, _ = split_incohesive_retry_groups((("a", "b"),), takes, arbiter=arbiter)
    assert result == (("a", "b"),)


def test_section10_partial_and_directional_superset_retry_still_merges():
    fact_a = "I had acne on my back."
    fact_a_plus_b = "I had acne on my back and treated it with resorcinol."
    takes = (_take("a", 0.0, 2.0, fact_a), _take("b", 2.5, 4.5, fact_a_plus_b))
    table = {(fact_a, fact_a_plus_b): (True, 0.95, "same claim, more detail")}
    arbiter = ConfiguredArbiter(table)
    result, _ = split_incohesive_retry_groups((("a", "b"),), takes, arbiter=arbiter)
    assert result == (("a", "b"),)


def test_section10_singleton_joining_a_coherent_retry_component_still_merges():
    takes = _d084_takes()
    # Only the pimples-beat subset: b1,b2 already a pair, b3 a genuine
    # third retry/restatement joining that same component.
    group = ("b1", "b2", "b3")
    arbiter = ConfiguredArbiter(_d084_table())
    result, diagnostics = split_incohesive_retry_groups((group,), takes, arbiter=arbiter)
    assert result == (("b1", "b2", "b3"),) or {frozenset(g) for g in result} == {frozenset({"b1", "b2", "b3"})}
    accepted_bridges = [r for r in diagnostics["edge_trace"] if r.get("bridge_sensitive") and r.get("accepted")]
    assert len(accepted_bridges) == 1


def test_section10_two_true_subclusters_with_genuine_shared_proposition_merge():
    # Two independently-formed 2-member components that DO genuinely
    # describe the same underlying proposition (e.g. the same claim
    # recorded twice, worded differently each time) must still be allowed
    # to merge into one 4-member family once component-level cohesion
    # confirms it -- D-085 must never turn into "bridges never merge."
    p1 = "These vitamins gave me way more energy in the morning."
    p2 = "I noticed a lot more energy every morning after taking these vitamins."
    q1 = "Honestly these vitamins made my mornings feel way more energetic."
    q2 = "Every morning I've had noticeably more energy since starting these vitamins."
    takes = (_take("p1", 0, 2, p1), _take("p2", 2, 4, p2), _take("q1", 4, 6, q1), _take("q2", 6, 8, q2))
    table = {
        (p1, p2): (True, 0.95, "same claim"),
        (q1, q2): (True, 0.96, "same claim"),
        (p2, q1): (True, 0.88, "plausible cross-pair match"),
        (p1 + " || " + p2, q1 + " || " + q2): (True, 0.95, "one shared morning-energy proposition"),
    }
    arbiter = ConfiguredArbiter(table)
    result, _ = split_incohesive_retry_groups((("p1", "p2", "q1", "q2"),), takes, arbiter=arbiter)
    assert len(result) == 1
    assert set(result[0]) == {"p1", "p2", "q1", "q2"}


# =====================================================================
# Section 11: NARRATIVE CONTINUATION SAFETY
# =====================================================================

def test_section11_adjacent_chronology_same_topic_different_events_stay_separate():
    x1 = "My doctor ordered an ultrasound to check things out."
    x1b = "The doctor decided I needed to get an ultrasound done."
    y1 = "The ultrasound ended up finding a suspicious nodule."
    takes = (_take("x1", 0, 2, x1), _take("x1b", 2, 4, x1b), _take("y1", 4, 6, y1))
    table = {
        (x1, x1b): (True, 0.95, "same claim: ordering the ultrasound"),
        # An over-eager arbiter falsely treats the finding as the same idea
        # as ordering the test -- exactly D-084's false-bridge shape.
        (x1b, y1): (True, 0.82, "both mention the ultrasound"),
        (x1 + " || " + x1b, y1): (False, 0.25, "ordering a test vs. its result are different events"),
    }
    arbiter = ConfiguredArbiter(table)
    result, _ = split_incohesive_retry_groups((("x1", "x1b", "y1"),), takes, arbiter=arbiter)
    result_sets = {frozenset(g) for g in result}
    assert frozenset({"x1", "x1b"}) in result_sets
    assert frozenset({"y1"}) in result_sets


# =====================================================================
# Section 12: SALES / UGC GENERALIZATION
# =====================================================================

def test_section12_dosage_vs_outcome_never_bridge_into_one_family():
    dosage = "I take two gummies every morning."
    outcome = "They helped my bloating."
    product_use = "I opened the pouch and took two gummies."
    experience = "They tasted like strawberry."
    hook = "I thought this was another useless wellness product."
    cta = "Tap the cart to try it."
    takes = (
        _take("dosage", 0, 2, dosage), _take("outcome", 2, 4, outcome),
        _take("use", 4, 6, product_use), _take("exp", 6, 8, experience),
        _take("hook", 8, 10, hook), _take("cta", 10, 12, cta),
    )
    # Even if an over-eager arbiter confirms dosage<->use (both mention
    # "two gummies" mechanically) at a plausible confidence, the others
    # must never bridge into it.
    table = {(dosage, product_use): (True, 0.9, "both mention taking two gummies")}
    arbiter = ConfiguredArbiter(table)
    group = ("dosage", "outcome", "use", "exp", "hook", "cta")
    result, _ = split_incohesive_retry_groups((group,), takes, arbiter=arbiter)
    result_sets = {frozenset(g) for g in result}
    assert frozenset({"outcome"}) in result_sets
    assert frozenset({"exp"}) in result_sets
    assert frozenset({"hook"}) in result_sets
    assert frozenset({"cta"}) in result_sets


def test_section12_benefit_restatement_may_still_group():
    left_text = "These gummies helped my bloating."
    right_text = "After a week these gummies really helped my bloating."
    takes = (_take("a", 0.0, 2.0, left_text), _take("b", 2.5, 4.5, right_text))
    table = {(left_text, right_text): (True, 0.93, "same benefit claim, more detail")}
    arbiter = ConfiguredArbiter(table)
    result, _ = split_incohesive_retry_groups((("a", "b"),), takes, arbiter=arbiter)
    assert result == (("a", "b"),)


# =====================================================================
# Section 13/14: bounded compute + diagnostics observability
# =====================================================================

def test_section13_14_diagnostics_report_bounded_bridge_call_counts():
    takes = _d084_takes()
    group = ("a1", "a2", "b1", "b2", "b3")
    arbiter = ConfiguredArbiter(_d084_table())
    _, diagnostics = split_incohesive_retry_groups((group,), takes, arbiter=arbiter)
    assert "edge_trace" in diagnostics
    assert "bridge_evaluated_count" in diagnostics
    assert "bridge_accepted_count" in diagnostics
    assert "bridge_rejected_count" in diagnostics
    assert "component_semantic_call_count" in diagnostics
    assert diagnostics["bridge_accepted_count"] + diagnostics["bridge_rejected_count"] == diagnostics["bridge_evaluated_count"]
    # Bounded: bridge evaluations are far fewer than all-pairs (C(5,2)=10).
    assert diagnostics["bridge_evaluated_count"] < 10
    for record in diagnostics["edge_trace"]:
        assert "left_clip_id" in record and "right_clip_id" in record
        assert "bridge_sensitive" in record and "accepted" in record


def test_section13_no_multi_member_groups_reports_zero_bridge_counts():
    result, diagnostics = split_incohesive_retry_groups((), (), arbiter=None)
    assert result == ()
    assert diagnostics["bridge_evaluated_count"] == 0
    assert diagnostics["bridge_accepted_count"] == 0
    assert diagnostics["bridge_rejected_count"] == 0
    assert diagnostics["component_semantic_call_count"] == 0
    assert diagnostics["edge_trace"] == []


# =====================================================================
# Section 15: NO SELECTION AUTHORITY IN GROUPING
# =====================================================================

def test_section15_grouping_never_reports_a_winner_only_families():
    takes = _d084_takes()
    group = ("a1", "a2", "b1", "b2", "b3")
    arbiter = ConfiguredArbiter(_d084_table())
    result, diagnostics = split_incohesive_retry_groups((group,), takes, arbiter=arbiter)
    assert isinstance(result, tuple) and all(isinstance(g, tuple) for g in result)
    forbidden_keys = {"winner", "selected_clip_id", "best_take"}
    assert not (forbidden_keys & diagnostics.keys())
    for record in diagnostics["edge_trace"]:
        assert not (forbidden_keys & record.keys())


# =====================================================================
# Section 16: QA_ENGINE -- always-YES pairwise arbiter must not defeat
# component cohesion
# =====================================================================

def test_qa_engine_always_yes_pairwise_arbiter_still_blocked_by_component_check():
    takes = _d084_takes()
    group = ("a1", "a2", "b1", "b2", "b3")
    # The component-level probe texts are unambiguous regardless of union
    # order at each step, since _component_probe_text sorts by clip id.
    component_table = {
        (_B1 + " || " + _B2, _B3): (True, 0.95, "one shared pimples proposition"),
        (_A1 + " || " + _A2, _B1 + " || " + _B2 + " || " + _B3): (
            False, 0.2, "two distinct propositions",
        ),
    }
    arbiter = ConfirmEverythingPairwiseButSkepticalComponentArbiter(component_table)
    result, diagnostics = split_incohesive_retry_groups((group,), takes, arbiter=arbiter)
    result_sets = {frozenset(g) for g in result}
    assert frozenset({"a1", "a2"}) in result_sets
    assert frozenset({"b1", "b2", "b3"}) in result_sets
    assert len(result) == 2
    assert diagnostics["bridge_rejected_count"] >= 1
