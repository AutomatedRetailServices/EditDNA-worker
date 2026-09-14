"""D-108 -- PIMPLES FAMILY GRANULARITY / RETRY-FAMILY TRANSITIVITY.

Root defect (docs/CUTSELL_DECISIONS.md D-108, forensic continuation of
D-104/D-107's pimples family `tg_edb72c9305a16337b5`): D-085's bridge-aware
cohesion pass (`_bridge_aware_components`) only ever asks "does the MERGED
component still share one proposition?" (via `_accept_restart_singleton_
bridge` / `_accept_complete_pairwise_bridge` / `_evaluate_bridge_cohesion`).
None of those three acceptance paths ever asks "did THIS SAME run already
determine two of the members about to be merged are NOT the same idea?" --
so an explicit, already-computed `content_divergence_blocked` (or a strong
`arbiter_rejected_pairs`) verdict between a complementary beat (A) and a
retry-competitor (B) never stops A from being pulled into B's retry family
through a THIRD member (C) that both a weak pairwise edge AND a component-
level probe judge "close enough" on coarse topical similarity alone.

Fixtures are generic (no Video00 clip ids, text, or timestamps) -- the
texts model the SHAPE of the real defect: A is a short, complete,
audience-facing beat; B is a longer realization carrying an explicit
distinct-addition discourse marker and sharing substantial specific content
with C; C is a high-confidence retry/equivalent of B. A and B are already
found explicitly non-equivalent by this run's own marker-gated divergence
check; A and C are only weakly, coarsely confirmed same-idea (neither
carries the marker, so that confirmation is never itself divergence-
checked) -- the general shape any future coarse-similarity confirmation
could reproduce, not a hardcoded transcript.
"""
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.semantic_idea_equivalence import IdeaEquivalenceDecision, IdeaEquivalenceResult
from cutsell_worker.take_grouping_provider import (
    SemanticEquivalenceGatePolicy,
    _RetryEdge,
    _bridge_aware_components,
    _component_probe_text,
    split_incohesive_retry_groups,
)

# A: complementary beat, unique/distinct addition ("rash, allergy").
A_TEXT = "I also noticed some spots, it felt like a rash, an allergy."
# B: another realization, explicit distinct-addition marker, shares the
# location/duration detail with C.
B_TEXT = (
    "Another symptom was that I had marks like an allergy on this part "
    "right here, behind my ear and down my neck. It happened in cycles."
)
# C: high-confidence retry/equivalent of B (same location detail, same
# eventual "hormonal" correction shape as a real retry would carry).
C_TEXT = (
    "I also noticed some marks on this part right here, behind my ear and "
    "down my neck, which I always assumed was an allergy, but it turned "
    "out to be marks from a hormonal issue."
)


def _take(clip_id, start, end, text):
    return CandidateTake(clip_id, "src", 0, start, end, text)


class TableArbiter:
    """Answers pairwise AND component-probe requests from a fixed table --
    the same style `test_cutsell_d097_a_retry_family_completeness.py` uses,
    extended to also answer `_evaluate_bridge_cohesion`'s own component-
    level probe (built from `_component_probe_text`), so a test can prove
    the veto holds even when the SAME coarse-similarity mistake that
    confirmed the weak pairwise edge would ALSO fool the component check."""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def check(self, request):
        self.calls += 1
        decisions = []
        for i, pair in enumerate(request.pairs):
            entry = self.table.get((pair.left_text, pair.right_text)) or self.table.get(
                (pair.right_text, pair.left_text)
            )
            same, conf, reason = entry or (False, 0.0, "declined")
            decisions.append(IdeaEquivalenceDecision(pair_index=i, same_idea=same, confidence=conf, reason=reason))
        return IdeaEquivalenceResult(
            decisions=tuple(decisions), provider="fake", model="fake", requested=True, available=True,
            estimated_input_tokens=10, estimated_output_tokens=5,
        )


def _pimples_shape_takes():
    a = _take("a", 0.0, 5.0, A_TEXT)
    b = _take("b", 6.0, 12.0, B_TEXT)
    c = _take("c", 13.0, 20.0, C_TEXT)
    return a, b, c


def _pimples_shape_table(a, b, c):
    """Every pairwise verdict this run would produce, PLUS the component
    probe `_evaluate_bridge_cohesion` asks once A tries to bridge into the
    {b, c} component -- deliberately answered the SAME coarse "same topic"
    way a real LLM arbiter's own mistake would answer it, so the test
    proves the veto works even when the softer component check would not."""
    take_map = {t.clip_id: t for t in (a, b, c)}
    probe_left = _component_probe_text(take_map, (a.clip_id,))
    probe_right = _component_probe_text(take_map, (b.clip_id, c.clip_id))
    return {
        (A_TEXT, B_TEXT): (True, 0.9, "topically overlapping symptom mention"),
        (B_TEXT, C_TEXT): (True, 0.95, "same allergy-on-neck realization"),
        (A_TEXT, C_TEXT): (True, 0.80, "both mention skin spots appearing"),
        (probe_left, probe_right): (True, 0.95, "both describe the same skin symptom near the ear/neck"),
    }


# --- preconditions: the marker-gated divergence check behaves as the fix
# assumes (confirms the fixture reproduces the real evidentiary shape) ----

def test_precondition_a_vs_b_is_marker_gated_divergent():
    from cutsell_worker.take_grouping_provider import _has_distinct_addition_marker, _marked_side_diverges_in_content
    assert _has_distinct_addition_marker(B_TEXT) is True
    assert _has_distinct_addition_marker(A_TEXT) is False
    assert _marked_side_diverges_in_content(A_TEXT, B_TEXT) is True


def test_precondition_b_vs_c_is_not_divergent():
    from cutsell_worker.take_grouping_provider import _marked_side_diverges_in_content
    assert _marked_side_diverges_in_content(B_TEXT, C_TEXT) is False


# --- the structural fix itself --------------------------------------------

def test_pimples_shape_a_stays_independent_b_c_compete():
    """POSITIVE: split_incohesive_retry_groups on one pre-formed 3-member
    group must isolate A (explicitly non-equivalent to B) into its own
    singleton and leave B/C as the one retry-competing pair -- never a
    hardcoded "keep A+B, drop C"; the B-vs-C winner is left to BestTake."""
    a, b, c = _pimples_shape_takes()
    arbiter = TableArbiter(_pimples_shape_table(a, b, c))
    groups = ((a.clip_id, b.clip_id, c.clip_id),)
    result_groups, diagnostics = split_incohesive_retry_groups(groups, (a, b, c), arbiter)

    result_sets = {frozenset(g) for g in result_groups}
    assert frozenset({a.clip_id}) in result_sets
    assert frozenset({b.clip_id, c.clip_id}) in result_sets
    assert len(result_groups) == 2

    assert diagnostics["blocked_pair_veto_count"] == 1
    blocked_ids = {(row["left_clip_id"], row["right_clip_id"]) for row in diagnostics["content_divergence_blocked"]}
    assert (a.clip_id, b.clip_id) in blocked_ids or (b.clip_id, a.clip_id) in blocked_ids
    assert diagnostics["splits"] == [
        {"original_group_ids": [a.clip_id, b.clip_id, c.clip_id],
         "resulting_groups": [[a.clip_id], [b.clip_id, c.clip_id]]}
    ]


def test_bridge_aware_components_before_and_after_the_veto():
    """Direct proof at the mechanism level: WITHOUT the D-108 veto, the
    exact same edges plus a component-probe arbiter answer that repeats the
    same coarse "same topic" mistake the pairwise edge already made would
    merge all three into one family (the bug). WITH the veto (the fix),
    the bridge is rejected before the component probe is even asked."""
    a, b, c = _pimples_shape_takes()
    take_map = {t.clip_id: t for t in (a, b, c)}
    arbiter = TableArbiter(_pimples_shape_table(a, b, c))
    policy = SemanticEquivalenceGatePolicy()
    edges = [
        _RetryEdge(b.clip_id, c.clip_id, "semantic", 0.95, "same allergy-on-neck realization"),
        _RetryEdge(a.clip_id, c.clip_id, "semantic", 0.80, "both mention skin spots appearing"),
    ]

    before_trace = []
    before = _bridge_aware_components(
        (a.clip_id, b.clip_id, c.clip_id), edges, protected_ids=frozenset(), take_map=take_map,
        arbiter=arbiter, policy=policy, edge_trace=before_trace, blocked_pairs=frozenset(),
    )
    assert before == ((a.clip_id, b.clip_id, c.clip_id),)  # the bug: all three merge

    after_trace = []
    blocked_pairs = frozenset({frozenset({a.clip_id, b.clip_id})})
    after = _bridge_aware_components(
        (a.clip_id, b.clip_id, c.clip_id), edges, protected_ids=frozenset(), take_map=take_map,
        arbiter=arbiter, policy=policy, edge_trace=after_trace, blocked_pairs=blocked_pairs,
    )
    assert {frozenset(g) for g in after} == {frozenset({a.clip_id}), frozenset({b.clip_id, c.clip_id})}
    rejected = [r for r in after_trace if r.get("reason_rejected") == "cross_component_explicit_non_equivalence"]
    assert len(rejected) == 1
    # the component-level probe was never even asked once the veto fired
    assert rejected[0]["component_cohesion_evaluated"] is False


# --- negative controls (D-108 directive, items 1/2/7) ---------------------

def test_negative_control_1_valid_three_member_family_stays_merged():
    """A-B equivalent (no divergence) AND B-C equivalent -> a genuinely
    cohesive 3-member family must remain intact; the veto must never fire
    when nothing was ever blocked."""
    a = _take("a", 0.0, 5.0, "I had seasonal back acne that I treated with an ointment my dermatologist gave me.")
    b = _take("b", 6.0, 12.0, "Every season I would get back breakouts and I used an ointment my dermatologist recommended for it.")
    c = _take("c", 13.0, 20.0, "That seasonal back acne kept coming back so I kept using the ointment my dermatologist gave me.")
    take_map = {t.clip_id: t for t in (a, b, c)}
    # a/c score high enough on plain lexical containment to merge as a
    # deterministic pair on their own; b then bridges in and must clear
    # `_evaluate_bridge_cohesion`'s own component-level probe -- supply an
    # honest "yes, same recurring back-acne story" answer for it, exactly
    # what a genuinely cohesive 3-member family requires.
    probe_text = _component_probe_text(take_map, (a.clip_id, c.clip_id))
    table = {
        (a.text, b.text): (True, 0.92, "same recurring back-acne story, genuine paraphrase"),
        (b.text, c.text): (True, 0.93, "same recurring back-acne story, genuine paraphrase"),
        (a.text, c.text): (True, 0.90, "same recurring back-acne story, genuine paraphrase"),
        (b.text, probe_text): (True, 0.95, "all three describe the same recurring back-acne story"),
    }
    arbiter = TableArbiter(table)
    groups = ((a.clip_id, b.clip_id, c.clip_id),)
    result_groups, diagnostics = split_incohesive_retry_groups(groups, (a, b, c), arbiter)
    assert len(result_groups) == 1
    assert frozenset(result_groups[0]) == frozenset({a.clip_id, b.clip_id, c.clip_id})
    assert diagnostics["blocked_pair_veto_count"] == 0
    assert diagnostics["content_divergence_blocked"] == []


def test_negative_control_2_bare_rejection_never_vetoes_even_at_high_confidence():
    """A bare same_idea=False verdict -- weak (0.3) OR strong (0.95) -- must
    NEVER by itself veto an otherwise-legitimate bridge: it is the ROUTINE
    outcome for any topically-unrelated pair inside a larger group (D-108's
    own module comment), not evidence of genuine non-equivalence. Only a
    `content_divergence_blocked` entry (same_idea=True, then structurally
    overridden) counts. Regression target: `test_regression_full_five_
    member_conflated_group_resolves_to_three_families`
    (test_cutsell_d083_distinct_idea_grouping_safety.py) -- a fixed-
    confidence test arbiter's routine 0.9-confidence decline of an unrelated
    pair must never veto that suite's own legitimate 3-member family."""
    a, b, c = _pimples_shape_takes()
    for confidence in (0.3, 0.95):
        table = _pimples_shape_table(a, b, c)
        # Replace the marker-gated block with a bare arbiter rejection
        # instead (same_idea=False), simulating an ordinary "not confirmed"
        # verdict rather than the structurally-strong marker+divergence block.
        table[(A_TEXT, B_TEXT)] = (False, confidence, "declined")
        arbiter = TableArbiter(table)
        groups = ((a.clip_id, b.clip_id, c.clip_id),)
        result_groups, diagnostics = split_incohesive_retry_groups(groups, (a, b, c), arbiter)
        assert len(result_groups) == 1, f"confidence={confidence}"
        assert frozenset(result_groups[0]) == frozenset({a.clip_id, b.clip_id, c.clip_id})
        assert diagnostics["blocked_pair_veto_count"] == 0


def test_negative_control_7_unknown_relationship_alone_does_not_veto():
    """A-C's relationship being simply UNKNOWN (never merged, never
    rejected -- e.g. an arbiter that only answers B-C) still isolates A
    (no positive edge connects it to anyone), but `blocked_pair_veto_count`
    must be 0: the split here comes from an absence of evidence, never from
    the D-108 veto treating "unknown" as "blocked"."""
    a, b, c = _pimples_shape_takes()
    table = {(B_TEXT, C_TEXT): (True, 0.95, "same allergy-on-neck realization")}
    arbiter = TableArbiter(table)
    groups = ((a.clip_id, b.clip_id, c.clip_id),)
    result_groups, diagnostics = split_incohesive_retry_groups(groups, (a, b, c), arbiter)
    result_sets = {frozenset(g) for g in result_groups}
    assert frozenset({a.clip_id}) in result_sets
    assert frozenset({b.clip_id, c.clip_id}) in result_sets
    assert diagnostics["blocked_pair_veto_count"] == 0
    assert diagnostics["content_divergence_blocked"] == []


def test_negative_control_3_two_member_retry_pair_unaffected():
    """No third, divergent party at all: an ordinary two-member retry pair
    (no bridge ever occurs) merges exactly as before D-108 -- the veto has
    nothing to do with this shape, it must never fire here."""
    a = _take("a", 0.0, 6.0, "When my contract ended I spoke with my doctor and asked for a full checkup and every test available.")
    b = _take("b", 8.0, 15.0, "When my contract ended I switched to a new doctor and asked her to run every test she could think of.")
    table = {(a.text, b.text): (True, 0.9, "same request, self-corrected wording")}
    arbiter = TableArbiter(table)
    groups = ((a.clip_id, b.clip_id),)
    result_groups, diagnostics = split_incohesive_retry_groups(groups, (a, b), arbiter)
    assert len(result_groups) == 1
    assert frozenset(result_groups[0]) == frozenset({a.clip_id, b.clip_id})
    assert diagnostics["blocked_pair_veto_count"] == 0
