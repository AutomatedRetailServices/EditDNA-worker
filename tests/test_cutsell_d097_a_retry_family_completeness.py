"""D-097.A -- RETRY-FAMILY COMPLETENESS: same-opening restart evidence.

Run 34008386434 (D-096 collision C-1): a failed take, its abandoned restart
and the clean retry all began with the same words within seconds; the group
was formed upstream, then `split_incohesive_retry_groups` isolated the clean
retry because the semantic arbiter declined the pair (a self-correction
rewrote the middle of the sentence, so `retry_similarity` scored 0.0 too).
"Ungrouped = keep" then played the failed take AND its retry.

Fix under test (grouping authority only, no winner decision here):
- `take_grouping.same_opening_restart`: deterministic recording-process
  evidence (same opening, adjacent, shared content beyond the opening);
- `group_takes` clusters on it like it already does for exact prefixes;
- the cohesion pass adds it as a deterministic edge (D-083 gate retained);
- a singleton attaching to a restart-cohesive component skips the D-085
  probe (deterministic restart edge, or >= 0.90 confirmation against that
  component), with the cross-component contradiction safety net;
- component-to-component bridges and semantic attaches to semantically
  formed components keep the D-085 probe unchanged;
- arbiter rejections are traced (`arbiter_rejected_pairs`).

Fixtures are generic (no Video00 clip ids); the texts model the SHAPE of a
self-corrected restart, which is a general recording-process pattern.
"""
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.pipeline import _semantic_best_take
from cutsell_worker.semantic_idea_equivalence import IdeaEquivalenceDecision, IdeaEquivalenceResult
from cutsell_worker.take_grouping import group_takes, retry_similarity, same_opening_restart
from cutsell_worker.take_grouping_provider import (
    _RetryEdge,
    _bridge_aware_components,
    reconcile_semantic_idea_equivalence,
    split_incohesive_retry_groups,
)
from cutsell_worker.take_judge import rank_takes


def _take(clip_id, start, end, text, source="src"):
    return CandidateTake(clip_id, source, 0, start, end, text)


# One sentence, three attempts: complete-but-failed first delivery, an
# abandoned restart, and the corrected clean retry (a detail changes:
# "spoke with" -> "switched to"; the request is rephrased).
FAILED = "When my contract ended I spoke with my doctor and asked her for every test she could possibly imagine or recommend."
ABANDONED = "When my contract ended I asked my doctor"
CLEAN = "When my contract ended I switched to a new doctor and asked her to run one test for everything she could imagine and recommend. Then she sent me for scans."
# A narrative continuation sharing only a discourse-connector opening.
CONNECTOR_A = "And after that I went to the clinic and they told me everything was fine"
CONNECTOR_B = "And after that I went on vacation to the beach with my family"


class RejectingArbiter:
    """Answers NOT-same-idea for every pair (the live shape: a semantic judge
    that treats a self-corrected detail as a different idea)."""

    def __init__(self):
        self.calls = 0

    def check(self, request):
        self.calls += 1
        return IdeaEquivalenceResult(
            decisions=tuple(
                IdeaEquivalenceDecision(pair_index=i, same_idea=False, confidence=0.9, reason="detail differs")
                for i, _ in enumerate(request.pairs)
            ),
            provider="fake", model="fake", requested=True, available=True,
            estimated_input_tokens=10, estimated_output_tokens=5,
        )


class TableArbiter:
    def __init__(self, table):
        self.table = table

    def check(self, request):
        decisions = []
        for i, pair in enumerate(request.pairs):
            entry = self.table.get((pair.left_text, pair.right_text)) or self.table.get((pair.right_text, pair.left_text))
            same, conf, reason = entry or (False, 0.0, "declined")
            decisions.append(IdeaEquivalenceDecision(pair_index=i, same_idea=same, confidence=conf, reason=reason))
        return IdeaEquivalenceResult(
            decisions=tuple(decisions), provider="fake", model="fake", requested=True, available=True,
            estimated_input_tokens=10, estimated_output_tokens=5,
        )


# --- the deterministic evidence itself -----------------------------------

def test_lexical_similarity_alone_cannot_see_the_self_corrected_restart():
    # Below `group_takes`'s 0.72 clustering threshold (the live pair scored
    # 0.0 on the 0.60 containment floor): lexical similarity alone does not
    # cluster a self-corrected restart.
    assert retry_similarity(FAILED, CLEAN) < 0.72


def test_same_opening_restart_recognises_failed_vs_clean_retry():
    failed = _take("f", 0.0, 6.0, FAILED)
    clean = _take("c", 10.0, 18.0, CLEAN)
    assert same_opening_restart(failed, clean) == "same_opening_restart"


def test_same_opening_abandoned_start_recognises_the_short_unfinished_restart():
    failed = _take("f", 0.0, 6.0, FAILED)
    abandoned = _take("a", 6.6, 8.5, ABANDONED)
    clean = _take("c", 10.0, 18.0, CLEAN)
    assert same_opening_restart(failed, abandoned) in {"same_opening_restart", "same_opening_abandoned_start"}
    assert same_opening_restart(abandoned, clean) in {"same_opening_restart", "same_opening_abandoned_start"}
    # A genuinely unfinished start that shares only ONE content word beyond
    # the opening is still recognised through the abandoned-start shape.
    barely = _take("b", 6.6, 7.9, "When my contract ended I asked her")
    assert same_opening_restart(barely, clean) is not None
    # The abandoned-start shape proper: too little content for the restart
    # rule (one shared content word), but a plainly unfinished half-length
    # start of the same sentence.
    stub = _take("s", 6.6, 7.9, "When my contract ended I told my")
    assert same_opening_restart(stub, _take("c2", 8.5, 16.0, "When my contract ended I told my doctor that I wanted every test she could think of.")) == "same_opening_abandoned_start"


def test_a_shared_discourse_connector_is_not_restart_evidence():
    assert same_opening_restart(_take("x", 0.0, 3.0, CONNECTOR_A), _take("y", 4.0, 7.0, CONNECTOR_B)) is None


def test_restart_evidence_is_bounded_by_adjacency_and_source():
    failed = _take("f", 0.0, 6.0, FAILED)
    assert same_opening_restart(failed, _take("c", 40.0, 48.0, CLEAN)) is None  # 34 s later: not a restart
    assert same_opening_restart(failed, _take("c", 10.0, 18.0, CLEAN, source="other")) is None


def test_restart_evidence_needs_content_beyond_the_opening():
    left = _take("l", 0.0, 2.0, "When my contract ended I")
    right = _take("r", 3.0, 8.0, CLEAN)
    assert same_opening_restart(left, right) is None  # too short to carry content


def test_different_items_with_the_same_list_opening_never_qualify():
    first = _take("a", 0.0, 4.0, "Another symptom was that I would get pimples like an allergy on my neck")
    second = _take("b", 5.0, 9.0, "Another symptom was that my hair would fall out whenever I washed it")
    assert same_opening_restart(first, second) is None


# --- clustering -----------------------------------------------------------

def test_group_takes_clusters_the_three_attempts_and_leaves_continuations_apart():
    takes = (
        _take("f", 0.0, 6.0, FAILED), _take("a", 6.6, 8.5, ABANDONED), _take("c", 10.0, 18.0, CLEAN),
        _take("x", 30.0, 33.0, CONNECTOR_A), _take("y", 34.0, 37.0, CONNECTOR_B),
    )
    clusters = [tuple(m.clip_id for m in members) for members in group_takes(takes).values()]
    assert ("f", "a", "c") in clusters
    assert ("x",) in clusters and ("y",) in clusters


# --- cohesion pass ----------------------------------------------------------

def _family():
    return (_take("f", 0.0, 6.0, FAILED), _take("a", 6.6, 8.5, ABANDONED), _take("c", 10.0, 18.0, CLEAN))


def test_cohesion_pass_keeps_the_family_whole_when_the_arbiter_declines_every_pair():
    arbiter = RejectingArbiter()
    groups, diag = split_incohesive_retry_groups((("f", "a", "c"),), _family(), arbiter)
    assert groups == (("f", "a", "c"),)
    assert diag["groups_split"] == 0
    reasons = {row.get("reason") or row.get("triggering_reason") for row in diag["edge_trace"]}
    assert "same_opening_restart" in reasons
    assert any(
        row.get("accepted") and row.get("accepted_by") == "deterministic_restart_evidence"
        for row in diag["edge_trace"] if row.get("bridge_sensitive")
    )
    assert diag["arbiter_rejected_pair_count"] == 0  # every pair had deterministic evidence; nothing was asked


def test_cohesion_pass_traces_arbiter_rejections_for_pairs_without_deterministic_evidence():
    takes = (
        _take("p", 0.0, 4.0, "The first thing the doctor found was a nodule on the left side of my thyroid"),
        _take("q", 5.0, 9.0, "Later the biopsy confirmed that it was papillary thyroid cancer"),
    )
    groups, diag = split_incohesive_retry_groups((("p", "q"),), takes, RejectingArbiter())
    assert groups == (("p",), ("q",))
    assert diag["arbiter_rejected_pair_count"] == 1
    assert diag["arbiter_rejected_pairs"][0]["reason"] == "detail differs"


def test_d083_marker_gate_still_blocks_a_restart_shaped_edge(monkeypatch):
    from cutsell_worker import take_grouping_provider as module
    monkeypatch.setattr(module, "_within_group_arbiter_confirmation_diverges", lambda take_map, l, r: True)
    groups, diag = split_incohesive_retry_groups((("f", "c"),), _family()[::2], RejectingArbiter())
    assert groups == (("f",), ("c",))  # no deterministic edge once the gate blocks it


def test_restart_singleton_bridge_is_rejected_on_cross_component_contradiction():
    from cutsell_worker.semantic_idea_equivalence import SemanticEquivalenceGatePolicy
    negated = "When my contract ended I spoke with my doctor and did not ask her for any test she could imagine or recommend."
    take_map = {"f": _take("f", 0.0, 6.0, FAILED), "c": _take("c", 7.0, 9.5, CLEAN), "n": _take("n", 10.0, 14.0, negated)}
    assert same_opening_restart(take_map["f"], take_map["n"]) == "same_opening_restart"
    edges = [
        _RetryEdge("f", "c", "deterministic", 1.0, "same_opening_restart"),
        _RetryEdge("f", "n", "deterministic", 1.0, "same_opening_restart"),
    ]
    trace = []
    components = _bridge_aware_components(
        ("f", "c", "n"), edges, protected_ids=frozenset(), take_map=take_map,
        arbiter=RejectingArbiter(), policy=SemanticEquivalenceGatePolicy(), edge_trace=trace,
    )
    assert {frozenset(c) for c in components} == {frozenset({"f", "c"}), frozenset({"n"})}
    rejected = [row for row in trace if row.get("reason_rejected")]
    assert rejected and rejected[0]["reason_rejected"] == "cross_component_contradiction"
    assert rejected[0]["accepted_by"] == "deterministic_restart_evidence"


def test_semantic_attach_to_a_restart_cohesive_component_skips_the_probe_at_high_confidence():
    restated = "Another symptom I had was that when my contract ended I asked a doctor for every possible test."
    takes = (*_family()[::2], _take("s", 20.0, 24.0, restated))
    arbiter = TableArbiter({(CLEAN, restated): (True, 0.95, "same request")})
    groups, diag = split_incohesive_retry_groups((("f", "c", "s"),), takes, arbiter)
    assert groups == (("f", "c", "s"),)
    assert any(
        row.get("accepted_by") == "semantic_confirmation_against_restart_cohesive_component"
        for row in diag["edge_trace"]
    )


def test_semantic_attach_below_the_bridge_floor_still_needs_the_probe():
    restated = "Another symptom I had was that when my contract ended I asked a doctor for every possible test."
    takes = (*_family()[::2], _take("s", 20.0, 24.0, restated))
    arbiter = TableArbiter({(CLEAN, restated): (True, 0.85, "same request")})  # < 0.90: D-085 probe applies, declined
    groups, diag = split_incohesive_retry_groups((("f", "c", "s"),), takes, arbiter)
    assert ("s",) in groups


def test_component_to_component_bridges_keep_the_d085_probe():
    second_family_a = "Later that week I had a scan of my thyroid and they found a suspicious nodule of three centimetres."
    second_family_b = "Later that week I had a scan of my thyroid and they saw a suspicious nodule they wanted to biopsy."
    takes = (*_family()[::2], _take("g", 20.0, 25.0, second_family_a), _take("h", 26.0, 31.0, second_family_b))
    # Pairwise the arbiter would bridge the two families at 0.95; the
    # component-level probe (" || " texts) is declined -> stays split.
    arbiter = TableArbiter({(CLEAN, second_family_a): (True, 0.95, "coarse topical")})
    groups, diag = split_incohesive_retry_groups((("f", "c", "g", "h"),), takes, arbiter)
    assert {frozenset(g) for g in groups} == {frozenset({"f", "c"}), frozenset({"g", "h"})}
    assert any(row.get("reason_rejected") for row in diag["edge_trace"])


def test_semantic_attach_to_a_semantically_formed_component_is_unchanged_from_d085():
    # Component formed by semantic evidence only (no restart edge) still
    # requires the probe for a further singleton -- D-085 byte-for-byte.
    edges = [
        _RetryEdge("m", "r", "semantic", 0.95, "same idea"),
        _RetryEdge("m", "s", "semantic", 0.95, "same idea"),
    ]
    take_map = {
        "m": _take("m", 0.0, 4.0, "The pimples showed up behind my ear and along my neck and looked like an allergy."),
        "r": _take("r", 5.0, 9.0, "Those pimples behind my ear looked exactly like an allergy but were hormonal."),
        "s": _take("s", 10.0, 13.0, "I also used to break out in a rash that looked like an allergy."),
    }
    from cutsell_worker.semantic_idea_equivalence import SemanticEquivalenceGatePolicy
    trace = []
    components = _bridge_aware_components(
        ("m", "r", "s"), edges, protected_ids=frozenset(), take_map=take_map,
        arbiter=RejectingArbiter(), policy=SemanticEquivalenceGatePolicy(), edge_trace=trace,
    )
    assert {frozenset(c) for c in components} == {frozenset({"m", "r"}), frozenset({"s"})}
    assert any(row.get("reason_rejected") == "component_cohesion_declined" for row in trace)


# --- reconcile observability --------------------------------------------------

def test_reconcile_traces_arbiter_rejections():
    takes = (
        _take("p", 0.0, 4.0, "The first thing the doctor found was a nodule on the left side of my thyroid"),
        _take("q", 5.0, 9.0, "Later the biopsy confirmed that it was papillary thyroid cancer"),
    )
    groups, diag = reconcile_semantic_idea_equivalence((("p",), ("q",)), takes, RejectingArbiter())
    assert groups == (("p",), ("q",))
    assert diag["arbiter_rejected_pair_count"] == 1


# --- the editorial consequence: the clean retry survives, the failed take does not --

def test_failed_attempt_plus_clean_retry_resolve_to_the_clean_retry_once_grouped():
    family = _family()
    groups, _ = split_incohesive_retry_groups((("f", "a", "c"),), family, RejectingArbiter())
    assert groups == (("f", "a", "c"),)
    ranked = rank_takes(family)
    labels = {"f": ("failed", 0.90), "a": ("failed", 0.95), "c": ("winner", 0.95)}
    selected, preferred, reason = _semantic_best_take(family, labels, ranked[0].clip_id, ranked)
    assert selected == "c" and reason == "single_semantic_winner"
