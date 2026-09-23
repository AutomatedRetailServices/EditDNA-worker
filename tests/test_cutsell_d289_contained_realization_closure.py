"""D-289 -- redundant realization closure BEFORE Selection Freeze, through
the existing IdeaClusterer / family-competition authorities.

Evidence (RAW #122 = Modal RAW run 35799404391 at head `f012beed`: the
four-way quality-ladder region map printed in its own job log; the frozen
baseline `benchmarks/video00_regression_qa.json` / `video00_selection_
lock.json`; and the recorded semantic verdicts for this same cluster in
`docs/CUTSELL_DECISIONS.md` D-056.5/D-058 -- all QA-ONLY material, never
read by production code):

* CASE 2 (conclusion vs. redundant percentage/heredity restatement): the
  complete family-context realization (295.52-313.50, family winner) was
  kept, and a LATER restatement of its percentage/heredity claim (327.78-
  334.24, "...solo un 5-10% de los") was kept too, ungrouped
  (`fam=None`; ladder: LEVEL_1 false_keep, "never grouped ... although
  their content overlaps", authority IdeaClusterer). Human Gold removes
  the restatement; the frozen baseline lock (23 segments) never contained
  it -- RAW #122's own "Verify frozen Selection lock" step failed.
  Recorded verdicts: the SemanticEquivalenceArbiter merged this exact
  pair ("Same personal cancer statistics and hereditary beliefs
  discussed.", 0.9 -- D-056.5), and the restatement's own claim vs. the
  winner is the D-058 canary paraphrase ("estoy convencida y la ciencia
  lo avala" ~ "está comprobado científicamente") that `claim_coverage`
  routes to the bounded claim-equivalence arbiter. Mechanism (D-288 audit
  item 5; D-094.F4's own comment on this family): reconcile merges the
  confirmed pair, then the cohesion pass's D-085 joined-text probe fails
  closed and splits the restatement back out. Closed here by
  `take_grouping_provider._accept_contained_restatement_singleton_bridge`.

* CASE 1 (pimples/allergy fragments): the short earlier fragments
  ("También me salían espinillas. / Era como un rush, / una alergia.")
  are kept beside the later, fuller "Otro síntoma..." winner. The SAME
  evidence shows this is NOT a defect: Human Gold keeps both (ladder
  regions 83/85/91 consensus_keep), the baseline QA REQUIRES the three
  fragments (`pimples_micro_1/2/3_present`, `pimples_micro_order`) and
  D-109 records their removal as a FALSE DELETE. The separating mechanism
  is D-048's own marker-gated divergence veto (winner carries "otro
  síntoma", shared content 4 < floor 6). The new path honors it (guard 2)
  -- the test below pins that the fragment stays, as both references do.

The production rule reads no Video00 phrase/id/timestamp; the real texts
and timings appear ONLY in this QA fixture, exactly as `benchmarks/
video00_regression_qa.json` already does.
"""
from __future__ import annotations

from cutsell_worker import take_grouping_provider as tgp
from cutsell_worker.claim_coverage_best_take import apply_claim_coverage_best_take
from cutsell_worker.contracts import CandidateTake, DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.deterministic_best_take_authority import apply_deterministic_best_take_authority
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation
from cutsell_worker.semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalenceResult,
    SemanticEquivalenceGatePolicy,
)
from cutsell_worker.take_grouping_provider import (
    _RetryEdge,
    _bridge_aware_components,
    reconcile_semantic_idea_equivalence,
    safe_group_takes,
    split_incohesive_retry_groups,
)
from cutsell_worker.take_judge import rank_takes


def _take(cid, start, end, text, *, complete=True, source="src"):
    return CandidateTake(cid, source, 0, start, end, text, complete_idea=complete)


class RecordedAnswersArbiter:
    """Pairwise same-idea answers from an explicit table (the recorded
    verdicts a replay is driven by). A component-level joined-text probe
    (" || ") DECLINES at 0.2 whenever it involves one of `unstable_probe_
    texts` -- the exact live probe shape D-094.2 recorded on run
    33983880111 and the D-288 audit traced RAW #122's restatement survival
    to; every other probe confirms, so the rest of the family keeps its
    real shape."""

    def __init__(self, table, *, unstable_probe_texts=()):
        self.table = {frozenset(k): v for k, v in table.items()}
        self.unstable = tuple(unstable_probe_texts)
        self.asked = []

    def check(self, request):
        out = []
        for i, pair in enumerate(request.pairs):
            self.asked.append((pair.left_text, pair.right_text))
            if " || " in pair.left_text or " || " in pair.right_text:
                joined = pair.left_text + " || " + pair.right_text
                if any(t in joined for t in self.unstable):
                    out.append(IdeaEquivalenceDecision(i, False, 0.2, "component probe declined"))
                else:
                    out.append(IdeaEquivalenceDecision(i, True, 0.95, "component probe confirmed"))
                continue
            same, conf, reason = self.table.get(frozenset((pair.left_text, pair.right_text)), (False, 0.0, "unconfigured"))
            out.append(IdeaEquivalenceDecision(i, same, conf, reason))
        return IdeaEquivalenceResult(tuple(out), "fake", "fake", True, True, 50, 10)


class RecordedClaimArbiter:
    """`ClaimEquivalenceArbiter.claim_covered` driven by a table of
    (claim_text, realization_text) -> covered -- the recorded D-058 canary
    verdict for this cluster; anything unlisted is NOT covered."""

    def __init__(self, table):
        self.table = dict(table)
        self.asked = []

    def claim_covered(self, claim_text, realization_text):
        self.asked.append((claim_text, realization_text))
        return (self.table.get((claim_text, realization_text), False), 0.9, "recorded")


def _real_chain(takes, arbiter, *, claim_arbiter=None):
    """The real pre-Freeze chain in pipeline order: lexical grouping ->
    reconcile (cross-group, arbiter-gated) -> cohesion pass fed the
    reconcile stage's own confirmations as prior evidence (exactly as
    `pipeline.py` does) -> ranking -> deterministic BestTake -> claim-
    coverage BestTake (with the bounded claim-equivalence arbiter, as
    `universal_clean_cut.py` wires it) -> final story coherence."""
    baseline = safe_group_takes(None, takes)
    merged, rec_diag = reconcile_semantic_idea_equivalence(baseline.groups, takes, arbiter)
    prior = {
        frozenset((r["left_clip_id"], r["right_clip_id"])): (float(r["confidence"]), str(r["reason"]))
        for r in (rec_diag.get("merges") or ()) if r.get("left_clip_id") and r.get("right_clip_id")
    }
    groups, coh_diag = split_incohesive_retry_groups(merged, takes, arbiter, prior_confirmations=prior)
    by_id = {t.clip_id: t for t in takes}
    judge_groups, selected_ids = [], []
    for index, ids in enumerate(groups):
        members = [by_id[c] for c in ids]
        ranked = rank_takes(members)
        if len(members) >= 2:
            judge_groups.append({"group_id": f"g{index}", "ranked": [
                {"clip_id": r.clip_id, "score": r.score, "reason": r.reason} for r in ranked]})
        selected_ids.extend(m.clip_id for m in members)
    selected = tuple(DraftClip(
        clip_id=t.clip_id, source_asset_id=t.source_asset_id, source_order=t.source_order,
        start=t.start, end=t.end, text=t.text, caption_text=t.text, selected=True,
    ) for t in (by_id[c] for c in selected_ids))
    # Exactly what `pipeline.py` stamps for the downstream authorities: the
    # reconcile stage's own merge records (D-061's paraphrase credit reads
    # them, never re-asks) and the cohesion pass's diagnostics.
    draft = DraftTimeline(schema_version=SCHEMA_VERSION, project_id="d289", strategy=EditStrategy.STORYTELLING,
                          selected=selected, alternates=(), discarded=(), diagnostics={
                              "take_judge_groups": judge_groups,
                              "semantic_idea_equivalence": rec_diag,
                              "distinct_idea_grouping_safety": coh_diag,
                          })
    draft = apply_deterministic_best_take_authority(draft, swap_enabled=False)
    draft = apply_claim_coverage_best_take(draft, claim_equivalence_arbiter=claim_arbiter)
    draft = apply_final_story_coherence_validation(
        draft, semantic_equivalence_arbiter=arbiter, claim_equivalence_arbiter=claim_arbiter,
    )
    return draft, groups, rec_diag, coh_diag


def _kept(draft):
    return {c.clip_id for c in draft.selected}


def _discarded(draft):
    return {c.clip_id for c in draft.discarded}


def _disable_new_path(monkeypatch):
    """The exact pre-D-289 dispatch: the new acceptance path never applies."""
    monkeypatch.setattr(tgp, "_accept_contained_restatement_singleton_bridge", lambda **kwargs: (False, None))


# =============================================================================
# CASE 2 -- RAW #122 conclusion cluster, real texts and timings (QA fixture)
# =============================================================================

WINNER = ("Esta es mi experiencia. Soy la única en mi familia que tiene este tipo de cáncer. Por eso no creo y está "
          "comprobado científicamente que los cánceres son hereditarios. Más bien solo un 5 -10 % son de hereditario. "
          "Mayormente son nuestras elecciones de vida así que cuídate.")
ASIDE = "Soy la primera en mi familia con este tipo de cáncer. Nadie en mi familia tiene un carcinoma papilar en la tiroides ni sufre de la tiroides."
RESTATEMENT = "Así que estoy convencida y la ciencia lo avala que solo un 5 -10 % de los"
TAIL = "cánceres son hereditarios."
PREFIX = "Soy la única en mi familia que tiene este tipo de cáncer."
CTA = "Por eso cuídate. Aliméntate bien. Hidrátate. Haz ejercicio."
HAIR = "Se me caía mucho el pelo cuando me lavaba el pelo perdía mucho pelo y pensaba que era por el estrés."

# The D-058 canary: the restatement's own claim clause vs. the winner.
RESTATEMENT_CLAIM = RESTATEMENT


def _case2_takes(*, with_prefix=False):
    takes = [
        _take("H", 226.74, 233.18, HAIR),
        _take("W", 295.52, 313.50, WINNER),
        _take("R", 327.78, 334.24, RESTATEMENT, complete=False),
        _take("C", 356.21, 361.61, CTA),
    ]
    if with_prefix:
        takes.insert(3, _take("P", 342.90, 346.52, PREFIX))
    return tuple(sorted(takes, key=lambda t: t.start))


def _case2_arbiters():
    pairwise = RecordedAnswersArbiter(
        {(WINNER, RESTATEMENT): (True, 0.9, "Same personal cancer statistics and hereditary beliefs discussed.")},
        unstable_probe_texts=(RESTATEMENT,),
    )
    claims = RecordedClaimArbiter({(RESTATEMENT_CLAIM, WINNER): True})
    return pairwise, claims


def test_repro_case2_before_the_fix_the_restatement_survives_ungrouped(monkeypatch):
    """The live RAW #122 shape, reproduced through the real chain with the
    new path switched off: reconcile merges W-R on the recorded
    confirmation, the cohesion pass's joined-text probe declines, R is
    split back out as its own singleton family and BOTH realizations of
    one editorial function are kept -- the winner and its restatement."""
    _disable_new_path(monkeypatch)
    pairwise, claims = _case2_arbiters()
    draft, groups, rec_diag, coh_diag = _real_chain(_case2_takes(with_prefix=True), pairwise, claim_arbiter=claims)

    assert any(frozenset((r["left_clip_id"], r["right_clip_id"])) == frozenset({"W", "R"}) for r in rec_diag["merges"])
    assert any(r.get("reason_rejected") == "component_cohesion_declined" for r in coh_diag["edge_trace"])
    assert ("R",) in groups
    assert {"W", "R"} <= _kept(draft)


def test_case2_after_the_fix_the_restatement_competes_and_loses_with_traceable_proof():
    """The live RAW #122 family shape (the winner's lexical group also holds
    its own strict prefix, so the restatement's confirmed edge is a BRIDGE
    into a multi-member component): the new acceptance path folds the
    restatement in with a traceable preservation proof, the family
    competes, the complete realization wins and the redundant formulation
    is discarded with its percentage proven preserved."""
    pairwise, claims = _case2_arbiters()
    draft, groups, rec_diag, coh_diag = _real_chain(_case2_takes(with_prefix=True), pairwise, claim_arbiter=claims)

    family = next(g for g in groups if "W" in g)
    assert "R" in family  # one retry family: the winner and its restatement compete
    bridge = next(r for r in coh_diag["edge_trace"] if r.get("accepted_by") == "contained_restatement_of_complete_realization")
    assert bridge["accepted"] is True and bridge["restated_clip_id"] == "R"
    assert bridge["component_cohesion_evaluated"] is False  # no synthetic joined-text probe spent
    proof = bridge["preservation_evidence"]
    assert proof["preserving_member_clip_id"] == "W"
    assert proof["digit_values_preserved"] == ["10", "5"]  # the percentage is preserved in the winner
    assert proof["claims_preserved"] == [{"claim_type": "MEASUREMENT_QUANTITY", "preserved_by": "numeric_restatement", "digit_values": ["10", "5"]}]

    assert _kept(draft) == {"H", "W", "C"}
    assert _discarded(draft) == {"P", "R"}  # the prefix AND the redundant formulation lose the competition
    coherence = draft.diagnostics["final_story_coherence_validation"]
    assert coherence["contradiction_findings"] == [] and coherence["missing_idea_coverage"] == []
    rows = {a["clip_id"]: a for a in coherence.get("lost_semantic_atoms", [])}
    assert not any(a.get("blocking") for a in rows.values())
    # The ledger credits the discard from the reconcile stage's own merge
    # record (D-061) -- no new call, and the percentage is not a lost atom.
    assert rows["R"]["content_loss_suppressed_by"] == "same_idea_semantic_equivalence"
    assert rows["R"]["missing_critical_atoms"] == []


def test_case2_pair_only_family_needs_no_bridge_and_resolves_the_same_way():
    """Without the prefix in the winner's lexical group the confirmed
    (winner, restatement) edge forms a plain two-member family: no bridge
    is evaluated (the new path is not needed and not consulted), the
    competition still hands the family to the complete realization."""
    pairwise, claims = _case2_arbiters()
    draft, groups, rec_diag, coh_diag = _real_chain(_case2_takes(), pairwise, claim_arbiter=claims)

    assert ("W", "R") in groups
    assert all(r.get("accepted_by") is None for r in coh_diag["edge_trace"])
    assert coh_diag["bridge_evaluated_count"] == 0
    assert _kept(draft) == {"H", "W", "C"}
    assert _discarded(draft) == {"R"}
    assert not any(a.get("blocking") for a in draft.diagnostics["final_story_coherence_validation"].get("lost_semantic_atoms", []))


def test_case2_without_the_recorded_claim_verdict_nothing_is_silently_lost():
    """"WHEN UNCERTAIN, KEEP": with no claim-equivalence answer available,
    the restatement's canary claim (D-058: "estoy convencida y la ciencia
    lo avala" vs "comprobado científicamente") stays 'not covered'
    lexically, so the family is not resolved silently -- the existing
    safety nets keep the loss visible (blocking) rather than fabricating
    a clean winner. Grouping the restatement into the family (this
    directive) never weakens that: the claim ledger still blocks."""
    pairwise, _ = _case2_arbiters()
    draft, groups, rec_diag, coh_diag = _real_chain(_case2_takes(with_prefix=True), pairwise, claim_arbiter=None)
    coherence = draft.diagnostics["final_story_coherence_validation"]
    # The family is NOT resolved to one clean winner: ambiguous coverage
    # keeps both the winner and its prefix, and the restatement's CRITICAL
    # percentage claim is recorded as a blocking loss against the winning
    # realization -- visible, owned by BestTakeResolver, never silent.
    assert {"W", "P"} <= _kept(draft)
    lost = [c for c in coherence.get("lost_critical_claims", []) if c.get("source_clip_id") == "R"]
    assert lost and all(c["blocking"] and c["claim_type"] == "MEASUREMENT_QUANTITY" for c in lost)


def test_case2_bare_negation_scoped_tail_is_not_folded_conservatively():
    """"cánceres son hereditarios." standing alone asserts the opposite of
    the winner's own negated clause ("no creo ... que los cánceres son
    hereditarios"). D-085's contradiction safety net (guard 7) keeps it
    out of the family even when a recorded verdict pairs it with the
    winner -- "conserva negaciones": the 2.4 s tail is a documented
    residual of this gate, not something this path decides."""
    takes = (_take("W", 295.52, 313.50, WINNER), _take("A", 319.38, 327.44, ASIDE), _take("T", 340.18, 342.58, TAIL))
    edges = [_RetryEdge("W", "A", "semantic", 0.95, "same"), _RetryEdge("W", "T", "semantic", 0.95, "same")]
    trace = []
    components = _bridge_aware_components(
        ("W", "A", "T"), edges, protected_ids=frozenset(), take_map={t.clip_id: t for t in takes},
        arbiter=RecordedAnswersArbiter({}, unstable_probe_texts=(TAIL,)), policy=SemanticEquivalenceGatePolicy(),
        edge_trace=trace,
    )
    assert ("T",) in components
    rows = [r for r in trace if r.get("bridge_sensitive") and "T" in (r.get("left_clip_id"), r.get("right_clip_id"))]
    assert rows and rows[0]["accepted"] is False
    assert rows[0].get("reason_rejected") in {"cross_component_contradiction", "component_cohesion_declined"}


# =============================================================================
# CASE 1 -- pimples/allergy fragments: kept, as BOTH references keep them
# =============================================================================

PIMPLES_FRAGMENT = "También me salían espinillas. Era como un rush, una alergia."
PIMPLES_MONOLITH = "También me salían espinillas en esta parte de aquí detrás de la oreja y todo el cuello que yo pensaba que era alergia pero era como espinillas."
PIMPLES_WINNER = "Otro síntoma era que me salían espinillas como si fuera una alergia de esta parte aquí detrás de la oreja y en el cuello. Me salía por temporadas."


def test_case1_pimples_fragment_is_not_folded_into_the_marked_winner_even_with_an_adversarial_verdict():
    """Human Gold keeps the short fragment AND the later fuller take; the
    baseline QA requires the fragment; D-109 records its deletion as a
    false delete. The separating evidence is the speaker's own distinct-
    addition marker on the later take ("otro síntoma") with thin shared
    content (D-048): guard 2 refuses this shape outright, and D-048's own
    veto applies to the marked pair as before, so even an over-eager
    same-idea verdict for the fragment cannot fold it."""
    takes = (
        _take("F", 192.44, 198.12, PIMPLES_FRAGMENT),
        _take("M", 198.88, 211.02, PIMPLES_MONOLITH, complete=False),
        _take("K", 213.34, 222.98, PIMPLES_WINNER),
    )
    arbiter = RecordedAnswersArbiter({
        (PIMPLES_MONOLITH, PIMPLES_WINNER): (True, 0.95, "same symptom, same location"),
        (PIMPLES_FRAGMENT, PIMPLES_WINNER): (True, 0.95, "adversarial: over-eager same-idea"),
    })
    draft, groups, rec_diag, coh_diag = _real_chain(takes, arbiter)

    assert "F" in _kept(draft) and "K" in _kept(draft)
    assert not any(set(g) >= {"F", "K"} for g in groups)  # never one family with the marked winner
    assert not any(r.get("restated_clip_id") == "F" and r.get("accepted") for r in coh_diag["edge_trace"])
    assert "M" in _discarded(draft)  # the marked fuller take wins ITS family over the abandoned monolith


# =============================================================================
# Generic negative controls (no Video00 content)
# =============================================================================

def _bridge(newcomer, member, *, confidence=0.95, extra=None, arbiter=None, policy=None):
    """A singleton newcomer attaching, via one semantic edge to `member`,
    to a component {member, extra}; returns (components, trace). The
    default `extra` mirrors the member's own completeness so it can never
    stand in as a preserving member on its own."""
    extra = extra or _take("E", member.start - 6.0, member.start - 1.0, member.text + " Said once more before.",
                           complete=member.complete_idea)
    takes = (member, extra, newcomer)
    edges = [_RetryEdge(member.clip_id, extra.clip_id, "semantic", 0.95, "same"),
             _RetryEdge(member.clip_id, newcomer.clip_id, "semantic", confidence, "same")]
    trace = []
    comps = _bridge_aware_components(
        tuple(t.clip_id for t in takes), edges, protected_ids=frozenset(), take_map={t.clip_id: t for t in takes},
        arbiter=arbiter or RecordedAnswersArbiter({}, unstable_probe_texts=(newcomer.text,)),
        policy=policy or SemanticEquivalenceGatePolicy(), edge_trace=trace,
    )
    return comps, trace


def _row(trace, cid):
    return next(r for r in trace if r.get("bridge_sensitive") and cid in (r.get("left_clip_id"), r.get("right_clip_id")))


def test_generic_numeric_restatement_after_a_complete_realization_is_folded():
    member = _take("M", 10.0, 16.0, "Only about 5 to 10 percent of these cancers are hereditary, so most of it comes down to lifestyle.")
    newcomer = _take("N", 20.0, 23.0, "So I am convinced that only 5 to 10 percent are hereditary", complete=False)
    comps, trace = _bridge(newcomer, member)
    assert len(comps) == 1
    row = _row(trace, "N")
    assert row["accepted"] and row["accepted_by"] == "contained_restatement_of_complete_realization"
    assert row["preservation_evidence"]["digit_values_preserved"] == ["10", "5"]


def test_generic_unique_numeric_fact_is_never_folded():
    member = _take("M", 10.0, 14.0, "Let's talk about how this product is made and why it lasts.")
    newcomer = _take("N", 20.0, 23.0, "This product uses exactly 12 grams of a patented compound per unit.")
    comps, trace = _bridge(newcomer, member)
    assert ("N",) in comps
    assert _row(trace, "N")["accepted"] is False  # fell through to the (declining) probe


def test_generic_contradicting_number_is_rejected_not_averaged():
    member = _take("M", 10.0, 16.0, "Only about 5 to 10 percent of these cancers are hereditary, so most of it comes down to lifestyle.")
    newcomer = _take("N", 20.0, 23.0, "So I am convinced that a full 30 percent of these cancers are hereditary")
    comps, trace = _bridge(newcomer, member)
    assert ("N",) in comps
    row = _row(trace, "N")
    assert row["accepted"] is False and row.get("accepted_by") != "contained_restatement_of_complete_realization"


def test_generic_continuation_adding_an_uncovered_claim_is_never_folded():
    member = _take("M", 10.0, 14.0, "The biopsy confirmed that it was papillary thyroid cancer.")
    newcomer = _take("N", 15.0, 19.0, "Looking back now there were signs I dismissed at the time.")
    comps, trace = _bridge(newcomer, member)
    assert ("N",) in comps
    assert _row(trace, "N")["accepted"] is False


def test_generic_shared_vocabulary_alone_is_never_a_preservation_proof():
    """D-097.A's own invariant: a further symptom sharing only 'looked like
    an allergy' with a semantically-formed component still needs the
    probe -- its own information (a rash) is not in the member."""
    member = _take("M", 0.0, 4.0, "The pimples showed up behind my ear and along my neck and looked like an allergy.")
    extra = _take("E", 5.0, 9.0, "Those pimples behind my ear looked exactly like an allergy but were hormonal.")
    newcomer = _take("N", 10.0, 13.0, "I also used to break out in a rash that looked like an allergy.")
    comps, trace = _bridge(newcomer, member, extra=extra)
    assert ("N",) in comps
    assert _row(trace, "N")["accepted"] is False


def test_generic_distinct_addition_marker_on_either_side_is_not_this_path():
    member = _take("M", 10.0, 16.0, "Only 5 to 10 percent of these cancers are hereditary.")
    marked = _take("N", 20.0, 23.0, "Another thing: only 5 to 10 percent of these cancers are hereditary", complete=False)
    comps, trace = _bridge(marked, member)
    assert _row(trace, "N").get("accepted_by") != "contained_restatement_of_complete_realization"


def test_generic_earlier_abandoned_attempt_is_not_this_path_chronology_guard():
    """The abandoned-attempt-before-completion shape stays with D-097.12/
    D-287/D-150 and D-094.2's default-OFF policy -- this path only ever
    folds a restatement that FOLLOWS the realization it restates."""
    complete = _take("M", 20.0, 26.0, "Only about 5 to 10 percent of these cancers are hereditary, so most of it comes down to lifestyle.")
    extra = _take("E", 27.0, 30.0, "Only about 5 to 10 percent are hereditary, as I said.")
    earlier = _take("N", 10.0, 13.0, "Only about 5 to 10 percent of these cancers are", complete=False)
    comps, trace = _bridge(earlier, complete, extra=extra)
    assert _row(trace, "N").get("accepted_by") != "contained_restatement_of_complete_realization"
    assert ("N",) in comps


def test_generic_newcomer_carrying_more_content_than_the_member_is_not_a_restatement():
    member = _take("M", 10.0, 13.0, "Only 5 to 10 percent are hereditary.")
    fuller = _take("N", 20.0, 27.0, "Only 5 to 10 percent of these cancers are hereditary, the rest comes down to diet, sleep and exercise habits.")
    comps, trace = _bridge(fuller, member)
    assert _row(trace, "N").get("accepted_by") != "contained_restatement_of_complete_realization"


def test_generic_incomplete_member_cannot_preserve_anything():
    member = _take("M", 10.0, 16.0, "Only about 5 to 10 percent of these cancers are", complete=False)
    newcomer = _take("N", 20.0, 23.0, "So only 5 to 10 percent are hereditary", complete=False)
    comps, trace = _bridge(newcomer, member)
    assert _row(trace, "N").get("accepted_by") != "contained_restatement_of_complete_realization"


def test_generic_below_confidence_floor_is_not_this_path():
    member = _take("M", 10.0, 16.0, "Only about 5 to 10 percent of these cancers are hereditary, so most of it comes down to lifestyle.")
    newcomer = _take("N", 20.0, 23.0, "So I am convinced that only 5 to 10 percent are hereditary", complete=False)
    comps, trace = _bridge(newcomer, member, confidence=0.85)
    assert _row(trace, "N").get("accepted_by") != "contained_restatement_of_complete_realization"


def test_dispatch_order_restart_path_still_wins_and_pairwise_policy_stays_off():
    """The new path sits AFTER the D-097.A restart path and BEFORE the
    default-OFF D-094.2 pairwise path; the policy flag is untouched."""
    assert SemanticEquivalenceGatePolicy().accept_complete_pairwise_singleton_bridge is False
    src = open(tgp.__file__, encoding="utf-8").read()
    i_restart = src.index("_accept_restart_singleton_bridge(\n                    left_members=")
    i_new = src.index("_accept_contained_restatement_singleton_bridge(\n                left_members=")
    i_pairwise = src.index("policy.accept_complete_pairwise_singleton_bridge and min(")
    assert i_restart < i_new < i_pairwise


def test_no_video00_material_in_the_production_rule():
    src = open(tgp.__file__, encoding="utf-8").read()
    start = src.index("_CONTAINED_RESTATEMENT_MIN_TOKEN_COVERAGE")
    end = src.index("def _cross_component_blocked_pair")
    block = src[start:end].lower()
    for forbidden in ("hereditari", "espinilla", "alergia", "5 -10", "clip_7a89", "clip_da42", "clip_38c2",
                      "clip_06c8", "clip_5928", "295.5", "327.7", "tg_"):
        assert forbidden not in block
