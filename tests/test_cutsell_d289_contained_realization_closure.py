"""D-289 / D-289.1 -- redundant realization closure BEFORE Selection Freeze,
through the existing IdeaClusterer / family-competition authorities.

FAITHFUL RAW #122 REPLAY (D-289.1). Every member, text, timing, arbiter
answer, arbiter rejection and hybrid label in section A below is copied
from the run's own result JSON (Modal RAW run 35799404391 at head
`f012beed`, `diagnostics.semantic_idea_equivalence` / `distinct_idea_
grouping_safety` / `hybrid_editorial_chunks` / `take_judge_groups`) --
identical before and after the fix. QA-ONLY material, never read by
production code (the production rule is scanned for it below).

Recorded facts of the conclusion cluster:
* family after reconcile: {W (complete conclusion, 295.52-313.50),
  A (aside, 319.38-327.44)} merged with R (incomplete restatement,
  327.78-334.24) -- pairwise verdicts W-A 0.9 "Both discuss being the only
  family member with the condition.", W-R 0.85 "Second text starts
  restating the hereditary statistics argument.", A-R 0.8 "Overlapping
  themes of familial uniqueness and hereditary stats.";
* the cohesion pass evaluated BOTH bridges into {W, A} with the D-085
  joined-text probe and the arbiter DECLINED at 0.9 (`component_
  cohesion_declined`, `cohesion_confidence` 0.9) -> R split back out as
  its own family, kept; W won {W, A} by `critical_coverage_dominance`
  (hybrid labels W winner 0.95, A alternate 0.8);
* T ("cánceres son hereditarios.", 340.18-342.58) is R's own sentence
  continuation (R stops on "... de los", T starts lower case, 5.94 s
  `real_speech_pause` boundary in `attempt_reconstruction`); it never
  reached the arbiter (3 words: below the candidate-pair floor), sat in its
  own singleton family, and was kept -- a predicate without its quantifier;
* the strict prefix P ("Soy la única en mi familia ...", 342.90-346.52)
  was REMOVED BEFORE GROUPING by the hybrid pass (`hybrid_cross_group_
  retry_integrity`, covered 1.0 by W) and is therefore NOT a family
  member on this run (D-289's first fixture wrongly placed it in the
  family; that fixture is kept in section E as HISTORICAL);
* hybrid labels: W winner 0.95, A alternate 0.8, R failed 0.9, T keep 0.8;
* the claim-equivalence arbiter was NEVER asked about R vs W on this run
  (R never entered W's family), so the JSON holds NO recorded verdict for
  the D-058 canary. The after-replay is therefore shown in all three
  states of that arbiter: absent, confirming, declining. Only the
  confirming state changes the selection; it is a HYPOTHESIS about a
  bounded provider answer, not a recorded fact, and is labelled so.

The three D-289.1 findings and where each is reproduced:
1. the shipped D-289 path required 0.90 on the attaching edge (the D-085
   COMPONENT-probe bar) and so declined the run's own 0.85 pairwise edge
   -> `test_raw122_finding1_*`: the corrected bar is the engine's existing
   PAIRWISE same-idea bar (D-058 Phase 2 / D-061), shared as
   `semantic_idea_equivalence.SAME_IDEA_HIGH_CONFIDENCE_THRESHOLD`;
2. "same digits = preserved" accepted "20 grams / 5 dollars" vs "20 months
   / 5 repairs" -> `test_finding2_*`: removed; preservation is decided by
   `semantic_claims.resolve_ambiguous_coverage` only;
3. R and T judged apart -> `test_raw122_finding3_*` and section F: the
   chain is one realization at grouping, in the bridge proof, in the
   contradiction net, in ranking/labels and in every bucket move.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from cutsell_worker import final_story_coherence_validation as fscv
from cutsell_worker import pipeline as pipeline_module
from cutsell_worker import realization_resolver
from cutsell_worker import semantic_idea_equivalence as sie
from cutsell_worker import take_grouping
from cutsell_worker import take_grouping_provider as tgp
from cutsell_worker.claim_coverage_best_take import apply_claim_coverage_best_take
from cutsell_worker.continuation_chain import (
    CONTINUATION_MEMBER_IDS_KEY,
    REALIZATION_TEXT_KEY,
    bind_continuation_tails,
    chain_tails_by_head,
    fold_family_members,
)
from cutsell_worker.contracts import CandidateTake, DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.deterministic_best_take_authority import apply_deterministic_best_take_authority
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation
from cutsell_worker.semantic_claims import AMBIGUOUS_COVERAGE_FLOOR, COVERAGE_THRESHOLD, claim_coverage, extract_claims
from cutsell_worker.semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalenceResult,
    SemanticEquivalenceGatePolicy,
)
from cutsell_worker.take_grouping import continuation_pairs, sentence_continuation
from cutsell_worker.take_grouping_provider import (
    CONTAINED_RESTATEMENT_ACCEPTANCE,
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
    """Pairwise same-idea answers from an explicit table (the run's recorded
    verdicts). A component-level joined-text probe (" || ") answers with
    the run's recorded probe rejection whenever it involves one of
    `declined_probe_texts` (same_idea=False at `probe_confidence`, the
    exact `cohesion_confidence` 0.9 the run logged); every other probe
    confirms so the rest of a family keeps its real shape."""

    def __init__(self, table, *, declined_probe_texts=(), probe_confidence=0.9):
        self.table = {frozenset(k): v for k, v in table.items()}
        self.declined = tuple(declined_probe_texts)
        self.probe_confidence = probe_confidence
        self.asked = []

    def check(self, request):
        out = []
        for i, pair in enumerate(request.pairs):
            self.asked.append((pair.left_text, pair.right_text))
            if " || " in pair.left_text or " || " in pair.right_text:
                joined = pair.left_text + " || " + pair.right_text
                if any(t in joined for t in self.declined):
                    out.append(IdeaEquivalenceDecision(i, False, self.probe_confidence, "component probe declined"))
                else:
                    out.append(IdeaEquivalenceDecision(i, True, 0.95, "component probe confirmed"))
                continue
            same, conf, reason = self.table.get(frozenset((pair.left_text, pair.right_text)), (False, 0.0, "unconfigured"))
            out.append(IdeaEquivalenceDecision(i, same, conf, reason))
        return IdeaEquivalenceResult(tuple(out), "fake", "fake", True, True, 50, 10)

    @property
    def probes(self):
        return [pair for pair in self.asked if " || " in pair[0] or " || " in pair[1]]


class ClaimArbiter:
    """`ClaimEquivalenceArbiter.claim_covered` with ONE fixed verdict --
    used to show the after-state under a confirming and under a declining
    bounded claim arbiter (no recorded verdict exists for RAW #122)."""

    def __init__(self, verdict):
        self.verdict = bool(verdict)
        self.asked = []

    def claim_covered(self, claim_text, realization_text):
        self.asked.append((claim_text, realization_text))
        return (self.verdict, 0.9, "hypothetical verdict, not recorded on RAW #122")


def _real_chain(takes, arbiter, *, claim_arbiter=None, semantic_labels=None):
    """The real pre-Freeze chain in pipeline order: lexical grouping ->
    reconcile (cross-group, arbiter-gated, continuation pass) -> cohesion
    pass fed the reconcile stage's own confirmations as prior evidence and
    the claim arbiter (exactly as `pipeline.py` does) -> continuation
    chains folded onto their heads -> ranking -> the semantic BestTake with
    the run's recorded hybrid labels (`pipeline._semantic_best_take`) ->
    deterministic BestTake -> claim-coverage BestTake -> final story
    coherence. Returns (draft, groups, reconcile_diag, cohesion_diag)."""
    baseline = safe_group_takes(None, takes)
    merged, rec_diag = reconcile_semantic_idea_equivalence(baseline.groups, takes, arbiter)
    prior = {
        frozenset((r["left_clip_id"], r["right_clip_id"])): (float(r["confidence"]), str(r["reason"]))
        for r in (rec_diag.get("merges") or ()) if r.get("left_clip_id") and r.get("right_clip_id")
    }
    groups, coh_diag = split_incohesive_retry_groups(
        merged, takes, arbiter, prior_confirmations=prior, claim_equivalence_arbiter=claim_arbiter,
    )
    by_id = {t.clip_id: t for t in takes}
    chain_tails = chain_tails_by_head(coh_diag.get("continuation_chains") or (), by_id)

    judge_groups, family_winner_ids, grouped_ids, no_usable_ids = [], set(), set(), set()
    for index, ids in enumerate(groups):
        members, family_chain_tails = fold_family_members([by_id[c] for c in ids], chain_tails, by_id)
        ranked = rank_takes(members)
        local_id = ranked[0].clip_id
        selected_id, reason = local_id, "deterministic_local"
        if semantic_labels and len(members) >= 2:
            decisions = {m.clip_id: semantic_labels[m.clip_id] for m in members if m.clip_id in semantic_labels}
            selected_id, _preferred, reason = pipeline_module._semantic_best_take(
                tuple(members), decisions, local_id, tuple(ranked),
            )
        grouped_ids.update(ids)
        if selected_id is None:
            no_usable_ids.update(ids)
        else:
            family_winner_ids.add(selected_id)
        if len(members) >= 2:
            judge_groups.append({
                "group_id": f"g{index}", "selected_clip_id": selected_id or "",
                "semantic_best_take_reason": reason,
                "no_usable_realization": selected_id is None,
                "ranked": [{
                    "clip_id": r.clip_id, "score": r.score, "reason": r.reason,
                    **({
                        CONTINUATION_MEMBER_IDS_KEY: list(family_chain_tails[r.clip_id]),
                        REALIZATION_TEXT_KEY: next(m.text for m in members if m.clip_id == r.clip_id),
                    } if r.clip_id in family_chain_tails else {}),
                } for r in ranked],
            })
    # compose_selected: one winner per family + every ungrouped take, then
    # the tails of every selected chain head (pipeline order).
    selected_takes = [t for t in takes if t.clip_id in family_winner_ids or t.clip_id not in grouped_ids]
    selected_takes = bind_continuation_tails(selected_takes, takes, chain_tails)
    selected_ids = {t.clip_id for t in selected_takes}

    def clip(t, selected):
        return DraftClip(
            clip_id=t.clip_id, source_asset_id=t.source_asset_id, source_order=t.source_order,
            start=t.start, end=t.end, text=t.text, caption_text=t.text, selected=selected,
        )

    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="d289", strategy=EditStrategy.STORYTELLING,
        selected=tuple(clip(t, True) for t in takes if t.clip_id in selected_ids),
        alternates=tuple(clip(t, False) for t in takes if t.clip_id not in selected_ids and t.clip_id not in no_usable_ids),
        discarded=tuple(clip(t, False) for t in takes if t.clip_id in no_usable_ids),
        diagnostics={
            "take_judge_groups": judge_groups,
            "semantic_idea_equivalence": rec_diag,
            "distinct_idea_grouping_safety": coh_diag,
        },
    )
    draft = apply_deterministic_best_take_authority(draft, swap_enabled=False)
    draft = apply_claim_coverage_best_take(draft, claim_equivalence_arbiter=claim_arbiter)
    draft = apply_final_story_coherence_validation(
        draft, semantic_equivalence_arbiter=arbiter, claim_equivalence_arbiter=claim_arbiter,
    )
    return draft, groups, rec_diag, coh_diag


def _kept(draft):
    return {c.clip_id for c in draft.selected}


def _discarded(draft):
    return {c.clip_id for c in draft.discarded} | {c.clip_id for c in draft.alternates}


def _bridge_rows(coh_diag, cid):
    return [r for r in coh_diag["edge_trace"] if r.get("bridge_sensitive") and cid in (r.get("left_clip_id"), r.get("right_clip_id"))]


def _disable_new_path(monkeypatch):
    monkeypatch.setattr(tgp, "_accept_contained_restatement_singleton_bridge", lambda **kwargs: (False, None))


def _disable_continuation(monkeypatch):
    """Pre-D-289.1 grouping: no continuation relation anywhere."""
    monkeypatch.setattr(tgp, "continuation_pairs", lambda takes: frozenset())


# =============================================================================
# A. RAW #122 -- faithful fixture from the run's own result JSON
# =============================================================================

WINNER = ("Esta es mi experiencia. Soy la única en mi familia que tiene este tipo de cáncer. Por eso no creo y está "
          "comprobado científicamente que los cánceres son hereditarios. Más bien solo un 5 -10 % son de hereditario. "
          "Mayormente son nuestras elecciones de vida así que cuídate.")
ASIDE = ("Soy la primera en mi familia con este tipo de cáncer. Nadie en mi familia tiene un carcinoma papilar en la "
         "tiroides ni sufre de la tiroides.")
RESTATEMENT = "Así que estoy convencida y la ciencia lo avala que solo un 5 -10 % de los"
TAIL = "cánceres son hereditarios."
PREFIX = "Soy la única en mi familia que tiene este tipo de cáncer."  # hybrid-deleted BEFORE grouping on RAW #122
CTA = "Por eso cuídate. Aliméntate bien. Hidrátate. Haz ejercicio."
HAIR = "Se me caía mucho el pelo cuando me lavaba el pelo perdía mucho pelo y pensaba que era por el estrés."
SENTENCE = RESTATEMENT + " " + TAIL

RECORDED_PAIRWISE = {
    (WINNER, ASIDE): (True, 0.9, "Both discuss being the only family member with the condition."),
    (WINNER, RESTATEMENT): (True, 0.85, "Second text starts restating the hereditary statistics argument."),
    (ASIDE, RESTATEMENT): (True, 0.8, "Overlapping themes of familial uniqueness and hereditary stats."),
}
RECORDED_LABELS = {"W": ("winner", 0.95), "A": ("alternate", 0.8), "R": ("failed", 0.9), "T": ("keep", 0.8)}


def _raw122_takes(*, with_tail=True, with_winner=True):
    takes = [
        _take("H", 226.74, 233.18, HAIR),
        _take("A", 319.38, 327.44, ASIDE),
        _take("R", 327.78, 334.24, RESTATEMENT, complete=False),
        _take("C", 356.21, 361.61, CTA),
    ]
    if with_winner:
        takes.append(_take("W", 295.52, 313.50, WINNER))
    if with_tail:
        takes.append(_take("T", 340.18, 342.58, TAIL))
    return tuple(sorted(takes, key=lambda t: t.start))


def _raw122_arbiter():
    return RecordedAnswersArbiter(RECORDED_PAIRWISE, declined_probe_texts=(RESTATEMENT,), probe_confidence=0.9)


def test_raw122_before_the_fix_reproduces_the_run_selection(monkeypatch):
    """Pre-fix engine (no continuation relation, new path off): the run's
    recorded shape byte for byte -- W-A one family, both bridges into it
    declined by the joined-text probe at 0.9, R and T each their own
    singleton family and BOTH kept next to the winner."""
    _disable_new_path(monkeypatch)
    _disable_continuation(monkeypatch)
    arbiter = _raw122_arbiter()
    draft, groups, rec_diag, coh_diag = _real_chain(_raw122_takes(), arbiter, semantic_labels=RECORDED_LABELS)

    merges = {frozenset((r["left_clip_id"], r["right_clip_id"])): r["confidence"] for r in rec_diag["merges"]}
    assert merges == {frozenset("WA"): 0.9, frozenset("WR"): 0.85, frozenset("AR"): 0.8}
    rejected = [r for r in coh_diag["edge_trace"] if r.get("reason_rejected") == "component_cohesion_declined"]
    assert {(r["left_clip_id"], r["right_clip_id"], r["triggering_confidence"], r["cohesion_confidence"]) for r in rejected} \
        == {("W", "R", 0.85, 0.9), ("A", "R", 0.8, 0.9)}
    assert ("R",) in groups and ("T",) in groups and set(next(g for g in groups if "W" in g)) == {"W", "A"}
    assert _kept(draft) == {"H", "W", "R", "T", "C"}
    assert "A" in _discarded(draft)
    assert draft.diagnostics["take_judge_groups"][0]["semantic_best_take_reason"] == "single_semantic_winner"


def test_raw122_finding1_the_run_edge_is_085_and_the_probe_bar_wrongly_declined_it(monkeypatch):
    """Finding 1: with D-289's 0.90 bar (the COMPONENT-probe bar) the path
    returns (False, None) for the run's own 0.85 and 0.80 edges, so the
    probe is asked and declines as recorded. With the corrected PAIRWISE
    bar -- the engine's existing 0.85 (D-058 Phase 2 / D-061), not a new
    number -- the same edge is judged on its preservation evidence."""
    claims = ClaimArbiter(True)
    monkeypatch.setattr(tgp, "SAME_IDEA_HIGH_CONFIDENCE_THRESHOLD", tgp._BRIDGE_MIN_COHESION_CONFIDENCE)
    arbiter = _raw122_arbiter()
    _draft, groups, _rec, coh_diag = _real_chain(_raw122_takes(), arbiter, claim_arbiter=claims, semantic_labels=RECORDED_LABELS)
    assert all(r.get("reason_rejected") == "component_cohesion_declined" for r in _bridge_rows(coh_diag, "R"))
    assert not claims.asked and set(next(g for g in groups if "W" in g)) == {"W", "A"}

    monkeypatch.undo()
    assert tgp.SAME_IDEA_HIGH_CONFIDENCE_THRESHOLD == sie.SAME_IDEA_HIGH_CONFIDENCE_THRESHOLD == 0.85
    assert fscv._SAME_IDEA_HIGH_CONFIDENCE_THRESHOLD == sie.SAME_IDEA_HIGH_CONFIDENCE_THRESHOLD
    assert realization_resolver._HIGH_CONFIDENCE_SEMANTIC_WINNER_THRESHOLD == sie.SAME_IDEA_HIGH_CONFIDENCE_THRESHOLD
    claims = ClaimArbiter(True)
    arbiter = _raw122_arbiter()
    _draft, groups, _rec, coh_diag = _real_chain(_raw122_takes(), arbiter, claim_arbiter=claims, semantic_labels=RECORDED_LABELS)
    accepted = [r for r in _bridge_rows(coh_diag, "R") if r.get("accepted")]
    assert accepted and accepted[0]["accepted_by"] == CONTAINED_RESTATEMENT_ACCEPTANCE
    assert accepted[0]["triggering_confidence"] == 0.85 and accepted[0]["component_cohesion_evaluated"] is False
    assert not arbiter.probes  # no synthetic joined-text probe spent on that edge


def test_raw122_after_with_a_confirming_claim_arbiter_the_chain_competes_and_loses_traceably():
    """AFTER (claim arbiter confirming the D-058 canary -- a HYPOTHESIS,
    the run never asked it): reconcile joins R and T as one realization
    and merges the group; the corrected path accepts the 0.85 bridge on a
    preservation proof over the COMPLETE sentence (coverage 0.56 vs the
    winner: ambiguous band -> bounded claim arbiter -> covered; digits 5/10
    present in the winner; no contradiction); the family competition then
    keeps the winner and discards the aside and the whole chain -- T moves
    with R, never alone. No contradiction, missing idea or blocking loss."""
    claims = ClaimArbiter(True)
    arbiter = _raw122_arbiter()
    draft, groups, rec_diag, coh_diag = _real_chain(_raw122_takes(), arbiter, claim_arbiter=claims, semantic_labels=RECORDED_LABELS)

    assert [(r["left_clip_id"], r["right_clip_id"]) for r in rec_diag["continuation_merges"]] == [("R", "T")]
    assert coh_diag["continuation_chains"] == [["R", "T"]]
    assert set(next(g for g in groups if "W" in g)) == {"W", "A", "R", "T"}
    bridge = next(r for r in coh_diag["edge_trace"] if r.get("accepted_by") == CONTAINED_RESTATEMENT_ACCEPTANCE)
    assert (bridge["left_clip_id"], bridge["right_clip_id"], bridge["triggering_confidence"]) == ("W", "R", 0.85)
    assert bridge["restated_unit_member_ids"] == ["R", "T"] and bridge["accepted"] is True
    proof = bridge["preservation_evidence"]
    assert proof["preserving_member_clip_id"] == "W" and proof["digit_values_preserved"] == ["10", "5"]
    assert proof["claim_arbiter_consulted"] is True
    assert proof["claims_preserved"] == [{
        "claim_type": "MEASUREMENT_QUANTITY", "importance": "CRITICAL", "coverage": 0.5556,
        "resolution": "claim_equivalence_arbiter", "covered": True,
    }]
    assert claims.asked[0] == (SENTENCE, WINNER)  # the complete sentence was judged, not the dangling head
    assert not arbiter.probes

    row = next(r for g in draft.diagnostics["take_judge_groups"] for r in g["ranked"] if r["clip_id"] == "R")
    assert row[CONTINUATION_MEMBER_IDS_KEY] == ["T"] and row[REALIZATION_TEXT_KEY] == SENTENCE
    assert _kept(draft) == {"H", "W", "C"}
    assert _discarded(draft) == {"A", "R", "T"}
    coherence = draft.diagnostics["final_story_coherence_validation"]
    assert coherence["contradiction_findings"] == [] and coherence["missing_idea_coverage"] == []
    assert not any(a.get("blocking") for a in coherence.get("lost_semantic_atoms", []))
    assert not any(c.get("blocking") for c in coherence.get("lost_critical_claims", []))
    assert coherence["freeze_blocked"] is False


@pytest.mark.parametrize("claim_arbiter", [None, ClaimArbiter(False)], ids=["absent", "declining"])
def test_raw122_after_without_a_confirming_claim_verdict_only_the_chain_changes(claim_arbiter):
    """AFTER, claim arbiter absent or declining: the preservation proof
    fails closed, the path falls through to D-085's probe, which declines
    exactly as recorded (0.9) -- the winner's family and the selection are
    the run's own, except that R and T are now ONE family kept together:
    the tail is never again a standalone kept fragment."""
    arbiter = _raw122_arbiter()
    draft, groups, _rec, coh_diag = _real_chain(_raw122_takes(), arbiter, claim_arbiter=claim_arbiter, semantic_labels=RECORDED_LABELS)
    rows = _bridge_rows(coh_diag, "R")
    assert rows and all(r["reason_rejected"] == "component_cohesion_declined" and r["cohesion_confidence"] == 0.9 for r in rows)
    assert all(r["component_cohesion_evaluated"] is True for r in rows)
    assert set(next(g for g in groups if "W" in g)) == {"W", "A"} and ("R", "T") in groups
    assert _kept(draft) == {"H", "W", "R", "T", "C"}
    if claim_arbiter is not None:
        assert claim_arbiter.asked == [(SENTENCE, WINNER)]


def test_raw122_finding3_the_chain_is_kept_or_dropped_as_a_unit_never_the_tail_alone():
    """Finding 3, generic outcome on the real texts: with no complete
    winner in reach, the chain wins its own family as a unit (head AND
    tail kept, in source order); the tail is never selected without its
    quantifier and the head is never judged without its predicate."""
    arbiter = _raw122_arbiter()
    draft, groups, _rec, coh_diag = _real_chain(_raw122_takes(with_winner=False), arbiter, semantic_labels=RECORDED_LABELS)
    assert ("R", "T") in groups
    assert {"R", "T"} <= _kept(draft)
    ordered = [c.clip_id for c in draft.selected]
    assert ordered.index("R") + 1 == ordered.index("T")


def test_raw122_finding3_a_tail_without_its_head_is_still_refused_by_the_contradiction_net():
    """A lone "cánceres son hereditarios." (no head present) asserts the
    opposite of the winner's negated clause: guard 7 rejects it even with
    a confirming verdict. Only the sentence it belongs to (head + tail) is
    judged compatible with the winner."""
    takes = (_take("W", 295.52, 313.50, WINNER), _take("A", 319.38, 327.44, ASIDE), _take("T", 340.18, 342.58, TAIL))
    edges = [_RetryEdge("W", "A", "semantic", 0.95, "same"), _RetryEdge("W", "T", "semantic", 0.95, "same")]
    trace = []
    components = _bridge_aware_components(
        ("W", "A", "T"), edges, protected_ids=frozenset(), take_map={t.clip_id: t for t in takes},
        arbiter=RecordedAnswersArbiter({}, declined_probe_texts=(TAIL,)), policy=SemanticEquivalenceGatePolicy(),
        edge_trace=trace, claim_equivalence_arbiter=ClaimArbiter(True),
    )
    assert ("T",) in components
    rows = [r for r in trace if r.get("bridge_sensitive") and "T" in (r.get("left_clip_id"), r.get("right_clip_id"))]
    assert rows and rows[0]["accepted"] is False and rows[0]["reason_rejected"] == "cross_component_contradiction"


def test_raw122_prefix_is_not_a_family_member_on_this_run():
    """The strict prefix was hybrid-deleted before grouping on RAW #122
    (`hybrid_cross_group_retry_integrity`, coverage 1.0 by the winner):
    the faithful fixture never places it in the family. Documented so
    D-289's historical fixture (section E) is not mistaken for the run."""
    assert PREFIX in WINNER
    assert all(t.text != PREFIX for t in _raw122_takes())


# =============================================================================
# B. Finding 2 -- equal digits are never a preservation proof
# =============================================================================

def _bridge(newcomer, member, *, confidence=0.95, extra=None, arbiter=None, claim_arbiter=None, policy=None):
    """A newcomer attaching, via one semantic edge to `member`, to a
    component {member, extra}; returns (components, trace). The default
    `extra` mirrors the member's own completeness so it can never stand in
    as a preserving member on its own."""
    extra = extra or _take("E", member.start - 6.0, member.start - 1.0, member.text + " Said once more before.",
                           complete=member.complete_idea)
    takes = (member, extra, newcomer)
    edges = [_RetryEdge(member.clip_id, extra.clip_id, "semantic", 0.95, "same"),
             _RetryEdge(member.clip_id, newcomer.clip_id, "semantic", confidence, "same")]
    trace = []
    comps = _bridge_aware_components(
        tuple(t.clip_id for t in takes), edges, protected_ids=frozenset(), take_map={t.clip_id: t for t in takes},
        arbiter=arbiter or RecordedAnswersArbiter({}, declined_probe_texts=(newcomer.text,)),
        policy=policy or SemanticEquivalenceGatePolicy(), edge_trace=trace, claim_equivalence_arbiter=claim_arbiter,
    )
    return comps, trace


def _row(trace, cid):
    return next(r for r in trace if r.get("bridge_sensitive") and cid in (r.get("left_clip_id"), r.get("right_clip_id")))


DEVICE = "The device weighs 20 grams and costs 5 dollars."
WARRANTY = "The warranty lasts 20 months and covers 5 repairs."


def test_finding2_reproduction_equal_digits_with_no_claim_coverage_are_not_preserved():
    """The Product Owner's reproduction: every digit of the newcomer
    appears in the member, `claim_is_covered` is False (0.0 overlap) --
    D-289's "numeric restatement" shortcut accepted this; D-289.1 refuses
    it: the ONLY preservation authority is `resolve_ambiguous_coverage`,
    and 0.0 is below the ambiguous floor (confidently lost, no arbiter
    consulted)."""
    member = _take("M", 10.0, 14.0, DEVICE)
    newcomer = _take("N", 20.0, 24.0, WARRANTY)
    claim = extract_claims("N", WARRANTY)[0]
    assert claim_coverage(claim, DEVICE) < AMBIGUOUS_COVERAGE_FLOOR
    assert tgp._digit_values(WARRANTY) <= tgp._digit_values(DEVICE)  # the digit veto alone would let it through
    assert tgp._contained_restatement_member(("N",), (("M",),), {"M": member, "N": newcomer}, ClaimArbiter(True)) is None
    claims = ClaimArbiter(True)
    comps, trace = _bridge(newcomer, member, claim_arbiter=claims)
    assert ("N",) in comps
    assert _row(trace, "N").get("accepted_by") != CONTAINED_RESTATEMENT_ACCEPTANCE
    assert not claims.asked


def test_finding2_preservation_is_decided_by_the_coverage_authority_in_each_band():
    member = _take("M", 10.0, 16.0, "Only about 5 to 10 percent of these cancers are hereditary, so most of it comes down to lifestyle.")
    # deterministic band: lexically covered, no arbiter needed
    lexical = _take("N", 20.0, 23.0, "Only about 5 to 10 percent of these cancers are hereditary", complete=False)
    assert claim_coverage(extract_claims("N", lexical.text)[0], member.text) >= COVERAGE_THRESHOLD
    comps, trace = _bridge(lexical, member)
    assert len(comps) == 1
    assert _row(trace, "N")["preservation_evidence"]["claims_preserved"][0]["resolution"] == "deterministic_coverage"
    # ambiguous band: the bounded claim arbiter decides; absent -> not this path
    paraphrase = _take("N", 20.0, 24.0, "So I am convinced and the science backs it that only 5 to 10 percent of the cancers are hereditary")
    coverage = claim_coverage(extract_claims("N", paraphrase.text)[0], member.text)
    assert AMBIGUOUS_COVERAGE_FLOOR <= coverage < COVERAGE_THRESHOLD
    comps, trace = _bridge(paraphrase, member)
    assert ("N",) in comps and _row(trace, "N").get("accepted_by") != CONTAINED_RESTATEMENT_ACCEPTANCE
    comps, trace = _bridge(paraphrase, member, claim_arbiter=ClaimArbiter(False))
    assert ("N",) in comps and _row(trace, "N").get("accepted_by") != CONTAINED_RESTATEMENT_ACCEPTANCE
    comps, trace = _bridge(paraphrase, member, claim_arbiter=ClaimArbiter(True))
    assert len(comps) == 1
    proof = _row(trace, "N")["preservation_evidence"]
    assert proof["claims_preserved"][0]["resolution"] == "claim_equivalence_arbiter" and proof["claim_arbiter_consulted"]
    assert "numeric_restatement" not in {c["resolution"] for c in proof["claims_preserved"]}


def test_finding2_a_newcomer_stating_a_number_the_member_lacks_is_vetoed_before_any_arbiter():
    member = _take("M", 10.0, 16.0, "Only about 5 to 10 percent of these cancers are hereditary, so most of it comes down to lifestyle.")
    newcomer = _take("N", 20.0, 23.0, "So I am convinced that a full 30 percent of these cancers are hereditary")
    claims = ClaimArbiter(True)
    comps, trace = _bridge(newcomer, member, claim_arbiter=claims)
    assert ("N",) in comps and _row(trace, "N").get("accepted_by") != CONTAINED_RESTATEMENT_ACCEPTANCE
    assert not claims.asked


# =============================================================================
# C. Generic negative controls (no Video00 content)
# =============================================================================

def test_generic_unique_numeric_fact_is_never_folded():
    member = _take("M", 10.0, 14.0, "Let's talk about how this product is made and why it lasts.")
    newcomer = _take("N", 20.0, 23.0, "This product uses exactly 12 grams of a patented compound per unit.")
    comps, trace = _bridge(newcomer, member, claim_arbiter=ClaimArbiter(True))
    assert ("N",) in comps and _row(trace, "N")["accepted"] is False


def test_generic_continuation_adding_an_uncovered_claim_is_never_folded():
    member = _take("M", 10.0, 14.0, "The biopsy confirmed that it was papillary thyroid cancer.")
    newcomer = _take("N", 15.0, 19.0, "Looking back now there were signs I dismissed at the time.")
    comps, trace = _bridge(newcomer, member, claim_arbiter=ClaimArbiter(True))
    assert ("N",) in comps and _row(trace, "N")["accepted"] is False


def test_generic_shared_vocabulary_alone_is_never_a_preservation_proof():
    member = _take("M", 0.0, 4.0, "The pimples showed up behind my ear and along my neck and looked like an allergy.")
    extra = _take("E", 5.0, 9.0, "Those pimples behind my ear looked exactly like an allergy but were hormonal.")
    newcomer = _take("N", 10.0, 13.0, "I also used to break out in a rash that looked like an allergy.")
    comps, trace = _bridge(newcomer, member, extra=extra)
    assert ("N",) in comps and _row(trace, "N")["accepted"] is False


def test_generic_distinct_addition_marker_on_either_side_is_not_this_path():
    member = _take("M", 10.0, 16.0, "Only 5 to 10 percent of these cancers are hereditary.")
    marked = _take("N", 20.0, 23.0, "Another thing: only 5 to 10 percent of these cancers are hereditary", complete=False)
    comps, trace = _bridge(marked, member, claim_arbiter=ClaimArbiter(True))
    assert _row(trace, "N").get("accepted_by") != CONTAINED_RESTATEMENT_ACCEPTANCE


def test_generic_earlier_abandoned_attempt_is_not_this_path_chronology_guard():
    complete = _take("M", 20.0, 26.0, "Only about 5 to 10 percent of these cancers are hereditary, so most of it comes down to lifestyle.")
    extra = _take("E", 27.0, 30.0, "Only about 5 to 10 percent are hereditary, as I said.")
    earlier = _take("N", 10.0, 13.0, "Only about 5 to 10 percent of these cancers are", complete=False)
    comps, trace = _bridge(earlier, complete, extra=extra, claim_arbiter=ClaimArbiter(True))
    assert _row(trace, "N").get("accepted_by") != CONTAINED_RESTATEMENT_ACCEPTANCE and ("N",) in comps


def test_generic_newcomer_carrying_more_content_than_the_member_is_not_a_restatement():
    member = _take("M", 10.0, 13.0, "Only 5 to 10 percent are hereditary.")
    fuller = _take("N", 20.0, 27.0, "Only 5 to 10 percent of these cancers are hereditary, the rest comes down to diet, sleep and exercise habits.")
    comps, trace = _bridge(fuller, member, claim_arbiter=ClaimArbiter(True))
    assert _row(trace, "N").get("accepted_by") != CONTAINED_RESTATEMENT_ACCEPTANCE


def test_generic_incomplete_member_cannot_preserve_anything():
    member = _take("M", 10.0, 16.0, "Only about 5 to 10 percent of these cancers are", complete=False)
    newcomer = _take("N", 20.0, 23.0, "So only 5 to 10 percent are hereditary", complete=False)
    comps, trace = _bridge(newcomer, member, claim_arbiter=ClaimArbiter(True))
    assert _row(trace, "N").get("accepted_by") != CONTAINED_RESTATEMENT_ACCEPTANCE


def test_generic_below_the_pairwise_bar_is_not_this_path():
    member = _take("M", 10.0, 16.0, "Only about 5 to 10 percent of these cancers are hereditary, so most of it comes down to lifestyle.")
    newcomer = _take("N", 20.0, 23.0, "Only about 5 to 10 percent of these cancers are hereditary", complete=False)
    comps, trace = _bridge(newcomer, member, confidence=0.8)
    assert _row(trace, "N").get("accepted_by") != CONTAINED_RESTATEMENT_ACCEPTANCE


def test_generic_component_to_component_merge_of_two_units_each_side_keeps_the_probe():
    """Two realizations on each side is D-084's transitive-contamination
    shape: never this path, always the D-085 probe."""
    a = _take("A", 0.0, 3.0, "Only about 5 to 10 percent of these cancers are hereditary.")
    b = _take("B", 4.0, 7.0, "Only about 5 to 10 percent of cancers are hereditary, really.")
    c = _take("C", 20.0, 22.0, "Only 5 to 10 percent are hereditary", complete=False)
    d = _take("D", 23.0, 25.0, "Only 5 to 10 percent hereditary, yes", complete=False)
    takes = (a, b, c, d)
    edges = [_RetryEdge("A", "B", "semantic", 0.95, "s"), _RetryEdge("C", "D", "semantic", 0.95, "s"),
             _RetryEdge("A", "C", "semantic", 0.9, "s")]
    trace = []
    _bridge_aware_components(
        ("A", "B", "C", "D"), edges, protected_ids=frozenset(), take_map={t.clip_id: t for t in takes},
        arbiter=RecordedAnswersArbiter({}, declined_probe_texts=(c.text,)), policy=SemanticEquivalenceGatePolicy(),
        edge_trace=trace, claim_equivalence_arbiter=ClaimArbiter(True),
    )
    row = next(r for r in trace if r.get("bridge_sensitive"))
    assert row["component_cohesion_evaluated"] is True and row.get("accepted_by") is None


def test_dispatch_order_restart_path_still_wins_and_pairwise_policy_stays_off():
    assert SemanticEquivalenceGatePolicy().accept_complete_pairwise_singleton_bridge is False
    src = open(tgp.__file__, encoding="utf-8").read()
    i_restart = src.index("_accept_restart_singleton_bridge(\n                    left_members=")
    i_new = src.index("_accept_contained_restatement_singleton_bridge(\n                left_members=")
    i_pairwise = src.index("policy.accept_complete_pairwise_singleton_bridge and min(")
    assert i_restart < i_new < i_pairwise


def test_no_video00_material_in_the_production_rules():
    for module, start_marker, end_marker in (
        (tgp, "_CONTAINED_RESTATEMENT_MIN_TOKEN_COVERAGE", "def _cross_component_blocked_pair"),
        (take_grouping, "_DANGLING_FUNCTION_WORDS", "def group_takes("),
    ):
        src = open(module.__file__, encoding="utf-8").read()
        block = src[src.index(start_marker):src.index(end_marker)].lower()
        for forbidden in ("hereditari", "espinilla", "alergia", "5 -10", "clip_7a89", "clip_da42", "clip_38c2",
                          "clip_5fde", "clip_06c8", "clip_5928", "295.5", "327.7", "340.1", "tg_"):
            assert forbidden not in block, (module.__name__, forbidden)


# =============================================================================
# D. Case 1 -- pimples/allergy fragments: kept, as BOTH references keep them
# =============================================================================

PIMPLES_FRAGMENT = "También me salían espinillas. Era como un rush, una alergia."
PIMPLES_MONOLITH = "También me salían espinillas en esta parte de aquí detrás de la oreja y todo el cuello que yo pensaba que era alergia pero era como espinillas."
PIMPLES_WINNER = "Otro síntoma era que me salían espinillas como si fuera una alergia de esta parte aquí detrás de la oreja y en el cuello. Me salía por temporadas."


def test_case1_pimples_fragment_is_not_folded_into_the_marked_winner_even_with_an_adversarial_verdict():
    """Human Gold keeps the short fragment AND the later fuller take; the
    baseline QA requires the fragment; D-109 records its deletion as a
    false delete. Guard 2 (the speaker's own distinct-addition marker)
    refuses this shape outright even with a confirming claim arbiter and
    an over-eager same-idea verdict."""
    takes = (
        _take("F", 192.44, 198.12, PIMPLES_FRAGMENT),
        _take("M", 198.88, 211.02, PIMPLES_MONOLITH, complete=False),
        _take("K", 213.34, 222.98, PIMPLES_WINNER),
    )
    arbiter = RecordedAnswersArbiter({
        (PIMPLES_MONOLITH, PIMPLES_WINNER): (True, 0.95, "same symptom, same location"),
        (PIMPLES_FRAGMENT, PIMPLES_WINNER): (True, 0.95, "adversarial: over-eager same-idea"),
    })
    draft, groups, _rec, coh_diag = _real_chain(takes, arbiter, claim_arbiter=ClaimArbiter(True))
    assert "F" in _kept(draft) and "K" in _kept(draft)
    assert not any(set(g) >= {"F", "K"} for g in groups)
    assert not any(r.get("restated_clip_id") == "F" and r.get("accepted") for r in coh_diag["edge_trace"])
    assert "M" in _discarded(draft)


# =============================================================================
# E. HISTORICAL fixture (D-289's first cut) -- NOT the RAW #122 run
# =============================================================================
#
# D-289 replayed this cluster with the D-056.5-era pairwise answer (0.9 for
# W-R) and with the strict prefix P inside the winner's family. On RAW #122
# the recorded W-R answer is 0.85 and P was hybrid-deleted before grouping
# (section A). Kept, clearly labelled, so the two are never confused.

HISTORICAL_PAIRWISE = {(WINNER, RESTATEMENT): (True, 0.9, "Same personal cancer statistics and hereditary beliefs discussed.")}


def _historical_takes():
    return tuple(sorted((
        _take("H", 226.74, 233.18, HAIR),
        _take("W", 295.52, 313.50, WINNER),
        _take("R", 327.78, 334.24, RESTATEMENT, complete=False),
        _take("P", 342.90, 346.52, PREFIX),
        _take("C", 356.21, 361.61, CTA),
    ), key=lambda t: t.start))


def test_historical_fixture_resolves_the_same_way_under_the_corrected_path():
    arbiter = RecordedAnswersArbiter(HISTORICAL_PAIRWISE, declined_probe_texts=(RESTATEMENT,))
    draft, groups, _rec, coh_diag = _real_chain(_historical_takes(), arbiter, claim_arbiter=ClaimArbiter(True))
    assert set(next(g for g in groups if "W" in g)) == {"W", "P", "R"}
    assert next(r for r in coh_diag["edge_trace"] if r.get("accepted_by") == CONTAINED_RESTATEMENT_ACCEPTANCE)["restated_unit_member_ids"] == ["R"]
    assert _kept(draft) == {"H", "W", "C"} and _discarded(draft) == {"P", "R"}


def test_historical_fixture_without_a_claim_verdict_keeps_the_loss_visible():
    arbiter = RecordedAnswersArbiter(HISTORICAL_PAIRWISE, declined_probe_texts=(RESTATEMENT,))
    draft, groups, _rec, _coh = _real_chain(_historical_takes(), arbiter, claim_arbiter=None)
    assert ("R",) in groups and "R" in _kept(draft)  # WHEN UNCERTAIN, KEEP: the restatement stays


# =============================================================================
# F. The continuation relation itself (deterministic, generic)
# =============================================================================

def test_continuation_relation_accepts_a_dangling_head_finished_by_a_lowercase_tail():
    head = _take("R", 10.0, 16.0, "So I am convinced and the science backs it that only 5 to 10 percent of the", complete=False)
    tail = _take("T", 21.5, 23.0, "cancers are hereditary.")
    assert sentence_continuation(head, tail)
    assert continuation_pairs((tail, head)) == frozenset({frozenset({"R", "T"})})


@pytest.mark.parametrize("head_text, head_complete, tail_text, tail_start", [
    ("So I am convinced that only 5 to 10 percent of the", False, "Cancers are hereditary.", 21.5),  # capitalised: new sentence
    ("So I am convinced that only 5 to 10 percent are hereditary.", True, "cancers are hereditary.", 21.5),  # complete head
    ("So I am convinced that only 5 to 10 percent are hereditary", False, "cancers are hereditary.", 21.5),  # not a dangling word
    ("So I am convinced that only 5 to 10 percent of the", False, "cancers are hereditary.", 16.0 + 8.5),  # beyond the retry bound
], ids=["capitalised_tail", "complete_head", "non_dangling_ending", "too_far"])
def test_continuation_relation_negative_controls(head_text, head_complete, tail_text, tail_start):
    head = _take("R", 10.0, 16.0, head_text, complete=head_complete)
    tail = _take("T", tail_start, tail_start + 1.5, tail_text)
    assert not sentence_continuation(head, tail)


def test_continuation_relation_requires_adjacency_and_same_source():
    head = _take("R", 10.0, 16.0, "So I am convinced that only 5 to 10 percent of the", complete=False)
    between = _take("X", 17.0, 18.0, "Wait, let me start again.")
    tail = _take("T", 21.5, 23.0, "cancers are hereditary.")
    assert continuation_pairs((head, between, tail)) == frozenset()
    assert not sentence_continuation(head, replace(tail, source_asset_id="other"))


def test_chain_folding_presents_the_sentence_and_binding_moves_the_tail_with_its_head():
    head = _take("R", 10.0, 16.0, "So I am convinced that only 5 to 10 percent of the", complete=False)
    tail = _take("T", 21.5, 23.0, "cancers are hereditary.")
    other = _take("W", 0.0, 5.0, "Only 5 to 10 percent of cancers are hereditary, the rest is lifestyle.")
    by_id = {t.clip_id: t for t in (head, tail, other)}
    tails = chain_tails_by_head([["T", "R"]], by_id)
    assert tails == {"R": ("T",)}
    folded, family_tails = fold_family_members([other, head, tail], tails, by_id)
    assert [m.clip_id for m in folded] == ["W", "R"] and family_tails == {"R": ("T",)}
    evaluated = next(m for m in folded if m.clip_id == "R")
    assert evaluated.text == head.text + " " + tail.text and evaluated.end == tail.end and evaluated.complete_idea
    assert bind_continuation_tails([head], [head, tail, other], tails) == (head, tail)
    assert bind_continuation_tails([other], [head, tail, other], tails) == (other,)
