"""D-289.10 -- RAW #124 forensic (run 35886004885 on `693c7b25`): the
pimples complete-window conflict fallback and the stomach initial-merge /
component-probe split, reproduced from the Product Owner's evidence
package (`raw124-review-evidence`, `raw124-mp4-review`) and corrected in
the existing authorities.

PIMPLES (recorded): family {monolith M, later take L}; two family-complete
windows disagree (chunk 3: M winner .92 / L alternate .80; chunk 4: L
winner .95 / M alternate .80). `family_scoped_semantic_decisions` merges
per clip by `_decision_priority` -> two "winner" labels; the D-150 gate
reports `ABSTAIN_CONFLICT`; the ladder (no critical claims, no dominance,
no contradiction) falls to the DeliveryScorer tie-break: M 0.6671 vs L
0.621 -> `DELIVERYSCORE_PATH`, terminal `NON_DECISIVE`
(`raw_score_difference_without_structured_dominance`, margin 0.0461). L is
discarded; both references keep L and reject M.
Why nothing intervened: `label_conflict_routed` is D-097.B's ALL-DELETE-
RECOMMENDED routing (false here by definition); the bounded finalist
authority (the only existing consumer of NON_DECISIVE) was flag-OFF on the
run; and the resolver's own `conflicting_high_confidence_semantic_winner_
evidence` branch never saw the disagreement because the Ledger records
winner evidence only when the fast path APPLIED an override.
Fix: the ladder stops before the delivery tie-break on a recorded
complete-window winner conflict (`unresolved_semantic_winner_conflict`,
CONFLICTED); the Ledger records each window's winner verdict as
SEMANTIC_WINNER_CONFLICT_EVIDENCE; the resolver's existing branch decides
REVIEW_REQUIRED. No "later window wins", no "higher confidence wins", no
text/clip exception, no flag activation.

STOMACH (recorded): the abandoned attempt S1 was merged with the
incomplete attempt E by DETERMINISTIC restart evidence
(`measured_pause_bridged_retry`, "arbiter not consulted") and E with the
gastritis take G (`incomplete_attempt_completed_by_retry`); in the cohesion
pass the E-S1 edge became a bridge and the D-085 component probe DECLINED
it (RAW #122: the identical probe ACCEPTED). Demonstrated cause: prior
confirmations reach the cohesion pass as (confidence, reason) only, so a
deterministic restart merge re-enters as a `semantic` edge and is
re-examined by a run-varying arbiter answer; `measured_pause_bridged_
retry` was also missing from `_RESTART_EVIDENCE_KINDS`. Fix: the merge's
`accepted_by` kind travels with the confirmation and a restart-kind prior
stays a deterministic edge (D-108 blocked-pair veto, D-083 divergence gate
and the cross-component contradiction net still apply).

QA: `pimples_later_winner_present` passed on RAW #124 because
`required_exact` is a token-coverage search (9/14 of the later take's
tokens sit in the selected monolith); `required_realization` asks whether
THE realization is present. Baselines untouched.

RAW #124/#122 texts, times, labels and answers below are QA fixtures from
the packages; production code reads none of them. Every arbiter answer a
fake gives is labelled; a simulated answer proves the PATH, never the
real outcome.
"""
from __future__ import annotations

import types
from dataclasses import replace

import pytest

from benchmarks.validate_video00_regression_qa import _find_present_realization, _find_semantic, _norm, realization_present
from cutsell_worker.contracts import CandidateTake, DraftClip, DraftTimeline, EditStrategy, RankedTake, SCHEMA_VERSION
from cutsell_worker.final_story_coherence_validation import _classify_family_under_authority, fold_alternates_into_discarded
from cutsell_worker.pipeline import _complete_window_winner_conflict, _semantic_best_take, family_scoped_semantic_decisions
from cutsell_worker.realization_resolver import apply_authoritative_realization_resolution, resolve_realizations_shadow
from cutsell_worker.semantic_authority_observability import family_authority_diagnostics, semantic_authority_gate_diagnostics
from cutsell_worker.semantic_ledger import SEMANTIC_WINNER_CONFLICT_EVIDENCE, build_semantic_ledger_shadow
from cutsell_worker.take_grouping_provider import _RESTART_EVIDENCE_KINDS, split_incohesive_retry_groups

from tests.test_cutsell_d289_contained_realization_closure import RecordedAnswersArbiter, _discarded, _kept, _real_chain


def _take(cid, start, end, text, *, complete=True, source="src"):
    return CandidateTake(cid, source, 0, start, end, text, complete_idea=complete)


# =============================================================================
# Fixtures -- RAW #124 / RAW #122 package texts
# =============================================================================
M_TEXT = ("También me salían espinillas en esta parte de aquí detrás de la oreja y todo el cuello que yo pensaba que era "
          "alergia pero era como espinillas de personas con problemas hormonales.")
L_TEXT = ("Otro síntoma era que me salían espinillas como si fuera una alergia de esta parte aquí detrás de la oreja y en "
          "el cuello. Me salía por temporadas.")
F1_TEXT, F2_TEXT = "También me salían espinillas.", "Era como un rush, una alergia."
E_TEXT = "Tuve problemas estomacales a un tiempo en donde se me hizo una endoscopía y me diagnosticaron con..."
S1_TEXT = "Tuve problemas de estómago en una temporada, en 2023, no hay que preguntar."
G_TEXT = ("Tuve problemas de digestión en donde me hicieron una endoscopía y dijeron que tenía gastritis. Nada severo pero "
          "tenía gastritis y me mandaron tres meses con pastillas.")

M = _take("M", 198.88, 211.02, M_TEXT)
L = _take("L", 213.34, 222.98, L_TEXT)
E = _take("E", 236.23, 244.19, E_TEXT, complete=False)
S1 = _take("S1", 245.39, 251.61, S1_TEXT)
G = _take("G", 258.87, 269.37, G_TEXT)

# the recorded reconcile merges of the stomach cluster (both runs identical)
PRIOR_E_S1 = (1.0, "deterministic restart evidence (measured_pause_bridged_retry); arbiter not consulted", "measured_pause_bridged_retry")
PRIOR_E_G = (1.0, "deterministic restart evidence (incomplete_attempt_completed_by_retry); arbiter not consulted", "incomplete_attempt_completed_by_retry")
PRIOR_S1_G_124 = (0.9, "Retries describing stomach digestion issues.", "")
STOMACH_PRIOR = {frozenset(("E", "S1")): PRIOR_E_S1, frozenset(("E", "G")): PRIOR_E_G, frozenset(("S1", "G")): PRIOR_S1_G_124}
LABELS_124 = {"E": ("failed", 0.95), "S1": ("winner", 0.95), "G": ("winner", 0.95)}
LABELS_122 = {"E": ("failed", 0.95), "S1": ("alternate", 0.7), "G": ("winner", 0.95)}


def _window(chunk, window_id, member_ids, decisions):
    return {"session_id": window_id, "partition_index": 0, "chunk_index": chunk, "member_ids": list(member_ids),
            "provider": "google", "model": "gemini-3.5-flash-lite", "request_hash": f"rh_{window_id}",
            "decisions": [{"clip_id": c, "label": l, "confidence": conf} for c, l, conf in decisions]}


# RAW #124 recorded complete windows (chunk 3 and 4) and RAW #122's
WINDOWS_124 = [_window(3, "hc_c81b9adfd5516bb012", ["X1", "M", "L"], [("M", "winner", 0.92), ("L", "alternate", 0.8)]),
               _window(4, "hc_463d6b56445bec0b14", ["M", "L", "X2"], [("L", "winner", 0.95), ("M", "alternate", 0.8)])]
WINDOWS_122 = [_window(3, "hc_75a744101587135c4f", ["X1", "M", "L"], [("L", "winner", 0.92), ("M", "alternate", 0.85)]),
               _window(4, "hc_311e83d3d13014d956", ["M", "L", "X2"], [("L", "winner", 0.9), ("M", "alternate", 0.75)])]
RANKED = (RankedTake("M", 0.6671, "watch_listen_baseline"), RankedTake("L", 0.621, "watch_listen_baseline"))


def _pimples_decision(windows, *, conflict_aware, members=(M, L), ranked=RANKED):
    glob = {}
    for w in windows:
        for d in w["decisions"]:
            glob[d["clip_id"]] = (d["label"], d["confidence"])
    decisions, source = family_scoped_semantic_decisions(members, glob, windows)
    ids = [m.clip_id for m in members]
    obs = family_authority_diagnostics(ids, windows, source)
    gate = semantic_authority_gate_diagnostics(ids, decisions, obs)
    conflict = _complete_window_winner_conflict(members, obs, gate)
    out: dict = {}
    kwargs = dict(semantic_comparative_authority=gate["semantic_authority_gate_status"], terminal_confidence_out=out)
    if conflict_aware:
        kwargs["complete_window_winner_conflict_ids"] = frozenset(conflict["conflicting_clip_ids"])
    selected, preferred, reason = _semantic_best_take(members, decisions, ranked[0].clip_id, ranked, **kwargs)
    conflict["routed"] = reason == "unresolved_semantic_winner_conflict"
    return {"decisions": decisions, "gate": gate, "obs": obs, "conflict": conflict, "selected": selected,
            "preferred": preferred, "reason": reason, "terminal": out["terminal_besttake_confidence"]}


def _pimples_draft(result, members=(M, L)):
    selected_id = result["selected"]
    def clip(t, selected):
        return DraftClip(clip_id=t.clip_id, source_asset_id=t.source_asset_id, source_order=0, start=t.start, end=t.end,
                         text=t.text, caption_text=t.text, selected=selected, semantic_idea_id="tg_pimples")
    row = {"group_id": "tg_pimples", "selected_clip_id": selected_id, "local_selected_clip_id": "M",
           "semantic_preferred_clip_id": result["preferred"], "semantic_override_applied": selected_id != "M",
           "semantic_best_take_reason": result["reason"],
           "semantic_candidates": [{"clip_id": m.clip_id, "label": result["decisions"][m.clip_id][0],
                                    "confidence": result["decisions"][m.clip_id][1]} for m in members],
           "ranked": [{"clip_id": r.clip_id, "score": r.score, "reason": r.reason} for r in RANKED],
           "complete_window_winner_conflict": result["conflict"], "no_usable_realization": False}
    return DraftTimeline(schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
                         selected=tuple(clip(t, True) for t in members if t.clip_id == selected_id),
                         alternates=tuple(clip(t, False) for t in members if t.clip_id != selected_id), discarded=(),
                         diagnostics={"take_judge_groups": [row], "take_group_members": [[m.clip_id for m in members]]})


# =============================================================================
# 1. PIMPLES -- the recorded path, reproduced
# =============================================================================

def test_pimples_recorded_path_conflict_then_deliveryscore_monolith_non_decisive():
    r = _pimples_decision(WINDOWS_124, conflict_aware=False)
    assert r["decisions"] == {"M": ("winner", 0.92), "L": ("winner", 0.95)}  # the per-clip merge: two winners
    assert r["obs"]["complete_window_agreement_status"] == "MULTIPLE_COMPLETE_WINDOWS_DISAGREE"
    assert r["gate"]["semantic_authority_gate_status"] == "ABSTAIN_CONFLICT"
    assert r["gate"]["semantic_authority_before"] == r["gate"]["semantic_authority_after"] == "NON_DECISIVE"
    assert (r["selected"], r["preferred"], r["reason"]) == ("M", None, "delivery_tie_break_among_survivors")
    assert r["terminal"].confidence_state == "NON_DECISIVE"
    assert r["terminal"].reason == "raw_score_difference_without_structured_dominance"
    assert round(r["terminal"].top_score - r["terminal"].runner_up_score, 4) == 0.0461


def test_pimples_why_no_existing_authority_intervened():
    """`label_conflict_routed` is the D-097.B all-delete-recommended flag
    (pipeline.py: `all_delete_recommended and not no_usable_realization`)
    -- nothing to do with window disagreement; and the resolver's own
    conflicting-winner branch reads Ledger winner evidence that, before
    D-289.10, was only ever written when the fast path APPLIED an
    override (`semantic_override_applied`), which a delivery tie-break
    never does."""
    r = _pimples_decision(WINDOWS_124, conflict_aware=False)
    draft = _pimples_draft(r)
    ledger = build_semantic_ledger_shadow(draft)
    assert not [d for d in ledger.decisions() if d.decision_type == SEMANTIC_WINNER_CONFLICT_EVIDENCE]
    report = resolve_realizations_shadow(ledger)
    (res,) = report.idea_resolutions.values()
    assert res.decision_status == "RESOLVED_WINNER" and res.winner_realization_id == "M"


# =============================================================================
# 2. PIMPLES -- after: the ladder stops, the Ledger carries the evidence,
#    the resolver's EXISTING conflict branch decides REVIEW_REQUIRED
# =============================================================================

def test_pimples_after_fix_conflict_routed_to_the_existing_resolver_branch():
    r = _pimples_decision(WINDOWS_124, conflict_aware=True)
    assert r["conflict"]["conflicting_clip_ids"] == ["L", "M"] and r["conflict"]["routed"] is True
    assert [(e["window_id"], e["clip_id"], e["confidence"]) for e in r["conflict"]["window_evidence"]] == [
        ("hc_c81b9adfd5516bb012", "M", 0.92), ("hc_463d6b56445bec0b14", "L", 0.95)]
    # no forced pick: the selection stays on the existing safe fallback, nothing is preferred
    assert (r["selected"], r["preferred"], r["reason"]) == ("M", None, "unresolved_semantic_winner_conflict")
    assert r["terminal"].confidence_state == "CONFLICTED" and r["terminal"].reason == "unresolved_semantic_winner_conflict"
    draft = _pimples_draft(r)
    ledger = build_semantic_ledger_shadow(draft)
    evidence = sorted((d.subject_realization_id, d.evidence["confidence"], d.evidence["window_id"])
                      for d in ledger.decisions() if d.decision_type == SEMANTIC_WINNER_CONFLICT_EVIDENCE)
    assert evidence == [("L", 0.95, "hc_463d6b56445bec0b14"), ("M", 0.92, "hc_c81b9adfd5516bb012")]
    report = resolve_realizations_shadow(ledger)
    (res,) = report.idea_resolutions.values()
    assert res.decision_status == "REVIEW_REQUIRED"
    assert res.decision_reason == "conflicting_high_confidence_semantic_winner_evidence"
    assert res.evidence["conflicting_realization_ids"] == ["L", "M"] and res.winner_realization_id is None
    assert set(res.retained_for_contextual_value) == {"L", "M"}
    auth = apply_authoritative_realization_resolution(draft, ledger, report)
    assert auth.status == "REVIEW_REQUIRED"
    assert [c.clip_id for c in auth.draft.selected] == ["M"] and [c.clip_id for c in auth.draft.alternates] == ["L"]
    accepted, classification = _classify_family_under_authority(
        types.SimpleNamespace(decision_status=res.decision_status, accepted_as_resolved=False))
    assert (accepted, classification) == (False, "authoritative_review_required")  # D-090: blocks Freeze


def test_pimples_raw122_agreeing_windows_are_byte_identical_before_and_after():
    before = _pimples_decision(WINDOWS_122, conflict_aware=False)
    after = _pimples_decision(WINDOWS_122, conflict_aware=True)
    for r in (before, after):
        assert r["gate"]["semantic_authority_gate_status"] == "AUTHORITATIVE"
        assert (r["selected"], r["preferred"], r["reason"]) == ("L", "L", "single_semantic_winner")
        assert r["terminal"].confidence_state == "DECISIVE"
        assert r["conflict"] == {"routed": False, "conflicting_clip_ids": [], "window_evidence": []}
    draft = _pimples_draft(after)
    report = resolve_realizations_shadow(build_semantic_ledger_shadow(draft))
    (res,) = report.idea_resolutions.values()
    assert (res.decision_status, res.winner_realization_id) == ("RESOLVED_WINNER", "L")


# --- controls: what never routes, and what still comes first --------------

def test_control_a_window_winner_below_the_floor_is_not_a_conflict_pair():
    windows = [_window(3, "w1", ["M", "L"], [("M", "winner", 0.84), ("L", "alternate", 0.8)]),
               _window(4, "w2", ["M", "L"], [("L", "winner", 0.95), ("M", "alternate", 0.8)])]
    r = _pimples_decision(windows, conflict_aware=True)
    assert r["gate"]["semantic_authority_gate_status"] == "ABSTAIN_CONFLICT"  # D-149 still sees the disagreement
    assert r["conflict"]["conflicting_clip_ids"] == [] and r["conflict"]["routed"] is False
    assert r["reason"] == "delivery_tie_break_among_survivors"  # unchanged pre-D-289.10 fall-through


def test_control_partial_window_conflict_and_incomplete_context_never_route():
    windows = [_window(3, "w1", ["M"], [("M", "winner", 0.95)]), _window(4, "w2", ["L"], [("L", "winner", 0.95)])]
    r = _pimples_decision(windows, conflict_aware=True)
    assert r["gate"]["semantic_authority_gate_status"] == "ABSTAIN_INCOMPLETE_CONTEXT"
    assert r["conflict"]["conflicting_clip_ids"] == [] and r["reason"] == "delivery_tie_break_among_survivors"


def test_control_critical_coverage_dominance_still_decides_before_the_conflict_stop():
    """D-062.1's ruling: layer 4 first. A conflicted pair where one member
    holds every CRITICAL claim resolves by dominance, never by the stop."""
    a = _take("A", 0.0, 6.0, "La biopsia confirmó que era un cáncer papilar de tiroides y me mandaron a cirugía.")
    b = _take("B", 8.0, 12.0, "Me mandaron a cirugía después de la biopsia de la tiroides.")
    windows = [_window(3, "w1", ["A", "B"], [("A", "winner", 0.92), ("B", "alternate", 0.8)]),
               _window(4, "w2", ["A", "B"], [("B", "winner", 0.95), ("A", "alternate", 0.8)])]
    ranked = (RankedTake("B", 0.7, "watch_listen_baseline"), RankedTake("A", 0.6, "watch_listen_baseline"))
    r = _pimples_decision(windows, conflict_aware=True, members=(a, b), ranked=ranked)
    assert r["conflict"]["conflicting_clip_ids"] == ["A", "B"]
    assert (r["selected"], r["reason"]) == ("A", "critical_coverage_dominance") and r["conflict"]["routed"] is False
    assert r["terminal"].confidence_state == "DECISIVE"


def test_control_contradiction_still_stops_first_and_no_last_or_highest_rule_exists():
    a = _take("A", 0.0, 6.0, "The medication worked well for her symptoms.")
    b = _take("B", 8.0, 12.0, "The medication never worked well for her symptoms.")
    windows = [_window(3, "w1", ["A", "B"], [("A", "winner", 0.99), ("B", "alternate", 0.8)]),
               _window(4, "w2", ["A", "B"], [("B", "winner", 0.90), ("A", "alternate", 0.8)])]
    ranked = (RankedTake("B", 0.7, "watch_listen_baseline"), RankedTake("A", 0.6, "watch_listen_baseline"))
    r = _pimples_decision(windows, conflict_aware=True, members=(a, b), ranked=ranked)
    # the negated pair is caught by step 5's own structured checks (the
    # negation is a CRITICAL claim only one side carries) BEFORE the
    # conflict stop -- never by it, never by delivery
    assert r["reason"] in ("unresolved_unique_fact_asymmetry", "unresolved_contradiction")
    assert r["terminal"].confidence_state == "CONFLICTED" and r["conflict"]["routed"] is False
    # swapping which window is later or which confidence is higher never changes the stop's answer
    for w in (WINDOWS_124, list(reversed(WINDOWS_124))):
        r2 = _pimples_decision(w, conflict_aware=True)
        assert (r2["selected"], r2["preferred"], r2["reason"]) == ("M", None, "unresolved_semantic_winner_conflict")


def test_control_singleton_and_omitted_kwarg_are_unchanged():
    out = {}
    assert _semantic_best_take((M,), {"M": ("keep", 0.9)}, "M", RANKED[:1], terminal_confidence_out=out) == ("M", None, "single_member_no_contest")
    assert _semantic_best_take((M,), {"M": ("keep", 0.9)}, "M", RANKED[:1], complete_window_winner_conflict_ids=frozenset({"M", "L"})) == ("M", None, "single_member_no_contest")
    r = _pimples_decision(WINDOWS_124, conflict_aware=False)
    assert r["reason"] == "delivery_tie_break_among_survivors"


# =============================================================================
# 3. STOMACH -- reproduction: with the kind dropped, the probe decides
# =============================================================================

def _cohesion(prior, arbiter):
    groups, diag = split_incohesive_retry_groups((("E", "S1", "G"),), (E, S1, G), arbiter, prior_confirmations=prior)
    rows = [(r["left_clip_id"], r["right_clip_id"], r["evidence"], r["accepted"], r.get("accepted_by"), r.get("reason_rejected"))
            for r in diag["edge_trace"]]
    return groups, rows, diag


def _kind_dropped(prior):
    return {k: v[:2] for k, v in prior.items()}  # the pre-D-289.10 shape pipeline.py built


def test_stomach_reproduction_raw122_probe_accepts_and_raw124_probe_declines_the_same_inputs():
    accepting = RecordedAnswersArbiter({}, probe_confidence=0.9)  # RAW #122's recorded probe answer shape (accept 0.9)
    declining = RecordedAnswersArbiter({}, declined_probe_texts=(S1_TEXT,), probe_confidence=0.9)  # RAW #124's (decline 0.9)
    g122, rows122, _ = _cohesion(_kind_dropped(STOMACH_PRIOR), accepting)
    g124, rows124, _ = _cohesion(_kind_dropped(STOMACH_PRIOR), declining)
    assert g122 == (("E", "S1", "G"),) and len(accepting.probes) == 1
    assert g124 == (("E", "G"), ("S1",)) and len(declining.probes) == 2
    assert [r[:4] for r in rows124] == [("E", "G", "semantic", True), ("E", "S1", "semantic", False), ("S1", "G", "semantic", False)]
    assert [r[5] for r in rows124[1:]] == ["component_cohesion_declined", "component_cohesion_declined"]
    # the deterministic restart merge had become a SEMANTIC edge: the probe decided a recording-process relation
    assert rows124[1][2] == "semantic" and "measured_pause_bridged_retry" not in str(rows124[1][4])


# =============================================================================
# 4. STOMACH -- after: the restart kind travels; no probe is asked
# =============================================================================

@pytest.mark.parametrize("arbiter_shape", ["accepting", "declining"])
def test_stomach_after_fix_the_deterministic_relation_never_reaches_the_probe(arbiter_shape):
    arbiter = (RecordedAnswersArbiter({}, probe_confidence=0.9) if arbiter_shape == "accepting"
               else RecordedAnswersArbiter({}, declined_probe_texts=(S1_TEXT,), probe_confidence=0.9))
    groups, rows, diag = _cohesion(STOMACH_PRIOR, arbiter)
    assert groups == (("E", "S1", "G"),) and arbiter.probes == []
    assert ("E", "S1", "deterministic", True, "deterministic_restart_evidence", None) in rows
    assert ("E", "G", "deterministic", True, None, None) in rows  # non-bridge deterministic union
    reused = {(r["left_clip_id"], r["right_clip_id"]): r for r in diag["prior_confirmations_reused"]}
    assert reused[("E", "S1")]["source"] == "prior_restart_evidence" and reused[("E", "S1")]["accepted_by"] == "measured_pause_bridged_retry"
    assert reused[("E", "G")]["accepted_by"] == "incomplete_attempt_completed_by_retry"
    assert "measured_pause_bridged_retry" in _RESTART_EVIDENCE_KINDS


def test_stomach_after_fix_downstream_the_gastritis_take_wins_and_the_attempt_is_discarded():
    """Through the real chain with RAW #124's labels (S1 winner .95 in one
    window) and RAW #122's: grouping is deterministic (no probe), the
    family competes, the authoritative resolver picks G, and the KEEP/
    DISCARD fold (D-092) discards S1 and E -- RAW #122's recorded
    outcome. The pairwise S1-G confirmation is the run's recorded answer;
    the probe fake is never reached."""
    for labels in (LABELS_124, LABELS_122):
        arbiter = RecordedAnswersArbiter({(S1_TEXT, G_TEXT): (True, 0.9, "Retries describing stomach digestion issues.")},
                                         declined_probe_texts=(S1_TEXT,), probe_confidence=0.9, unlisted_reason="not_recorded__fake_decline")
        draft, groups, rec, coh = _real_chain((E, S1, G), arbiter, claim_arbiter=None, semantic_labels=labels)
        assert groups == (("E", "S1", "G"),) and arbiter.probes == []
        assert coh["continuation_chains"] == [] and rec["merges"][0]["accepted_by"] == "incomplete_attempt_completed_by_retry"
        gid = draft.diagnostics["take_judge_groups"][0]["group_id"]
        stamped = replace(draft, selected=tuple(replace(c, semantic_idea_id=gid) for c in draft.selected),
                          alternates=tuple(replace(c, semantic_idea_id=gid) for c in draft.alternates),
                          discarded=tuple(replace(c, semantic_idea_id=gid) for c in draft.discarded))
        ledger = build_semantic_ledger_shadow(stamped)
        report = resolve_realizations_shadow(ledger)
        (res,) = report.idea_resolutions.values()
        assert (res.decision_status, res.decision_reason, res.winner_realization_id) == (
            "RESOLVED_WINNER", "single_realization_full_critical_coverage", "G")
        auth = apply_authoritative_realization_resolution(stamped, ledger, report)
        final = fold_alternates_into_discarded(auth.draft)
        assert [c.clip_id for c in final.selected] == ["G"]
        assert {c.clip_id for c in final.discarded} == {"S1", "E"} and final.alternates == ()


# --- controls: the carried kind never widens what merges -------------------

def test_control_a_semantic_prior_without_a_kind_still_goes_to_the_probe():
    """A plain arbiter confirmation (no restart kind) below D-097.A's 0.90
    component bar is still a semantic bridge: the probe decides, as
    before. (At >= 0.90 into a RESTART-COHESIVE component the EXISTING
    D-097.A rule b accepts it without a probe -- unchanged rule, now
    reachable because the component's own restart kind is known.)"""
    prior = {frozenset(("E", "S1")): (0.85, "same idea per arbiter", ""), frozenset(("E", "G")): PRIOR_E_G}
    declining = RecordedAnswersArbiter({}, declined_probe_texts=(S1_TEXT,), probe_confidence=0.9)
    groups, rows, _ = _cohesion(prior, declining)
    assert groups == (("E", "G"), ("S1",)) and len(declining.probes) == 1
    high = {frozenset(("E", "S1")): (0.95, "same idea per arbiter", ""), frozenset(("E", "G")): PRIOR_E_G}
    accepting = RecordedAnswersArbiter({}, probe_confidence=0.9)
    groups, rows, _ = _cohesion(high, accepting)
    assert groups == (("E", "S1", "G"),) and accepting.probes == []
    assert ("E", "S1", "semantic", True, "semantic_confirmation_against_restart_cohesive_component", None) in rows


def test_control_an_unknown_kind_is_treated_as_a_semantic_prior():
    prior = {frozenset(("E", "S1")): (0.85, "confirmed (not_a_kind)", "not_a_kind"), frozenset(("E", "G")): PRIOR_E_G}
    declining = RecordedAnswersArbiter({}, declined_probe_texts=(S1_TEXT,), probe_confidence=0.9)
    groups, rows, _ = _cohesion(prior, declining)
    assert groups == (("E", "G"), ("S1",))
    assert any(r[:3] == ("E", "S1", "semantic") for r in rows)


def test_control_restart_kind_prior_still_refused_on_a_cross_component_contradiction():
    """The restart path keeps D-085's deterministic contradiction net: a
    newcomer that negates the component's shared proposition never joins."""
    a = _take("A", 0.0, 4.0, "The treatment worked for every patient in the", complete=False)
    b = _take("B", 5.0, 9.0, "The treatment worked for every patient in the trial.")
    c = _take("C", 12.0, 16.0, "The treatment never worked for any patient in the trial.")
    prior = {frozenset(("A", "B")): (1.0, "deterministic restart evidence (incomplete_attempt_completed_by_retry); arbiter not consulted", "incomplete_attempt_completed_by_retry"),
             frozenset(("A", "C")): (1.0, "deterministic restart evidence (measured_pause_bridged_retry); arbiter not consulted", "measured_pause_bridged_retry")}
    arbiter = RecordedAnswersArbiter({}, probe_confidence=0.9)
    groups, diag = split_incohesive_retry_groups((("A", "B", "C"),), (a, b, c), arbiter, prior_confirmations=prior)
    assert ("C",) in groups
    rows = [r for r in diag["edge_trace"] if {r["left_clip_id"], r["right_clip_id"]} == {"A", "C"}]
    assert rows and rows[0]["accepted"] is False and rows[0]["reason_rejected"] == "cross_component_contradiction"


def test_control_restart_kind_prior_still_blocked_by_the_d083_divergence_gate():
    a = _take("A", 0.0, 4.0, "Tuve problemas de estómago y me hicieron una endoscopía", complete=False)
    b = _take("B", 5.0, 9.0, "Tuve problemas de estómago y me hicieron una endoscopía que salió bien.")
    c = _take("C", 12.0, 16.0, "Otro síntoma: tuve problemas de estómago con la piel llena de manchas rojas en los brazos.")
    prior = {frozenset(("A", "B")): (1.0, "r", "incomplete_attempt_completed_by_retry"),
             frozenset(("A", "C")): (1.0, "r", "measured_pause_bridged_retry")}
    groups, diag = split_incohesive_retry_groups((("A", "B", "C"),), (a, b, c), RecordedAnswersArbiter({}), prior_confirmations=prior)
    blocked = {(r["left_clip_id"], r["right_clip_id"]) for r in diag["content_divergence_blocked"]}
    assert ("A", "C") in blocked and ("C",) in groups


# =============================================================================
# 5. QA -- shared content vs presence of the realization
# =============================================================================

def _result(selected_texts, discarded_texts=()):
    return {"selected": [{"clip_id": f"s{i}", "text": t} for i, t in enumerate(selected_texts)],
            "discarded": [{"clip_id": f"d{i}", "text": t} for i, t in enumerate(discarded_texts)]}


def test_qa_required_exact_passes_by_shared_content_while_the_later_take_is_discarded():
    raw124 = _result([F1_TEXT, F2_TEXT, M_TEXT], discarded_texts=[L_TEXT])
    texts = [_norm(r["text"]) for r in raw124["selected"]]
    assert _find_semantic(texts, L_TEXT) is not None  # what `pimples_later_winner_present` measured: coverage 9/14
    assert _find_present_realization([(r["clip_id"], _norm(r["text"])) for r in raw124["selected"]], L_TEXT) is None
    assert realization_present(M_TEXT, L_TEXT) is False and realization_present(L_TEXT, M_TEXT) is False


def test_qa_required_realization_passes_when_the_later_take_is_actually_selected():
    raw122 = _result([F1_TEXT, F2_TEXT, L_TEXT], discarded_texts=[M_TEXT])
    rows = [(r["clip_id"], _norm(r["text"])) for r in raw122["selected"]]
    assert _find_present_realization(rows, L_TEXT) == ("s2", _norm(L_TEXT))
    assert realization_present(L_TEXT, L_TEXT) and realization_present(L_TEXT.replace(".", "").lower(), L_TEXT)
    # the monolith's own presence is still a separate question -- never inferred from the later take
    assert _find_present_realization(rows, M_TEXT) is None


def test_qa_required_realization_kind_in_the_validator_without_touching_the_baseline(tmp_path):
    import json
    from benchmarks.validate_video00_regression_qa import validate
    manifest = {"schema_version": "test", "checks": [
        {"id": "later_present_exact", "kind": "required_exact", "text": L_TEXT},
        {"id": "later_present_realization", "kind": "required_realization", "text": L_TEXT},
        {"id": "monolith_absent", "kind": "forbidden_contains", "text": "También me salían espinillas en esta parte de aquí detrás de la oreja y todo el cuello"},
    ]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "raw124.json").write_text(json.dumps(_result([F1_TEXT, F2_TEXT, M_TEXT], [L_TEXT])), encoding="utf-8")
    ok, report = validate(str(tmp_path / "raw124.json"), str(tmp_path / "manifest.json"))
    assert ok is False and "later_present_exact" in report["passed_checks"]
    assert {f["id"]: f["reason"] for f in report["failed_checks"]} == {
        "later_present_realization": "realization_not_present_only_shared_content", "monolith_absent": "historical_bad_take_returned"}
    (tmp_path / "raw122.json").write_text(json.dumps(_result([F1_TEXT, F2_TEXT, L_TEXT], [M_TEXT])), encoding="utf-8")
    ok, report = validate(str(tmp_path / "raw122.json"), str(tmp_path / "manifest.json"))
    assert ok is True and set(report["passed_checks"]) >= {"later_present_exact", "later_present_realization", "monolith_absent"}
    baseline = json.load(open("benchmarks/video00_regression_qa.json", encoding="utf-8"))
    assert all(c["kind"] != "required_realization" for c in baseline["checks"])  # baseline untouched
