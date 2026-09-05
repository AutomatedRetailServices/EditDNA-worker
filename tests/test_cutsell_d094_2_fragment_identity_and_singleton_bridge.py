"""D-094.2 -- run 33983880111 (b8b11de) live defects.

F7  A completion fragment ("resorcina.") mechanically deleted by the hybrid
    pass was restored by `final_delivery_integrity` by MERGING its text into
    the truncated candidate. The CompositeResolver chain's outer hooks
    rebuild kept/deleted from the SOURCE takes by clip id whenever they fire,
    so the merge was reverted: the kept take ended mid-sentence, the fragment
    fell back into `deleted`, the Ledger recorded a hybrid delete with no
    replacement, the Resolver raised an orphan REVIEW_REQUIRED, Freeze was
    blocked. Both restore guards now restore the fragment as its OWN take.
F3b The D-085 component probe declined ("A || B" vs a truncated retry, 0.2)
    a singleton bridge whose three pairwise confirmations were 0.90-0.95, so
    the D-020 pair (abandoned retry + complete delivery) was split and both
    were kept. Policy-gated (default OFF) singleton clique acceptance.

Generic fixtures; no Video00 clip ids.
"""
from __future__ import annotations

import pytest

import cutsell_worker  # noqa: F401  -- package init wires the chain prerequisites
from cutsell_worker import composite_resolver, hybrid_session_cleanup
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.final_delivery_integrity import restore_immediate_completion_fragments
from cutsell_worker.hybrid_session_cleanup import HybridSessionCleanupResult
from cutsell_worker.semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalenceResult,
    SemanticEquivalenceGatePolicy,
)
from cutsell_worker.take_grouping_provider import split_incohesive_retry_groups
from cutsell_worker.terminal_delivery_reconciliation import restore_tiny_completion_suffixes


def _take(cid, start, end, text, complete=True):
    return CandidateTake(cid, "src", 0, start, end, text, complete_idea=complete)


# ---------------------------------------------------------------------------
# F7: completion-fragment restore preserves identity
# ---------------------------------------------------------------------------

CANDIDATE = "Por temporada me salió un acné en la espalda con la que yo resolvía con"
FRAGMENT = "resorcina."


@pytest.mark.parametrize("guard", [restore_immediate_completion_fragments, restore_tiny_completion_suffixes])
def test_f7_fragment_is_restored_as_its_own_take_never_merged(guard):
    a = _take("A", 100.0, 104.0, CANDIDATE, complete=False)
    b = _take("B", 104.3, 104.9, FRAGMENT, complete=True)
    kept, deleted, diag = guard((a,), (b,), (("A", "failed", 0.92), ("B", "failed", 0.88)), None)
    assert [t.clip_id for t in kept] == ["A", "B"]
    assert kept[0].text == CANDIDATE and kept[0].end == 104.0 and kept[0].complete_idea is False
    assert kept[1] is b  # identity preserved: same object, same span, same text
    assert deleted == ()
    assert diag[0]["restored_fragment_id"] == "B" and diag[0]["restored_as"] == "separate_take"
    assert diag[0]["restored_text"] == f"{CANDIDATE} {FRAGMENT}"


def _stub_base(kept_ids, deleted_ids, decisions):
    def base(takes_in, context, judge, **kw):
        takes_in = tuple(takes_in)
        kept = tuple(t for t in takes_in if t.clip_id in kept_ids)
        deleted = tuple(t for t in takes_in if t.clip_id in deleted_ids)
        rows = [{
            "clip_id": c, "label": l, "confidence": k, "reason_code": "",
            "local_failure_corroborated": c in deleted_ids,
            "local_failure_reasons": ["dense_physical_reset:7"] if c in deleted_ids else [],
            "delete_basis": "micro_failed_plus_local_performance" if c in deleted_ids else "kept_fail_open",
            "applied_delete": c in deleted_ids, "semantic_delete_recommended": False,
        } for c, l, k in decisions]
        diag = ({"partition_index": 0, "chunk_index": 0, "member_ids": [t.clip_id for t in takes_in],
                 "requested": True, "available": True, "provider": "google", "model": "m",
                 "deleted_ids": list(deleted_ids), "decisions": rows},)
        return HybridSessionCleanupResult(kept, deleted, 1, 1, diag, tuple(decisions))
    return base


def test_f7_fragment_survives_the_full_composite_resolver_chain_when_an_outer_hook_fires(monkeypatch):
    """Live shape: hook 12 restores the fragment, a LATER hook fires and
    rebuilds kept/deleted from the source takes. The fragment must still be
    kept (own identity), the candidate untouched, nothing in deleted."""
    h = _take("H", 0.0, 3.0, "Tenía cáncer de tiroides y no lo sabía.")
    a = _take("A", 100.0, 104.0, CANDIDATE, complete=False)
    b = _take("B", 104.3, 104.9, FRAGMENT)
    c = _take("C", 110.0, 116.0, "También me salían espinillas detrás de la oreja y en el cuello que yo pensaba que era alergia.")
    d = _take("D", 120.0, 125.0, "Se me caía mucho el pelo cuando me lavaba y pensaba que era por el estrés.")
    # X: deleted by the base pass but carrying conflicting winner 0.95 / failed
    # 0.80 labels -> hybrid_semantic_conflict_arbitration (the OUTERMOST hook)
    # restores it and rebuilds kept/deleted from the source takes.
    x = _take("X", 130.0, 134.0, "Esta es mi experiencia y por eso cuídate y hazte tus chequeos.")
    takes = (h, a, b, c, d, x)
    decisions = (("H", "winner", 0.95), ("A", "failed", 0.92), ("B", "failed", 0.88), ("C", "winner", 0.96),
                 ("D", "winner", 0.95), ("X", "failed", 0.80), ("X", "winner", 0.95))
    monkeypatch.setattr(hybrid_session_cleanup, "apply_hybrid_session_cleanup",
                        _stub_base({"H", "A", "C", "D"}, {"B", "X"}, decisions))
    monkeypatch.setattr(composite_resolver, "_take_level_chain", None)
    result, _split_ids = composite_resolver.apply_composite_resolution(takes, None, object())
    monkeypatch.setattr(composite_resolver, "_take_level_chain", None)

    kept_by_id = {t.clip_id: t for t in result.kept}
    assert "B" in kept_by_id and kept_by_id["B"].text == FRAGMENT and kept_by_id["B"].end == 104.9
    assert kept_by_id["A"].text == CANDIDATE and kept_by_id["A"].end == 104.0
    assert "X" in kept_by_id  # proves the outer hook actually fired
    assert [t.clip_id for t in result.deleted] == []
    # Identity invariant across the whole chain: every kept take IS a source take.
    source_by_id = {t.clip_id: t for t in takes}
    for take in result.kept:
        assert take.text == source_by_id[take.clip_id].text and take.end == source_by_id[take.clip_id].end
    restore_rows = [row for diag in result.diagnostics for row in (diag.get("final_delivery_integrity") or [])]
    assert any(r.get("restored_fragment_id") == "B" and r.get("restored_as") == "separate_take" for r in restore_rows)


# ---------------------------------------------------------------------------
# F3b: singleton clique bridge (policy-gated)
# ---------------------------------------------------------------------------

ABANDONED = "Tuve problemas estomacales a un tiempo en donde se me hizo una endoscopía y me diagnosticaron con..."
ASIDE = "Tuve problemas de estómago en una temporada, en 2023, no hay que preguntar."
COMPLETE = "Tuve problemas de digestión en donde me hicieron una endoscopía y dijeron que tenía gastritis. Nada severo pero tenía gastritis."


class PairwiseYesComponentNoArbiter:
    """Confirms every plain pair (order-insensitive table); declines every
    component-level probe (any text containing ' || ') at 0.2 -- the exact
    live shape of run 33983880111's gastritis family."""

    def __init__(self, table):
        self.table = table
        self.pairs_asked = []

    def check(self, request):
        out = []
        for i, pair in enumerate(request.pairs):
            self.pairs_asked.append((pair.left_text, pair.right_text))
            if " || " in pair.left_text or " || " in pair.right_text:
                out.append(IdeaEquivalenceDecision(i, False, 0.2, "component probe declined"))
                continue
            entry = self.table.get((pair.left_text, pair.right_text)) or self.table.get((pair.right_text, pair.left_text))
            same, conf, reason = entry if entry else (False, 0.0, "unconfigured")
            out.append(IdeaEquivalenceDecision(i, same, conf, reason))
        return IdeaEquivalenceResult(tuple(out), "fake", "fake", True, True, 50, 10)


def _gastritis():
    takes = (_take("G1", 100.0, 106.0, ABANDONED, complete=False), _take("G2", 107.0, 111.0, ASIDE), _take("G3", 112.0, 120.0, COMPLETE))
    table = {(ABANDONED, ASIDE): (True, 0.9, "same stomach story"), (ABANDONED, COMPLETE): (True, 0.95, "same story"),
             (ASIDE, COMPLETE): (True, 0.95, "same story")}
    return takes, table


def test_f3b_default_off_reproduces_the_live_split_shape():
    takes, table = _gastritis()
    arbiter = PairwiseYesComponentNoArbiter(table)
    groups, diag = split_incohesive_retry_groups((("G1", "G2", "G3"),), takes, arbiter)
    assert len(groups) == 2  # the probe's 0.2 wins over three >= 0.90 pairwise answers
    assert any(r.get("reason_rejected") == "component_cohesion_declined" for r in diag["edge_trace"])
    assert any(" || " in l or " || " in r for l, r in arbiter.pairs_asked)


def test_f3b_policy_on_singleton_clique_is_accepted_without_the_component_probe():
    takes, table = _gastritis()
    arbiter = PairwiseYesComponentNoArbiter(table)
    policy = SemanticEquivalenceGatePolicy(accept_complete_pairwise_singleton_bridge=True)
    groups, diag = split_incohesive_retry_groups((("G1", "G2", "G3"),), takes, arbiter, policy=policy)
    assert len(groups) == 1 and set(groups[0]) == {"G1", "G2", "G3"}
    assert not any(" || " in l or " || " in r for l, r in arbiter.pairs_asked)  # no paid probe
    bridge = next(r for r in diag["edge_trace"] if r.get("bridge_sensitive"))
    assert bridge["accepted"] is True and bridge["accepted_by"] == "complete_pairwise_confirmation"
    assert bridge["component_cohesion_evaluated"] is False
    assert len(bridge["cross_pair_confirmations"]) == 2
    assert all(c["confidence"] >= 0.9 for c in bridge["cross_pair_confirmations"])
    assert diag["component_semantic_call_count"] == 0


def test_f3b_policy_on_incomplete_pairwise_evidence_still_goes_through_the_probe():
    takes, table = _gastritis()
    del table[(ABANDONED, ASIDE)]  # G1<->G2 never confirmed -> clique incomplete
    arbiter = PairwiseYesComponentNoArbiter(table)
    policy = SemanticEquivalenceGatePolicy(accept_complete_pairwise_singleton_bridge=True)
    groups, diag = split_incohesive_retry_groups((("G1", "G2", "G3"),), takes, arbiter, policy=policy)
    assert len(groups) == 2
    assert any(r.get("component_cohesion_evaluated") for r in diag["edge_trace"])


def test_f3b_policy_on_below_floor_pairwise_confidence_still_goes_through_the_probe():
    takes, table = _gastritis()
    table[(ABANDONED, ASIDE)] = (True, 0.85, "weak")  # D-084's false-bridge band
    arbiter = PairwiseYesComponentNoArbiter(table)
    policy = SemanticEquivalenceGatePolicy(accept_complete_pairwise_singleton_bridge=True)
    groups, diag = split_incohesive_retry_groups((("G1", "G2", "G3"),), takes, arbiter, policy=policy)
    assert len(groups) == 2
    assert any(r.get("component_cohesion_evaluated") for r in diag["edge_trace"])


def test_f3b_policy_on_cross_component_contradiction_still_rejects_the_clique():
    left = "Me mandaron tres meses con pastillas y no fue nada severo."
    m1 = "Me mandaron seis meses con pastillas y sí fue severo."
    m2 = "Estuve seis meses con pastillas, fue severo."
    takes = (_take("L", 100.0, 104.0, left), _take("M1", 105.0, 109.0, m1), _take("M2", 110.0, 114.0, m2))
    table = {(left, m1): (True, 0.95, "x"), (left, m2): (True, 0.95, "x"), (m1, m2): (True, 0.96, "x")}
    arbiter = PairwiseYesComponentNoArbiter(table)
    policy = SemanticEquivalenceGatePolicy(accept_complete_pairwise_singleton_bridge=True)
    groups, diag = split_incohesive_retry_groups((("L", "M1", "M2"),), takes, arbiter, policy=policy)
    assert len(groups) == 2
    bridge = next(r for r in diag["edge_trace"] if r.get("bridge_sensitive"))
    assert bridge["accepted"] is False and bridge["reason_rejected"] == "cross_component_contradiction"
    assert bridge["accepted_by"] == "complete_pairwise_confirmation"


def test_f3b_policy_on_component_to_component_bridge_always_keeps_the_probe():
    a1, a2 = "Primera idea sobre la tiroides dicha una vez.", "Primera idea sobre la tiroides dicha otra vez."
    b1, b2 = "Segunda idea sobre el pelo dicha una vez.", "Segunda idea sobre el pelo dicha otra vez."
    takes = (_take("a1", 1.0, 3.0, a1), _take("a2", 4.0, 6.0, a2), _take("b1", 7.0, 9.0, b1), _take("b2", 10.0, 12.0, b2))
    table = {(a1, a2): (True, 0.98, "s"), (b1, b2): (True, 0.98, "s"), (a1, b1): (True, 0.95, "s"), (a1, b2): (True, 0.95, "s"),
             (a2, b1): (True, 0.95, "s"), (a2, b2): (True, 0.95, "s")}
    arbiter = PairwiseYesComponentNoArbiter(table)
    policy = SemanticEquivalenceGatePolicy(accept_complete_pairwise_singleton_bridge=True)
    groups, diag = split_incohesive_retry_groups((("a1", "a2", "b1", "b2"),), takes, arbiter, policy=policy)
    assert {frozenset(g) for g in groups} == {frozenset({"a1", "a2"}), frozenset({"b1", "b2"})}
    assert any(r.get("component_cohesion_evaluated") and not r.get("accepted") for r in diag["edge_trace"])


def test_f3b_pipeline_reads_the_flag_from_env(monkeypatch):
    import cutsell_worker.pipeline as pipeline
    monkeypatch.delenv("CUTSELL_BRIDGE_COMPLETE_PAIRWISE_SINGLETON", raising=False)
    assert pipeline._env_flag_enabled("CUTSELL_BRIDGE_COMPLETE_PAIRWISE_SINGLETON") is False
    monkeypatch.setenv("CUTSELL_BRIDGE_COMPLETE_PAIRWISE_SINGLETON", "1")
    assert pipeline._env_flag_enabled("CUTSELL_BRIDGE_COMPLETE_PAIRWISE_SINGLETON") is True
    monkeypatch.setenv("CUTSELL_BRIDGE_COMPLETE_PAIRWISE_SINGLETON", "0")
    assert pipeline._env_flag_enabled("CUTSELL_BRIDGE_COMPLETE_PAIRWISE_SINGLETON") is False
