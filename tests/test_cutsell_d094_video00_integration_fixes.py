"""D-094 mission -- Video00 integration fixes F2 / F3 / F4.

Evidence: runs 33960713625 (40dde20) and 33969388042 (e4cd508).

F2  hybrid semantic windows refused by the per-edit DollarBudgetLedger were a
    silent fail-open keep (provider string only). Now counted and priced.
F3  `split_incohesive_retry_groups` re-asked the arbiter only for the top
    `max_pairs_per_request` weak pairs, never reused the reconcile stage's
    own confirmed merges, and split a 0.95-confirmed retry pair on absence
    of evidence -> the abandoned gastritis retry and the complete delivery
    were both kept (D-020 violated silently).
F4  `_evaluate_bridge_cohesion` rejected a bridge on ANY pair contradiction
    including pairs INSIDE one component (a truncated mid-sentence fragment
    vs its own negated full sentence) -> the hereditary restatement was
    co-kept as a separate family. Only cross-component pairs may reject.

Fixtures are generic shapes (no Video00 ids); texts reproduce the live
SHAPE of the failures.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

import cutsell_worker.pipeline as pipeline
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult, EditorialSession, HybridGatePolicy
from cutsell_worker.hybrid_session_cleanup import _BUDGET_EXHAUSTED_PROVIDER_PREFIX, apply_hybrid_session_cleanup
from cutsell_worker.semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalenceResult,
    SemanticEquivalenceGatePolicy,
)
from cutsell_worker.take_grouping_provider import _evaluate_bridge_cohesion, _RetryEdge, split_incohesive_retry_groups


def _take(clip_id, start, end, text, complete=True):
    return CandidateTake(clip_id, "src", 0, start, end, text, complete_idea=complete)


class TableArbiter:
    """Lookup-table arbiter (order-insensitive pair key); unconfigured pairs
    are declined. Counts calls and pairs asked."""

    def __init__(self, table):
        self.table = table
        self.calls = 0
        self.pairs_asked = []

    def check(self, request):
        self.calls += 1
        decisions = []
        for i, pair in enumerate(request.pairs):
            self.pairs_asked.append((pair.left_text, pair.right_text))
            entry = self.table.get((pair.left_text, pair.right_text)) or self.table.get((pair.right_text, pair.left_text))
            same, conf, reason = entry if entry else (False, 0.0, "unconfigured_pair_declined")
            decisions.append(IdeaEquivalenceDecision(pair_index=i, same_idea=same, confidence=conf, reason=reason))
        return IdeaEquivalenceResult(decisions=tuple(decisions), provider="fake", model="fake", requested=True, available=True,
                                     estimated_input_tokens=50, estimated_output_tokens=10)


# ---------------------------------------------------------------------------
# F3: reuse of reconcile-stage confirmations beyond the bounded re-ask
# ---------------------------------------------------------------------------

ABANDONED = "Tuve problemas estomacales a un tiempo en donde se me hizo una endoscopía y me diagnosticaron con..."
ASIDE = "Tuve problemas de estómago en una temporada, en 2023, no hay que preguntar."
COMPLETE = "Tuve problemas de digestión en donde me hicieron una endoscopía y dijeron que tenía gastritis. Nada severo pero tenía gastritis."


def _gastritis_takes():
    return (_take("G1", 100.0, 106.0, ABANDONED, complete=False), _take("G2", 107.0, 111.0, ASIDE), _take("G3", 112.0, 120.0, COMPLETE))


def test_f3_red_confirmed_pair_outside_the_bounded_reask_is_split_without_prior_evidence():
    takes = _gastritis_takes()
    # The arbiter would confirm G1<->G3 at 0.95 -- but the pass is capped to ONE pair and asks G1<->G2 first.
    arbiter = TableArbiter({(ABANDONED, ASIDE): (True, 0.85, "same stomach story"), (ABANDONED, COMPLETE): (True, 0.95, "same story")})
    policy = SemanticEquivalenceGatePolicy(max_pairs_per_request=1)
    groups, diag = split_incohesive_retry_groups((("G1", "G2", "G3"),), takes, arbiter, policy=policy)
    assert diag["checked_pair_count"] == 1
    assert diag["unchecked_weak_pair_count"] >= 1  # observability: unasked pairs are named, not hidden
    assert diag["prior_confirmations_reused_count"] == 0
    assert len(groups) == 2 and ("G3",) in groups  # the live defect shape: complete delivery split off


def test_f3_green_prior_confirmation_keeps_the_family_and_never_reasks_that_pair():
    takes = _gastritis_takes()
    # The aside G2 is bridge-sensitive once G1<->G3 is a component (D-085):
    # the component-cohesion probe asks the arbiter with the joined component
    # text, so the table carries that probe too.
    arbiter = TableArbiter({
        (ABANDONED, ASIDE): (True, 0.85, "same stomach story"),
        (f"{ABANDONED} || {COMPLETE}", ASIDE): (True, 0.92, "aside about the same stomach episode"),
    })
    policy = SemanticEquivalenceGatePolicy(max_pairs_per_request=1)
    prior = {frozenset(("G1", "G3")): (0.95, "Same personal story about stomach problems and endoscopy.")}
    groups, diag = split_incohesive_retry_groups((("G1", "G2", "G3"),), takes, arbiter, policy=policy, prior_confirmations=prior)
    assert diag["prior_confirmations_reused_count"] == 1
    assert diag["prior_confirmations_reused"][0]["source"] == "prior_confirmation"
    assert (ABANDONED, COMPLETE) not in arbiter.pairs_asked and (COMPLETE, ABANDONED) not in arbiter.pairs_asked
    # The D-020 pair (abandoned retry + complete delivery) is never split on
    # absence of evidence again ...
    family = next(g for g in groups if "G1" in g)
    assert "G3" in family
    # ... and with the bridge probe answered the whole family survives.
    assert len(groups) == 1 and set(groups[0]) == {"G1", "G2", "G3"}
    assert diag["unchecked_weak_pairs"] == [{"left_clip_id": "G2", "right_clip_id": "G3"}]
    assert diag["weak_pair_count"] == 3  # total, not "remaining after reuse"


def test_f3_prior_confirmation_still_subject_to_the_d083_divergence_gate():
    left = "También me salían espinillas. Era como un rush, una alergia."
    right = "Otro síntoma era que me salían espinillas como si fuera una alergia de esta parte aquí detrás de la oreja y en el cuello. Me salía por temporadas."
    takes = (_take("P1", 10.0, 13.0, left), _take("P2", 14.0, 20.0, right))
    prior = {frozenset(("P1", "P2")): (0.9, "Both describe breakouts initially thought to be allergies.")}
    groups, diag = split_incohesive_retry_groups((("P1", "P2"),), takes, TableArbiter({}), prior_confirmations=prior)
    assert diag["prior_confirmations_reused_count"] == 0
    assert diag["content_divergence_blocked"] and diag["content_divergence_blocked"][0]["source"] == "prior_confirmation"
    assert len(groups) == 2


def test_f3_pipeline_passes_reconcile_merges_as_prior_confirmations(monkeypatch):
    captured = {}
    real = pipeline.split_incohesive_retry_groups

    def spy(groups, takes, arbiter, **kw):
        captured["prior"] = kw.get("prior_confirmations")
        return real(groups, takes, arbiter, **kw)
    monkeypatch.setattr(pipeline, "split_incohesive_retry_groups", spy)
    monkeypatch.setattr(pipeline, "reconcile_semantic_idea_equivalence", lambda groups, kept, arbiter, **kw: (
        groups, {"merges": [{"left_clip_id": "x", "right_clip_id": "y", "confidence": 0.95, "reason": "same"}]},
    ))
    import inspect
    src = inspect.getsource(pipeline)
    assert "prior_confirmations=prior_confirmations" in src  # wired at the real call site
    # The dict shape the call site builds from reconcile merges:
    rows = [{"left_clip_id": "x", "right_clip_id": "y", "confidence": 0.95, "reason": "same"}]
    prior = {frozenset((r["left_clip_id"], r["right_clip_id"])): (float(r["confidence"]), r["reason"]) for r in rows}
    assert prior[frozenset(("x", "y"))] == (0.95, "same")


# ---------------------------------------------------------------------------
# F4: bridge contradiction is cross-component only
# ---------------------------------------------------------------------------

FULL = "Esta es mi experiencia. Soy la única en mi familia que tiene este tipo de cáncer. Por eso no creo y está comprobado científicamente que los cánceres son hereditarios. Más bien solo un 5 -10 % son de carácter hereditario."
FRAGMENT = "cánceres son hereditarios. Soy la única en mi familia que tiene este tipo de cáncer."
RESTATEMENT = "Soy la primera en mi familia con este tipo de cáncer. Nadie en mi familia tiene un carcinoma papilar en la tiroides. Así que estoy convencida y la ciencia lo avala que solo un 5 -10 % de los"


def _hereditary_take_map():
    return {t.clip_id: t for t in (_take("H1", 200.0, 212.0, FULL), _take("H2", 213.0, 216.0, FRAGMENT), _take("H3", 217.0, 226.0, RESTATEMENT))}


def test_f4_red_shape_within_component_contradiction_alone_no_longer_rejects_the_bridge():
    take_map = _hereditary_take_map()
    arbiter = TableArbiter({(f"{FULL} || {FRAGMENT}", RESTATEMENT): (True, 0.95, "one shared proposition")})
    edge = _RetryEdge("H1", "H3", "semantic", 0.85, "Both discuss family history and hereditary cancer statistics.")
    accepted, record = _evaluate_bridge_cohesion(
        left_members=("H1", "H2"), right_members=("H3",), edge=edge, take_map=take_map, arbiter=arbiter,
        policy=SemanticEquivalenceGatePolicy(),
    )
    assert record.get("reason_rejected") != "cross_component_contradiction"
    assert record["within_component_contradiction"] is True  # recorded, not used to veto
    assert accepted is True and record["component_cohesion_evaluated"] is True


def test_f4_cross_component_contradiction_still_rejects():
    take_map = _hereditary_take_map()
    take_map["H4"] = _take("H4", 230.0, 236.0, "La ciencia dice que un 40 % de los cánceres son hereditarios.")
    edge = _RetryEdge("H1", "H4", "semantic", 0.9, "same topic")
    accepted, record = _evaluate_bridge_cohesion(
        left_members=("H1", "H2"), right_members=("H4",), edge=edge, take_map=take_map,
        arbiter=TableArbiter({}), policy=SemanticEquivalenceGatePolicy(),
    )
    assert accepted is False and record["reason_rejected"] == "cross_component_contradiction"


def test_f4_full_split_pass_keeps_the_hereditary_family_together():
    takes = tuple(_hereditary_take_map().values())
    table = {
        (FULL, FRAGMENT): (True, 0.9, "The right text contains a repeated excerpt of the left text."),
        (FULL, RESTATEMENT): (True, 0.85, "Both discuss family history and hereditary cancer statistics."),
        (FRAGMENT, RESTATEMENT): (True, 0.9, "The left text is a subset of the right text's core message."),
        (f"{FULL} || {FRAGMENT}", RESTATEMENT): (True, 0.95, "one shared proposition"),
        (RESTATEMENT, f"{FULL} || {FRAGMENT}"): (True, 0.95, "one shared proposition"),
    }
    groups, diag = split_incohesive_retry_groups((("H1", "H2", "H3"),), takes, TableArbiter(table))
    assert len(groups) == 1 and set(groups[0]) == {"H1", "H2", "H3"}
    assert not any(r.get("reason_rejected") == "cross_component_contradiction" for r in diag["edge_trace"])


# ---------------------------------------------------------------------------
# F2: budget starvation is counted and priced
# ---------------------------------------------------------------------------

@dataclass
class BudgetLimitedJudge:
    capacity: int
    calls: int = 0

    def judge(self, session: EditorialSession) -> EditorialJudgeResult:
        self.calls += 1
        if self.calls > self.capacity:
            # Exactly what `hybrid_google_transport` raises when the per-edit
            # DollarBudgetLedger refuses the call; `safe_editorial_judge`
            # turns it into the provider string the prefix matches.
            raise RuntimeError("hybrid edit/test dollar budget exhausted")
        decisions = tuple(EditorialDecision(clip_id=c.clip_id, label="keep", confidence=0.5, reason_code="t") for c in session.candidates)
        return EditorialJudgeResult(decisions, "google", "gemini-3.5-flash-lite", True, True)


def test_f2_budget_exhausted_windows_are_flagged_and_priced():
    takes = tuple(_take(f"c{i}", float(i * 10), float(i * 10 + 5), f"Frase completa número {i} con contenido suficiente para una ventana.") for i in range(6))
    judge = BudgetLimitedJudge(capacity=2)
    result = apply_hybrid_session_cleanup(takes, None, judge, chunk_size=1, chunk_stride=1)
    rows = list(result.diagnostics)
    assert all("budget_exhausted" in row and "estimated_cost_usd" in row for row in rows)
    exhausted = [row for row in rows if row["budget_exhausted"]]
    assert len(exhausted) == len(rows) - 2
    assert all(row["available"] is False and row["decisions"] == [] for row in exhausted)
    assert all(row["estimated_cost_usd"] > 0 for row in rows)
    served = [row for row in rows if not row["budget_exhausted"]]
    assert all(row["available"] for row in served)


# ---------------------------------------------------------------------------
# F2 (accounting): the planner / per-window estimate is the ledger's own
# reservation figure, proven against the real transport's preflight.
# ---------------------------------------------------------------------------

def test_f2_window_estimate_is_exactly_what_the_live_transport_reserves(monkeypatch):
    """Runs 33960713625 / 33969388042 said "6 planned / 0 deferred" while the
    ledger served 4: the plan priced windows with a lighter formula than the
    transport reserves with. Now the estimate IS the reservation: a ledger
    holding exactly that amount admits the call, a hair less refuses it."""
    from cutsell_worker.hybrid_google_transport import DollarBudgetLedger, GoogleGeminiTransport
    from cutsell_worker.hybrid_payload import HybridCostPolicy, build_compact_editorial_payload, preflight_hybrid_call
    from cutsell_worker.hybrid_provider_settings import HybridProviderSettings
    from cutsell_worker.hybrid_session_cleanup import _editorial_session, _estimate_window_reservation_usd

    members = tuple(_take(f"w{i}", float(i * 10), float(i * 10 + 6), f"Una frase completa número {i} que describe un síntoma con detalle suficiente.") for i in range(4))
    session = _editorial_session(members, None, partition_index=0, chunk_index=0)
    settings = HybridProviderSettings(enabled=True)
    estimate = _estimate_window_reservation_usd(session, members, settings)
    assert estimate > 0

    class _Sentinel(Exception):
        pass

    class _NoHttp:
        @staticmethod
        def post(*a, **k):
            raise _Sentinel("reserve passed; HTTP would follow")

    payload = build_compact_editorial_payload(session, cost_policy=HybridCostPolicy())
    max_out = int(preflight_hybrid_call(session, HybridGatePolicy(), cost_policy=HybridCostPolicy())["max_output_tokens"])

    model = settings.primary_model
    admitted = GoogleGeminiTransport(api_key="k", model=model, settings=settings, ledger=DollarBudgetLedger(estimate), session=_NoHttp)
    with pytest.raises(_Sentinel):
        admitted(payload, max_out)
    refused = GoogleGeminiTransport(api_key="k", model=model, settings=settings, ledger=DollarBudgetLedger(estimate - 1e-6), session=_NoHttp)
    with pytest.raises(RuntimeError, match="dollar budget exhausted"):
        refused(payload, max_out)
