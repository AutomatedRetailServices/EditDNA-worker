"""D-052 Part B: semantic compute planner + priority-ordered dispatch.

See docs/CUTSELL_DECISIONS.md D-051 (audit: a fixed, order-consumed dollar
ledger starves whichever chunk is requested Nth, with zero regard for
content) / D-052 (this fix). Covers:
- semantic_compute_planner.py unit tests (Section 13's 9 required cases)
- hybrid_session_cleanup.py's CUTSELL_SEMANTIC_COMPUTE_PLANNER flag: OFF by
  default (parity with pre-D-052 behavior), and when ON, P0 work is never
  starved by original chunk position.
"""
from __future__ import annotations

from dataclasses import dataclass

from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_editorial import (
    EditorialDecision,
    EditorialJudge,
    EditorialJudgeResult,
    EditorialSession,
)
from cutsell_worker.hybrid_session_cleanup import apply_hybrid_session_cleanup
from cutsell_worker.semantic_compute_planner import (
    SemanticWorkItem,
    SemanticWorkPriority,
    build_cost_contract_report,
    build_semantic_compute_plan,
)

P0 = SemanticWorkPriority.P0_SAFETY_CRITICAL
P1 = SemanticWorkPriority.P1_RETRY_EQUIVALENCE
P2 = SemanticWorkPriority.P2_EDITORIAL_QUALITY


def _item(work_id: str, priority: SemanticWorkPriority, cost: float = 0.001) -> SemanticWorkItem:
    return SemanticWorkItem(work_id=work_id, priority=priority, estimated_cost_usd=cost, reason="test")


# ---------------------------------------------------------------------------
# Section 13: semantic_compute_planner.py unit tests
# ---------------------------------------------------------------------------

def test_four_work_items_sufficient_budget_all_planned():
    items = [_item(str(i), P2, 0.001) for i in range(4)]
    plan = build_semantic_compute_plan(items, cost_ceiling_usd=0.01)
    assert set(plan.planned_calls) == {"0", "1", "2", "3"}
    assert plan.deferred_optional_calls == ()


def test_six_work_items_sufficient_budget_all_planned():
    items = [_item(str(i), P2, 0.001) for i in range(6)]
    plan = build_semantic_compute_plan(items, cost_ceiling_usd=0.01)
    assert len(plan.planned_calls) == 6
    assert plan.deferred_optional_calls == ()


def test_six_work_items_constrained_budget_defers_the_rest():
    items = [_item(str(i), P2, 0.002) for i in range(6)]
    plan = build_semantic_compute_plan(items, cost_ceiling_usd=0.0075)
    # 0.0075 / 0.002 -> exactly 3 fit (0.006 <= 0.0075, a 4th would be 0.008 > 0.0075)
    assert len(plan.planned_calls) == 3
    assert len(plan.deferred_optional_calls) == 3


def test_p0_always_reserved_before_p2_even_when_p0_arrives_last():
    items = [_item("p2_a", P2, 0.003), _item("p2_b", P2, 0.003), _item("p2_c", P2, 0.003), _item("p0_last", P0, 0.003)]
    plan = build_semantic_compute_plan(items, cost_ceiling_usd=0.006)
    assert "p0_last" in plan.planned_calls
    assert plan.planned_calls[0] == "p0_last"
    assert "p2_c" in plan.deferred_optional_calls


def test_changing_input_order_does_not_change_which_priority_items_execute():
    items_order_a = [_item("a", P0), _item("b", P1), _item("c", P2), _item("d", P2)]
    items_order_b = [_item("d", P2), _item("c", P2), _item("b", P1), _item("a", P0)]
    plan_a = build_semantic_compute_plan(items_order_a, cost_ceiling_usd=0.0025)
    plan_b = build_semantic_compute_plan(items_order_b, cost_ceiling_usd=0.0025)
    assert set(plan_a.planned_calls) == set(plan_b.planned_calls)
    assert set(plan_a.deferred_optional_calls) == set(plan_b.deferred_optional_calls)


def test_estimated_cost_ceiling_respected():
    items = [_item(str(i), P2, 0.01) for i in range(10)]
    plan = build_semantic_compute_plan(items, cost_ceiling_usd=0.025)
    assert plan.planned_cost_usd <= plan.cost_ceiling_usd
    assert len(plan.planned_calls) == 2


def test_p0_exhaustion_fails_closed():
    items = [_item("p0_a", P0, 0.01), _item("p0_b", P0, 0.01)]
    plan = build_semantic_compute_plan(items, cost_ceiling_usd=0.01)
    deferred_p0 = plan.outcome_for("p0_b")
    assert deferred_p0 is not None
    assert deferred_p0.planned is False
    assert deferred_p0.exhaustion_action == "REVIEW_REQUIRED_fail_closed"


def test_p2_exhaustion_preserves_content_via_keep_fallback_action():
    items = [_item("p2_a", P2, 0.01), _item("p2_b", P2, 0.01)]
    plan = build_semantic_compute_plan(items, cost_ceiling_usd=0.01)
    deferred_p2 = plan.outcome_for("p2_b")
    assert deferred_p2 is not None
    assert deferred_p2.planned is False
    assert deferred_p2.exhaustion_action == "KEEP_deterministic_fallback"


def test_no_fifth_call_positional_starvation():
    """The exact D-051 shape: 6 items requested in plain order, a $0.0075
    budget that only fits ~4 -- but item #5 (index 4, zero-based) is P0.
    Under the old call-order model it would have been silently starved
    purely by position; the planner must reserve it regardless."""
    items = [_item(str(i), P2, 0.0015) for i in range(6)]
    items[4] = _item("4", P0, 0.0015)  # the "fifth call" is safety-critical
    plan = build_semantic_compute_plan(items, cost_ceiling_usd=0.0075)
    assert "4" in plan.planned_calls
    # P0 is reserved first, before any P2 competes for the remaining budget.
    assert plan.planned_calls[0] == "4"


def test_cost_contract_report_shape():
    items = [_item("a", P0, 0.002), _item("b", P2, 0.002)]
    plan = build_semantic_compute_plan(items, cost_ceiling_usd=0.003)
    report = build_cost_contract_report(plan)
    assert report["estimated_semantic_cost_usd"] == plan.planned_cost_usd
    assert "actual_semantic_cost_usd" not in report
    report_after = build_cost_contract_report(plan, actual_cost_usd=0.0021)
    assert report_after["actual_semantic_cost_usd"] == 0.0021


# ---------------------------------------------------------------------------
# hybrid_session_cleanup.py integration: flag OFF parity + P0 starvation fix
# ---------------------------------------------------------------------------

@dataclass
class _RecordingBudgetLimitedJudge:
    """Fake EditorialJudge: succeeds for the first `capacity` calls (in the
    order .judge() is actually invoked), then fails open exactly like a
    real DollarBudgetLedger exhaustion would (available=False)."""
    capacity: int
    call_order: list = None

    def __post_init__(self):
        if self.call_order is None:
            self.call_order = []

    def judge(self, session: EditorialSession) -> EditorialJudgeResult:
        self.call_order.append(session.session_id)
        if len(self.call_order) > self.capacity:
            return EditorialJudgeResult((), "google", "gemini-3.5-flash-lite", True, False)
        decisions = tuple(
            EditorialDecision(clip_id=c.clip_id, label="keep", confidence=0.5, reason_code="test")
            for c in session.candidates
        )
        return EditorialJudgeResult(decisions, "google", "gemini-3.5-flash-lite", True, True)


def _take(clip_id: str, start: float, text: str, source_asset_id: str = "src1") -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id=source_asset_id,
        source_order=0,
        start=start,
        end=start + 1.0,
        text=text,
        complete_idea=True,
    )


def _make_takes_with_negation_conflict_in_position(position: int, total: int) -> tuple[CandidateTake, ...]:
    """Build `total` single-member windows (chunk_size=1 forces one window
    per take) where exactly the take at `position` carries a stand-alone
    negation-conflict pair against its neighbor -- i.e. genuinely P0 -- and
    everything else is ordinary P2 filler text."""
    takes = []
    for i in range(total):
        if i == position:
            text = "No tengo dolor de estomago en esta zona hoy"
        elif i == position + 1 if position + 1 < total else False:
            text = "Si tengo dolor de estomago en esta zona hoy"
        else:
            text = f"clip de relleno ordinario numero {i} sin nada especial"
        takes.append(_take(f"clip{i}", float(i) * 10.0, text))
    return tuple(takes)


def test_flag_off_preserves_plain_chunk_index_order():
    takes = tuple(_take(f"clip{i}", float(i) * 10.0, f"contenido {i}") for i in range(4))
    judge = _RecordingBudgetLimitedJudge(capacity=100)
    result = apply_hybrid_session_cleanup(
        takes, None, judge, chunk_size=1, chunk_stride=1, env={},
    )
    assert result.semantic_compute_plan is None
    diag_positions = [(d["partition_index"], d["chunk_index"]) for d in result.diagnostics]
    assert diag_positions == sorted(diag_positions)
    assert all(d["planner_priority"] is None for d in result.diagnostics)


def test_flag_on_p0_window_is_attempted_before_later_p2_windows_regardless_of_original_position():
    # chunk_size=2/stride=2 -> 3 non-overlapping two-member windows
    # (chunk_index 0: clips 0-1, 1: clips 2-3, 2: clips 4-5). Put the
    # negation-conflict PAIR in the LAST window -- exactly the D-051
    # "later chunk loses purely by position" shape -- and give the judge
    # only enough capacity for 2 of the 3 windows to succeed (mirrors the
    # real $0.0075 -> a handful of calls shape).
    filler = "clip de relleno ordinario sin nada especial que decir aqui hoy"
    takes = (
        _take("clip0", 0.0, filler),
        _take("clip1", 10.0, filler),
        _take("clip2", 20.0, filler),
        _take("clip3", 30.0, filler),
        _take("clip4", 40.0, "No tengo dolor de estomago en esta zona para nada"),
        _take("clip5", 50.0, "Si tengo dolor de estomago en esta zona todo el tiempo"),
    )

    judge_off = _RecordingBudgetLimitedJudge(capacity=2)
    result_off = apply_hybrid_session_cleanup(
        takes, None, judge_off, chunk_size=2, chunk_stride=2, env={"CUTSELL_SEMANTIC_COMPUTE_PLANNER": "0"},
    )
    # Flag off: plain order means the negation-conflict window (chunk_index
    # 2, requested 3rd) starves under a 2-call capacity.
    off_diag_by_chunk = {d["chunk_index"]: d for d in result_off.diagnostics}
    assert off_diag_by_chunk[2]["available"] is False

    judge_on = _RecordingBudgetLimitedJudge(capacity=2)
    result_on = apply_hybrid_session_cleanup(
        takes, None, judge_on, chunk_size=2, chunk_stride=2, env={"CUTSELL_SEMANTIC_COMPUTE_PLANNER": "1"},
    )
    assert result_on.semantic_compute_plan is not None
    on_diag_by_chunk = {d["chunk_index"]: d for d in result_on.diagnostics}
    # The P0 (negation-conflict) window is now planned FIRST and so is
    # among the first 2 calls attempted -- no longer starved purely by
    # having originally been chunk_index 2 (the last one requested).
    assert on_diag_by_chunk[2]["planner_priority"] == "P0_SAFETY_CRITICAL"
    assert on_diag_by_chunk[2]["available"] is True
    # An ordinary P2 filler window is what now loses out instead.
    assert any(
        d["planner_priority"] == "P2_EDITORIAL_QUALITY" and d["available"] is False
        for d in result_on.diagnostics
    )


def test_flag_on_with_no_risk_signals_everything_is_p2_and_order_is_unaffected():
    takes = tuple(_take(f"clip{i}", float(i) * 10.0, f"contenido ordinario {i}") for i in range(4))
    judge = _RecordingBudgetLimitedJudge(capacity=100)
    result = apply_hybrid_session_cleanup(
        takes, None, judge, chunk_size=1, chunk_stride=1, env={"CUTSELL_SEMANTIC_COMPUTE_PLANNER": "1"},
    )
    assert all(d["planner_priority"] == "P2_EDITORIAL_QUALITY" for d in result.diagnostics)
    assert all(d["available"] for d in result.diagnostics)
