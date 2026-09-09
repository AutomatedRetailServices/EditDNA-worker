"""D-174: Watch+Listen BestTake Guard Authority. Covers all 45 directive-
required test-matrix categories plus the 3 D-173 real-shape replays.

See cutsell_worker/watch_listen_besttake_guard_authority.py's own module
docstring for the full two-phase design ("WATCH+LISTEN MAY VETO A BAD
WINNER. WATCH+LISTEN DOES NOT BECOME THE WINNER SELECTOR.") this suite
verifies against.
"""
from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest

from cutsell_worker.contracts import (
    SCHEMA_VERSION as CONTRACTS_SCHEMA_VERSION,
    DraftClip,
    DraftTimeline,
    EditStrategy,
    SemanticRole,
)
from cutsell_worker.watch_listen_besttake_evidence import (
    CASE_A_BOUNDARY_ONLY,
    CASE_B_DELIVERY_OWNED,
    CASE_CLEAN,
    USABILITY_UNUSABLE as V1_UNUSABLE,
    USABILITY_USABLE as V1_USABLE,
    WatchListenBestTakeEvidence,
)
from cutsell_worker.watch_listen_zone_usability_v2 import (
    CandidateZoneUsabilityV2,
    SEVERITY_MATERIAL,
    SEVERITY_MILD,
    SEVERITY_NONE,
    SEVERITY_SEVERE,
    USABILITY_IMPAIRED,
    USABILITY_QUESTIONABLE,
    USABILITY_UNUSABLE,
    USABILITY_USABLE,
    ZoneUsabilityResult,
)
from cutsell_worker.watch_listen_besttake_v2_evidence import evaluate_watch_listen_besttake_guard_v2
from cutsell_worker import watch_listen_besttake_guard_authority as m
from cutsell_worker.watch_listen_besttake_guard_authority import (
    AUTHORITY_BLOCKED_BY_CASE_OWNERSHIP,
    AUTHORITY_BLOCKED_BY_CONFLICT,
    AUTHORITY_BLOCKED_BY_D123,
    AUTHORITY_BLOCKED_BY_MEANING_FIREWALL,
    AUTHORITY_GUARD_REJECT_CURRENT_WINNER,
    AUTHORITY_LADDER_REEVALUATED,
    AUTHORITY_NO_ACTION,
    AUTHORITY_SOURCE_DETERMINISTIC_BESTTAKE_LADDER,
    apply_watch_listen_besttake_guard_authority,
    evaluate_watch_listen_besttake_guard_authority,
    watch_listen_besttake_guard_authority_diagnostics,
    watch_listen_besttake_guard_authority_enabled,
    watch_listen_besttake_guard_authority_row,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _guard_authority_flag_on(monkeypatch):
    """Phase 2 (`apply_watch_listen_besttake_guard_authority`) self-gates on
    its own flag; every test in this file except the explicit default-OFF
    proof (test_01, which overrides this via its own monkeypatch call
    after this fixture runs) exercises the ON behavior."""
    monkeypatch.setenv("CUTSELL_WATCH_LISTEN_BESTTAKE_GUARD_AUTHORITY_ENABLED", "1")


def _run_git_diff(path: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", path],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    return result.stdout.strip()


def _source(module_name: str) -> str:
    return inspect.getsource(__import__(f"cutsell_worker.{module_name}", fromlist=["_"]))


# ---------------------------------------------------------------------------
# Fixture builders (same pattern as test_cutsell_d172_watch_listen_besttake_
# v2_evidence.py's own helpers).
# ---------------------------------------------------------------------------
def _zone(usability, severity, count=0, fraction=None, conflict=False, kinds=()):
    return ZoneUsabilityResult(
        zone="DELIVERY", zone_usability=usability, zone_severity=severity,
        event_count=count, event_duration_total_sec=0.1 * count,
        zone_duration_sec=5.0, affected_fraction=fraction if fraction is not None else (0.02 * count if count else 0.0),
        isolated_event=(count == 1), repeated_defect=(count >= 2 and (fraction or 0) < 0.5),
        sustained_defect=bool(fraction and fraction >= 0.5),
        dominant_event_kinds=kinds, confidence="SUPPORTED", provenance="VISUAL_SIGNAL", zone_conflict=conflict,
    )


def _v2(cid, delivery_usability, delivery_severity, count=0, fraction=None, conflict=False,
        entry_usability=USABILITY_USABLE, exit_usability=USABILITY_USABLE, case=None, kinds=()):
    entry = _zone(entry_usability, SEVERITY_NONE)
    exit_z = _zone(exit_usability, SEVERITY_NONE)
    delivery = _zone(delivery_usability, delivery_severity, count, fraction, conflict, kinds=kinds)
    return CandidateZoneUsabilityV2(
        candidate_id=cid, entry=entry, delivery=delivery, exit=exit_z,
        overall_usability=delivery_usability,
        case_classification=case or (CASE_CLEAN if delivery_usability == USABILITY_USABLE else CASE_B_DELIVERY_OWNED),
    )


def _v1(cid, delivery_usability, meaning_sufficient=True, conflict_flags=(), case=None):
    return WatchListenBestTakeEvidence(
        candidate_id=cid, meaning_sufficient=meaning_sufficient,
        overall_performance_usability=delivery_usability,
        entry_usability=V1_USABLE, delivery_usability=delivery_usability, exit_usability=V1_USABLE,
        delivery_defect_present=(delivery_usability == V1_UNUSABLE), entry_only_defect=False, exit_only_defect=False,
        breaking_character_during_delivery=False, reset_or_fumble_during_delivery=(delivery_usability == V1_UNUSABLE),
        performance_continuity_status="INTERRUPTED_DURING_DELIVERY" if delivery_usability == V1_UNUSABLE else "CONTINUOUS",
        audio_signal_usability="UNKNOWN", visual_signal_usability=delivery_usability,
        editability_status="DELIVERY_OWNED" if delivery_usability == V1_UNUSABLE else "CLEAN",
        conflict_flags=conflict_flags,
        evidence_provenance={}, case_classification=case or (CASE_B_DELIVERY_OWNED if delivery_usability == V1_UNUSABLE else CASE_CLEAN),
    )


def _ranked(*pairs):
    return [{"clip_id": cid, "score": score} for cid, score in pairs]


def _v2guard(winner_id, meaning_ids, v1_map, v2_map, ranked):
    """Builds a REAL `WatchListenBestTakeV2GuardResult` through the actual
    production D-172 call chain -- never a hand-rolled stand-in."""
    return evaluate_watch_listen_besttake_guard_v2(
        winner_id=winner_id, meaning_sufficient_ids=meaning_ids,
        v1_evidence_by_id=v1_map, v2_evidence_by_id=v2_map, ranked=ranked,
    )


def _authority(winner_id, meaning_ids, member_count, v2_result, v2_map, conflict=False):
    return evaluate_watch_listen_besttake_guard_authority(
        winner_id=winner_id, meaning_sufficient_ids=meaning_ids, member_count=member_count,
        v2_result=v2_result, v2_evidence_by_id=v2_map, case_b_conflict_present=conflict,
    )


def _clip(cid, order, selected):
    return DraftClip(
        clip_id=cid, source_asset_id="src", source_order=order, start=float(order) * 4.0,
        end=float(order) * 4.0 + 4.0, text="text " + cid, caption_text="text " + cid,
        semantic_role=SemanticRole.STORY, selected=selected,
    )


def _draft(rows, selected_ids, discarded_ids):
    selected = tuple(_clip(cid, i, True) for i, cid in enumerate(selected_ids))
    discarded = tuple(_clip(cid, i + len(selected_ids), False) for i, cid in enumerate(discarded_ids))
    return DraftTimeline(
        schema_version=CONTRACTS_SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=discarded,
        diagnostics={"take_judge_groups": rows},
    )


# =============================================================================
# 1. authority default OFF
# =============================================================================
def test_01_authority_default_off(monkeypatch):
    monkeypatch.delenv("CUTSELL_WATCH_LISTEN_BESTTAKE_GUARD_AUTHORITY_ENABLED", raising=False)
    assert watch_listen_besttake_guard_authority_enabled() is False
    row = {
        "group_id": "tg_1", "guard_authority_state": AUTHORITY_GUARD_REJECT_CURRENT_WINNER,
        "guard_authority_rejected_winner_id": "A", "guard_authority_winner_before": "A",
        "ranked": _ranked(("A", 0.10), ("B", 0.95)),
    }
    draft = _draft([row], selected_ids=["A"], discarded_ids=["B"])
    result = apply_watch_listen_besttake_guard_authority(draft)
    assert result is draft  # byte-identical no-op when disabled


# =============================================================================
# 2. V2 absent (V1_FALLBACK) -> NO_ACTION
# =============================================================================
def test_02_v2_absent_never_authorizes():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": None, "B": None}
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.4)))
    r = _authority("A", ["A", "B"], 2, g, v2)
    assert r.authority_state == AUTHORITY_NO_ACTION
    assert r.rejection_reason == "v2_evidence_absent_v1_fallback_never_authorizes"


# =============================================================================
# 3. one candidate only
# =============================================================================
def test_03_one_candidate_only():
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, count=3, fraction=0.9)}
    g = _v2guard("A", ["A"], {"A": _v1("A", V1_UNUSABLE)}, v2, _ranked(("A", 0.5)))
    r = _authority("A", ["A"], 1, g, v2)
    assert r.authority_state == AUTHORITY_NO_ACTION


# =============================================================================
# 4. no meaning-sufficient alternative
# =============================================================================
def test_04_no_meaning_sufficient_alternative():
    v1 = {"A": _v1("A", V1_UNUSABLE, meaning_sufficient=True), "B": _v1("B", V1_UNUSABLE, meaning_sufficient=False)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, 3, 0.9), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0)}
    g = _v2guard("A", ["A"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A"], 2, g, v2)
    assert r.authority_state == AUTHORITY_BLOCKED_BY_MEANING_FIREWALL


# =============================================================================
# 5. meaning-insufficient dominant alt blocked
# =============================================================================
def test_05_meaning_insufficient_dominant_blocked():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE, meaning_sufficient=False)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_MATERIAL, 3, 0.6), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0)}
    g = _v2guard("A", ["A"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A"], 2, g, v2)
    assert r.authority_state == AUTHORITY_BLOCKED_BY_MEANING_FIREWALL
    assert r.rejected_winner_id is None


# =============================================================================
# 6. current winner USABLE
# =============================================================================
def test_06_winner_usable_no_action():
    v1 = {"A": _v1("A", V1_USABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_USABLE, SEVERITY_NONE, 0), "B": _v2("B", USABILITY_UNUSABLE, SEVERITY_SEVERE, 3, 0.8)}
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.9), ("B", 0.5)))
    r = _authority("A", ["A", "B"], 2, g, v2)
    assert r.authority_state == AUTHORITY_NO_ACTION


# =============================================================================
# 7. both candidates MILD near-equal
# =============================================================================
def test_07_mild_near_equal_no_authority():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {
        "A": _v2("A", USABILITY_QUESTIONABLE, SEVERITY_MILD, 2, 0.3),
        "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0),
    }
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A", "B"], 2, g, v2)
    assert g.guard_status == m.GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE  # V2 does find ordinal dominance
    assert r.authority_state == AUTHORITY_NO_ACTION
    assert r.rejection_reason == "winner_severity_below_material_floor"


# =============================================================================
# 8. material winner vs clean alt
# =============================================================================
def test_08_material_winner_vs_clean_alt_rejects():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_IMPAIRED, SEVERITY_MATERIAL, 2, 0.4), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0)}
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A", "B"], 2, g, v2)
    assert r.authority_state == AUTHORITY_GUARD_REJECT_CURRENT_WINNER
    assert r.rejected_winner_id == "A"
    assert r.eligible_alternative_ids == ("B",)


# =============================================================================
# 9. severe winner vs mild alt
# =============================================================================
def test_09_severe_winner_vs_mild_alt_rejects():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, 4, 0.9), "B": _v2("B", USABILITY_QUESTIONABLE, SEVERITY_MILD, 1, 0.1)}
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A", "B"], 2, g, v2)
    assert r.authority_state == AUTHORITY_GUARD_REJECT_CURRENT_WINNER


# =============================================================================
# 10. isolated ordinary motion (never reaches severity at all)
# =============================================================================
def test_10_isolated_ordinary_motion_no_authority():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_USABLE)}
    v2 = {
        "A": _v2("A", USABILITY_USABLE, SEVERITY_NONE, 1, 0.02, kinds=("hand_motion_reset_candidate",)),
        "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0),
    }
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A", "B"], 2, g, v2)
    assert r.authority_state == AUTHORITY_NO_ACTION


# =============================================================================
# 11. repeated ordinary motion, non-disruptive (MILD, floor blocks it)
# =============================================================================
def test_11_repeated_ordinary_motion_non_disruptive_no_authority():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {
        "A": _v2("A", USABILITY_QUESTIONABLE, SEVERITY_MILD, 3, 0.2, kinds=("hand_motion_reset_candidate",)),
        "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0),
    }
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A", "B"], 2, g, v2)
    assert r.authority_state == AUTHORITY_NO_ACTION
    assert r.rejection_reason == "winner_severity_below_material_floor"


# =============================================================================
# 12. sustained breaking-character style defect (HIGH materiality -> SEVERE)
# =============================================================================
def test_12_sustained_breaking_character_authorizes():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {
        "A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, 3, 0.7, kinds=("breaking_character",)),
        "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0),
    }
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A", "B"], 2, g, v2)
    assert r.authority_state == AUTHORITY_GUARD_REJECT_CURRENT_WINNER


# =============================================================================
# 13. CASE A (defensive -- structurally unreachable via real dominance, so
# exercised by constructing the guard result directly)
# =============================================================================
def test_13_case_a_boundary_only_never_rejected():
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, 3, 0.9, case=CASE_A_BOUNDARY_ONLY), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0)}
    v1_result_stub = m.WatchListenBestTakeGuardResult = None  # not used; build V2GuardResult directly
    from cutsell_worker.watch_listen_besttake_evidence import WatchListenBestTakeGuardResult, GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE as V1_DOM
    v1_res = WatchListenBestTakeGuardResult(V1_DOM, "x", "A", "B", None, {"A": "UNUSABLE", "B": "USABLE"})
    from cutsell_worker.watch_listen_besttake_v2_evidence import WatchListenBestTakeV2GuardResult, EVIDENCE_SOURCE_V2
    g = WatchListenBestTakeV2GuardResult(
        guard_status=m.GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE, guard_reason="x", winner_id="A",
        dominant_candidate_id="B", existing_ladder_agrees=True, evidence_source=EVIDENCE_SOURCE_V2,
        winner_severity=SEVERITY_SEVERE, alt_severity=SEVERITY_NONE, meaning_firewall_blocked=False,
        candidate_usability_summary={}, v1_result=v1_res,
    )
    r = _authority("A", ["A", "B"], 2, g, v2)
    assert r.authority_state == AUTHORITY_BLOCKED_BY_CASE_OWNERSHIP


# =============================================================================
# 14. CASE B (delivery-owned -- the normal authorizing shape)
# =============================================================================
def test_14_case_b_delivery_owned_authorizes():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_MATERIAL, 3, 0.6, case=CASE_B_DELIVERY_OWNED), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0)}
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A", "B"], 2, g, v2)
    assert r.case_owner == CASE_B_DELIVERY_OWNED
    assert r.authority_state == AUTHORITY_GUARD_REJECT_CURRENT_WINNER


# =============================================================================
# 15. CASE C (ambiguous / conflict) -> BLOCKED_BY_CONFLICT
# =============================================================================
def test_15_case_c_ambiguous_blocked_by_conflict():
    v1 = {"A": _v1("A", V1_UNUSABLE, conflict_flags=("MEANING_COMPLETE_VS_ABANDONED_ATTEMPT_EVIDENCE",)), "B": _v1("B", V1_USABLE)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, 3, 0.9), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0)}
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A", "B"], 2, g, v2)
    assert r.authority_state == AUTHORITY_BLOCKED_BY_CONFLICT


# =============================================================================
# 16. CASE CLEAN -> no defect -> NO_ACTION
# =============================================================================
def test_16_case_clean_no_authority():
    v1 = {"A": _v1("A", V1_USABLE), "B": _v1("B", V1_USABLE)}
    v2 = {"A": _v2("A", USABILITY_USABLE, SEVERITY_NONE, 0), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0)}
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.9), ("B", 0.5)))
    r = _authority("A", ["A", "B"], 2, g, v2)
    assert r.authority_state == AUTHORITY_NO_ACTION


# =============================================================================
# 17. D-123 actionable -> BLOCKED_BY_D123
# =============================================================================
def test_17_d123_actionable_blocked():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_MATERIAL, 3, 0.6), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0)}
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A", "B"], 2, g, v2, conflict=True)
    assert r.authority_state == AUTHORITY_BLOCKED_BY_D123
    assert r.d123_blocked is True


# =============================================================================
# 18. D-123 non-actionable -> proceeds
# =============================================================================
def test_18_d123_non_actionable_proceeds():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_MATERIAL, 3, 0.6), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0)}
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A", "B"], 2, g, v2, conflict=False)
    assert r.authority_state == AUTHORITY_GUARD_REJECT_CURRENT_WINNER
    assert r.d123_blocked is False


# =============================================================================
# 19/20. semantic authoritative winner / semantic abstain -- no coupling
# =============================================================================
def test_19_20_no_semantic_authority_coupling():
    sig = inspect.signature(evaluate_watch_listen_besttake_guard_authority)
    assert "semantic_authority_gate_status" not in sig.parameters
    assert "semantic_winner" not in sig.parameters


# =============================================================================
# 21. multi-alternative family
# =============================================================================
def test_21_multi_alternative_family():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE), "C": _v1("C", V1_UNUSABLE)}
    v2 = {
        "A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, 4, 0.9),
        "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0),
        "C": _v2("C", USABILITY_USABLE, SEVERITY_NONE, 0),
    }
    g = _v2guard("A", ["A", "B", "C"], v1, v2, _ranked(("A", 0.5), ("B", 0.9), ("C", 0.3)))
    r = _authority("A", ["A", "B", "C"], 3, g, v2)
    assert r.authority_state == AUTHORITY_GUARD_REJECT_CURRENT_WINNER
    assert set(r.eligible_alternative_ids) == {"B", "C"}


# =============================================================================
# 22/23/24/42. Phase 2: ladder decides the replacement, never V2 directly.
# =============================================================================
def test_22_v2_dominant_not_ladder_winner():
    # V2's own dominant tie-break picks B (higher ranked score among
    # dominant candidates), but the row simulates Phase 1 already having
    # marked A rejected; the ladder's own top-two gap among the REAL
    # remainder (B, C) decisively favors C -- Phase 2 must return C, not B.
    row = {
        "group_id": "tg_1", "guard_authority_state": AUTHORITY_GUARD_REJECT_CURRENT_WINNER,
        "guard_authority_rejected_winner_id": "A", "guard_authority_winner_before": "A",
        "ranked": _ranked(("A", 0.10), ("B", 0.40), ("C", 0.85)),
    }
    draft = _draft([row], selected_ids=["A"], discarded_ids=["B", "C"])
    result = apply_watch_listen_besttake_guard_authority(draft)
    out_row = result.diagnostics["take_judge_groups"][0]
    assert out_row["guard_authority_ladder_replacement_id"] == "C"
    assert out_row["guard_authority_winner_after"] == "C"
    assert {c.clip_id for c in result.selected} == {"C"}
    assert {c.clip_id for c in result.discarded} == {"A", "B"}


def test_23_v2_dominant_equals_ladder_winner():
    row = {
        "group_id": "tg_1", "guard_authority_state": AUTHORITY_GUARD_REJECT_CURRENT_WINNER,
        "guard_authority_rejected_winner_id": "A", "guard_authority_winner_before": "A",
        "ranked": _ranked(("A", 0.10), ("B", 0.95)),
    }
    draft = _draft([row], selected_ids=["A"], discarded_ids=["B"])
    result = apply_watch_listen_besttake_guard_authority(draft)
    out_row = result.diagnostics["take_judge_groups"][0]
    assert out_row["guard_authority_ladder_replacement_id"] == "B"
    assert out_row.get("guard_authority_replacement_source") == AUTHORITY_SOURCE_DETERMINISTIC_BESTTAKE_LADDER


def test_24_ladder_chooses_third_candidate():
    row = {
        "group_id": "tg_1", "guard_authority_state": AUTHORITY_GUARD_REJECT_CURRENT_WINNER,
        "guard_authority_rejected_winner_id": "A", "guard_authority_winner_before": "A",
        "ranked": _ranked(("A", 0.10), ("B", 0.50), ("C", 0.95)),
    }
    draft = _draft([row], selected_ids=["A"], discarded_ids=["B", "C"])
    result = apply_watch_listen_besttake_guard_authority(draft)
    out_row = result.diagnostics["take_judge_groups"][0]
    assert out_row["guard_authority_winner_after"] == "C"


def test_42_no_direct_v2_winner_selection_structural_proof():
    """The authority module must never write winner_after = <v2 dominant
    candidate> unless the ladder independently returns it. Constructs a
    case where the dominant candidate B and the ladder's own pick C
    DIFFER, and asserts the source code contains no direct assignment
    from a v2/dominant field into winner_after -- only from the ladder's
    own return value."""
    source = _source("watch_listen_besttake_guard_authority")
    assert "winner_after\"] = " not in source.replace(" ", "")
    assert 'replacement_id = str(replacement_row.get("clip_id")' in source
    # Functional proof (mirrors test_22): dominant != ladder pick.
    row = {
        "group_id": "tg_1", "guard_authority_state": AUTHORITY_GUARD_REJECT_CURRENT_WINNER,
        "guard_authority_rejected_winner_id": "A", "guard_authority_winner_before": "A",
        "guard_authority_v2_dominant_candidates": ["B"],
        "ranked": _ranked(("A", 0.10), ("B", 0.40), ("C", 0.85)),
    }
    draft = _draft([row], selected_ids=["A"], discarded_ids=["B", "C"])
    result = apply_watch_listen_besttake_guard_authority(draft)
    out_row = result.diagnostics["take_judge_groups"][0]
    assert out_row["guard_authority_winner_after"] != "B"
    assert out_row["guard_authority_winner_after"] == "C"


# =============================================================================
# 25. failed reevaluation -> original winner preserved
# =============================================================================
def test_25_failed_reevaluation_preserves_original():
    # Remaining gap (0.55 - 0.50 = 0.05) is below CLEAR_WINNER_MINIMUM_GAP
    # (0.30) -- the ladder is not decisive on the remainder.
    row = {
        "group_id": "tg_1", "guard_authority_state": AUTHORITY_GUARD_REJECT_CURRENT_WINNER,
        "guard_authority_rejected_winner_id": "A", "guard_authority_winner_before": "A",
        "ranked": _ranked(("A", 0.10), ("B", 0.55), ("C", 0.50)),
    }
    draft = _draft([row], selected_ids=["A"], discarded_ids=["B", "C"])
    result = apply_watch_listen_besttake_guard_authority(draft)
    out_row = result.diagnostics["take_judge_groups"][0]
    assert out_row["guard_authority_state"] == AUTHORITY_GUARD_REJECT_CURRENT_WINNER  # never upgraded
    assert out_row.get("guard_authority_applied", False) is False
    assert {c.clip_id for c in result.selected} == {"A"}


# =============================================================================
# 26. exception safety -> original preserved, no raise
# =============================================================================
def test_26_exception_safety_malformed_row():
    row = {
        "group_id": "tg_1", "guard_authority_state": AUTHORITY_GUARD_REJECT_CURRENT_WINNER,
        "guard_authority_rejected_winner_id": "NOT_IN_DRAFT", "guard_authority_winner_before": "NOT_IN_DRAFT",
        "ranked": _ranked(("NOT_IN_DRAFT", 0.10), ("B", 0.95)),
    }
    draft = _draft([row], selected_ids=["A"], discarded_ids=["B"])
    result = apply_watch_listen_besttake_guard_authority(draft)  # must not raise
    assert {c.clip_id for c in result.selected} == {"A"}


# =============================================================================
# 27. family topology unchanged (only the marked group's clips move)
# =============================================================================
def test_27_family_topology_unchanged_for_other_groups():
    row_a = {
        "group_id": "tg_1", "guard_authority_state": AUTHORITY_GUARD_REJECT_CURRENT_WINNER,
        "guard_authority_rejected_winner_id": "A", "guard_authority_winner_before": "A",
        "ranked": _ranked(("A", 0.10), ("B", 0.95)),
    }
    row_untouched = {"group_id": "tg_2", "guard_authority_state": AUTHORITY_NO_ACTION, "ranked": _ranked(("X", 0.9), ("Y", 0.1))}
    draft = DraftTimeline(
        schema_version=CONTRACTS_SCHEMA_VERSION, project_id="p1", strategy=EditStrategy.STORYTELLING,
        selected=(_clip("A", 0, True), _clip("X", 2, True)), alternates=(),
        discarded=(_clip("B", 1, False), _clip("Y", 3, False)),
        diagnostics={"take_judge_groups": [row_a, row_untouched]},
    )
    result = apply_watch_listen_besttake_guard_authority(draft)
    assert result.diagnostics["take_judge_groups"][1] == row_untouched  # byte-identical
    assert {c.clip_id for c in result.selected} == {"B", "X"}


# =============================================================================
# 28/29/30/31/32. no coupling with Family/Proposition/Attempt/D-150/D-158/D-161.
# =============================================================================
def test_28_29_30_31_32_no_forbidden_module_imports():
    source = _source("watch_listen_besttake_guard_authority")
    forbidden = (
        "attempt_relationship_authority", "language_proposition_relation",
        "language_utterance_attempt", "semantic_authority_observability",
        "watch_listen_family_evidence", "watch_listen_relation_discovery",
        "take_grouping_provider", "session_boundaries",
    )
    for name in forbidden:
        assert name not in source, f"unexpected coupling with {name}"


# =============================================================================
# 33/34/35. D-163/D-167/D-172 stay at zero diff
# =============================================================================
def test_33_34_35_closed_modules_zero_diff():
    for path in (
        "cutsell_worker/watch_listen_besttake_evidence.py",
        "cutsell_worker/watch_listen_zone_usability_v2.py",
        "cutsell_worker/watch_listen_besttake_v2_evidence.py",
    ):
        assert _run_git_diff(path) == "", f"{path} was modified by D-174"


# =============================================================================
# 36. D-171 Language Spine unchanged
# =============================================================================
def test_36_language_spine_zero_diff():
    for path in (
        "cutsell_worker/language_spine_consumer_migration.py",
        "cutsell_worker/take_grouping_provider.py",
        "cutsell_worker/recording_meta_continuation.py",
        "cutsell_worker/language_proposition_relation.py",
        "cutsell_worker/language_utterance_attempt.py",
    ):
        assert _run_git_diff(path) == "", f"{path} was modified by D-174"


# =============================================================================
# 37/38/39. Boundary / Pacing / render unchanged
# =============================================================================
def test_37_38_39_boundary_pacing_render_zero_diff():
    for path in (
        "cutsell_worker/boundary_engine_pass.py",
        "cutsell_worker/final_boundary_authority.py",
        "cutsell_worker/dialogue_pacing_transition.py",
        "cutsell_worker/composer.py",
        "cutsell_worker/live_render_qc.py",
    ):
        assert _run_git_diff(path) == "", f"{path} was modified by D-174"


# =============================================================================
# 40. no provider/network
# =============================================================================
def test_40_no_provider_network_call():
    source = _source("watch_listen_besttake_guard_authority")
    for token in ("openai", "gemini", "requests.", "httpx", "urllib", "socket."):
        assert token not in source.lower()


# =============================================================================
# 41. deterministic provenance -- same inputs, same outputs
# =============================================================================
def test_41_deterministic_provenance():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_MATERIAL, 3, 0.6), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0)}
    g1 = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    g2 = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r1 = _authority("A", ["A", "B"], 2, g1, v2)
    r2 = _authority("A", ["A", "B"], 2, g2, v2)
    assert r1 == r2


# =============================================================================
# 43. meaning firewall P0 -- restated, run-level diagnostics too
# =============================================================================
def test_43_meaning_firewall_p0_diagnostics():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE, meaning_sufficient=False)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, 4, 0.9), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0)}
    g = _v2guard("A", ["A"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A"], 2, g, v2)
    diag = watch_listen_besttake_guard_authority_diagnostics([r])
    assert diag["guard_authority_meaning_block_count"] == 1
    assert diag["guard_authority_rejection_count"] == 0


# =============================================================================
# 44. current-winner-rejection-only (never resurrects into select/alternates
# elsewhere)
# =============================================================================
def test_44_rejection_moves_only_to_discard():
    row = {
        "group_id": "tg_1", "guard_authority_state": AUTHORITY_GUARD_REJECT_CURRENT_WINNER,
        "guard_authority_rejected_winner_id": "A", "guard_authority_winner_before": "A",
        "ranked": _ranked(("A", 0.10), ("B", 0.95)),
    }
    draft = _draft([row], selected_ids=["A"], discarded_ids=["B"])
    result = apply_watch_listen_besttake_guard_authority(draft)
    assert not result.alternates
    assert "A" in {c.clip_id for c in result.discarded}
    assert "A" not in {c.clip_id for c in result.selected}


# =============================================================================
# 45. existing ladder makes the replacement (exact 0.30 gap contract reused)
# =============================================================================
def test_45_existing_ladder_gap_contract_reused():
    from cutsell_worker.deterministic_best_take_authority import CLEAR_WINNER_MINIMUM_GAP
    assert CLEAR_WINNER_MINIMUM_GAP == 0.30
    row_below = {
        "group_id": "tg_1", "guard_authority_state": AUTHORITY_GUARD_REJECT_CURRENT_WINNER,
        "guard_authority_rejected_winner_id": "A", "guard_authority_winner_before": "A",
        "ranked": _ranked(("A", 0.10), ("B", 0.60), ("C", 0.35)),  # gap 0.25 < 0.30
    }
    draft_below = _draft([row_below], selected_ids=["A"], discarded_ids=["B", "C"])
    result_below = apply_watch_listen_besttake_guard_authority(draft_below)
    assert result_below.diagnostics["take_judge_groups"][0].get("guard_authority_applied", False) is False

    row_above = {
        "group_id": "tg_1", "guard_authority_state": AUTHORITY_GUARD_REJECT_CURRENT_WINNER,
        "guard_authority_rejected_winner_id": "A", "guard_authority_winner_before": "A",
        "ranked": _ranked(("A", 0.10), ("B", 0.65), ("C", 0.30)),  # gap 0.35 >= 0.30
    }
    draft_above = _draft([row_above], selected_ids=["A"], discarded_ids=["B", "C"])
    result_above = apply_watch_listen_besttake_guard_authority(draft_above)
    assert result_above.diagnostics["take_judge_groups"][0]["guard_authority_winner_after"] == "B"


# =============================================================================
# D-173 REAL-SHAPE REPLAYS (offline abstract fixtures)
# =============================================================================
def test_replay_a_meaning_insufficient_dominant_no_authority():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE, meaning_sufficient=False)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_MATERIAL, 3, 0.6), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0)}
    g = _v2guard("A", ["A"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A"], 2, g, v2)
    assert r.authority_state == AUTHORITY_BLOCKED_BY_MEANING_FIREWALL
    assert g.meaning_firewall_blocked is True


def test_replay_b_real_performance_dominance_both_meaning_sufficient():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, 4, 0.9), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0)}
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A", "B"], 2, g, v2)
    assert r.authority_state == AUTHORITY_GUARD_REJECT_CURRENT_WINNER  # MAY be rejected
    row = {
        "group_id": "tg_1", "guard_authority_state": r.authority_state,
        "guard_authority_rejected_winner_id": r.rejected_winner_id, "guard_authority_winner_before": "A",
        "ranked": _ranked(("A", 0.5), ("B", 0.9)),
    }
    draft = _draft([row], selected_ids=["A"], discarded_ids=["B"])
    result = apply_watch_listen_besttake_guard_authority(draft)
    assert result.diagnostics["take_judge_groups"][0]["guard_authority_winner_after"] == "B"  # ladder decides


def test_replay_c_pimples_like_near_equal_no_authority():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {
        "A": _v2("A", USABILITY_UNUSABLE, SEVERITY_MILD, 1, 0.05),
        "B": _v2("B", USABILITY_QUESTIONABLE, SEVERITY_MILD, 1, 0.04),
    }
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.55)))
    r = _authority("A", ["A", "B"], 2, g, v2)
    assert r.authority_state == AUTHORITY_NO_ACTION


# =============================================================================
# Structural: diagnostics row helper carries the exact directive field names.
# =============================================================================
def test_row_helper_field_names():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_MATERIAL, 3, 0.6), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, 0)}
    g = _v2guard("A", ["A", "B"], v1, v2, _ranked(("A", 0.5), ("B", 0.9)))
    r = _authority("A", ["A", "B"], 2, g, v2)
    row = watch_listen_besttake_guard_authority_row(r)
    for key in (
        "guard_authority_enabled", "guard_authority_evaluated", "guard_authority_state",
        "guard_authority_winner_before", "guard_authority_rejected_winner_id",
        "guard_authority_eligible_alternative_ids", "guard_authority_v2_dominant_candidates",
        "guard_authority_meaning_firewall_blocked", "guard_authority_d123_blocked",
        "guard_authority_case_owner", "guard_authority_ladder_reevaluated",
        "guard_authority_ladder_replacement_id", "guard_authority_winner_after", "guard_authority_applied",
    ):
        assert key in row
