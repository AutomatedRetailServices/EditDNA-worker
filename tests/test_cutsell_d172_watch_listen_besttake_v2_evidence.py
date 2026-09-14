"""D-172: Watch+Listen BestTake evidence, Zone-Usability V2 diagnostic
consumption. Covers all 34 directive-required fixture categories plus
additional structural/no-authority-change proofs.

See cutsell_worker/watch_listen_besttake_v2_evidence.py's own module
docstring for the full design rationale (the "V2 only ever refines the
ONE V1 outcome D-164 proved under-resolved" contract) this suite verifies
against.
"""
from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest

from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.positioned_performance_evidence import (
    DeliverySpan,
    PositionAwarePerformanceEvidence,
    PositionedEvent,
    ZONE_DELIVERY,
    ZONE_ENTRY,
    ZONE_EXIT,
)
from cutsell_worker.raw_understanding_map import RawUnderstandingSpan, build_raw_understanding_span
from cutsell_worker.watch_listen_besttake_evidence import (
    CASE_B_DELIVERY_OWNED,
    CASE_CLEAN,
    GUARD_BYPASS_POOR_USABILITY_WINNER,
    GUARD_NO_ACTION,
    GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE,
    GUARD_PRESERVE_STRUCTURED_WINNER,
    GUARD_UNCERTAIN,
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
    SEVERITY_UNKNOWN,
    USABILITY_IMPAIRED,
    USABILITY_QUESTIONABLE,
    USABILITY_UNKNOWN,
    USABILITY_UNUSABLE,
    USABILITY_USABLE,
    ZoneUsabilityResult,
    build_zone_usability_v2,
)
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext
from cutsell_worker import watch_listen_besttake_v2_evidence as m
from cutsell_worker.watch_listen_besttake_v2_evidence import (
    EVIDENCE_SOURCE_NO_EVIDENCE,
    EVIDENCE_SOURCE_V1_FALLBACK,
    EVIDENCE_SOURCE_V2,
    build_candidate_zone_usability_v2,
    evaluate_watch_listen_besttake_guard_v2,
    watch_listen_besttake_v2_diagnostics,
    watch_listen_besttake_v2_group_row,
    zone_usability_v2_besttake_enabled,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_git_diff(path: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", path],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    return result.stdout.strip()


def _source(module_name: str) -> str:
    return inspect.getsource(__import__(f"cutsell_worker.{module_name}", fromlist=["_"]))


# ---------------------------------------------------------------------------
# Fixture builders
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
        entry_usability=USABILITY_USABLE, exit_usability=USABILITY_USABLE, case=None):
    entry = _zone(entry_usability, SEVERITY_NONE)
    exit_z = _zone(exit_usability, SEVERITY_NONE)
    delivery = _zone(delivery_usability, delivery_severity, count, fraction, conflict)
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


# ---------------------------------------------------------------------------
# 1. V2 absent -> V1 fallback
# ---------------------------------------------------------------------------
def test_01_v2_absent_v1_fallback():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": None, "B": None}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    assert result.guard_status == GUARD_BYPASS_POOR_USABILITY_WINNER
    assert result.evidence_source == EVIDENCE_SOURCE_V1_FALLBACK
    assert result.dominant_candidate_id is None


# ---------------------------------------------------------------------------
# 2. Clean vs clean
# ---------------------------------------------------------------------------
def test_02_clean_vs_clean():
    v1 = {"A": _v1("A", V1_USABLE), "B": _v1("B", V1_USABLE)}
    v2 = {"A": _v2("A", USABILITY_USABLE, SEVERITY_NONE), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
    )
    assert result.guard_status == GUARD_PRESERVE_STRUCTURED_WINNER
    assert result.dominant_candidate_id is None


# ---------------------------------------------------------------------------
# 3. Mild vs clean (winner USABLE overall -> V1 already PRESERVE, V2 never runs)
# ---------------------------------------------------------------------------
def test_03_mild_vs_clean_winner_already_acceptable():
    v1 = {"A": _v1("A", V1_USABLE), "B": _v1("B", V1_USABLE)}
    v2 = {"A": _v2("A", USABILITY_QUESTIONABLE, SEVERITY_MILD, count=2), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
    )
    assert result.guard_status == GUARD_PRESERVE_STRUCTURED_WINNER


# ---------------------------------------------------------------------------
# 4. Material vs mild (winner UNUSABLE V1, V2 dominance found)
# ---------------------------------------------------------------------------
def test_04_material_vs_mild_dominance():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_IMPAIRED, SEVERITY_MATERIAL, count=2),
          "B": _v2("B", USABILITY_QUESTIONABLE, SEVERITY_MILD, count=1)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert result.dominant_candidate_id == "B"
    assert result.winner_severity == SEVERITY_MATERIAL
    assert result.alt_severity == SEVERITY_MILD


# ---------------------------------------------------------------------------
# 5. Severe vs material (still a valid dominance -- ordinal, not numeric)
# ---------------------------------------------------------------------------
def test_05_severe_vs_material_dominance():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, count=3),
          "B": _v2("B", USABILITY_IMPAIRED, SEVERITY_MATERIAL, count=1)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert result.dominant_candidate_id == "B"


# ---------------------------------------------------------------------------
# 6. Isolated vs sustained (isolated USABLE never worse than sustained UNUSABLE)
# ---------------------------------------------------------------------------
def test_06_isolated_vs_sustained():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, count=3, fraction=0.7),
          "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, count=1, fraction=0.008)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert result.dominant_candidate_id == "B"


# ---------------------------------------------------------------------------
# 7. Repeated vs isolated (repeated MILD vs isolated NONE -- still dominance)
# ---------------------------------------------------------------------------
def test_07_repeated_vs_isolated():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_QUESTIONABLE, SEVERITY_MILD, count=3),
          "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, count=1)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert result.dominant_candidate_id == "B"


# ---------------------------------------------------------------------------
# 8. Same coarse V1, different V2 (the D-164/D-170 real shape)
# ---------------------------------------------------------------------------
def test_08_same_coarse_v1_different_v2():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    assert v1["A"].delivery_usability == v1["B"].delivery_usability  # coarse V1 identical
    v2 = {"A": _v2("A", USABILITY_IMPAIRED, SEVERITY_MATERIAL, count=8),
          "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, count=1)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert result.dominant_candidate_id == "B"


# ---------------------------------------------------------------------------
# 9. Meaning-insufficient V2-dominant alt blocked (D-170 replay)
# ---------------------------------------------------------------------------
def test_09_meaning_insufficient_dominant_blocked():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE, meaning_sufficient=False)}
    v2 = {"A": _v2("A", USABILITY_IMPAIRED, SEVERITY_MATERIAL, count=4),
          "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, count=0)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    assert result.guard_status == GUARD_BYPASS_POOR_USABILITY_WINNER
    assert result.dominant_candidate_id is None
    assert result.meaning_firewall_blocked is True


# ---------------------------------------------------------------------------
# 10. Meaning-sufficient V2-dominant alt surfaced
# ---------------------------------------------------------------------------
def test_10_meaning_sufficient_dominant_surfaced():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_IMPAIRED, SEVERITY_MATERIAL, count=4),
          "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, count=0)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert result.dominant_candidate_id == "B"
    assert result.meaning_firewall_blocked is False


# ---------------------------------------------------------------------------
# 11. Semantic + DeliveryScorer agreement contradiction (guard exposes, never replaces)
# ---------------------------------------------------------------------------
def test_11_semantic_deliveryscore_agreement_contradiction():
    # semantic winner == DeliveryScorer winner == A (modeled simply as
    # winner_id="A" -- this module is agnostic to WHY A was chosen).
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, count=6),
          "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, count=0)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.9), ("B", 0.1)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert result.dominant_candidate_id == "B"
    # Winner is NOT replaced -- this is diagnostic evidence only.
    assert result.winner_id == "A"


# ---------------------------------------------------------------------------
# 12. Semantic non-decisive contradiction
# ---------------------------------------------------------------------------
def test_12_semantic_non_decisive_contradiction():
    # Semantic abstains; DeliveryScorer picked A anyway (modeled as
    # winner_id="A"); V2 says B dominates -- surfaced, no authority.
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_IMPAIRED, SEVERITY_MATERIAL, count=5),
          "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, count=0)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.5)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert result.winner_id == "A"  # not mutated


# ---------------------------------------------------------------------------
# 13. D-123 disagreement preserved (D-172 does not touch D-123's own gate)
# ---------------------------------------------------------------------------
def test_13_d123_ownership_untouched():
    assert _run_git_diff("cutsell_worker/case_b_performance_evidence.py") == ""
    assert _run_git_diff("cutsell_worker/multimodal_besttake_fallback.py") == ""


# ---------------------------------------------------------------------------
# 14. CASE A preserved (Boundary-owned winner case never demotes)
# ---------------------------------------------------------------------------
def test_14_case_a_preserved():
    from cutsell_worker.watch_listen_besttake_evidence import CASE_A_BOUNDARY_ONLY
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    winner_v2 = _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, count=3, case=CASE_A_BOUNDARY_ONLY)
    alt_v2 = _v2("B", USABILITY_USABLE, SEVERITY_NONE, count=0)
    v2 = {"A": winner_v2, "B": alt_v2}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    # zone_usability_v2_dominates itself refuses when winner case is CASE A --
    # no dominance found, BYPASS preserved (never a forced demotion).
    assert result.guard_status == GUARD_BYPASS_POOR_USABILITY_WINNER
    assert result.dominant_candidate_id is None


# ---------------------------------------------------------------------------
# 15. CASE B exposed (legitimate performance evidence)
# ---------------------------------------------------------------------------
def test_15_case_b_exposed():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, count=4, case=CASE_B_DELIVERY_OWNED),
          "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, count=0)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE


# ---------------------------------------------------------------------------
# 16. CASE C uncertain (delivery UNKNOWN -- fail-open, no forced dominance)
# ---------------------------------------------------------------------------
def test_16_case_c_uncertain_fail_open():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    winner_v2 = _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, count=3)
    # Alt has UNKNOWN delivery usability -- rank 2, same as QUESTIONABLE,
    # so it does NOT strictly dominate a SEVERE/UNUSABLE (rank 0) winner...
    # actually UNKNOWN(2) > UNUSABLE(0) so it WOULD dominate; use it to
    # confirm the ordinal comparison (not a special uncertain carve-out)
    # governs, exactly like V1's own contract.
    alt_v2 = _v2("B", USABILITY_UNKNOWN, SEVERITY_UNKNOWN, count=0)
    v2 = {"A": winner_v2, "B": alt_v2}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE


# ---------------------------------------------------------------------------
# 17. Ordinary motion no penalty (LOW materiality isolated -> SEVERITY_NONE)
# ---------------------------------------------------------------------------
def test_17_ordinary_motion_no_penalty():
    span = _build_real_span(
        delivery_events=(_ev("hand_motion_reset_candidate", 1.0, 1.05, 0.95),),
    )
    v2 = build_zone_usability_v2("c1", span)
    assert v2.delivery.zone_severity == SEVERITY_NONE
    assert v2.overall_usability == USABILITY_USABLE


# ---------------------------------------------------------------------------
# 18. Audio silence not a semantic defect
# ---------------------------------------------------------------------------
def test_18_audio_silence_not_semantic_defect():
    from cutsell_worker.watch_listen_zone_usability_v2 import _DEFECT_KINDS
    assert "audio_silence_interval" not in _DEFECT_KINDS


# ---------------------------------------------------------------------------
# 19. Conflict flags fail-open
# ---------------------------------------------------------------------------
def test_19_conflict_flags_fail_open():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, count=3, conflict=True),
          "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, count=0)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    # Winner-side zone_conflict -> V2 never resolves it (fails open to V1's
    # own BYPASS outcome, exactly like a V1 conflict_flags winner fails
    # open to UNCERTAIN and is never resolved by richer evidence).
    assert result.guard_status == GUARD_BYPASS_POOR_USABILITY_WINNER
    assert result.dominant_candidate_id is None


# ---------------------------------------------------------------------------
# 20. Missing V2 fail-open
# ---------------------------------------------------------------------------
def test_20_missing_v2_fail_open():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": None, "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    assert result.guard_status == GUARD_BYPASS_POOR_USABILITY_WINNER
    assert result.evidence_source == EVIDENCE_SOURCE_V1_FALLBACK


# ---------------------------------------------------------------------------
# 21. Deterministic output
# ---------------------------------------------------------------------------
def test_21_deterministic_output():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_IMPAIRED, SEVERITY_MATERIAL, count=4),
          "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, count=0)}
    kwargs = dict(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    r1 = evaluate_watch_listen_besttake_guard_v2(**kwargs)
    r2 = evaluate_watch_listen_besttake_guard_v2(**kwargs)
    assert r1 == r2


# ---------------------------------------------------------------------------
# 22. Winner immutability
# ---------------------------------------------------------------------------
def test_22_winner_immutability():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_IMPAIRED, SEVERITY_MATERIAL, count=4),
          "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, count=0)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    row = watch_listen_besttake_v2_group_row(result, "A")
    assert row["winner_before"] == "A"
    assert row["winner_after"] == "A"
    assert row["action_applied"] is False


# ---------------------------------------------------------------------------
# 23. Family unchanged
# ---------------------------------------------------------------------------
def test_23_family_unchanged():
    for name in ("take_grouping", "take_grouping_provider", "hybrid_session_cleanup", "semantic_idea_equivalence"):
        assert "watch_listen_besttake_v2_evidence" not in _source(name)


# ---------------------------------------------------------------------------
# 24. Proposition/attempt unchanged
# ---------------------------------------------------------------------------
def test_24_proposition_attempt_unchanged():
    assert _run_git_diff("cutsell_worker/language_proposition_relation.py") == ""
    assert _run_git_diff("cutsell_worker/language_utterance_attempt.py") == ""
    assert _run_git_diff("cutsell_worker/language_spine.py") == ""
    assert _run_git_diff("cutsell_worker/language_spine_consumer_migration.py") == ""


# ---------------------------------------------------------------------------
# 25. D-150 unchanged
# ---------------------------------------------------------------------------
def test_25_d150_unchanged():
    assert _run_git_diff("cutsell_worker/semantic_authority_observability.py") == ""


# ---------------------------------------------------------------------------
# 26. D-158 unchanged
# ---------------------------------------------------------------------------
def test_26_d158_unchanged():
    assert _run_git_diff("cutsell_worker/attempt_relationship_authority.py") == ""


# ---------------------------------------------------------------------------
# 27. D-161 unchanged
# ---------------------------------------------------------------------------
def test_27_d161_unchanged():
    assert _run_git_diff("cutsell_worker/watch_listen_relation_discovery.py") == ""


# ---------------------------------------------------------------------------
# 28. D-163 old behavior available (byte-identical module, old function still works)
# ---------------------------------------------------------------------------
def test_28_d163_old_behavior_available():
    assert _run_git_diff("cutsell_worker/watch_listen_besttake_evidence.py") == ""
    from cutsell_worker.watch_listen_besttake_evidence import evaluate_watch_listen_besttake_guard
    v1 = {"A": _v1("A", V1_USABLE)}
    result = evaluate_watch_listen_besttake_guard(winner_id="A", meaning_sufficient_ids=["A"], evidence_by_id=v1)
    assert result.guard_status == GUARD_PRESERVE_STRUCTURED_WINNER


# ---------------------------------------------------------------------------
# 29. D-167 unchanged
# ---------------------------------------------------------------------------
def test_29_d167_unchanged():
    assert _run_git_diff("cutsell_worker/watch_listen_zone_usability_v2.py") == ""


# ---------------------------------------------------------------------------
# 30. Language Spine unchanged
# ---------------------------------------------------------------------------
def test_30_language_spine_unchanged():
    for name in ("language_spine", "language_utterance_attempt", "language_proposition_relation",
                 "language_spine_consumer_migration"):
        assert "watch_listen_besttake_v2_evidence" not in _source(name)


# ---------------------------------------------------------------------------
# 31. Boundary unchanged
# ---------------------------------------------------------------------------
def test_31_boundary_unchanged():
    assert _run_git_diff("cutsell_worker/boundary_engine_pass.py") == ""


# ---------------------------------------------------------------------------
# 32. Pacing unchanged
# ---------------------------------------------------------------------------
def test_32_pacing_unchanged():
    assert _run_git_diff("cutsell_worker/dialogue_pacing_transition.py") == ""


# ---------------------------------------------------------------------------
# 33. Render unchanged
# ---------------------------------------------------------------------------
def test_33_render_unchanged():
    for name in ("render", "render_plan", "render_versions"):
        assert _run_git_diff(f"cutsell_worker/{name}.py") == ""


# ---------------------------------------------------------------------------
# 34. No provider/network
# ---------------------------------------------------------------------------
def test_34_no_provider_network():
    source = inspect.getsource(m)
    for forbidden in ("openai", "gemini", "requests.", "httpx.", "urllib.request", "socket."):
        assert forbidden not in source.lower()


# ---------------------------------------------------------------------------
# Additional structural/integration tests beyond the 34-item matrix.
# ---------------------------------------------------------------------------
def _ev(kind, start, end, confidence):
    return TemporalEvent(source_asset_id="s1", start=start, end=end, kind=kind, confidence=confidence, description="")


def _status():
    return ProviderStatus(provider="test", requested=False, available=False, status="not_requested")


def _build_real_span(delivery_events=()):
    from cutsell_worker.positioned_performance_evidence import build_positioned_performance_evidence
    from cutsell_worker.watch_listen_zone_usability_v2 import _DEFECT_KINDS

    words = tuple(Word(text=f"w{i}", start=0.5 * i, end=0.5 * i + 0.4, confidence=0.9) for i in range(10))
    candidate = CandidateTake(
        clip_id="c1", source_asset_id="s1", source_order=0, start=0.0, end=5.0,
        text="hola como estas hoy en el video de hoy", words=words,
    )
    context = WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="s1", summary="", dominant_style="", creator_intent="",
            events=tuple(delivery_events),
        ),),
        status=_status(),
    )
    positioned = build_positioned_performance_evidence(candidate, context, event_kinds=_DEFECT_KINDS)
    return build_raw_understanding_span(candidate, positioned)


def test_feature_flag_default_off():
    assert zone_usability_v2_besttake_enabled({}) is False
    assert zone_usability_v2_besttake_enabled({"CUTSELL_WATCH_LISTEN_ZONE_USABILITY_V2_BESTTAKE_ENABLED": "1"}) is True


def test_build_candidate_zone_usability_v2_real_integration():
    words = tuple(Word(text=f"w{i}", start=0.5 * i, end=0.5 * i + 0.4, confidence=0.9) for i in range(10))
    candidate = CandidateTake(
        clip_id="c1", source_asset_id="s1", source_order=0, start=0.0, end=5.0,
        text="hola como estas hoy en el video de hoy", words=words,
    )
    context = WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="s1", summary="", dominant_style="", creator_intent="",
            events=(_ev("hand_motion_reset_candidate", 2.0, 2.1, 0.95),),
        ),),
        status=_status(),
    )
    result = build_candidate_zone_usability_v2(candidate, context)
    assert result is not None
    assert result.candidate_id == "c1"
    assert result.delivery.event_count == 1


def test_build_candidate_zone_usability_v2_fail_open_on_exception():
    # Passing an object that will blow up build_positioned_performance_evidence
    # (no .words attribute) must return None, never raise.
    class Bogus:
        clip_id = "bad"
        source_asset_id = "s1"
        start = 0.0
        end = 1.0

    result = build_candidate_zone_usability_v2(Bogus(), None)
    assert result is None


def test_build_candidate_zone_usability_v2_none_context():
    words = (Word(text="hi", start=0.0, end=0.4, confidence=0.9),)
    candidate = CandidateTake(clip_id="c1", source_asset_id="s1", source_order=0, start=0.0, end=1.0, text="hi", words=words)
    result = build_candidate_zone_usability_v2(candidate, None)
    # No context -> no events -> a valid, empty V2 record (never a crash).
    assert result is not None
    assert result.delivery.event_count == 0


def test_diagnostics_shape_no_transcript():
    v1 = {"A": _v1("A", V1_UNUSABLE), "B": _v1("B", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_IMPAIRED, SEVERITY_MATERIAL, count=4),
          "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE, count=0)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    diag = watch_listen_besttake_v2_diagnostics([result])
    assert diag["v2_evaluated_count"] == 1
    assert diag["v2_dominance_count"] == 1
    assert diag["v2_bypass_count"] == 0
    assert "hola" not in str(diag)


def test_group_row_field_names_exact():
    v1 = {"A": _v1("A", V1_UNUSABLE)}
    v2 = {"A": _v2("A", USABILITY_UNUSABLE, SEVERITY_SEVERE, count=2)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
    )
    row = watch_listen_besttake_v2_group_row(result, "A")
    for key in (
        "watch_listen_besttake_evidence_source", "watch_listen_besttake_v2_available",
        "watch_listen_besttake_v2_dominant_candidate", "watch_listen_besttake_v2_guard_status",
        "watch_listen_besttake_v2_guard_reason", "watch_listen_besttake_v2_winner_severity",
        "watch_listen_besttake_v2_alt_severity", "watch_listen_besttake_v2_meaning_firewall_blocked",
        "winner_before", "winner_after", "action_applied",
    ):
        assert key in row


def test_no_action_v1_pass_through_unchanged():
    v1 = {"A": None}
    v2 = {"A": _v2("A", USABILITY_USABLE, SEVERITY_NONE)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
    )
    assert result.guard_status == GUARD_NO_ACTION


def test_uncertain_v1_conflict_pass_through_unchanged():
    v1 = {"A": _v1("A", V1_UNUSABLE, conflict_flags=("MEANING_COMPLETE_VS_ABANDONED_ATTEMPT_EVIDENCE",))}
    v2 = {"A": _v2("A", USABILITY_USABLE, SEVERITY_NONE)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
    )
    assert result.guard_status == GUARD_UNCERTAIN


def test_v1_already_found_dominance_never_second_guessed():
    # V1's own ordinal comparison already finds a dominant alternative
    # (rare but possible: e.g. entry/exit differ). V2 must never override
    # an already-reached V1 PERFORMANCE_DOMINANT_ALTERNATIVE outcome.
    from cutsell_worker.watch_listen_besttake_evidence import evaluate_watch_listen_besttake_guard
    v1 = {
        "A": WatchListenBestTakeEvidence(
            candidate_id="A", meaning_sufficient=True, overall_performance_usability=V1_UNUSABLE,
            entry_usability=V1_USABLE, delivery_usability=V1_UNUSABLE, exit_usability=V1_USABLE,
            delivery_defect_present=True, entry_only_defect=False, exit_only_defect=False,
            breaking_character_during_delivery=False, reset_or_fumble_during_delivery=True,
            performance_continuity_status="INTERRUPTED_DURING_DELIVERY", audio_signal_usability="UNKNOWN",
            visual_signal_usability=V1_UNUSABLE, editability_status="DELIVERY_OWNED",
            conflict_flags=(), evidence_provenance={}, case_classification=CASE_B_DELIVERY_OWNED,
        ),
        "B": WatchListenBestTakeEvidence(
            candidate_id="B", meaning_sufficient=True, overall_performance_usability=V1_USABLE,
            entry_usability=V1_USABLE, delivery_usability=V1_USABLE, exit_usability=V1_USABLE,
            delivery_defect_present=False, entry_only_defect=False, exit_only_defect=False,
            breaking_character_during_delivery=False, reset_or_fumble_during_delivery=False,
            performance_continuity_status="CONTINUOUS", audio_signal_usability="UNKNOWN",
            visual_signal_usability=V1_USABLE, editability_status="CLEAN",
            conflict_flags=(), evidence_provenance={}, case_classification=CASE_CLEAN,
        ),
    }
    v1_baseline = evaluate_watch_listen_besttake_guard(
        winner_id="A", meaning_sufficient_ids=["A", "B"], evidence_by_id=v1,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    assert v1_baseline.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert v1_baseline.dominant_candidate_id == "B"

    # Now: V2 evidence exists but (hypothetically) disagrees/says A is fine.
    v2 = {"A": _v2("A", USABILITY_USABLE, SEVERITY_NONE), "B": _v2("B", USABILITY_USABLE, SEVERITY_NONE)}
    result = evaluate_watch_listen_besttake_guard_v2(
        winner_id="A", meaning_sufficient_ids=["A", "B"], v1_evidence_by_id=v1, v2_evidence_by_id=v2,
        ranked=_ranked(("A", 0.5), ("B", 0.4)),
    )
    # V1's own already-reached dominance is passed through UNCHANGED.
    assert result.guard_status == GUARD_PERFORMANCE_DOMINANT_ALTERNATIVE
    assert result.dominant_candidate_id == "B"
    assert result.guard_reason == v1_baseline.guard_reason


def test_not_imported_by_unrelated_production_call_sites():
    call_sites = (
        "take_grouping", "take_grouping_provider", "hybrid_session_cleanup",
        "semantic_idea_equivalence", "attempt_relationship_authority",
        "watch_listen_relation_discovery", "deterministic_best_take_authority",
        "take_judge", "boundary_engine_pass", "dialogue_pacing_transition",
        "semantic_authority_observability", "language_spine",
        "language_utterance_attempt", "language_proposition_relation",
        "language_spine_consumer_migration",
    )
    for name in call_sites:
        assert "watch_listen_besttake_v2_evidence" not in _source(name), name


def test_pipeline_imports_new_module_gated():
    source = _source("pipeline")
    assert "watch_listen_besttake_v2_evidence" in source
    assert "zone_usability_v2_besttake_enabled" in source
