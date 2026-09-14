"""D-239F: LIVE LOST-ATOM OWNERSHIP -> MATERIALITY -> FREEZE
OBSERVABILITY -- OFFLINE ONLY.

Covers this task's own required matrix: the D-239-shape offline replay
(real recovered clip_id/word-index/attempt/proposition numbers), the 7
fixture scenarios (meaning-critical, editorial-required, critical-
conflict, unknown-context, retry/process, non-material, ambiguity), the
D-235S/D-235T capture-not-recompute proof, and no-policy-change proofs.
Mirrors the established D-235-series source-code-truth + fixture-matrix
test style.
"""
from __future__ import annotations

import inspect
import subprocess

import cutsell_worker.lost_atom_ownership_materiality_diagnostics as loamd
import cutsell_worker.repair_loop as repair_loop_module
from cutsell_worker.complete_lost_semantic_atom_materiality import (
    assess_complete_lost_semantic_atom_materiality,
)
from cutsell_worker.exact_lost_atom_ownership import (
    LanguageAttemptWordEvidence,
    assess_exact_lost_atom_ownership,
)
from cutsell_worker.lost_atom_ownership_materiality_diagnostics import (
    build_lost_atom_ownership_materiality_diagnostics,
    lost_atom_repair_suppression_by_provenance_diagnostics,
)
from cutsell_worker.lost_atom_repair_suppression import (
    LostAtomRepairSuppressionDecision,
    PRESERVE_REPAIR_ESCALATION,
    SUPPRESS_SAME_NON_MATERIAL_ATOM,
)
from cutsell_worker.repair_loop import RepairAttempt

REPO_ROOT = __file__.rsplit("/tests/", 1)[0]


def _diff_stat(*paths: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", *paths],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return result.stdout.strip()


def _row(**overrides):
    base = {
        "clip_id": "clip_1d9d2cebaf8ed3004836",
        "text": "too many people ready set these are the",
        "classification": "REAL_CONTENT_LOSS",
        "blocking": True,
        "missing_critical_atoms": (),
        "atom_classifications": (),
        "own_content_token_count": 5,
        "coverage_against_final_keep": 0.9,
        "lost_atom_provenance_id": "latom_clip_1d9d2cebaf8ed3004836_0",
    }
    base.update(overrides)
    return base


def _real_d239_ownership(clip_id="clip_1d9d2cebaf8ed3004836"):
    """Reproduces the EXACT real D-239 shape recovered from run 34676546250's
    externally-retrieved artifact: candidate 41..48 (8 words), sole
    containing LanguageAttempt latt_be3141887f7b37d1bb3a spanning 31..258
    (228 words), one proposition, same source."""
    return assess_exact_lost_atom_ownership(
        clip_id=clip_id,
        candidate_source_asset_id="src_8f6265cea717b4ac1467",
        candidate_word_indices=range(41, 49),
        language_attempts=[
            LanguageAttemptWordEvidence(
                attempt_id="latt_be3141887f7b37d1bb3a",
                source_asset_id="src_8f6265cea717b4ac1467",
                word_indices=tuple(range(31, 259)),
                proposition_candidate_ids=("prop_475b7c5a7d7ebc6e6e8c",),
            ),
        ],
    )


# ===========================================================================
# 1-3: the D-239-shape offline replay -- ownership serializes correctly.
# ===========================================================================
class TestD239ShapeOfflineReplay:
    def test_01_real_shape_reproduces_exact_singleton(self):
        ownership = _real_d239_ownership()
        assert ownership.ownership_status == "EXACT_SINGLETON_OWNERSHIP"
        assert ownership.is_exact_singleton is True
        assert ownership.containing_language_attempt_id == "latt_be3141887f7b37d1bb3a"
        assert ownership.proposition_candidate_ids == ("prop_475b7c5a7d7ebc6e6e8c",)

    def test_02_diagnostics_serializes_ownership_fields_for_real_shape(self):
        ownership = _real_d239_ownership()
        diag = build_lost_atom_ownership_materiality_diagnostics(
            [_row()], lost_atom_ownership_by_clip_id={"clip_1d9d2cebaf8ed3004836": ownership},
        )
        assert diag["atom_count"] == 1
        atom = diag["atoms"][0]
        assert atom["ownership_input_present"] is True
        assert atom["ownership_status"] == "EXACT_SINGLETON_OWNERSHIP"
        assert atom["containing_language_attempt_id"] == "latt_be3141887f7b37d1bb3a"
        assert atom["proposition_candidate_ids"] == ["prop_475b7c5a7d7ebc6e6e8c"]
        assert atom["ownership_ambiguity_reason"] == []
        assert atom["bounded_excerpt"] == "too many people ready set these are the"
        assert atom["lost_atom_provenance_id"] == "latom_clip_1d9d2cebaf8ed3004836_0"

    def test_03_without_ownership_input_present_is_false(self):
        diag = build_lost_atom_ownership_materiality_diagnostics([_row()])
        atom = diag["atoms"][0]
        assert atom["ownership_input_present"] is False
        assert atom["ownership_status"] is None
        assert atom["containing_language_attempt_id"] is None
        assert atom["proposition_candidate_ids"] == []


# ===========================================================================
# 4-10: the 7 required fixture scenarios -- D-235Q/R states distinguished.
# ===========================================================================
class TestSevenFixtureScenarios:
    def _materiality_and_freeze(self, ownership=None, **assess_kwargs):
        m = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=ownership, **assess_kwargs,
        )
        diag = build_lost_atom_ownership_materiality_diagnostics(
            [_row()],
            lost_atom_ownership_by_clip_id=({"clip_1d9d2cebaf8ed3004836": ownership} if ownership else None),
            materiality_by_clip_id={"clip_1d9d2cebaf8ed3004836": m},
        )
        return diag["atoms"][0]

    def test_04_meaning_critical(self):
        atom = self._materiality_and_freeze(
            ownership=_real_d239_ownership(), critical_claim_conflict=True,
        )
        assert atom["meaning_critical_state"] == "MEANING_CRITICAL"
        assert atom["blocking_recommendation"] == "BLOCK"
        assert atom["freeze_authority_status"] == "PRESERVE_BLOCK"
        assert atom["freeze_effective_blocking"] is True

    def test_05_editorial_required(self):
        from cutsell_worker.shared_attempt_word_identity import (
            AttemptLanguageIdentityMatch, RELATIONSHIP_EXACT_SAME_MEMBERSHIP, WordMembership,
        )
        rm = WordMembership("s1", "r1", (0, 1), "AVAILABLE")
        lms = (WordMembership("s1", "latt1", (0, 1), "AVAILABLE"),)
        match = AttemptLanguageIdentityMatch(
            reconstructed_attempt_id="r1", language_attempt_ids=("latt1",), source_asset_id="s1",
            reconstructed_word_membership=rm, language_word_memberships=lms,
            relationship_status=RELATIONSHIP_EXACT_SAME_MEMBERSHIP, exact_shared_word_count=2,
            reconstructed_word_count=2, language_word_count=2, provenance=("test",),
        )
        atom = self._materiality_and_freeze(
            ownership=_real_d239_ownership(), exact_match=match,
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "HOOK"}, idea_coverage_status=True,
        )
        assert atom["editorial_requirement_state"] == "REQUIRED"
        assert atom["blocking_recommendation"] == "BLOCK"
        assert atom["freeze_authority_status"] == "PRESERVE_BLOCK"

    def test_06_critical_conflict(self):
        atom = self._materiality_and_freeze(ownership=_real_d239_ownership(), critical_claim_conflict=True)
        assert atom["final_materiality_status"] == "MEANING_CRITICAL"
        assert atom["blocking_recommendation"] == "BLOCK"

    def test_07_unknown_context(self):
        # critical_claim_conflict left None (genuinely unknown) -- even with
        # ownership present, must stay ABSTAIN, never manufactured DO_NOT_BLOCK.
        atom = self._materiality_and_freeze(ownership=_real_d239_ownership())
        assert atom["final_materiality_status"] == "INSUFFICIENT_EVIDENCE"
        assert atom["blocking_recommendation"] == "ABSTAIN"
        assert atom["freeze_authority_status"] == "ABSTAIN_PRESERVE_BLOCK"

    def test_08_retry_process(self):
        atom = self._materiality_and_freeze(
            ownership=_real_d239_ownership(), critical_claim_conflict=False, recording_process_evidence=True,
        )
        assert atom["retry_process_state"] == "FOUND"
        assert atom["final_materiality_status"] == "RETRY_OR_RECORDING_RESIDUE"
        assert atom["blocking_recommendation"] == "DO_NOT_BLOCK"
        assert atom["freeze_authority_status"] == "SUPPRESS_NON_MATERIAL_BLOCK"
        assert atom["freeze_effective_blocking"] is False

    def test_09_non_material_with_ownership_do_not_block(self):
        atom = self._materiality_and_freeze(ownership=_real_d239_ownership(), critical_claim_conflict=False)
        assert atom["final_materiality_status"] == "NON_MATERIAL_REAL_CONTENT"
        assert atom["blocking_recommendation"] == "DO_NOT_BLOCK"
        assert atom["exact_ownership_available"] is True
        assert atom["freeze_authority_status"] == "SUPPRESS_NON_MATERIAL_BLOCK"
        assert atom["freeze_effective_blocking"] is False

    def test_10_ambiguity_multiple_attempts_never_singleton(self):
        ambiguous = assess_exact_lost_atom_ownership(
            clip_id="clip_1d9d2cebaf8ed3004836", candidate_source_asset_id="s1",
            candidate_word_indices=(1, 2, 3),
            language_attempts=[
                LanguageAttemptWordEvidence("a1", "s1", tuple(range(0, 10)), ("p1",)),
                LanguageAttemptWordEvidence("a2", "s1", tuple(range(0, 10)), ("p2",)),
            ],
        )
        diag = build_lost_atom_ownership_materiality_diagnostics(
            [_row()], lost_atom_ownership_by_clip_id={"clip_1d9d2cebaf8ed3004836": ambiguous},
        )
        atom = diag["atoms"][0]
        assert atom["ownership_status"] == "AMBIGUOUS_MULTIPLE_ATTEMPTS"
        assert atom["ownership_ambiguity_reason"] == ["multiple_language_attempts_each_fully_contain_target"]
        assert atom["exact_ownership_available"] is None  # no materiality supplied in this fixture
        # Feeding a non-singleton ownership through the real D-235Q gate
        # never flips exact_ownership_available True.
        m = assess_complete_lost_semantic_atom_materiality(_row(), lost_atom_ownership=ambiguous)
        assert m.exact_ownership_available is False


# ===========================================================================
# 11-14: D-235S/D-235T capture-not-recompute proof.
# ===========================================================================
class TestD235SD235TCapture:
    def test_11_repair_loop_result_captures_suppression_decisions_field(self):
        assert "suppression_decisions" in repair_loop_module.RepairLoopResult.__dataclass_fields__

    def test_12_projection_reads_decision_verbatim_never_recomputes(self):
        decision = LostAtomRepairSuppressionDecision(
            lost_atom_provenance_id="latom_x", reviewer_finding_kind="UNIQUE_FACT_LOST",
            repair_attempt_status="NO_REPAIR_STRATEGY_EXISTS", freeze_authority_status="SUPPRESS_NON_MATERIAL_BLOCK",
            materiality_status="NON_MATERIAL_REAL_CONTENT", suppression_status=SUPPRESS_SAME_NON_MATERIAL_ATOM,
            suppress_repair_escalation=True, reason="same_atom_already_qualifies_for_freeze_authority_suppression",
            provenance=("test",),
        )
        attempt = RepairAttempt(
            plan_id="p1", previous_plan_version=1, new_plan_version=1, finding_kind="UNIQUE_FACT_LOST",
            idea_id="i1", owning_authority="selection", previous_realization=("clip_1d9d2cebaf8ed3004836",),
            replacement_realization=("clip_1d9d2cebaf8ed3004836",), coverage_before="covered", coverage_after="covered",
            reason="no_repair_strategy_exists_for_this_finding_kind", unaffected_ideas_changed=False, repaired=False,
            source_lost_atom_provenance_id="latom_x",
        )
        diag = lost_atom_repair_suppression_by_provenance_diagnostics(
            suppression_decisions=(decision,), repair_attempts=(attempt,),
        )
        assert diag["entry_count"] == 1
        entry = diag["entries"][0]
        assert entry["lost_atom_provenance_id"] == "latom_x"
        assert entry["d235s_repair_attempt_provenance_confirmed"] is True
        assert entry["d235s_exact_link_status"] == "EXACT_MATCH"
        assert entry["d235t_precomputed_materiality_received"] is True
        assert entry["d235t_suppression_status"] == SUPPRESS_SAME_NON_MATERIAL_ATOM
        assert entry["d235t_suppress_repair_escalation"] is True
        assert diag["suppressed_count"] == 1
        assert diag["preserved_count"] == 0

    def test_13_preserved_decision_reports_preserve_status(self):
        decision = LostAtomRepairSuppressionDecision(
            lost_atom_provenance_id="latom_y", reviewer_finding_kind="UNIQUE_FACT_LOST",
            repair_attempt_status="NO_REPAIR_STRATEGY_EXISTS", freeze_authority_status="PRESERVE_BLOCK",
            materiality_status="MEANING_CRITICAL", suppression_status=PRESERVE_REPAIR_ESCALATION,
            suppress_repair_escalation=False, reason="freeze_authority_preserved_block",
            provenance=("test",),
        )
        diag = lost_atom_repair_suppression_by_provenance_diagnostics(suppression_decisions=(decision,))
        entry = diag["entries"][0]
        assert entry["d235t_suppress_repair_escalation"] is False
        assert diag["suppressed_count"] == 0
        assert diag["preserved_count"] == 1

    def test_14_ambiguous_provenance_link_reported_honestly(self):
        decision = LostAtomRepairSuppressionDecision(
            lost_atom_provenance_id="latom_z", reviewer_finding_kind="UNIQUE_FACT_LOST",
            repair_attempt_status="NO_REPAIR_STRATEGY_EXISTS", freeze_authority_status=None,
            materiality_status=None, suppression_status="ABSTAIN_PRESERVE_ESCALATION",
            suppress_repair_escalation=False, reason="provenance_link_ambiguous",
            provenance=("test",),
        )
        diag = lost_atom_repair_suppression_by_provenance_diagnostics(suppression_decisions=(decision,))
        entry = diag["entries"][0]
        assert entry["d235s_exact_link_status"] == "provenance_link_ambiguous"
        assert entry["d235t_precomputed_materiality_received"] is False


# ===========================================================================
# 15-16: distinguishability matrix (A-E from the task's own directive).
# ===========================================================================
class TestDistinguishabilityMatrix:
    def test_15_case_a_ownership_not_singleton(self):
        non_singleton = assess_exact_lost_atom_ownership(
            clip_id="c1", candidate_source_asset_id="s1", candidate_word_indices=(1, 2),
            language_attempts=[],
        )
        diag = build_lost_atom_ownership_materiality_diagnostics(
            [_row(clip_id="c1")], lost_atom_ownership_by_clip_id={"c1": non_singleton},
        )
        atom = diag["atoms"][0]
        assert atom["ownership_status"] != "EXACT_SINGLETON_OWNERSHIP"

    def test_16_case_b_ownership_exact_but_q_abstains(self):
        # critical_claim_conflict left unknown -> ABSTAIN despite ownership.
        m = assess_complete_lost_semantic_atom_materiality(_row(), lost_atom_ownership=_real_d239_ownership())
        assert m.exact_ownership_available is True
        assert m.blocking_recommendation == "ABSTAIN"

    def test_17_case_c_q_non_material_r_would_suppress(self):
        m = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=_real_d239_ownership(), critical_claim_conflict=False,
        )
        from cutsell_worker.lost_semantic_atom_freeze_authority import decide_lost_semantic_atom_freeze_authority
        d = decide_lost_semantic_atom_freeze_authority(_row(), m)
        assert m.final_materiality_status == "NON_MATERIAL_REAL_CONTENT"
        assert d.authority_status == "SUPPRESS_NON_MATERIAL_BLOCK"
        assert d.effective_blocking is False


# ===========================================================================
# 18-24: no-policy-change proofs.
# ===========================================================================
class TestNoPolicyChange:
    def test_18_no_authoritative_relationship_statuses_diff(self):
        diff = _diff_stat("cutsell_worker/shared_attempt_word_identity.py")
        assert diff == "", diff

    def test_19_no_ownership_policy_diff(self):
        # exact_lost_atom_ownership.py's own gate is untouched by D-239F.
        diff = _diff_stat("cutsell_worker/exact_lost_atom_ownership.py")
        assert diff == "", diff

    def test_20_diagnostics_module_never_imports_authoritative_statuses(self):
        imports = "\n".join(
            l for l in inspect.getsource(loamd).splitlines()
            if l.strip().startswith(("import ", "from "))
        )
        assert "AUTHORITATIVE_RELATIONSHIP_STATUSES" not in imports
        assert "shared_attempt_word_identity" not in imports

    def test_21_diagnostics_module_never_calls_assess_exact_lost_atom_ownership(self):
        source = inspect.getsource(loamd)
        assert "assess_exact_lost_atom_ownership(" not in source

    def test_22_diagnostics_module_never_calls_assess_complete_materiality(self):
        source = inspect.getsource(loamd)
        assert "assess_complete_lost_semantic_atom_materiality(" not in source

    def test_23_diagnostics_module_never_calls_decide_lost_atom_repair_suppression(self):
        source = inspect.getsource(loamd)
        assert "decide_lost_atom_repair_suppression(" not in source

    def test_24_diagnostics_module_reuses_freeze_authority_verbatim(self):
        # The ONE permitted pure re-derivation, matching this codebase's own
        # pre-existing precedent (_lost_atom_materiality_orchestration_diagnostics).
        source = inspect.getsource(loamd)
        assert "decide_lost_semantic_atom_freeze_authority(" in source


# ===========================================================================
# 25: bounded output / no transcript leakage.
# ===========================================================================
class TestBoundedOutput:
    def test_25_excerpt_bounded_never_full_transcript(self):
        long_text = "x" * 5000
        diag = build_lost_atom_ownership_materiality_diagnostics([_row(text=long_text)])
        assert len(diag["atoms"][0]["bounded_excerpt"]) <= 160
