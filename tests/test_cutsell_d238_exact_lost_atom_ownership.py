"""D-238: BOUNDED EXACT SINGLETON LOST-ATOM OWNERSHIP -- OFFLINE ONLY.

Covers this task's own required matrix: the 8-status vocabulary, the
9-condition structural gate, the real D-237M shape replay, the 21-item
counterexample matrix (as many as apply at each level), the D-235Q/R/T
integration/precedence proofs, and the no-false-authority proofs. Mirrors
the established D-235-series / D-237-series source-code-truth +
fixture-matrix test style (see test_cutsell_d237l_clip_key_namespace_fix.py).
"""
from __future__ import annotations

import inspect
import subprocess

import cutsell_worker.complete_lost_semantic_atom_materiality as clsam
import cutsell_worker.exact_lost_atom_ownership as elao
import cutsell_worker.lost_semantic_atom_freeze_authority as lsafa
from cutsell_worker.exact_lost_atom_ownership import (
    ExactLostAtomOwnership,
    LanguageAttemptWordEvidence,
    OWNERSHIP_ABSTAIN,
    OWNERSHIP_AMBIGUOUS_MULTIPLE_ATTEMPTS,
    OWNERSHIP_AMBIGUOUS_MULTIPLE_PROPOSITIONS,
    OWNERSHIP_EXACT_SINGLETON,
    OWNERSHIP_MISSING_WORD_PROVENANCE,
    OWNERSHIP_NO_CONTAINING_ATTEMPT,
    OWNERSHIP_PARTIAL_OVERLAP,
    OWNERSHIP_SOURCE_MISMATCH,
    OWNERSHIP_STATUSES_SUFFICIENT_FOR_IDENTITY_GATE,
    assess_exact_lost_atom_ownership,
    exact_lost_atom_ownership_diagnostics,
)
from cutsell_worker.shared_attempt_word_identity import AUTHORITATIVE_RELATIONSHIP_STATUSES

REPO_ROOT = __file__.rsplit("/tests/", 1)[0]
OWNERSHIP_PATH = "cutsell_worker/exact_lost_atom_ownership.py"
Q_PATH = "cutsell_worker/complete_lost_semantic_atom_materiality.py"
R_PATH = "cutsell_worker/lost_semantic_atom_freeze_authority.py"
T_PATH = "cutsell_worker/lost_atom_repair_suppression.py"
P_PATH = "cutsell_worker/shared_attempt_word_identity.py"


def _diff_stat(*paths: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", *paths],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return result.stdout.strip()


def _attempt(attempt_id, source, indices, props=()):
    return LanguageAttemptWordEvidence(
        attempt_id=attempt_id, source_asset_id=source,
        word_indices=tuple(indices), proposition_candidate_ids=tuple(props),
    )


def _assess_ownership(**kwargs):
    defaults = dict(
        clip_id="clip_1", candidate_source_asset_id="src_1",
        candidate_word_indices=(1, 2, 3), language_attempts=(),
    )
    defaults.update(kwargs)
    return assess_exact_lost_atom_ownership(**defaults)


# ===========================================================================
# 1-8: the 8-value status vocabulary is exhaustive and each is reachable.
# ===========================================================================
class TestStatusVocabulary:
    def test_01_all_eight_statuses_are_valid(self):
        assert elao._VALID_OWNERSHIP_STATUSES == frozenset({
            OWNERSHIP_EXACT_SINGLETON, OWNERSHIP_AMBIGUOUS_MULTIPLE_ATTEMPTS,
            OWNERSHIP_AMBIGUOUS_MULTIPLE_PROPOSITIONS, OWNERSHIP_SOURCE_MISMATCH,
            OWNERSHIP_MISSING_WORD_PROVENANCE, OWNERSHIP_NO_CONTAINING_ATTEMPT,
            OWNERSHIP_PARTIAL_OVERLAP, OWNERSHIP_ABSTAIN,
        })
        assert len(elao._VALID_OWNERSHIP_STATUSES) == 8

    def test_02_only_exact_singleton_sufficient_for_identity_gate(self):
        assert OWNERSHIP_STATUSES_SUFFICIENT_FOR_IDENTITY_GATE == frozenset({OWNERSHIP_EXACT_SINGLETON})

    def test_03_invalid_status_raises(self):
        import pytest
        with pytest.raises(ValueError):
            ExactLostAtomOwnership(
                clip_id="c", source_asset_id="s", candidate_word_indices=(1,),
                containing_language_attempt_id=None, proposition_candidate_ids=(),
                ownership_status="NOT_A_REAL_STATUS", reason_codes=(), provenance=(),
            )

    def test_04_missing_word_provenance_empty_target(self):
        res = _assess_ownership(candidate_word_indices=())
        assert res.ownership_status == OWNERSHIP_MISSING_WORD_PROVENANCE
        assert res.is_exact_singleton is False

    def test_05_no_containing_attempt_no_attempts_at_all(self):
        res = _assess_ownership(language_attempts=())
        assert res.ownership_status == OWNERSHIP_NO_CONTAINING_ATTEMPT

    def test_06_source_mismatch_only_other_source_attempts_exist(self):
        res = _assess_ownership(language_attempts=[_attempt("a1", "src_OTHER", range(0, 10))])
        assert res.ownership_status == OWNERSHIP_SOURCE_MISMATCH

    def test_07_partial_overlap_no_full_containment(self):
        res = _assess_ownership(
            candidate_word_indices=(5, 6, 7),
            language_attempts=[_attempt("a1", "src_1", (4, 5, 6))],
        )
        assert res.ownership_status == OWNERSHIP_PARTIAL_OVERLAP

    def test_08_abstain_when_containing_attempt_owns_zero_propositions(self):
        res = _assess_ownership(
            candidate_word_indices=(1, 2, 3),
            language_attempts=[_attempt("a1", "src_1", range(0, 10), props=())],
        )
        assert res.ownership_status == OWNERSHIP_ABSTAIN
        assert res.is_exact_singleton is False


# ===========================================================================
# 9-17: the 9-condition minimum structural gate, each condition isolated.
# ===========================================================================
class TestNineConditionGate:
    def test_09_condition1_empty_candidate_set_abstains(self):
        res = _assess_ownership(candidate_word_indices=())
        assert res.ownership_status == OWNERSHIP_MISSING_WORD_PROVENANCE

    def test_10_condition2_source_mismatch_blocks_ownership(self):
        res = _assess_ownership(
            candidate_source_asset_id="src_A",
            language_attempts=[_attempt("a1", "src_B", range(0, 20), props=("p1",))],
        )
        assert res.ownership_status == OWNERSHIP_SOURCE_MISMATCH

    def test_11_condition3_reconstructed_only_words_block_singleton(self):
        # candidate has a word (99) the attempt does not contain -> not a subset.
        res = _assess_ownership(
            candidate_word_indices=(1, 2, 99),
            language_attempts=[_attempt("a1", "src_1", range(0, 10), props=("p1",))],
        )
        assert res.ownership_status == OWNERSHIP_PARTIAL_OVERLAP

    def test_12_condition4_exactly_one_containing_attempt_required(self):
        res = _assess_ownership(
            candidate_word_indices=(1, 2, 3),
            language_attempts=[
                _attempt("a1", "src_1", range(0, 10), props=("p1",)),
                _attempt("a2", "src_1", range(0, 10), props=("p2",)),
            ],
        )
        assert res.ownership_status == OWNERSHIP_AMBIGUOUS_MULTIPLE_ATTEMPTS

    def test_13_condition5_second_partial_overlap_still_ambiguous(self):
        # a1 fully contains target; a2 only partially overlaps it. Still ambiguous.
        res = _assess_ownership(
            candidate_word_indices=(5, 6, 7),
            language_attempts=[
                _attempt("a1", "src_1", range(0, 20), props=("p1",)),
                _attempt("a2", "src_1", (6, 7, 8, 9)),
            ],
        )
        assert res.ownership_status == OWNERSHIP_AMBIGUOUS_MULTIPLE_ATTEMPTS

    def test_14_condition6_exactly_one_proposition_required(self):
        res = _assess_ownership(
            candidate_word_indices=(1, 2, 3),
            language_attempts=[_attempt("a1", "src_1", range(0, 10), props=("p1", "p2"))],
        )
        assert res.ownership_status == OWNERSHIP_AMBIGUOUS_MULTIPLE_PROPOSITIONS

    def test_15_condition7_cross_source_attempts_never_considered(self):
        # a1 (other source) fully contains target numerically but is filtered
        # out before any word-set comparison; only a2 (same source) counts.
        res = _assess_ownership(
            candidate_source_asset_id="src_1",
            candidate_word_indices=(1, 2, 3),
            language_attempts=[
                _attempt("a1", "src_OTHER", range(0, 100), props=("px",)),
                _attempt("a2", "src_1", range(0, 10), props=("p1",)),
            ],
        )
        assert res.ownership_status == OWNERSHIP_EXACT_SINGLETON
        assert res.containing_language_attempt_id == "a2"

    def test_16_condition8_empty_word_attempt_never_contains_but_participates(self):
        res = _assess_ownership(
            candidate_word_indices=(1, 2, 3),
            language_attempts=[
                _attempt("empty1", "src_1", ()),
                _attempt("a1", "src_1", range(0, 10), props=("p1",)),
            ],
        )
        assert res.ownership_status == OWNERSHIP_EXACT_SINGLETON

    def test_17_condition9_all_clear_yields_exact_singleton(self):
        res = _assess_ownership(
            candidate_word_indices=(1, 2, 3),
            language_attempts=[_attempt("a1", "src_1", range(0, 10), props=("p1",))],
        )
        assert res.ownership_status == OWNERSHIP_EXACT_SINGLETON
        assert res.is_exact_singleton is True
        assert res.containing_language_attempt_id == "a1"
        assert res.proposition_candidate_ids == ("p1",)


# ===========================================================================
# 18: the real D-237M shape (RAW 34673899271) -- proves genuine singleton
# containment on the actual recovered numbers, and that the containing
# attempt's own size (221/252 words) is irrelevant to the verdict.
# ===========================================================================
class TestRealD237MShapeReplay:
    def test_18_real_d237m_shape_is_exact_singleton(self):
        res = _assess_ownership(
            clip_id="clip_3b74992d8a5a6cb08b31",
            candidate_source_asset_id="src_52b317dea148de2cd084",
            candidate_word_indices=range(41, 50),
            language_attempts=[
                _attempt(
                    "latt_939d22f452dbf84d81cb", "src_52b317dea148de2cd084",
                    range(31, 252), props=("prop_66ecd5ebab7c5c5560f9",),
                ),
            ],
        )
        assert res.ownership_status == OWNERSHIP_EXACT_SINGLETON
        assert res.is_exact_singleton is True
        assert res.containing_language_attempt_id == "latt_939d22f452dbf84d81cb"
        assert res.proposition_candidate_ids == ("prop_66ecd5ebab7c5c5560f9",)
        assert res.candidate_word_indices == tuple(range(41, 50))

    def test_19_huge_containing_attempt_size_irrelevant_vs_small(self):
        huge = _assess_ownership(
            candidate_word_indices=(41, 42, 43),
            language_attempts=[_attempt("a_huge", "src_1", range(0, 1000), props=("p1",))],
        )
        small = _assess_ownership(
            candidate_word_indices=(41, 42, 43),
            language_attempts=[_attempt("a_small", "src_1", range(41, 44), props=("p1",))],
        )
        assert huge.ownership_status == OWNERSHIP_EXACT_SINGLETON
        assert small.ownership_status == OWNERSHIP_EXACT_SINGLETON
        assert huge.is_exact_singleton == small.is_exact_singleton is True


# ===========================================================================
# 20-30: remaining counterexample-matrix items at the ownership-module level.
# ===========================================================================
class TestCounterexampleMatrixOwnershipLevel:
    def test_20_disjoint_no_overlap_at_all(self):
        res = _assess_ownership(
            candidate_word_indices=(500, 501),
            language_attempts=[_attempt("a1", "src_1", range(0, 10), props=("p1",))],
        )
        assert res.ownership_status == OWNERSHIP_NO_CONTAINING_ATTEMPT

    def test_21_exact_same_membership_is_still_singleton(self):
        res = _assess_ownership(
            candidate_word_indices=(1, 2, 3),
            language_attempts=[_attempt("a1", "src_1", (1, 2, 3), props=("p1",))],
        )
        assert res.ownership_status == OWNERSHIP_EXACT_SINGLETON

    def test_22_target_crosses_two_attempts_partial_each(self):
        # No attempt fully contains the target; it straddles both.
        res = _assess_ownership(
            candidate_word_indices=(4, 5, 6, 7),
            language_attempts=[
                _attempt("a1", "src_1", (0, 1, 2, 3, 4, 5)),
                _attempt("a2", "src_1", (6, 7, 8, 9)),
            ],
        )
        assert res.ownership_status == OWNERSHIP_PARTIAL_OVERLAP

    def test_23_multiple_lost_atoms_independent_results(self):
        attempts = [_attempt("a1", "src_1", range(0, 10), props=("p1",)),
                    _attempt("a2", "src_1", range(20, 30), props=("p2",))]
        r1 = _assess_ownership(clip_id="c1", candidate_word_indices=(1, 2), language_attempts=attempts)
        r2 = _assess_ownership(clip_id="c2", candidate_word_indices=(21, 22), language_attempts=attempts)
        r3 = _assess_ownership(clip_id="c3", candidate_word_indices=(500,), language_attempts=attempts)
        assert r1.ownership_status == OWNERSHIP_EXACT_SINGLETON and r1.containing_language_attempt_id == "a1"
        assert r2.ownership_status == OWNERSHIP_EXACT_SINGLETON and r2.containing_language_attempt_id == "a2"
        assert r3.ownership_status == OWNERSHIP_NO_CONTAINING_ATTEMPT

    def test_24_multi_source_isolation(self):
        # Same numeric indices exist in two different sources; only the
        # candidate's own source's attempt is ever considered.
        attempts = [
            _attempt("a1", "src_A", range(0, 10), props=("pA",)),
            _attempt("a2", "src_B", range(0, 10), props=("pB",)),
        ]
        res = _assess_ownership(candidate_source_asset_id="src_A", candidate_word_indices=(1, 2), language_attempts=attempts)
        assert res.ownership_status == OWNERSHIP_EXACT_SINGLETON
        assert res.containing_language_attempt_id == "a1"

    def test_25_spanish_word_indices_trivially_supported(self):
        # Ownership is index-based, never text-based -- language is irrelevant.
        res = _assess_ownership(
            clip_id="clip_es", candidate_source_asset_id="src_es",
            candidate_word_indices=(2, 3), language_attempts=[
                _attempt("attempt_es_1", "src_es", range(0, 8), props=("prop_es",)),
            ],
        )
        assert res.ownership_status == OWNERSHIP_EXACT_SINGLETON

    def test_26_english_word_indices_trivially_supported(self):
        res = _assess_ownership(
            clip_id="clip_en", candidate_source_asset_id="src_en",
            candidate_word_indices=(2, 3), language_attempts=[
                _attempt("attempt_en_1", "src_en", range(0, 8), props=("prop_en",)),
            ],
        )
        assert res.ownership_status == OWNERSHIP_EXACT_SINGLETON

    def test_27_spanglish_mixed_word_indices_trivially_supported(self):
        res = _assess_ownership(
            clip_id="clip_spanglish", candidate_source_asset_id="src_mix",
            candidate_word_indices=(2, 3), language_attempts=[
                _attempt("attempt_mix_1", "src_mix", range(0, 8), props=("prop_mix",)),
            ],
        )
        assert res.ownership_status == OWNERSHIP_EXACT_SINGLETON

    def test_28_no_language_attempts_supplied_at_all(self):
        res = _assess_ownership(language_attempts=[])
        assert res.ownership_status == OWNERSHIP_NO_CONTAINING_ATTEMPT

    def test_29_missing_candidate_word_indices_is_missing_provenance(self):
        res = _assess_ownership(candidate_word_indices=[])
        assert res.ownership_status == OWNERSHIP_MISSING_WORD_PROVENANCE

    def test_30_diagnostics_batch_summary(self):
        results = [
            _assess_ownership(clip_id="c1", candidate_word_indices=(1,), language_attempts=[_attempt("a1", "src_1", range(0, 5), props=("p1",))]),
            _assess_ownership(clip_id="c2", candidate_word_indices=()),
        ]
        diag = exact_lost_atom_ownership_diagnostics(results)
        assert diag["row_count"] == 2
        assert diag["exact_singleton_count"] == 1
        assert diag["ownership_status_counts"][OWNERSHIP_EXACT_SINGLETON] == 1
        assert diag["ownership_status_counts"][OWNERSHIP_MISSING_WORD_PROVENANCE] == 1


# ===========================================================================
# 31-38: D-235Q integration -- ownership as a second, disjoint sufficiency
# source at the identity-sufficiency gate. Precedence is never jumped.
# ===========================================================================
def _q_row(**overrides):
    base = {
        "clip_id": "c1",
        "text": "some lost content",
        "classification": "REAL_CONTENT_LOSS",
        "blocking": True,
        "missing_critical_atoms": (),
        "atom_classifications": (),
        "own_content_token_count": 20,
        "coverage_against_final_keep": 0.9,
    }
    base.update(overrides)
    return base


def _real_d237m_ownership(clip_id="c1"):
    return _assess_ownership(
        clip_id=clip_id,
        candidate_source_asset_id="src_52b317dea148de2cd084",
        candidate_word_indices=range(41, 50),
        language_attempts=[
            _attempt(
                "latt_939d22f452dbf84d81cb", "src_52b317dea148de2cd084",
                range(31, 252), props=("prop_66ecd5ebab7c5c5560f9",),
            ),
        ],
    )


class TestD235QIntegration:
    def test_31_without_ownership_matches_pre_d238_abstain(self):
        res = clsam.assess_complete_lost_semantic_atom_materiality(_q_row(), critical_claim_conflict=False)
        assert res.exact_ownership_available is False
        assert res.lost_atom_ownership_status is None
        assert res.blocking_recommendation == "ABSTAIN"
        assert res.final_materiality_status == "INSUFFICIENT_EVIDENCE"

    def test_32_with_real_d237m_ownership_reaches_do_not_block(self):
        ownership = _real_d237m_ownership()
        res = clsam.assess_complete_lost_semantic_atom_materiality(
            _q_row(), critical_claim_conflict=False, lost_atom_ownership=ownership,
        )
        assert res.exact_ownership_available is True
        assert res.lost_atom_ownership_status == "EXACT_SINGLETON_OWNERSHIP"
        assert res.final_materiality_status == "NON_MATERIAL_REAL_CONTENT"
        assert res.blocking_recommendation == "DO_NOT_BLOCK"

    def test_33_ownership_never_overrides_meaning_critical_block(self):
        ownership = _real_d237m_ownership()
        res = clsam.assess_complete_lost_semantic_atom_materiality(
            _q_row(atom_classifications=[{"importance": "CRITICAL", "atom": "x"}]),
            lost_atom_ownership=ownership,
        )
        assert res.final_materiality_status == "MEANING_CRITICAL"
        assert res.blocking_recommendation == "BLOCK"

    def test_34_ownership_never_overrides_critical_claim_conflict_block(self):
        ownership = _real_d237m_ownership()
        res = clsam.assess_complete_lost_semantic_atom_materiality(
            _q_row(), critical_claim_conflict=True, lost_atom_ownership=ownership,
        )
        assert res.final_materiality_status == "MEANING_CRITICAL"
        assert res.blocking_recommendation == "BLOCK"

    def test_35_ownership_never_overrides_editorial_required_block(self):
        from cutsell_worker.shared_attempt_word_identity import (
            AttemptLanguageIdentityMatch, RELATIONSHIP_EXACT_SAME_MEMBERSHIP, WordMembership,
        )
        ownership = _real_d237m_ownership()
        rm = WordMembership("s1", "r1", (0, 1), "AVAILABLE")
        lms = (WordMembership("s1", "latt1", (0, 1), "AVAILABLE"),)
        match = AttemptLanguageIdentityMatch(
            reconstructed_attempt_id="r1", language_attempt_ids=("latt1",), source_asset_id="s1",
            reconstructed_word_membership=rm, language_word_memberships=lms,
            relationship_status=RELATIONSHIP_EXACT_SAME_MEMBERSHIP, exact_shared_word_count=2,
            reconstructed_word_count=2, language_word_count=2, provenance=("test",),
        )
        res = clsam.assess_complete_lost_semantic_atom_materiality(
            _q_row(), exact_match=match,
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "HOOK"},
            idea_coverage_status=True,
            lost_atom_ownership=ownership,
        )
        assert res.final_materiality_status == "EDITORIALLY_REQUIRED"
        assert res.blocking_recommendation == "BLOCK"

    def test_36_ownership_never_overrides_unknown_critical_context_abstain(self):
        # critical_claim_conflict left None (unknown) -- ownership present but
        # the underlying materiality assessment never clears to
        # NON_MATERIAL_REAL_CONTENT without an explicit critical_claim_
        # conflict=False, so ownership alone must not manufacture a
        # DO_NOT_BLOCK here.
        ownership = _real_d237m_ownership()
        res = clsam.assess_complete_lost_semantic_atom_materiality(
            _q_row(), lost_atom_ownership=ownership,
        )
        assert res.blocking_recommendation == "ABSTAIN"
        assert res.final_materiality_status == "INSUFFICIENT_EVIDENCE"

    def test_37_ownership_plus_retry_process_still_do_not_block(self):
        ownership = _real_d237m_ownership()
        res = clsam.assess_complete_lost_semantic_atom_materiality(
            _q_row(), critical_claim_conflict=False, recording_process_evidence=True,
            lost_atom_ownership=ownership,
        )
        assert res.blocking_recommendation == "DO_NOT_BLOCK"

    def test_38_ownership_alone_without_independent_non_material_clearance_abstains(self):
        # A row whose raw materiality is never cleared to NON_MATERIAL_REAL_
        # CONTENT (critical_claim_conflict left unknown, i.e. None) must
        # stay ABSTAIN/INSUFFICIENT_EVIDENCE even with ownership present --
        # ownership only ever changes `requirement_genuinely_clear`, never
        # the raw `materiality.materiality_status` D-235L independently
        # computes.
        ownership = _real_d237m_ownership()
        res_without = clsam.assess_complete_lost_semantic_atom_materiality(_q_row())
        res_with = clsam.assess_complete_lost_semantic_atom_materiality(_q_row(), lost_atom_ownership=ownership)
        assert res_without.final_materiality_status == res_with.final_materiality_status == "INSUFFICIENT_EVIDENCE"
        assert res_without.blocking_recommendation == res_with.blocking_recommendation == "ABSTAIN"


# ===========================================================================
# 39-43: D-235Q batch wrapper + diagnostics thread ownership by clip_id.
# ===========================================================================
class TestAssessManyAndDiagnostics:
    def test_39_assess_many_forwards_ownership_by_clip_id(self):
        ownership = _real_d237m_ownership(clip_id="c1")
        results = clsam.assess_many(
            [_q_row(clip_id="c1"), _q_row(clip_id="c2")],
            critical_claim_conflict_by_clip_id={"c1": False, "c2": False},
            lost_atom_ownership_by_clip_id={"c1": ownership},
        )
        by_id = {r.clip_id: r for r in results}
        assert by_id["c1"].exact_ownership_available is True
        assert by_id["c1"].blocking_recommendation == "DO_NOT_BLOCK"
        assert by_id["c2"].exact_ownership_available is False

    def test_40_diagnostics_include_ownership_fields(self):
        ownership = _real_d237m_ownership()
        res = clsam.assess_complete_lost_semantic_atom_materiality(
            _q_row(), critical_claim_conflict=False, lost_atom_ownership=ownership,
        )
        diag = clsam.complete_lost_semantic_atom_materiality_diagnostics(res)
        assert diag["exact_ownership_available"] is True
        assert diag["lost_atom_ownership_status"] == "EXACT_SINGLETON_OWNERSHIP"

    def test_41_as_dict_includes_ownership_fields(self):
        ownership = _real_d237m_ownership()
        res = clsam.assess_complete_lost_semantic_atom_materiality(
            _q_row(), critical_claim_conflict=False, lost_atom_ownership=ownership,
        )
        d = res.as_dict()
        assert d["exact_ownership_available"] is True
        assert d["lost_atom_ownership_status"] == "EXACT_SINGLETON_OWNERSHIP"

    def test_42_none_ownership_reproduces_pre_d238_byte_identical_fields(self):
        res = clsam.assess_complete_lost_semantic_atom_materiality(_q_row(), critical_claim_conflict=False)
        assert res.exact_ownership_available is False
        assert res.lost_atom_ownership_status is None

    def test_43_ownership_object_never_mutates_materiality_object(self):
        ownership = _real_d237m_ownership()
        before = ownership.as_dict()
        clsam.assess_complete_lost_semantic_atom_materiality(
            _q_row(), critical_claim_conflict=False, lost_atom_ownership=ownership,
        )
        assert ownership.as_dict() == before


# ===========================================================================
# 44-48: D-235R condition-10 extension -- ownership-alone sufficiency,
# and D-235R/D-235T "same result" consumption proofs.
# ===========================================================================
def _materiality_result(**overrides):
    base = dict(
        clip_id="clip_test",
        exact_identity_available=False,
        exact_language_attempt_ids=(),
        exact_proposition_candidate_ids=(),
        meaning_materiality_status="NON_MATERIAL_REAL_CONTENT",
        editorial_requirement_status="INSUFFICIENT_EVIDENCE",
        retry_or_process_status="UNKNOWN",
        redundancy_status="UNKNOWN",
        final_materiality_status="NON_MATERIAL_REAL_CONTENT",
        blocking_recommendation="DO_NOT_BLOCK",
        reason_codes=("test",),
        provenance=("test",),
        exact_ownership_available=False,
        lost_atom_ownership_status=None,
    )
    base.update(overrides)
    return clsam.CompleteLostSemanticAtomMateriality(**base)


_ROW = {"clip_id": "clip_test", "blocking": True}


class TestD235RIntegration:
    def test_44_ownership_alone_sufficient_for_condition_10(self):
        m = _materiality_result(exact_identity_available=False, exact_ownership_available=True,
                                 lost_atom_ownership_status="EXACT_SINGLETON_OWNERSHIP")
        d = lsafa.decide_lost_semantic_atom_freeze_authority(_ROW, m)
        assert d.authority_status == lsafa.AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK
        assert d.suppression_applied is True
        assert d.effective_blocking is False
        assert d.safety_block_reason != "identity_and_ownership_insufficient_for_suppression"

    def test_45_neither_identity_nor_ownership_still_preserves(self):
        m = _materiality_result(exact_identity_available=False, exact_ownership_available=False)
        d = lsafa.decide_lost_semantic_atom_freeze_authority(_ROW, m)
        assert d.authority_status == lsafa.AUTHORITY_PRESERVE_BLOCK
        assert d.effective_blocking is True
        assert d.safety_block_reason == "identity_and_ownership_insufficient_for_suppression"

    def test_46_identity_alone_still_sufficient_unchanged(self):
        m = _materiality_result(exact_identity_available=True, exact_ownership_available=False)
        d = lsafa.decide_lost_semantic_atom_freeze_authority(_ROW, m)
        assert d.authority_status == lsafa.AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK
        assert d.safety_block_reason != "identity_and_ownership_insufficient_for_suppression"

    def test_47_d235r_never_imports_exact_lost_atom_ownership(self):
        imports = "\n".join(
            l for l in inspect.getsource(lsafa).splitlines()
            if l.strip().startswith(("import ", "from "))
        )
        assert "exact_lost_atom_ownership" not in imports

    def test_48_d235t_needs_zero_changes(self):
        diff = _diff_stat(T_PATH)
        assert diff == "", f"lost_atom_repair_suppression.py has an unexpected diff: {diff}"

    def test_49_d235t_never_reads_ownership_fields_directly(self):
        import cutsell_worker.lost_atom_repair_suppression as t_module
        source = inspect.getsource(t_module)
        assert "exact_ownership_available" not in source
        assert "lost_atom_ownership_status" not in source
        assert "exact_lost_atom_ownership" not in source


# ===========================================================================
# 50-58: no-false-authority proofs.
# ===========================================================================
class TestNoFalseAuthority:
    def test_50_shared_attempt_word_identity_unchanged(self):
        diff = _diff_stat(P_PATH)
        assert diff == "", f"shared_attempt_word_identity.py has an unexpected diff: {diff}"

    def test_51_authoritative_relationship_statuses_unchanged(self):
        assert "EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED" not in AUTHORITATIVE_RELATIONSHIP_STATUSES

    def test_52_ownership_module_never_imports_p1_p2(self):
        imports = "\n".join(
            l for l in inspect.getsource(elao).splitlines()
            if l.strip().startswith(("import ", "from "))
        )
        for banned in ("editorial_moment_sequence_integration", "whole_video_editorial_reasoning"):
            assert banned not in imports

    def test_53_ownership_module_never_imports_language_spine(self):
        diff = _diff_stat(
            "cutsell_worker/language_spine.py",
            "cutsell_worker/language_utterance_attempt.py",
            "cutsell_worker/language_spine_live_integration.py",
        )
        assert diff == "", f"Language-Spine files have an unexpected diff: {diff}"
        imports = "\n".join(
            l for l in inspect.getsource(elao).splitlines()
            if l.strip().startswith(("import ", "from "))
        )
        assert "language_spine" not in imports

    def test_54_ownership_module_never_imports_boundary_bestTake_family_ordering(self):
        imports = "\n".join(
            l for l in inspect.getsource(elao).splitlines()
            if l.strip().startswith(("import ", "from "))
        )
        for banned in ("boundary", "best_take", "bestTake", "family_resolver", "ordering"):
            assert banned not in imports.lower()

    def test_55_ownership_module_never_imports_pacing_audio_join(self):
        imports = "\n".join(
            l for l in inspect.getsource(elao).splitlines()
            if l.strip().startswith(("import ", "from "))
        )
        for banned in ("pacing_v2", "dialogue_pacing_transition", "audio_join_treatment"):
            assert banned not in imports

    def test_56_no_new_threshold_constant_in_ownership_module(self):
        source = inspect.getsource(elao)
        for banned in ("_TOLERANCE", "_THRESHOLD", "DEFAULT_SPLIT_GAP_SEC"):
            assert banned not in source

    def test_57_no_provider_or_fuzzy_matching_import(self):
        imports = "\n".join(
            l for l in inspect.getsource(elao).splitlines()
            if l.strip().startswith(("import ", "from "))
        )
        for banned in ("openai", "gemini", "anthropic", "requests", "httpx", "difflib", "rapidfuzz", "fuzzywuzzy"):
            assert banned not in imports.lower()

    def test_58_no_raw_or_modal_runpod_reference(self):
        source = inspect.getsource(elao)
        for banned in ("modal", "runpod", "workflow_dispatch"):
            assert banned not in source.lower()
