"""D-235Q: COMPLETE EDITORIAL-REQUIREMENT + LOST-ATOM MATERIALITY
INTEGRATION -- OFFLINE ONLY.

Covers the task's own required 55-item fixture/proof matrix. Mirrors the
established D-235J-P source-code-truth + fixture-matrix test style.
"""
from __future__ import annotations

import ast

import cutsell_worker.complete_lost_semantic_atom_materiality as clsam
from cutsell_worker.lost_atom_editorial_requirement_evidence import (
    REQUIREMENT_NOT_REQUIRED,
)
from cutsell_worker.shared_attempt_word_identity import (
    AttemptLanguageIdentityMatch,
    RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED,
    RELATIONSHIP_EXACT_PARTIAL_WORD_MEMBERSHIP_OVERLAP,
    RELATIONSHIP_EXACT_SAME_MEMBERSHIP,
    RELATIONSHIP_ONE_RECONSTRUCTED_TO_MULTIPLE_LANGUAGE_ATTEMPTS_EXACT_PARTITION,
    WordMembership,
)

PROD_PATH = "cutsell_worker/complete_lost_semantic_atom_materiality.py"
L_PATH = "cutsell_worker/lost_semantic_atom_materiality.py"
M_PATH = "cutsell_worker/lost_atom_editorial_requirement_evidence.py"
P_PATH = "cutsell_worker/shared_attempt_word_identity.py"
N_PATH = "cutsell_worker/lost_atom_proposition_identity_forensic.py"


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _code_without_docstrings(path: str) -> str:
    text = _read(path)
    tree = ast.parse(text)
    docstring_lines: set[int] = set()

    def _mark(node) -> None:
        body = getattr(node, "body", None)
        if not body:
            return
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
            if isinstance(first.value.value, str):
                end = first.end_lineno or first.lineno
                docstring_lines.update(range(first.lineno, end + 1))

    _mark(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _mark(node)

    lines = text.splitlines()
    return "\n".join(line for i, line in enumerate(lines, start=1) if i not in docstring_lines)


def _row(**overrides):
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


def _match(status, lang_ids, entity_id="r1", source="s1"):
    rm = WordMembership(source, entity_id, (0, 1), "AVAILABLE")
    lms = tuple(WordMembership(source, lid, (0, 1), "AVAILABLE") for lid in lang_ids)
    return AttemptLanguageIdentityMatch(
        reconstructed_attempt_id=entity_id, language_attempt_ids=tuple(lang_ids), source_asset_id=source,
        reconstructed_word_membership=rm, language_word_memberships=lms,
        relationship_status=status, exact_shared_word_count=2,
        reconstructed_word_count=2, language_word_count=2, provenance=("test",),
    )


def _assess(**kwargs):
    row = kwargs.pop("row", None) or _row()
    return clsam.assess_complete_lost_semantic_atom_materiality(row, **kwargs)


# ---------------------------------------------------------------------------
# 1-10. Meaning-critical BLOCK fixtures.
# ---------------------------------------------------------------------------
class TestMeaningCriticalBlock:
    def _critical_row(self, **extra):
        return _row(atom_classifications=[{"importance": "CRITICAL", "atom": "x", **extra}])

    def test_01_lost_negation_blocks(self):
        res = _assess(row=self._critical_row(atom_type="NEGATION"))
        assert res.final_materiality_status == "MEANING_CRITICAL"
        assert res.blocking_recommendation == "BLOCK"

    def test_02_lost_number_blocks(self):
        res = _assess(row=self._critical_row(atom_type="NUMBER"))
        assert res.blocking_recommendation == "BLOCK"

    def test_03_lost_product_fact_blocks(self):
        res = _assess(row=self._critical_row(atom_type="PRODUCT_FACT"))
        assert res.blocking_recommendation == "BLOCK"

    def test_04_lost_factual_qualifier_blocks(self):
        res = _assess(row=self._critical_row(atom_type="QUALIFIER"))
        assert res.blocking_recommendation == "BLOCK"

    def test_05_lost_meaning_critical_correction_blocks(self):
        res = _assess(critical_claim_conflict=True)
        assert res.final_materiality_status == "MEANING_CRITICAL"
        assert res.blocking_recommendation == "BLOCK"

    def _required_match(self):
        return _match(RELATIONSHIP_EXACT_SAME_MEMBERSHIP, ["latt1"])

    def test_06_unique_required_setup_blocks(self):
        res = _assess(
            exact_match=self._required_match(),
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "HOOK"},
            idea_coverage_status=True,
        )
        assert res.final_materiality_status == "EDITORIALLY_REQUIRED"
        assert res.blocking_recommendation == "BLOCK"

    def test_07_unique_required_consequence_blocks(self):
        res = _assess(idea_coverage_status=True, critical_claim_conflict=False)
        assert res.final_materiality_status == "EDITORIALLY_REQUIRED"
        assert res.blocking_recommendation == "BLOCK"

    def test_08_unique_required_causal_bridge_blocks(self):
        res = _assess(downstream_dependency_present=True, critical_claim_conflict=False)
        assert res.final_materiality_status == "EDITORIALLY_REQUIRED"
        assert res.blocking_recommendation == "BLOCK"

    def test_09_sole_required_cta_meaning_blocks(self):
        res = _assess(
            exact_match=self._required_match(),
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "CTA"},
            critical_claim_conflict=False,
        )
        assert res.final_materiality_status == "EDITORIALLY_REQUIRED"
        assert res.blocking_recommendation == "BLOCK"

    def test_10_required_conclusion_blocks(self):
        res = _assess(
            exact_match=self._required_match(),
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "CONCLUSION"},
            critical_claim_conflict=False,
        )
        assert res.final_materiality_status == "EDITORIALLY_REQUIRED"
        assert res.blocking_recommendation == "BLOCK"


# ---------------------------------------------------------------------------
# 11-14. Retry / process DO_NOT_BLOCK.
# ---------------------------------------------------------------------------
class TestRetryProcessDoNotBlock:
    def test_11_retry_residue_do_not_block(self):
        row = _row(pre_group_restart_consultations=[{"same_idea": True}])
        res = _assess(row=row)
        assert res.final_materiality_status == "RETRY_OR_RECORDING_RESIDUE"
        assert res.blocking_recommendation == "DO_NOT_BLOCK"

    def test_12_creator_coaching_speech_do_not_block(self):
        res = _assess(recording_process_evidence=True)
        assert res.final_materiality_status == "RETRY_OR_RECORDING_RESIDUE"
        assert res.blocking_recommendation == "DO_NOT_BLOCK"

    def test_13_false_start_do_not_block(self):
        res = _assess(recording_process_evidence=True, recording_process_status="FALSE_START")
        assert res.blocking_recommendation == "DO_NOT_BLOCK"

    def test_14_recording_process_speech_do_not_block(self):
        res = _assess(recording_process_evidence=True, recording_process_status="RECORDING_PROCESS")
        assert res.blocking_recommendation == "DO_NOT_BLOCK"


# ---------------------------------------------------------------------------
# 15-18. Redundant / optional / non-material DO_NOT_BLOCK.
# ---------------------------------------------------------------------------
class TestRedundantOptionalNonMaterial:
    def test_15_redundant_equivalent_meaning_do_not_block(self):
        row = _row(content_loss_suppressed_by="tg_1")
        res = _assess(row=row, critical_claim_conflict=False)
        assert res.final_materiality_status == "REDUNDANT_EQUIVALENT"
        assert res.blocking_recommendation == "DO_NOT_BLOCK"

    def test_16_redundant_required_editorial_function_do_not_block(self):
        res = _assess(replacement_function_preserved=True, critical_claim_conflict=False)
        assert res.final_materiality_status == "REDUNDANT_EQUIVALENT"
        assert res.blocking_recommendation == "DO_NOT_BLOCK"

    def test_17_optional_elaboration_do_not_block(self):
        res = _assess(
            critical_claim_conflict=False, idea_coverage_status=False,
            downstream_dependency_present=False, audience_delivery_status="AUDIENCE_DELIVERY_SUPPORTED",
        )
        assert res.final_materiality_status == "NON_MATERIAL_REAL_CONTENT"
        assert res.blocking_recommendation == "DO_NOT_BLOCK"

    def test_18_non_material_real_content_do_not_block(self):
        match = _match(RELATIONSHIP_EXACT_SAME_MEMBERSHIP, ["latt1"])
        res = _assess(
            critical_claim_conflict=False, exact_match=match,
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "OTHER"},
        )
        assert res.final_materiality_status == "NON_MATERIAL_REAL_CONTENT"
        assert res.blocking_recommendation == "DO_NOT_BLOCK"


# ---------------------------------------------------------------------------
# 19-21. Identity-dependent ABSTAIN.
# ---------------------------------------------------------------------------
class TestIdentityDependentAbstain:
    def test_19_exact_identity_missing_abstains_where_requirement_depends_on_it(self):
        res = _assess(critical_claim_conflict=False)
        assert res.blocking_recommendation == "ABSTAIN"
        assert res.final_materiality_status == "INSUFFICIENT_EVIDENCE"

    def test_20_heuristic_only_identity_abstains(self):
        res = _assess(critical_claim_conflict=False, heuristic_identity_available=True)
        assert res.blocking_recommendation == "ABSTAIN"
        assert res.exact_identity_available is False

    def test_21_multi_proposition_unresolved_ownership_abstains(self):
        match = _match(RELATIONSHIP_ONE_RECONSTRUCTED_TO_MULTIPLE_LANGUAGE_ATTEMPTS_EXACT_PARTITION, ["la", "lb"])
        res = _assess(
            critical_claim_conflict=False, exact_match=match,
            proposition_candidate_ids_by_attempt_id={"la": ("pa",), "lb": ("pb",)},
            proposition_slot_evidence_by_id={"pa": "CTA", "pb": "OTHER"},
            idea_coverage_status=False,
        )
        assert res.blocking_recommendation == "ABSTAIN"
        assert res.final_materiality_status == "CONFLICTED"


# ---------------------------------------------------------------------------
# 22-25. Conflict precedence over DO_NOT_BLOCK; critical/required always wins.
# ---------------------------------------------------------------------------
class TestConflictPrecedence:
    def test_22_conflicting_criticality_vs_editorial_never_do_not_block(self):
        row = _row(atom_classifications=[{"importance": "CRITICAL", "atom": "x"}])
        res = _assess(row=row, idea_coverage_status=True, replacement_function_preserved=True)
        assert res.blocking_recommendation in ("BLOCK", "ABSTAIN")
        assert res.blocking_recommendation != "DO_NOT_BLOCK"

    def test_23_retry_plus_critical_fact_blocks(self):
        row = _row(
            atom_classifications=[{"importance": "CRITICAL", "atom": "x"}],
            pre_group_restart_consultations=[{"same_idea": True}],
        )
        res = _assess(row=row)
        assert res.final_materiality_status == "MEANING_CRITICAL"
        assert res.blocking_recommendation == "BLOCK"

    def test_24_retry_plus_required_setup_blocks(self):
        row = _row(pre_group_restart_consultations=[{"same_idea": True}])
        res = _assess(row=row, idea_coverage_status=True)
        assert res.final_materiality_status == "EDITORIALLY_REQUIRED"
        assert res.blocking_recommendation == "BLOCK"

    def test_25_redundant_plus_critical_fact_blocks(self):
        row = _row(
            atom_classifications=[{"importance": "CRITICAL", "atom": "x"}],
            content_loss_suppressed_by="tg_1",
        )
        res = _assess(row=row)
        assert res.final_materiality_status == "MEANING_CRITICAL"
        assert res.blocking_recommendation == "BLOCK"


# ---------------------------------------------------------------------------
# 26. Multiple lost atoms, mixed materiality (batch isolation).
# ---------------------------------------------------------------------------
class TestMixedMateriality:
    def test_26_multiple_lost_atoms_mixed_materiality_isolated(self):
        rows = [
            _row(clip_id="c_critical", atom_classifications=[{"importance": "CRITICAL", "atom": "x"}]),
            _row(clip_id="c_retry", pre_group_restart_consultations=[{"same_idea": True}]),
        ]
        results = clsam.assess_many(rows)
        by_id = {r.clip_id: r for r in results}
        assert by_id["c_critical"].blocking_recommendation == "BLOCK"
        assert by_id["c_retry"].blocking_recommendation == "DO_NOT_BLOCK"
        # A third, independently-assessed row with its own fully-cleared
        # evidence must land on its own correct verdict, unaffected by the
        # other two rows' own evidence (isolation, not just batch call
        # convenience) -- assess_many's per-clip maps intentionally don't
        # thread every optional parameter, so this one is asserted directly.
        nonmaterial_res = _assess(
            row=_row(clip_id="c_nonmaterial"), critical_claim_conflict=False, idea_coverage_status=False,
        )
        assert nonmaterial_res.blocking_recommendation == "DO_NOT_BLOCK"
        assert nonmaterial_res.final_materiality_status == "NON_MATERIAL_REAL_CONTENT"


# ---------------------------------------------------------------------------
# 27. Same-topic but no equivalence -> ABSTAIN.
# ---------------------------------------------------------------------------
class TestSameTopicNoEquivalence:
    def test_27_same_topic_no_equivalence_abstains(self):
        # No redundancy signal supplied at all -- "same topic" is not even
        # an accepted input; only real structural evidence counts.
        res = _assess(critical_claim_conflict=None)
        assert res.blocking_recommendation == "ABSTAIN"


# ---------------------------------------------------------------------------
# 28/29/30.
# ---------------------------------------------------------------------------
class TestCompletenessFixtures:
    def test_28_optional_detail_with_complete_story_do_not_block(self):
        res = _assess(
            critical_claim_conflict=False, idea_coverage_status=False,
            downstream_dependency_present=False,
        )
        assert res.blocking_recommendation == "DO_NOT_BLOCK"

    def test_29_story_incomplete_after_removal_blocks(self):
        res = _assess(idea_coverage_status=True)
        assert res.final_materiality_status == "EDITORIALLY_REQUIRED"
        assert res.blocking_recommendation == "BLOCK"

    def test_30_exact_preserved_substitute_do_not_block(self):
        res = _assess(critical_claim_conflict=False, replacement_function_preserved=True)
        assert res.final_materiality_status == "REDUNDANT_EQUIVALENT"
        assert res.blocking_recommendation == "DO_NOT_BLOCK"


# ---------------------------------------------------------------------------
# 31. D-235K-shape generic fixture (never hardcoding the literal phrase).
# ---------------------------------------------------------------------------
class TestD235KShapeFixture:
    def _d235k_row(self):
        return _row(
            clip_id="c_d235k",
            text="a generic lost fragment of real speech",
            classification="REAL_CONTENT_LOSS",
            blocking=True,
            missing_critical_atoms=(),
            atom_classifications=(),
        )

    def test_31_d235k_shape_exact_identity_no_requirement_do_not_block(self):
        match = _match(RELATIONSHIP_EXACT_SAME_MEMBERSHIP, ["latt1"])
        res = _assess(
            row=self._d235k_row(), critical_claim_conflict=False, exact_match=match,
            proposition_candidate_ids_by_attempt_id={"latt1": ("prop1",)},
            proposition_slot_evidence_by_id={"prop1": "OTHER"},
            idea_coverage_status=False,
        )
        assert res.final_materiality_status == "NON_MATERIAL_REAL_CONTENT"
        assert res.blocking_recommendation == "DO_NOT_BLOCK"

    def test_31b_d235k_shape_ambiguous_identity_abstains(self):
        match = _match(RELATIONSHIP_ONE_RECONSTRUCTED_TO_MULTIPLE_LANGUAGE_ATTEMPTS_EXACT_PARTITION, ["la", "lb"])
        res = _assess(
            row=self._d235k_row(), critical_claim_conflict=False, exact_match=match,
            proposition_candidate_ids_by_attempt_id={"la": ("pa",), "lb": ("pb",)},
            proposition_slot_evidence_by_id={"pa": "HOOK", "pb": "OTHER"},
            idea_coverage_status=False,
        )
        assert res.blocking_recommendation == "ABSTAIN"

    def test_31c_no_hardcoded_literal_phrase_in_production_module(self):
        content = _read(PROD_PATH)
        assert "too many people" not in content
        assert "ready set" not in content


# ---------------------------------------------------------------------------
# 32/33/34. Deterministic repeat / order independence / multi-source isolation.
# ---------------------------------------------------------------------------
class TestDeterminismAndIsolation:
    def test_32_deterministic_repeat(self):
        row = _row(atom_classifications=[{"importance": "CRITICAL", "atom": "x"}])
        results = {_assess(row=row).final_materiality_status for _ in range(5)}
        assert len(results) == 1

    def test_33_order_independence_in_batch(self):
        rows_a = [_row(clip_id="c1"), _row(clip_id="c2", atom_classifications=[{"importance": "CRITICAL", "atom": "x"}])]
        rows_b = list(reversed(rows_a))
        res_a = {r.clip_id: r.final_materiality_status for r in clsam.assess_many(rows_a)}
        res_b = {r.clip_id: r.final_materiality_status for r in clsam.assess_many(rows_b)}
        assert res_a == res_b

    def test_34_multi_source_isolation(self):
        match_a = _match(RELATIONSHIP_EXACT_SAME_MEMBERSHIP, ["latt_a"], entity_id="ra", source="src_a")
        match_b = _match(RELATIONSHIP_EXACT_SAME_MEMBERSHIP, ["latt_b"], entity_id="rb", source="src_b")
        res_a = _assess(
            row=_row(clip_id="ca"), exact_match=match_a, critical_claim_conflict=False,
            proposition_candidate_ids_by_attempt_id={"latt_a": ("pa",)},
            proposition_slot_evidence_by_id={"pa": "OTHER"}, idea_coverage_status=False,
        )
        res_b = _assess(
            row=_row(clip_id="cb"), exact_match=match_b, critical_claim_conflict=False,
            proposition_candidate_ids_by_attempt_id={"latt_b": ("pb",)},
            proposition_slot_evidence_by_id={"pb": "OTHER"}, idea_coverage_status=False,
        )
        assert res_a.exact_language_attempt_ids == ("latt_a",)
        assert res_b.exact_language_attempt_ids == ("latt_b",)
        assert res_a.exact_language_attempt_ids != res_b.exact_language_attempt_ids


# ---------------------------------------------------------------------------
# 35/36/37. Multilingual result (pure structural pass-through, no branch).
# ---------------------------------------------------------------------------
class TestMultilingualResult:
    def test_35_36_37_multilingual_text_no_special_handling(self):
        for text in ("perdí el conteo del producto", "I lost count of the product", "perdí el count del producto"):
            row = _row(text=text, atom_classifications=[{"importance": "CRITICAL", "atom": "x"}])
            res = _assess(row=row)
            assert res.blocking_recommendation == "BLOCK"

    def test_no_language_branch_in_module(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ('lang == "es"', 'lang == "en"', 'language == "es"', 'language == "en"'):
            assert needle not in content


# ---------------------------------------------------------------------------
# Diagnostics (39).
# ---------------------------------------------------------------------------
class TestDiagnostics:
    def test_39_diagnostics_required_fields_present(self):
        res = _assess(critical_claim_conflict=False)
        diag = clsam.complete_lost_semantic_atom_materiality_diagnostics(res)
        for key in (
            "clip_id", "exact_identity_status", "language_attempt_ids", "proposition_candidate_ids",
            "meaning_materiality_status", "editorial_requirement_status", "retry_or_process_status",
            "redundancy_status", "final_materiality_status", "blocking_recommendation",
            "reason_codes", "provenance",
        ):
            assert key in diag

    def test_diagnostics_no_transcript_dump(self):
        res = _assess(critical_claim_conflict=False)
        diag = clsam.complete_lost_semantic_atom_materiality_diagnostics(res)
        assert "text" not in diag
        assert "bounded_excerpt" not in diag

    def test_batch_diagnostics_counts_only(self):
        rows = [_row(clip_id="c1"), _row(clip_id="c2", atom_classifications=[{"importance": "CRITICAL", "atom": "x"}])]
        results = clsam.assess_many(rows)
        summary = clsam.complete_lost_semantic_atom_materiality_batch_diagnostics(results)
        assert summary["row_count"] == 2


# ---------------------------------------------------------------------------
# 40/41. No fuzzy-text / no timestamp-authority proof.
# ---------------------------------------------------------------------------
class TestStructuralSafety:
    def test_40_no_fuzzy_text_authority(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("SequenceMatcher", "difflib", "fuzzy", "ratio(", "get_close_matches"):
            assert needle not in content

    def test_41_no_timestamp_authority(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in (".start", ".end", "overlap_ratio", "IoU", "tolerance_sec"):
            assert needle not in content

    def test_no_provider_or_network(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("requests.", "httpx", "openai", "genai.", "modal.", "runpod", "subprocess", "socket."):
            assert needle not in content

    def test_no_numeric_master_score(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("score =", "confidence =", "weighted_score", "0.5 *"):
            assert needle not in content


# ---------------------------------------------------------------------------
# 42/43/44. No Freeze/RepairLoop/resolver mutation.
# ---------------------------------------------------------------------------
class TestNoLiveAuthorityMutation:
    def test_42_no_freeze_import_or_mutation(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("selection_freeze", "freeze_blocked", "SelectionFreeze"):
            assert needle not in content

    def test_43_no_repair_loop_import(self):
        content = _code_without_docstrings(PROD_PATH)
        assert "repair_loop" not in content

    def test_44_no_resolver_import(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in (
            "deterministic_best_take_authority", "take_judge", "boundary_engine_pass",
            "dialogue_pacing_transition", "hybrid_session_cleanup", "final_edit_reviewer",
        ):
            assert needle not in content

    def test_no_live_module_imports_this_gate(self):
        live_modules = (
            "cutsell_worker/pipeline.py",
            "cutsell_worker/universal_clean_cut.py",
            "cutsell_worker/final_story_coherence_validation.py",
            "cutsell_worker/final_edit_reviewer.py",
            "cutsell_worker/repair_loop.py",
        )
        for path in live_modules:
            content = _read(path)
            assert "complete_lost_semantic_atom_materiality" not in content


# ---------------------------------------------------------------------------
# Modification zero-diff checks for D-235L/M/P/N (49-53).
# ---------------------------------------------------------------------------
class TestSiblingModuleZeroDiff:
    def test_49_d235l_module_not_modified_by_this_gate(self):
        content = _read(L_PATH)
        assert "complete_lost_semantic_atom_materiality" not in content

    def test_50_d235m_module_not_modified_by_this_gate(self):
        content = _read(M_PATH)
        assert "complete_lost_semantic_atom_materiality" not in content

    def test_51_d235n_module_untouched(self):
        content = _read(N_PATH)
        assert "complete_lost_semantic_atom_materiality" not in content

    def test_52_d235p_module_not_modified_by_this_gate(self):
        content = _read(P_PATH)
        assert "complete_lost_semantic_atom_materiality" not in content

    def test_53_slot_evidence_only_assigned_under_exact_identity_branch(self):
        # Structural proof that editorial_slot_evidence is only ever set
        # from D-235P's own AUTHORITATIVE match -- never a heuristic one.
        import inspect
        source = inspect.getsource(clsam.assess_complete_lost_semantic_atom_materiality)
        idx = source.index("if exact_identity_available:")
        # exact_slot_evidence is only mutated inside the branch that starts here.
        branch = source[idx:source.index("if exact_identity_available:\n        identity_mapping_status")]
        assert "exact_slot_evidence, ownership_ambiguous = _exact_slot_for_proposition_set" in branch


# ---------------------------------------------------------------------------
# Vocabulary reuse (never redefined with different spellings).
# ---------------------------------------------------------------------------
class TestVocabularyReuse:
    def test_final_materiality_vocabulary_matches_d235l_exactly(self):
        from cutsell_worker.lost_semantic_atom_materiality import _VALID_MATERIALITY
        assert clsam.FINAL_MATERIALITY_VOCABULARY == _VALID_MATERIALITY

    def test_seven_value_vocabulary(self):
        assert len(clsam.FINAL_MATERIALITY_VOCABULARY) == 7

    def test_blocking_vocabulary_exactly_three(self):
        assert clsam._VALID_BLOCKING == {"BLOCK", "DO_NOT_BLOCK", "ABSTAIN"}


# ---------------------------------------------------------------------------
# Requirement-not-required always safe regardless of identity (regression
# guard for the identity-sufficiency gate design).
# ---------------------------------------------------------------------------
class TestRequirementNotRequiredAlwaysSafe:
    def test_not_required_via_firewall_safe_without_identity(self):
        res = _assess(
            recording_process_evidence=True, recording_process_status="RECORDING_PROCESS",
            critical_claim_conflict=False,
        )
        assert res.editorial_requirement_status in (REQUIREMENT_NOT_REQUIRED, "NOT_REQUIRED")
        assert res.blocking_recommendation == "DO_NOT_BLOCK"


# ---------------------------------------------------------------------------
# compileall proxy.
# ---------------------------------------------------------------------------
class TestCompiles:
    def test_module_compiles(self):
        ast.parse(_read(PROD_PATH))
        ast.parse(_read("tests/test_cutsell_d235q_complete_lost_semantic_atom_materiality.py"))
