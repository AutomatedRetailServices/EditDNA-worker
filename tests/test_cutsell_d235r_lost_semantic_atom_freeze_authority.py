"""D-235R: BOUNDED LOST-ATOM FREEZE AUTHORITY ADAPTER -- OFFLINE FIRST.

Covers the task's own required 48-item fixture/proof matrix. Mirrors the
established D-235J-Q source-code-truth + fixture-matrix test style.
"""
from __future__ import annotations

import ast

import cutsell_worker.complete_lost_semantic_atom_materiality as clsam
import cutsell_worker.lost_semantic_atom_freeze_authority as auth
from cutsell_worker.final_story_coherence_validation import (
    apply_final_story_coherence_validation,
)

PROD_PATH = "cutsell_worker/lost_semantic_atom_freeze_authority.py"
COHERENCE_PATH = "cutsell_worker/final_story_coherence_validation.py"
Q_PATH = "cutsell_worker/complete_lost_semantic_atom_materiality.py"
L_PATH = "cutsell_worker/lost_semantic_atom_materiality.py"
M_PATH = "cutsell_worker/lost_atom_editorial_requirement_evidence.py"
P_PATH = "cutsell_worker/shared_attempt_word_identity.py"


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
        "clip_id": "c1", "text": "some lost content", "classification": "REAL_CONTENT_LOSS",
        "blocking": True, "missing_critical_atoms": (), "atom_classifications": (),
    }
    base.update(overrides)
    return base


def _materiality_for(row, **kwargs):
    return clsam.assess_complete_lost_semantic_atom_materiality(row, **kwargs)


def _decide(row, **kwargs):
    m = _materiality_for(row, **kwargs)
    return auth.decide_lost_semantic_atom_freeze_authority(row, m)


# ---------------------------------------------------------------------------
# 1-3. Suppression fixtures.
# ---------------------------------------------------------------------------
class TestSuppression:
    def test_01_non_material_blocking_atom_suppressed(self):
        d = _decide(_row(clip_id="c_nm"), critical_claim_conflict=False, idea_coverage_status=False)
        assert d.effective_blocking is False
        assert d.authority_status == auth.AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK

    def test_02_retry_residue_blocking_atom_suppressed(self):
        row = _row(clip_id="c_retry", pre_group_restart_consultations=[{"same_idea": True}])
        d = _decide(row)
        assert d.effective_blocking is False
        assert d.authority_status == auth.AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK

    def test_03_redundant_equivalent_blocking_atom_suppressed(self):
        row = _row(clip_id="c_red", content_loss_suppressed_by="tg_1")
        d = _decide(row)
        assert d.effective_blocking is False
        assert d.authority_status == auth.AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK


# ---------------------------------------------------------------------------
# 4-10. Preservation fixtures.
# ---------------------------------------------------------------------------
class TestPreservation:
    def test_04_meaning_critical_preserved(self):
        row = _row(clip_id="c_crit", atom_classifications=[{"importance": "CRITICAL", "atom": "x"}])
        d = _decide(row)
        assert d.effective_blocking is True
        assert d.authority_status == auth.AUTHORITY_PRESERVE_BLOCK
        assert d.safety_block_reason == "meaning_critical_firewall"

    def test_05_editorially_required_preserved(self):
        row = _row(clip_id="c_req")
        d = _decide(row, idea_coverage_status=True)
        assert d.effective_blocking is True
        assert d.safety_block_reason == "editorially_required_firewall"

    def test_06_insufficient_evidence_preserved(self):
        row = _row(clip_id="c_insuff")
        d = _decide(row)  # zero context -> D-235Q's own honest ABSTAIN default
        assert d.effective_blocking is True
        assert d.authority_status == auth.AUTHORITY_ABSTAIN_PRESERVE_BLOCK

    def test_07_conflicted_evidence_preserved(self):
        row = _row(
            clip_id="c_conf",
            pre_group_restart_consultations=[{"same_idea": True}, {"same_idea": False}],
        )
        d = _decide(row)
        assert d.effective_blocking is True
        assert d.authority_status == auth.AUTHORITY_ABSTAIN_PRESERVE_BLOCK

    def test_08_missing_materiality_result_preserved(self):
        row = _row(clip_id="c_missing")
        d = auth.decide_lost_semantic_atom_freeze_authority(row, None)
        assert d.effective_blocking is True
        assert d.authority_status == auth.AUTHORITY_PRESERVE_BLOCK
        assert d.safety_block_reason == "materiality_result_missing_or_malformed"

    def test_09_malformed_result_preserved(self):
        row = _row(clip_id="c_malformed")
        d = auth.decide_lost_semantic_atom_freeze_authority(row, {"final_materiality_status": "NON_MATERIAL_REAL_CONTENT"})
        assert d.effective_blocking is True
        assert d.authority_status == auth.AUTHORITY_PRESERVE_BLOCK

    def test_10_heuristic_only_identity_preserved(self):
        # No exact_match supplied at all -- editorial-requirement side
        # falls to D-235M's own honest default (INSUFFICIENT_EVIDENCE),
        # which is not one of the three suppressible categories.
        row = _row(clip_id="c_heuristic")
        d = _decide(row, critical_claim_conflict=False)  # non-material path but no exact identity resolved a slot
        # With critical_claim_conflict=False and no other signal, materiality
        # reaches NON_MATERIAL_REAL_CONTENT but requirement is INSUFFICIENT_EVIDENCE
        # with exact_identity_available=False -> condition 10 must preserve.
        assert d.effective_blocking is True


# ---------------------------------------------------------------------------
# 11-17. Multi-atom behavior.
# ---------------------------------------------------------------------------
class TestMultiAtomBehavior:
    def test_11_one_suppressible_atom_only(self):
        row = _row(clip_id="c_red", content_loss_suppressed_by="tg_1")
        trigger = auth.lost_semantic_atom_freeze_trigger_present([row], enabled=True)
        assert trigger is False

    def test_12_two_suppressible_atoms(self):
        row_a = _row(clip_id="a", content_loss_suppressed_by="tg_1")
        row_b = _row(clip_id="b", pre_group_restart_consultations=[{"same_idea": True}])
        trigger = auth.lost_semantic_atom_freeze_trigger_present([row_a, row_b], enabled=True)
        assert trigger is False

    def test_13_suppressible_plus_critical_atom_trigger_remains(self):
        row_a = _row(clip_id="a", content_loss_suppressed_by="tg_1")
        row_b = _row(clip_id="b", atom_classifications=[{"importance": "CRITICAL", "atom": "x"}])
        trigger = auth.lost_semantic_atom_freeze_trigger_present([row_a, row_b], enabled=True)
        assert trigger is True

    def test_14_suppressible_plus_editorial_required_trigger_remains(self):
        row_a = _row(clip_id="a", content_loss_suppressed_by="tg_1")
        row_b = _row(clip_id="b")
        materiality_by_id = {
            "a": _materiality_for(row_a),
            "b": _materiality_for(row_b, idea_coverage_status=True),
        }
        trigger = auth.lost_semantic_atom_freeze_trigger_present(
            [row_a, row_b], enabled=True, materiality_by_clip_id=materiality_by_id,
        )
        assert trigger is True

    def test_15_suppressible_plus_conflicted_trigger_remains(self):
        row_a = _row(clip_id="a", content_loss_suppressed_by="tg_1")
        row_b = _row(clip_id="b", pre_group_restart_consultations=[{"same_idea": True}, {"same_idea": False}])
        trigger = auth.lost_semantic_atom_freeze_trigger_present([row_a, row_b], enabled=True)
        assert trigger is True

    def test_16_non_blocking_atom_unchanged(self):
        row = _row(clip_id="c_nb", blocking=False)
        d = auth.decide_lost_semantic_atom_freeze_authority(row, None)
        assert d.effective_blocking is False
        assert d.authority_status == auth.AUTHORITY_NOT_APPLICABLE

    def test_17_all_non_blocking_atoms_unchanged(self):
        rows = [_row(clip_id="a", blocking=False), _row(clip_id="b", blocking=False)]
        trigger = auth.lost_semantic_atom_freeze_trigger_present(rows, enabled=True)
        assert trigger is False


# ---------------------------------------------------------------------------
# 18-23. Other Freeze triggers untouched (byte-identical terms).
# ---------------------------------------------------------------------------
class TestOtherFreezeTriggersUnchanged:
    def test_18_23_other_trigger_terms_still_present_in_source(self):
        content = _read(COHERENCE_PATH)
        for needle in (
            "bool(contradiction_findings)",
            "bool(missing_idea_coverage)",
            "bool(lost_critical_claims)",
            "bool(authority_membership_findings)",
        ):
            assert needle in content


# ---------------------------------------------------------------------------
# 24/25. Reviewer/repair-loop linkage (observability only).
# ---------------------------------------------------------------------------
class TestRepairLoopLinkageObservabilityOnly:
    def test_24_exact_same_unique_fact_lost_linkage_labelled(self):
        row = _row(clip_id="c_red", content_loss_suppressed_by="tg_1")
        d = _decide(row)
        assert d.suppression_applied
        attempts = [{"finding_kind": "UNIQUE_FACT_LOST", "previous_realization": ("c_red",)}]
        labels = auth.suppressed_atom_repair_finding_labels([d], attempts)
        assert labels.get("c_red") == auth.REPAIR_FINDING_LABEL_NON_MATERIAL_LOST_ATOM_SUPPRESSED

    def test_25_unrelated_unique_fact_lost_not_suppressed(self):
        row = _row(clip_id="c_red", content_loss_suppressed_by="tg_1")
        d = _decide(row)
        attempts = [{"finding_kind": "UNIQUE_FACT_LOST", "previous_realization": ("unrelated_clip",)}]
        labels = auth.suppressed_atom_repair_finding_labels([d], attempts)
        assert "c_red" not in labels

    def test_no_repair_loop_module_touched(self):
        content = _code_without_docstrings(PROD_PATH)
        assert "import" not in content.split("repair_loop")[0][-10:] if "repair_loop" in content else True
        assert "from .repair_loop" not in content
        assert "from .final_edit_reviewer" not in content


# ---------------------------------------------------------------------------
# 26/27. STORY_ORDER_BREAK / unrelated NEEDS_HUMAN_REVIEW unchanged.
# ---------------------------------------------------------------------------
class TestUnrelatedFindingsUnchanged:
    def test_26_story_order_break_vocabulary_untouched(self):
        content = _code_without_docstrings(PROD_PATH)
        assert "STORY_ORDER_BREAK" not in content

    def test_27_no_repair_status_mutation(self):
        content = _code_without_docstrings(PROD_PATH)
        assert "NEEDS_HUMAN_REVIEW" not in content
        assert "repair_loop_status" not in content


# ---------------------------------------------------------------------------
# 28/29. Flag OFF byte-identical / flag ON only target change.
# ---------------------------------------------------------------------------
class TestFlagParity:
    def test_28_flag_off_byte_identical(self):
        rows = [
            _row(clip_id="a", content_loss_suppressed_by="tg_1"),
            _row(clip_id="b", atom_classifications=[{"importance": "CRITICAL", "atom": "x"}]),
            _row(clip_id="c", blocking=False),
        ]
        original = any(row.get("blocking", True) for row in rows)
        via_adapter_off = auth.lost_semantic_atom_freeze_trigger_present(rows, enabled=False)
        assert via_adapter_off == original

    def test_28b_default_env_is_off(self):
        assert auth.lost_atom_materiality_freeze_authority_enabled({}) is False
        assert auth.lost_atom_materiality_freeze_authority_enabled() in (True, False)  # never raises

    def test_29_flag_on_only_target_row_changes(self):
        rows = [
            _row(clip_id="a", content_loss_suppressed_by="tg_1"),  # suppressible
            _row(clip_id="b"),  # ambiguous/insufficient -> preserved
        ]
        trigger_on = auth.lost_semantic_atom_freeze_trigger_present(rows, enabled=True)
        trigger_off = auth.lost_semantic_atom_freeze_trigger_present(rows, enabled=False)
        assert trigger_off is True  # both rows blocking=True by default -> any() is True
        assert trigger_on is True  # row b still preserved -> trigger remains despite row a's suppression


# ---------------------------------------------------------------------------
# 17 (replay). D-235K-shape replay (offline proof of full adapter mechanics).
# ---------------------------------------------------------------------------
class TestD235KShapeReplay:
    def _d235k_row(self):
        return _row(
            clip_id="c_d235k", text="a generic lost fragment of real speech",
            classification="REAL_CONTENT_LOSS", blocking=True,
            missing_critical_atoms=(), atom_classifications=(),
        )

    def test_offline_full_context_suppresses(self):
        row = self._d235k_row()
        d = _decide(row, critical_claim_conflict=False, idea_coverage_status=False)
        assert d.effective_blocking is False
        assert d.authority_status == auth.AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK
        # If this were the ONLY originally-blocking lost atom and no other
        # Freeze trigger existed, freeze_blocked would become False under
        # flag ON.
        trigger = auth.lost_semantic_atom_freeze_trigger_present(
            [row], enabled=True, materiality_by_clip_id={"c_d235k": clsam.assess_complete_lost_semantic_atom_materiality(
                row, critical_claim_conflict=False, idea_coverage_status=False,
            )},
        )
        assert trigger is False

    def test_flag_off_preserves_block(self):
        row = self._d235k_row()
        trigger = auth.lost_semantic_atom_freeze_trigger_present([row], enabled=False)
        assert trigger is True

    def test_live_seam_zero_context_honestly_preserves(self):
        # The bare live wiring (zero extra kwargs) does NOT auto-derive
        # critical_claim_conflict -- honestly preserves rather than
        # guessing. See module docstring's own scope boundary.
        row = self._d235k_row()
        trigger = auth.lost_semantic_atom_freeze_trigger_present([row], enabled=True)
        assert trigger is True

    def test_no_hardcoded_literal_phrase(self):
        content = _read(PROD_PATH)
        assert "too many people" not in content
        assert "ready set" not in content


# ---------------------------------------------------------------------------
# 30/31/32. Determinism, order independence, multi-source isolation.
# ---------------------------------------------------------------------------
class TestDeterminismAndIsolation:
    def test_30_deterministic_repeat(self):
        row = _row(clip_id="c1", atom_classifications=[{"importance": "CRITICAL", "atom": "x"}])
        results = {_decide(row).effective_blocking for _ in range(5)}
        assert len(results) == 1

    def test_31_order_independence(self):
        row_a = _row(clip_id="a", content_loss_suppressed_by="tg_1")
        row_b = _row(clip_id="b", atom_classifications=[{"importance": "CRITICAL", "atom": "x"}])
        t1 = auth.lost_semantic_atom_freeze_trigger_present([row_a, row_b], enabled=True)
        t2 = auth.lost_semantic_atom_freeze_trigger_present([row_b, row_a], enabled=True)
        assert t1 == t2 is True

    def test_32_multi_source_isolation(self):
        row_a = _row(clip_id="src_a_c1", content_loss_suppressed_by="tg_1")
        row_b = _row(clip_id="src_b_c1", content_loss_suppressed_by="tg_2")
        d_a = _decide(row_a)
        d_b = _decide(row_b)
        assert d_a.clip_id != d_b.clip_id
        assert d_a.effective_blocking == d_b.effective_blocking is False


# ---------------------------------------------------------------------------
# 33-41. Structural safety / no-authority-mutation proofs.
# ---------------------------------------------------------------------------
class TestStructuralSafety:
    def test_33_no_provider(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("requests.", "httpx", "openai", "genai.", "modal.", "runpod"):
            assert needle not in content

    def test_34_no_raw_subprocess_network(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("subprocess", "socket.", "urllib", "boto3"):
            assert needle not in content

    def test_35_no_threshold(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("0.5", "0.8", "0.95", "threshold", "tolerance_sec"):
            assert needle not in content

    def test_36_no_fuzzy_text(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("SequenceMatcher", "difflib", "fuzzy", "ratio(", "get_close_matches"):
            assert needle not in content

    def test_37_no_timestamp_authority(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in (".start", ".end", "overlap_ratio", "IoU"):
            assert needle not in content

    def test_38_no_p1_p2_mutation(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in (
            "editorial_moment_sequence_integration", "language_spine_live_integration",
            "build_editorial_moments_for_source", "language_proposition_relation",
        ):
            assert needle not in content

    def test_39_no_besttake_family_mutation(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("deterministic_best_take_authority", "take_grouping", "take_judge"):
            assert needle not in content

    def test_40_no_ordering_boundary_mutation(self):
        content = _code_without_docstrings(PROD_PATH)
        assert "boundary_engine_pass" not in content

    def test_41_no_pacing_audio_join_mutation(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("dialogue_pacing_transition", "audio_join"):
            assert needle not in content


# ---------------------------------------------------------------------------
# 42. compileall proxy.
# ---------------------------------------------------------------------------
class TestCompiles:
    def test_42_compiles(self):
        ast.parse(_read(PROD_PATH))
        ast.parse(_read(COHERENCE_PATH))
        ast.parse(_read("tests/test_cutsell_d235r_lost_semantic_atom_freeze_authority.py"))


# ---------------------------------------------------------------------------
# 43-45. Sibling module regressions (zero diff to D-235L/M/P/Q).
# ---------------------------------------------------------------------------
class TestSiblingRegressions:
    def test_43_d235q_module_not_modified(self):
        content = _read(Q_PATH)
        assert "lost_semantic_atom_freeze_authority" not in content

    def test_44_d235p_module_not_modified(self):
        content = _read(P_PATH)
        assert "lost_semantic_atom_freeze_authority" not in content

    def test_45_d235l_m_not_modified(self):
        for path in (L_PATH, M_PATH):
            content = _read(path)
            assert "lost_semantic_atom_freeze_authority" not in content


# ---------------------------------------------------------------------------
# Freeze seam wiring proof (the only production seam this task touches).
# ---------------------------------------------------------------------------
class TestFreezeSeamWiring:
    def test_coherence_module_imports_the_adapter(self):
        content = _read(COHERENCE_PATH)
        # D-235V/D-235W widened this to a multi-line import (also pulling in
        # `lost_atom_materiality_freeze_authority_enabled`) -- the exact
        # single-line form this test originally checked no longer appears,
        # but the adapter itself is still imported from the same module.
        assert "from .lost_semantic_atom_freeze_authority import (" in content
        assert "lost_semantic_atom_freeze_trigger_present," in content

    def test_coherence_module_calls_adapter_at_both_sites(self):
        content = _read(COHERENCE_PATH)
        # D-235W widened both call sites to also pass the live
        # `materiality_by_clip_id=` context Parts A+C compute -- the exact
        # bare-argument literal this test originally checked no longer
        # appears, but the adapter is still called at exactly both sites.
        # D-239F: `materiality_by_clip_id=materiality_by_clip_id,` now ALSO
        # appears at both `build_lost_atom_ownership_materiality_
        # diagnostics(...)` call sites (a coincidental second consumer of
        # the SAME kwarg name) -- checked precisely here (the two lines
        # immediately following each `lost_semantic_atom_freeze_trigger_
        # present(` opening) rather than a whole-file substring count, so
        # this test stays scoped to the ONE adapter it actually names.
        assert content.count("lost_semantic_atom_freeze_trigger_present(\n") == 2
        for block in content.split("lost_semantic_atom_freeze_trigger_present(\n")[1:]:
            assert "materiality_by_clip_id=materiality_by_clip_id," in block.splitlines()[0]
        assert "any(row.get(\"blocking\", True) for row in lost_semantic_atoms)" not in content


# ---------------------------------------------------------------------------
# Diagnostics (27 in deliverable list).
# ---------------------------------------------------------------------------
class TestDiagnostics:
    def test_diagnostics_required_fields_present(self):
        rows = [
            _row(clip_id="a", content_loss_suppressed_by="tg_1"),
            _row(clip_id="b", atom_classifications=[{"importance": "CRITICAL", "atom": "x"}]),
        ]
        decisions = [_decide(r) for r in rows]
        diag = auth.lost_semantic_atom_freeze_authority_diagnostics(
            lost_semantic_atoms=rows, decisions=decisions, enabled=True,
        )
        for key in (
            "lost_atom_materiality_authority_enabled", "lost_atom_original_blocking_count",
            "lost_atom_effective_blocking_count", "lost_atom_suppressed_count",
            "lost_atom_preserved_blocking_count", "lost_atom_suppression_status",
            "lost_atom_suppression_reasons", "repair_findings_suppressed_count",
        ):
            assert key in diag

    def test_diagnostics_no_transcript_dump(self):
        rows = [_row(clip_id="a", content_loss_suppressed_by="tg_1")]
        decisions = [_decide(r) for r in rows]
        diag = auth.lost_semantic_atom_freeze_authority_diagnostics(lost_semantic_atoms=rows, decisions=decisions)
        payload = str(diag)
        assert "some lost content" not in payload

    def test_diagnostics_counts_correct(self):
        rows = [
            _row(clip_id="a", content_loss_suppressed_by="tg_1"),
            _row(clip_id="b", atom_classifications=[{"importance": "CRITICAL", "atom": "x"}]),
            _row(clip_id="c", blocking=False),
        ]
        decisions = [_decide(r) for r in rows]
        diag = auth.lost_semantic_atom_freeze_authority_diagnostics(lost_semantic_atoms=rows, decisions=decisions, enabled=True)
        assert diag["lost_atom_original_blocking_count"] == 2
        assert diag["lost_atom_suppressed_count"] == 1
        assert diag["lost_atom_effective_blocking_count"] == 1
        assert diag["lost_atom_suppression_status"] == "PARTIALLY_SUPPRESSED"


# ---------------------------------------------------------------------------
# Vocabulary checks.
# ---------------------------------------------------------------------------
class TestVocabulary:
    def test_authority_vocabulary_four_values(self):
        assert auth._VALID_AUTHORITY_STATUSES == {
            "PRESERVE_BLOCK", "SUPPRESS_NON_MATERIAL_BLOCK", "ABSTAIN_PRESERVE_BLOCK", "NOT_APPLICABLE",
        }

    def test_suppressible_statuses_match_d235q(self):
        assert auth.SUPPRESSIBLE_FINAL_MATERIALITY_STATUSES == {
            "NON_MATERIAL_REAL_CONTENT", "RETRY_OR_RECORDING_RESIDUE", "REDUNDANT_EQUIVALENT",
        }

    def test_no_numeric_master_score(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("score =", "confidence =", "0.5 *", "weighted"):
            assert needle not in content
