"""D-235T: BOUNDED SAME-ATOM REPAIR-LOOP SUPPRESSION ADAPTER -- OFFLINE FIRST.

Covers the task's own required fixture matrix: terminal-vocabulary audit,
absolute firewalls, provenance integrity, single- and multi-finding
suppression semantics, no-fake-repair proof, mandatory flag-off byte-
identical parity, structural safety (no fuzzy text/timestamp/provider/
P1/P2/BestTake/Ordering/Boundary/Pacing/Audio-Join reasoning), and full
end-to-end RepairLoop integration via the real `build_canonical_edit_plan`
/ `review` / `run_repair_loop` functions. Mirrors the established
D-235J-S source-code-truth + fixture-matrix test style.
"""
from __future__ import annotations

import ast

import cutsell_worker.lost_atom_repair_suppression as suppression
from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_edit_reviewer import (
    CONTRADICTION,
    STORY_ORDER_BREAK,
    UNIQUE_FACT_LOST,
    Finding,
    review,
)
from cutsell_worker.lost_semantic_atom_freeze_authority import (
    AUTHORITY_ABSTAIN_PRESERVE_BLOCK,
    AUTHORITY_PRESERVE_BLOCK,
    AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK,
)
from cutsell_worker.repair_loop import RepairLoopResult, run_repair_loop

PROD_PATH = "cutsell_worker/lost_atom_repair_suppression.py"
REPAIR_LOOP_PATH = "cutsell_worker/repair_loop.py"
FLAG = "CUTSELL_LOST_ATOM_MATERIALITY_FREEZE_AUTHORITY_ENABLED"


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
    base = {"clip_id": "c1", "blocking": True, "lost_atom_provenance_id": "latom_c1_0"}
    base.update(overrides)
    return base


def _retry_row(clip_id="c1", provenance_id="latom_c1_0", **overrides):
    return _row(
        clip_id=clip_id, lost_atom_provenance_id=provenance_id,
        pre_group_restart_consultations=[{"same_idea": True}], **overrides,
    )


def _critical_row(clip_id="c1", provenance_id="latom_c1_0", **overrides):
    return _row(
        clip_id=clip_id, lost_atom_provenance_id=provenance_id,
        atom_classifications=[{"importance": "CRITICAL"}], **overrides,
    )


def _finding(kind=UNIQUE_FACT_LOST, *, detail, clip_ids=("c1",), idea_id="i1"):
    return Finding(
        kind=kind, plan_id="p1", plan_version=1, idea_id=idea_id, clip_ids=tuple(clip_ids),
        detail=detail, owning_authority="StoryValidator", blocking=True,
    )


# ---------------------------------------------------------------------------
# 1. RepairLoop terminal-vocabulary audit (must be true BEFORE this task's
#    own seam can be trusted).
# ---------------------------------------------------------------------------
class TestRepairLoopTerminalVocabularyAudit:
    def test_01_only_two_status_values_exist_in_source(self):
        content = _read(REPAIR_LOOP_PATH)
        assert '"PASS"' in content
        assert '"NEEDS_HUMAN_REVIEW"' in content
        # No third literal status string introduced.
        assert "NO_REPAIR_NEEDED" not in content
        assert "SUPPRESSED_PASS" not in content
        assert "PARTIAL_PASS" not in content

    def test_02_status_reused_never_a_new_value(self):
        content = _code_without_docstrings(REPAIR_LOOP_PATH)
        # The suppression branch must set status via the SAME "PASS"
        # literal, never mint a new terminal value.
        assert 'status = "PASS" if (result.status == "PASS" or blocking_findings_suppressed)' in content

    def test_03_repair_loop_result_gains_one_additive_field_only(self):
        content = _read(REPAIR_LOOP_PATH)
        assert "blocking_findings_suppressed: bool = False" in content

    def test_04_final_review_field_never_reassigned(self):
        content = _code_without_docstrings(REPAIR_LOOP_PATH)
        # `final_review=result` -- always the real review() output, never
        # a replaced/mutated object.
        assert "final_review=result," in content
        assert "result.status = " not in content
        assert "replace(result" not in content


# ---------------------------------------------------------------------------
# 2. Suppression-status vocabulary + dataclass validation.
# ---------------------------------------------------------------------------
class TestSuppressionDecisionVocabulary:
    def test_05_four_status_values_defined(self):
        assert suppression.SUPPRESS_SAME_NON_MATERIAL_ATOM == "SUPPRESS_SAME_NON_MATERIAL_ATOM"
        assert suppression.PRESERVE_REPAIR_ESCALATION == "PRESERVE_REPAIR_ESCALATION"
        assert suppression.ABSTAIN_PRESERVE_ESCALATION == "ABSTAIN_PRESERVE_ESCALATION"
        assert suppression.NOT_APPLICABLE == "NOT_APPLICABLE"

    def test_06_invalid_suppression_status_raises(self):
        import pytest
        with pytest.raises(ValueError):
            suppression.LostAtomRepairSuppressionDecision(
                lost_atom_provenance_id="x", reviewer_finding_kind=UNIQUE_FACT_LOST,
                repair_attempt_status=None, freeze_authority_status=None, materiality_status=None,
                suppression_status="BOGUS", suppress_repair_escalation=False, reason="r", provenance=(),
            )

    def test_07_as_dict_shape(self):
        d = suppression.decide_lost_atom_repair_suppression(
            _finding(detail=_retry_row()), enabled=True,
        )
        as_dict = d.as_dict()
        for key in (
            "lost_atom_provenance_id", "reviewer_finding_kind", "repair_attempt_status",
            "freeze_authority_status", "materiality_status", "suppression_status",
            "suppress_repair_escalation", "reason", "provenance",
        ):
            assert key in as_dict


# ---------------------------------------------------------------------------
# 3. Flag-off: ALWAYS NOT_APPLICABLE / never suppress, regardless of row.
# ---------------------------------------------------------------------------
class TestFlagOffNeverSuppresses:
    def test_08_flag_off_explicit_param(self):
        d = suppression.decide_lost_atom_repair_suppression(
            _finding(detail=_retry_row()), enabled=False,
        )
        assert d.suppression_status == suppression.NOT_APPLICABLE
        assert d.suppress_repair_escalation is False

    def test_09_flag_off_env_default(self, monkeypatch):
        monkeypatch.delenv(FLAG, raising=False)
        d = suppression.decide_lost_atom_repair_suppression(_finding(detail=_retry_row()))
        assert d.suppress_repair_escalation is False

    def test_10_flag_off_even_for_obviously_non_material_row(self):
        d = suppression.decide_lost_atom_repair_suppression(
            _finding(detail=_row(pre_group_restart_consultations=[{"same_idea": True}])),
            enabled=False,
        )
        assert d.suppress_repair_escalation is False


# ---------------------------------------------------------------------------
# 4. Absolute firewalls (never suppress, flag ON).
# ---------------------------------------------------------------------------
class TestAbsoluteFirewalls:
    def test_11_meaning_critical_never_suppressed(self):
        d = suppression.decide_lost_atom_repair_suppression(
            _finding(detail=_critical_row()), enabled=True,
        )
        assert d.suppress_repair_escalation is False
        assert d.suppression_status == suppression.PRESERVE_REPAIR_ESCALATION
        assert d.freeze_authority_status == AUTHORITY_PRESERVE_BLOCK

    def test_12_editorially_required_never_suppressed(self):
        d = suppression.decide_lost_atom_repair_suppression(
            _finding(detail=_row(
                editorial_slot_evidence="HOOK", identity_mapping_status="IDENTITY_MAPPING_EXACT",
                idea_coverage_status=False,
            )),
            enabled=True,
        )
        assert d.suppress_repair_escalation is False

    def test_13_conflicted_or_insufficient_never_suppressed(self):
        d = suppression.decide_lost_atom_repair_suppression(
            _finding(detail=_row(pre_group_restart_consultations=[{"same_idea": True}, {"same_idea": False}])),
            enabled=True,
        )
        assert d.suppress_repair_escalation is False
        assert d.suppression_status == suppression.ABSTAIN_PRESERVE_ESCALATION

    def test_14_unsupported_finding_kind_never_suppressed(self):
        d = suppression.decide_lost_atom_repair_suppression(
            _finding(kind=CONTRADICTION, detail={"reason": "polarity"}), enabled=True,
        )
        assert d.suppression_status == suppression.NOT_APPLICABLE
        assert d.suppress_repair_escalation is False

    def test_15_story_order_break_never_evaluated_as_suppressible(self):
        d = suppression.decide_lost_atom_repair_suppression(
            _finding(kind=STORY_ORDER_BREAK, detail={"recording_order": ["a", "b"]}), enabled=True,
        )
        assert d.suppression_status == suppression.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# 5. Provenance integrity (this task's own independent gate on top of
#    D-235R's firewalls).
# ---------------------------------------------------------------------------
class TestProvenanceIntegrity:
    def test_16_missing_provenance_id_abstains(self):
        row = {"clip_id": "c1", "blocking": True, "pre_group_restart_consultations": [{"same_idea": True}]}
        d = suppression.decide_lost_atom_repair_suppression(_finding(detail=row), enabled=True)
        assert d.suppression_status == suppression.ABSTAIN_PRESERVE_ESCALATION
        assert "provenance_id_missing" in d.reason

    def test_17_empty_string_provenance_id_treated_as_missing(self):
        d = suppression.decide_lost_atom_repair_suppression(
            _finding(detail=_retry_row(provenance_id="")), enabled=True,
        )
        assert d.suppress_repair_escalation is False

    def test_18_ambiguous_shared_provenance_id_abstains(self):
        f1 = _finding(detail=_retry_row(clip_id="c1", provenance_id="latom_shared_0"), clip_ids=("c1",))
        f2 = _finding(detail=_retry_row(clip_id="c2", provenance_id="latom_shared_0"), clip_ids=("c2",))
        d = suppression.decide_lost_atom_repair_suppression(f1, all_findings=(f1, f2), enabled=True)
        assert d.suppression_status == suppression.ABSTAIN_PRESERVE_ESCALATION
        assert "ambiguous" in d.reason.lower()

    def test_19_malformed_detail_not_a_mapping_abstains(self):
        finding = Finding(
            kind=UNIQUE_FACT_LOST, plan_id="p", plan_version=1, idea_id="i1", clip_ids=("c1",),
            detail=None, owning_authority="StoryValidator", blocking=True,
        )
        d = suppression.decide_lost_atom_repair_suppression(finding, enabled=True)
        assert d.suppression_status == suppression.ABSTAIN_PRESERVE_ESCALATION

    def test_20_all_findings_defaults_to_singleton_when_omitted(self):
        d = suppression.decide_lost_atom_repair_suppression(
            _finding(detail=_retry_row()), enabled=True,
        )
        assert d.suppression_status == suppression.SUPPRESS_SAME_NON_MATERIAL_ATOM


# ---------------------------------------------------------------------------
# 6. Single-finding suppression eligibility (mirrors D-235R's own three
#    suppressible categories).
# ---------------------------------------------------------------------------
class TestSingleFindingSuppression:
    def test_21_retry_residue_suppressed(self):
        d = suppression.decide_lost_atom_repair_suppression(_finding(detail=_retry_row()), enabled=True)
        assert d.suppression_status == suppression.SUPPRESS_SAME_NON_MATERIAL_ATOM
        assert d.freeze_authority_status == AUTHORITY_SUPPRESS_NON_MATERIAL_BLOCK
        assert d.materiality_status == "RETRY_OR_RECORDING_RESIDUE"

    def test_22_redundant_equivalent_suppressed(self):
        d = suppression.decide_lost_atom_repair_suppression(
            _finding(detail=_row(preserving_realization_id="r1")), enabled=True,
        )
        assert d.suppression_status == suppression.SUPPRESS_SAME_NON_MATERIAL_ATOM

    def test_23_non_blocking_row_never_reaches_this_path_but_is_handled(self):
        # Defensive: even if a caller hands this a non-blocking row's
        # finding (should not occur, since review() never puts a
        # non-blocking row in `findings`), it must never suppress from a
        # NOT_APPLICABLE freeze-authority verdict.
        d = suppression.decide_lost_atom_repair_suppression(
            _finding(detail=_retry_row(**{"blocking": False})), enabled=True,
        )
        assert d.suppress_repair_escalation is False


# ---------------------------------------------------------------------------
# 7. Multiple-findings semantics -- no majority voting, all-or-nothing.
# ---------------------------------------------------------------------------
class TestMultipleFindingsSemantics:
    def test_24_two_suppressible_findings_both_suppress(self):
        f1 = _finding(detail=_retry_row(clip_id="c1", provenance_id="latom_c1_0"))
        f2 = _finding(detail=_retry_row(clip_id="c2", provenance_id="latom_c2_0"), clip_ids=("c2",))
        all_suppressed, decisions = suppression.all_blocking_findings_safely_suppressed((f1, f2), enabled=True)
        assert all_suppressed is True
        assert len(decisions) == 2
        assert all(d.suppress_repair_escalation for d in decisions)

    def test_25_one_suppressible_one_critical_preserves_escalation(self):
        f1 = _finding(detail=_retry_row(clip_id="c1", provenance_id="latom_c1_0"))
        f2 = _finding(detail=_critical_row(clip_id="c2", provenance_id="latom_c2_0"), clip_ids=("c2",))
        all_suppressed, decisions = suppression.all_blocking_findings_safely_suppressed((f1, f2), enabled=True)
        assert all_suppressed is False
        assert decisions[0].suppress_repair_escalation is True
        assert decisions[1].suppress_repair_escalation is False

    def test_26_one_suppressible_one_unrelated_kind_preserves_escalation(self):
        f1 = _finding(detail=_retry_row(clip_id="c1", provenance_id="latom_c1_0"))
        f2 = _finding(kind=CONTRADICTION, detail={"reason": "polarity"}, clip_ids=("c2", "c3"))
        all_suppressed, decisions = suppression.all_blocking_findings_safely_suppressed((f1, f2), enabled=True)
        assert all_suppressed is False

    def test_27_empty_findings_never_suppresses(self):
        all_suppressed, decisions = suppression.all_blocking_findings_safely_suppressed((), enabled=True)
        assert all_suppressed is False
        assert decisions == ()

    def test_28_never_any_always_all(self):
        # 3 suppressible + 1 preserved -> still False (never "majority").
        findings = tuple(
            _finding(detail=_retry_row(clip_id=f"c{i}", provenance_id=f"latom_c{i}_0"), clip_ids=(f"c{i}",))
            for i in range(3)
        ) + (_finding(detail=_critical_row(clip_id="cX", provenance_id="latom_cX_0"), clip_ids=("cX",)),)
        all_suppressed, decisions = suppression.all_blocking_findings_safely_suppressed(findings, enabled=True)
        assert all_suppressed is False
        assert sum(1 for d in decisions if d.suppress_repair_escalation) == 3


# ---------------------------------------------------------------------------
# 8. Full RepairLoop integration -- real build_canonical_edit_plan/review/
#    run_repair_loop, no mocks.
# ---------------------------------------------------------------------------
def clip(clip_id, start, end, text, *, selected, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def draft(*, selected=(), discarded=(), coherence=None):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=discarded,
        diagnostics={"final_story_coherence_validation": coherence or {}},
    )


class TestRepairLoopIntegration:
    def test_29_single_suppressible_atom_loop_pass_reviewer_still_fail(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        a = clip("a", 0.0, 5.0, "the winning delivery", selected=True)
        b = clip("b", 5.0, 10.0, "a failed retry", selected=False)
        d = draft(
            selected=(a,), discarded=(b,),
            coherence={
                "freeze_blocked": False,
                "lost_semantic_atoms": [_retry_row(clip_id="b", provenance_id="latom_b_0")],
                "contradiction_findings": [],
            },
        )
        result = run_repair_loop(d)
        assert isinstance(result, RepairLoopResult)
        assert result.status == "PASS"
        assert result.final_review.status == "FAIL"
        assert result.blocking_findings_suppressed is True
        assert len(result.attempts) == 1
        assert result.attempts[0].repaired is False
        assert result.attempts[0].reason == suppression.REASON_SUPPRESSED_SAME_ATOM
        assert result.attempts[0].source_lost_atom_provenance_id == "latom_b_0"

    def test_30_flag_off_byte_identical_to_pre_d235t(self, monkeypatch):
        monkeypatch.setenv(FLAG, "0")
        a = clip("a", 0.0, 5.0, "the winning delivery", selected=True)
        b = clip("b", 5.0, 10.0, "a failed retry", selected=False)
        d = draft(
            selected=(a,), discarded=(b,),
            coherence={
                "freeze_blocked": False,
                "lost_semantic_atoms": [_retry_row(clip_id="b", provenance_id="latom_b_0")],
                "contradiction_findings": [],
            },
        )
        result = run_repair_loop(d)
        assert result.status == "NEEDS_HUMAN_REVIEW"
        assert result.blocking_findings_suppressed is False
        assert len(result.attempts) == 1
        assert result.attempts[0].reason == suppression.REASON_NO_REPAIR_STRATEGY

    def test_31_flag_unset_defaults_off(self, monkeypatch):
        monkeypatch.delenv(FLAG, raising=False)
        a = clip("a", 0.0, 5.0, "the winning delivery", selected=True)
        b = clip("b", 5.0, 10.0, "a failed retry", selected=False)
        d = draft(
            selected=(a,), discarded=(b,),
            coherence={
                "freeze_blocked": False,
                "lost_semantic_atoms": [_retry_row(clip_id="b", provenance_id="latom_b_0")],
                "contradiction_findings": [],
            },
        )
        result = run_repair_loop(d)
        assert result.status == "NEEDS_HUMAN_REVIEW"

    def test_32_two_suppressible_atoms_both_suppress_end_to_end(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        a = clip("a", 0.0, 5.0, "the winning delivery", selected=True)
        b = clip("b", 5.0, 10.0, "failed retry one", selected=False)
        c = clip("c", 10.0, 15.0, "failed retry two", selected=False)
        d = draft(
            selected=(a,), discarded=(b, c),
            coherence={
                "freeze_blocked": False,
                "lost_semantic_atoms": [
                    _retry_row(clip_id="b", provenance_id="latom_b_0"),
                    _retry_row(clip_id="c", provenance_id="latom_c_0"),
                ],
                "contradiction_findings": [],
            },
        )
        result = run_repair_loop(d)
        assert result.status == "PASS"
        assert len(result.attempts) == 2
        assert all(a.repaired is False for a in result.attempts)

    def test_33_suppressible_plus_critical_atom_preserves_escalation_end_to_end(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        a = clip("a", 0.0, 5.0, "the winning delivery", selected=True)
        b = clip("b", 5.0, 10.0, "failed retry", selected=False)
        c = clip("c", 10.0, 15.0, "the only mention of the critical dosage", selected=False)
        d = draft(
            selected=(a,), discarded=(b, c),
            coherence={
                "freeze_blocked": True,
                "lost_semantic_atoms": [
                    _retry_row(clip_id="b", provenance_id="latom_b_0"),
                    _critical_row(clip_id="c", provenance_id="latom_c_0"),
                ],
                "contradiction_findings": [],
            },
        )
        result = run_repair_loop(d)
        assert result.status == "NEEDS_HUMAN_REVIEW"
        assert result.blocking_findings_suppressed is False

    def test_34_legacy_row_without_provenance_id_preserves_escalation(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        a = clip("a", 0.0, 5.0, "the winning delivery", selected=True)
        b = clip("b", 5.0, 10.0, "failed retry", selected=False)
        legacy_row = {"clip_id": "b", "blocking": True, "pre_group_restart_consultations": [{"same_idea": True}]}
        d = draft(
            selected=(a,), discarded=(b,),
            coherence={"freeze_blocked": False, "lost_semantic_atoms": [legacy_row], "contradiction_findings": []},
        )
        result = run_repair_loop(d)
        assert result.status == "NEEDS_HUMAN_REVIEW"

    def test_35_unrelated_contradiction_finding_mixed_in_preserves_escalation(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        a = clip("a", 0.0, 5.0, "winner", selected=True)
        b = clip("b", 5.0, 10.0, "failed retry", selected=False)
        c = clip("c", 10.0, 15.0, "claims X", selected=True)
        e = clip("e", 15.0, 20.0, "claims not X", selected=True)
        d = draft(
            selected=(a, c, e), discarded=(b,),
            coherence={
                "freeze_blocked": True,
                "lost_semantic_atoms": [_retry_row(clip_id="b", provenance_id="latom_b_0")],
                "contradiction_findings": [{"idea_id": "i1", "clip_ids": ["c", "e"], "reason": "polarity conflict"}],
            },
        )
        result = run_repair_loop(d)
        assert "CONTRADICTION" in [f.kind for f in result.final_review.findings]
        assert result.status == "NEEDS_HUMAN_REVIEW"
        assert result.blocking_findings_suppressed is False

    def test_36_story_order_break_repair_strategy_still_applies_unaffected(self, monkeypatch):
        # A genuinely repairable STORY_ORDER_BREAK finding must still be
        # repaired via the pre-existing strategy -- this task's own seam
        # only touches the "no repair strategy" branch.
        monkeypatch.setenv(FLAG, "1")
        x = clip("x", 10.0, 15.0, "second thing recorded first shown", selected=True)
        y = clip("y", 0.0, 5.0, "first thing recorded second shown", selected=True)
        d = draft(
            selected=(x, y), discarded=(),
            coherence={
                "freeze_blocked": False,
                "lost_semantic_atoms": [],
                "contradiction_findings": [],
                "story_order_breaks": [{"idea_id": "i1", "clip_ids": ["x", "y"], "recording_order": ["y", "x"]}],
            },
        )
        result = run_repair_loop(d)
        assert any(att.repaired for att in result.attempts) or result.status == "PASS"

    def test_37_clean_pass_untouched(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        a = clip("a", 0.0, 5.0, "the winning delivery", selected=True)
        d = draft(
            selected=(a,), discarded=(),
            coherence={"freeze_blocked": False, "lost_semantic_atoms": [], "contradiction_findings": []},
        )
        result = run_repair_loop(d)
        assert result.status == "PASS"
        assert result.blocking_findings_suppressed is False
        assert result.attempts == ()


# ---------------------------------------------------------------------------
# 9. No-fake-repair proof.
# ---------------------------------------------------------------------------
class TestNoFakeRepair:
    def test_38_suppressed_attempts_never_repaired_true(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        a = clip("a", 0.0, 5.0, "winner", selected=True)
        b = clip("b", 5.0, 10.0, "failed retry", selected=False)
        d = draft(
            selected=(a,), discarded=(b,),
            coherence={
                "freeze_blocked": False,
                "lost_semantic_atoms": [_retry_row(clip_id="b", provenance_id="latom_b_0")],
                "contradiction_findings": [],
            },
        )
        result = run_repair_loop(d)
        assert all(att.repaired is False for att in result.attempts)

    def test_39_reason_distinguishes_suppressed_from_no_strategy(self):
        assert suppression.REASON_SUPPRESSED_SAME_ATOM != suppression.REASON_NO_REPAIR_STRATEGY
        content = _read(REPAIR_LOOP_PATH)
        assert "REASON_SUPPRESSED_SAME_ATOM" in content
        assert "REASON_NO_REPAIR_STRATEGY" in content


# ---------------------------------------------------------------------------
# 10. Structural safety -- no fuzzy text/timestamp/provider/P1/P2/BestTake/
#     Ordering/Boundary/Pacing/Audio-Join reasoning introduced.
# ---------------------------------------------------------------------------
class TestStructuralSafety:
    def test_40_no_sequence_matcher_or_difflib(self):
        content = _code_without_docstrings(PROD_PATH)
        assert "SequenceMatcher" not in content
        assert "difflib" not in content

    def test_41_no_timestamp_window_arithmetic(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("start -", "end -", "abs(start", "abs(end", "time_window"):
            assert needle not in content

    def test_42_no_provider_or_llm_call(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("gemini", "openai", "anthropic", "genai.", "requests.post", "runpod", "modal."):
            assert needle.lower() not in content.lower()

    def test_43_no_p1_p2_language(self):
        content = _code_without_docstrings(PROD_PATH)
        assert "P1" not in content
        assert "P2" not in content

    def test_44_no_best_take_family_ordering_boundary_language(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("BestTake", "RetryFamily", "Ordering", "BoundaryEngine", "family_id"):
            assert needle not in content

    def test_45_no_pacing_audio_join_language(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("dialogue_overlap", "overlaps_delivery", "audio_join", "pacing"):
            assert needle not in content

    def test_46_no_numeric_master_score(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("master_score", "confidence_score =", "weighted_sum"):
            assert needle not in content

    def test_47_no_new_freeze_or_materiality_thresholds(self):
        content = _code_without_docstrings(PROD_PATH)
        # Only imports/reuses D-235R/Q constants -- never redefines a
        # threshold-shaped constant of its own.
        assert "SUPPRESSIBLE_FINAL_MATERIALITY_STATUSES = frozenset" not in content
        assert "MATERIALITY_NON_MATERIAL_REAL_CONTENT =" not in content

    def test_48_only_imports_existing_pure_functions(self):
        content = _read(PROD_PATH)
        assert "from .complete_lost_semantic_atom_materiality import" in content
        assert "from .lost_semantic_atom_freeze_authority import" in content
        assert "from .lost_atom_reviewer_finding_provenance import" in content
        assert "from .final_edit_reviewer import" in content

    def test_49_same_feature_flag_reused_no_new_flag(self):
        content = _code_without_docstrings(PROD_PATH)
        # Imports and calls D-235R's own checker function ...
        assert "lost_atom_materiality_freeze_authority_enabled" in content
        # ... but never re-declares its own copy of the env-var name or a
        # second os.environ.get call -- the flag is looked up exactly once,
        # inside lost_semantic_atom_freeze_authority.py, never duplicated
        # here.
        assert "_AUTHORITY_ENV" not in content
        assert "os.environ" not in content


# ---------------------------------------------------------------------------
# 11. Diagnostics.
# ---------------------------------------------------------------------------
class TestDiagnostics:
    def test_50_diagnostics_shape(self):
        f = _finding(detail=_retry_row())
        d = suppression.decide_lost_atom_repair_suppression(f, enabled=True)
        diag = suppression.lost_atom_repair_suppression_diagnostics(decisions=(d,), enabled=True)
        for key in ("enabled", "decision_count", "suppressed_count", "all_suppressed", "decisions", "provenance"):
            assert key in diag
        assert diag["decision_count"] == 1
        assert diag["suppressed_count"] == 1
        assert diag["all_suppressed"] is True

    def test_51_diagnostics_tail_safe_no_transcript(self):
        f = _finding(detail=_retry_row())
        d = suppression.decide_lost_atom_repair_suppression(f, enabled=True)
        diag = suppression.lost_atom_repair_suppression_diagnostics(decisions=(d,), enabled=True)
        import json
        json.dumps(diag)  # must be JSON-safe


# ---------------------------------------------------------------------------
# 12. D-235O snapshot family widening (this task's own file family growth).
# ---------------------------------------------------------------------------
class TestFileFamilySnapshot:
    def test_52_new_file_matches_established_naming_convention(self):
        import os
        assert os.path.exists(PROD_PATH)
        new_files = [
            f for f in os.listdir("cutsell_worker")
            if f.startswith("lost_atom_") or f.startswith("shared_attempt_")
        ]
        expected = {
            "lost_atom_editorial_requirement_evidence.py",
            "lost_atom_proposition_identity_forensic.py",
            "lost_atom_reviewer_finding_provenance.py",
            "lost_atom_repair_suppression.py",
            # D-239F (separately-authorized, a later gate) -- widened here
            # for the same reason as this file's own test_17-style comment.
            "lost_atom_ownership_materiality_diagnostics.py",
            "shared_attempt_word_identity.py",
        }
        assert set(new_files) == expected
