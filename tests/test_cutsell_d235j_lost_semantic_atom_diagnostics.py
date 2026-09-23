"""D-235J: Lost Semantic Atom DETAIL observability, OBSERVABILITY ONLY.

Tests `cutsell_worker.selection_freeze_diagnostics.build_lost_semantic_atom_
diagnostics` -- a pure re-projection of the caller's own already-computed
`coherence_diag.get("lost_semantic_atoms")` rows (`final_story_coherence_
validation.py::_lost_semantic_atoms()`'s own real row schema, unchanged by
this task) and the caller's own already-serialized `diagnostics["repair_
loop"]["attempts"]` list. This module recomputes NO semantic atom, NO
materiality, and never changes a `blocking` flag -- every assertion below
checks that a REAL, ALREADY-PRESENT field is copied through bounded and
verbatim, never that a new judgment was made.
"""
import ast
import json
import re

import pytest

from cutsell_worker.selection_freeze_diagnostics import (
    ABSENT_FIELDS_NOT_RETAINED_BY_ENGINE,
    LOST_ATOM_SCHEMA_VERSION,
    REPAIR_LINK_NOT_DIRECTLY_ATTEMPTED,
    REVIEWER_FINDING_KIND_UNIQUE_FACT_LOST,
    STATE_FOUND,
    STATE_NOT_FOUND,
    STATE_UNKNOWN,
    build_lost_semantic_atom_diagnostics,
)

MODULE_PATH = "cutsell_worker/selection_freeze_diagnostics.py"


def _code_only_lines(module_text: str) -> str:
    """Strips the module's own top-level docstring before substring-
    checking -- same pattern already established for D-230/D-231/D-235G's
    own false-positive fixes (a docstring explaining what this module is
    NOT can otherwise trip a naive substring check)."""
    tree = ast.parse(module_text)
    doc = ast.get_docstring(tree)
    if doc and doc in module_text:
        module_text = module_text.replace(doc, "", 1)
    return module_text


def _content() -> str:
    with open(MODULE_PATH) as f:
        return _code_only_lines(f.read())


# --- Fixtures: real row shapes, matching _lost_semantic_atoms()'s own two
# shapes exactly (see final_story_coherence_validation.py lines ~862-901). ---

def _content_loss_row(**overrides) -> dict:
    row = {
        "clip_id": "clip_1",
        "text": "the doctor said it was not benign, it was cancer, and the biopsy confirmed a stage 2 diagnosis",
        "missing_critical_atoms": ["stage 2"],
        "atom_classifications": [
            {
                "atom": "stage 2", "atom_type": "number", "importance": "CRITICAL",
                "evidence": "own clip text carries a dose/measurement marker", "resolved_by": "deterministic",
            },
        ],
        "missing_content_token_count": 6,
        "own_content_token_count": 12,
        "coverage_against_final_keep": 0.3,
        "blocking": True,
        "classification": "REAL_CONTENT_LOSS",
    }
    row.update(overrides)
    return row


def _no_usable_realization_row(**overrides) -> dict:
    row = {
        "clip_id": "clip_2",
        "text": "no wait, that's not right, let me start over" * 3,
        "kind": "LOST_IN_NO_USABLE_REALIZATION_FAMILY",
        "basis": "corroborated_bts_singleton",
        "blocking": False,
    }
    row.update(overrides)
    return row


def _repair_attempt(**overrides) -> dict:
    attempt = {
        "plan_id": "plan_1",
        "previous_plan_version": 1,
        "new_plan_version": 1,
        "finding_kind": REVIEWER_FINDING_KIND_UNIQUE_FACT_LOST,
        "idea_id": None,
        "owning_authority": "StoryValidator",
        "previous_realization": ("clip_1",),
        "replacement_realization": ("clip_1",),
        "coverage_before": "unknown",
        "coverage_after": "unknown",
        "reason": "no_repair_strategy_exists_for_this_finding_kind",
        "unaffected_ideas_changed": False,
        "repaired": False,
    }
    attempt.update(overrides)
    return attempt


class TestBoundingAndBlockingAtom:
    def test_01_one_blocking_atom_surfaced(self):
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[_content_loss_row()])
        assert out["atom_count"] == 1
        assert out["blocking_atom_count"] == 1
        assert out["atoms"][0]["blocking"] is True
        assert out["atoms"][0]["clip_id"] == "clip_1"

    def test_02_one_non_blocking_atom_surfaced(self):
        row = _content_loss_row(blocking=False, classification="SEMANTICALLY_COVERED_BY_SELECTED_REALIZATION")
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row])
        assert out["atom_count"] == 1
        assert out["blocking_atom_count"] == 0
        assert out["atoms"][0]["blocking"] is False

    def test_03_multiple_atoms_all_present(self):
        rows = [_content_loss_row(clip_id=f"clip_{i}") for i in range(5)]
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=rows)
        assert out["atom_count"] == 5
        assert len(out["atoms"]) == 5
        assert {a["clip_id"] for a in out["atoms"]} == {f"clip_{i}" for i in range(5)}

    def test_04_no_atoms_reports_not_found(self):
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[])
        assert out["atom_count"] == 0
        assert out["blocking_atom_count"] == 0
        assert out["atoms"] == []
        assert out["ledger_status"] == STATE_NOT_FOUND

    def test_05_none_reports_unknown_never_zero_atoms_as_false_pass(self):
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=None)
        assert out["ledger_status"] == STATE_UNKNOWN
        assert out["atom_count"] == 0

    def test_06_atom_with_text_gets_bounded_excerpt(self):
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[_content_loss_row()])
        excerpt = out["atoms"][0]["text_excerpt"]
        assert excerpt is not None
        assert len(excerpt) <= 161  # cap + 1 ellipsis char

    def test_07_atom_without_text_reports_none_not_empty_string(self):
        row = _content_loss_row(text="")
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row])
        assert out["atoms"][0]["text_excerpt"] is None

    def test_08_atom_with_clip_id_preserved_verbatim(self):
        row = _content_loss_row(clip_id="real_clip_id_xyz")
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row])
        assert out["atoms"][0]["clip_id"] == "real_clip_id_xyz"

    def test_09_atom_without_clip_id_reports_empty_string_never_none_crash(self):
        row = _content_loss_row(clip_id=None)
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row])
        assert out["atoms"][0]["clip_id"] == ""

    def test_10_blocking_flag_preserved_true(self):
        row = _content_loss_row(blocking=True)
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row])
        assert out["atoms"][0]["blocking"] is True

    def test_11_blocking_flag_preserved_false(self):
        row = _content_loss_row(blocking=False)
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row])
        assert out["atoms"][0]["blocking"] is False

    def test_12_blocking_defaults_true_when_absent_matches_source_default(self):
        row = _content_loss_row()
        del row["blocking"]
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row])
        assert out["atoms"][0]["blocking"] is True

    def test_13_no_usable_realization_shape_handled(self):
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[_no_usable_realization_row()])
        row = out["atoms"][0]
        assert row["row_kind"] == "LOST_IN_NO_USABLE_REALIZATION_FAMILY"
        assert row["no_usable_realization_basis"] == "corroborated_bts_singleton"
        assert row["blocking"] is False

    def test_14_content_loss_shape_row_kind_labeled(self):
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[_content_loss_row()])
        assert out["atoms"][0]["row_kind"] == "COVERAGE_LEDGER_CONTENT_LOSS"
        assert out["atoms"][0]["no_usable_realization_basis"] is None

    def test_15_mixed_multiple_trigger_context(self):
        rows = [_content_loss_row(clip_id="a"), _no_usable_realization_row(clip_id="b")]
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=rows)
        assert out["atom_count"] == 2
        assert out["blocking_atom_count"] == 1


class TestReviewerAndRepairLoopLinkage:
    def test_16_reviewer_finding_kind_is_structural_constant(self):
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[_content_loss_row()])
        assert out["atoms"][0]["reviewer_finding_kind"] == REVIEWER_FINDING_KIND_UNIQUE_FACT_LOST

    def test_17_no_repair_strategy_case_matched_by_real_clip_id(self):
        row = _content_loss_row(clip_id="clip_1")
        attempts = [_repair_attempt(previous_realization=("clip_1",))]
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row], repair_loop_attempts=attempts)
        atom = out["atoms"][0]
        assert atom["repair_loop_attempt_status"] == "MATCHED_BY_CLIP_ID"
        assert atom["repair_loop_reason"] == "no_repair_strategy_exists_for_this_finding_kind"
        assert atom["repair_loop_repaired"] is False

    def test_18_no_matching_attempt_reports_not_directly_attempted(self):
        row = _content_loss_row(clip_id="clip_unattempted")
        attempts = [_repair_attempt(previous_realization=("clip_1",))]
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row], repair_loop_attempts=attempts)
        atom = out["atoms"][0]
        assert atom["repair_loop_attempt_status"] == REPAIR_LINK_NOT_DIRECTLY_ATTEMPTED
        assert atom["repair_loop_attempt_index"] is None

    def test_19_no_attempts_at_all_reports_not_directly_attempted(self):
        row = _content_loss_row()
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row], repair_loop_attempts=())
        assert out["atoms"][0]["repair_loop_attempt_status"] == REPAIR_LINK_NOT_DIRECTLY_ATTEMPTED

    def test_20_never_matches_wrong_finding_kind(self):
        row = _content_loss_row(clip_id="clip_1")
        attempts = [_repair_attempt(previous_realization=("clip_1",), finding_kind="STORY_ORDER_BREAK")]
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row], repair_loop_attempts=attempts)
        assert out["atoms"][0]["repair_loop_attempt_status"] == REPAIR_LINK_NOT_DIRECTLY_ATTEMPTED

    def test_21_never_uses_text_similarity_only_exact_clip_id_membership(self):
        # A clip whose text is nearly identical to the attempted one, but a
        # DIFFERENT clip_id, must never be matched -- real-id-only linkage.
        row = _content_loss_row(clip_id="clip_lookalike", text=_content_loss_row()["text"])
        attempts = [_repair_attempt(previous_realization=("clip_1",))]
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row], repair_loop_attempts=attempts)
        assert out["atoms"][0]["repair_loop_attempt_status"] == REPAIR_LINK_NOT_DIRECTLY_ATTEMPTED


class TestAtomClassificationsAndCounts:
    def test_22_atom_classifications_bounded_and_evidence_not_dumped(self):
        row = _content_loss_row()
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row])
        classifications = out["atoms"][0]["atom_classifications"]
        assert len(classifications) == 1
        c = classifications[0]
        assert c["atom_type"] == "number"
        assert c["importance"] == "CRITICAL"
        assert c["evidence_present"] is True
        assert "evidence" not in c  # presence-only, never the free-text evidence itself

    def test_23_missing_critical_atom_count_matches_real_count(self):
        row = _content_loss_row(missing_critical_atoms=["a", "b", "c"])
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row])
        assert out["atoms"][0]["missing_critical_atom_count"] == 3

    def test_24_token_and_coverage_counts_pass_through_verbatim(self):
        row = _content_loss_row(own_content_token_count=12, missing_content_token_count=6, coverage_against_final_keep=0.3)
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row])
        atom = out["atoms"][0]
        assert atom["own_content_token_count"] == 12
        assert atom["missing_content_token_count"] == 6
        assert atom["coverage_against_final_keep"] == 0.3

    def test_25_content_loss_suppressed_by_and_preservation_fields_pass_through(self):
        row = _content_loss_row(
            content_loss_suppressed_by="GROUPED_SAME_IDEA",
            preserving_realization_id="realization_9",
            preserved_claim_ids=["claim_a", "claim_b"],
            nonrequired_omissions=[{"reason": "x"}],
        )
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row])
        atom = out["atoms"][0]
        assert atom["content_loss_suppressed_by"] == "GROUPED_SAME_IDEA"
        assert atom["preserving_realization_id"] == "realization_9"
        assert atom["preserved_claim_count"] == 2
        assert atom["nonrequired_omission_count"] == 1

    def test_26_present_before_after_selection_structural_constants(self):
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[_content_loss_row()])
        atom = out["atoms"][0]
        assert atom["present_before_selection"] is True
        assert atom["present_after_selection"] is False


class TestBoundingSizeAndTruncation:
    def test_27_atoms_truncated_beyond_max(self):
        rows = [_content_loss_row(clip_id=f"clip_{i}") for i in range(40)]
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=rows)
        assert out["atom_count"] == 40
        assert len(out["atoms"]) < 40
        assert out["atoms_truncated"] is True

    def test_28_no_truncation_flag_false_when_under_cap(self):
        rows = [_content_loss_row(clip_id=f"clip_{i}") for i in range(3)]
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=rows)
        assert out["atoms_truncated"] is False

    def test_29_no_transcript_dump_text_never_exceeds_bound(self):
        long_text = "word " * 500
        row = _content_loss_row(text=long_text)
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row])
        assert len(out["atoms"][0]["text_excerpt"]) < 200

    def test_30_json_safe_serializable(self):
        rows = [_content_loss_row(), _no_usable_realization_row()]
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=rows, repair_loop_attempts=[_repair_attempt()])
        json.dumps(out)  # must not raise

    def test_31_schema_version_present(self):
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[])
        assert out["schema_version"] == LOST_ATOM_SCHEMA_VERSION

    def test_32_absent_fields_list_reported_every_call(self):
        out = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[])
        assert out["absent_fields_not_retained_by_engine"] == list(ABSENT_FIELDS_NOT_RETAINED_BY_ENGINE)
        # Directive's own suggested-but-absent fields must actually be named.
        for field in ("atom_id", "source_span_id", "source_proposition_id", "semantic_role", "required_or_optional"):
            assert field in out["absent_fields_not_retained_by_engine"]

    def test_33_determinism_same_input_same_output(self):
        rows = [_content_loss_row(), _no_usable_realization_row()]
        out1 = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=rows, repair_loop_attempts=[_repair_attempt()])
        out2 = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=rows, repair_loop_attempts=[_repair_attempt()])
        assert out1 == out2


class TestBehaviorNeutrality:
    """Proves this module never touches Freeze/coherence/repair/resolver/
    Pacing/renderer decision logic -- same discipline as D-235G's own
    test_27/test_28-style code-only-lines checks."""

    def test_34_no_freeze_or_coherence_or_repair_or_resolver_import(self):
        content = _content()
        forbidden_imports = [
            "final_story_coherence_validation", "repair_loop", "realization_resolver",
            "canonical_edit_plan", "final_edit_reviewer", "post_authority_validation",
            "pacing_v2", "audio_join_treatment", "renderer",
        ]
        import_lines = [line for line in content.splitlines() if line.strip().startswith(("import ", "from "))]
        import_block = "\n".join(import_lines)
        for name in forbidden_imports:
            assert name not in import_block, f"unexpected import of {name!r} in {MODULE_PATH}"

    def test_35_never_mutates_blocking_flag_it_only_reads(self):
        row = _content_loss_row(blocking=True)
        original = dict(row)
        build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[row])
        assert row == original  # input row untouched

    def test_36_never_mutates_repair_attempt_input(self):
        attempt = _repair_attempt()
        original = dict(attempt)
        build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[_content_loss_row()], repair_loop_attempts=[attempt])
        assert attempt == original

    def test_37_no_provider_raw_modal_runpod_reference(self):
        content = _content()
        for token in ("modal.App", "runpod", "RunPod", "dispatch_workflow", "boto3"):
            assert token not in content

    def test_38_no_video00_golden_file_reference(self):
        content = _content()
        for token in ("video00_selection_lock", "video00_architecture", "expected_selected_count"):
            assert token.lower() not in content.lower()

    def test_39_no_semantic_atom_recomputation_helper_defined(self):
        # This module must never define its own atom-detection logic (e.g.
        # a numbers/negations extractor) -- only read already-built rows.
        content = _content()
        assert "_numbers(" not in content
        assert "_negations(" not in content
        assert re.search(r"def\s+classify_(negation|number)_atom", content) is None


class TestModuleAndRegressionQualification:
    def test_40_module_compiles(self):
        import py_compile
        py_compile.compile(MODULE_PATH, doraise=True)

    def test_41_selection_freeze_diagnostics_still_importable_together(self):
        from cutsell_worker.selection_freeze_diagnostics import build_selection_freeze_diagnostics
        # Both functions coexist; calling one never requires the other.
        out1 = build_selection_freeze_diagnostics(freeze_blocked=False)
        out2 = build_lost_semantic_atom_diagnostics(lost_semantic_atoms=[])
        assert out1["schema_version"] != out2["schema_version"]

    def test_42_universal_clean_cut_imports_both_functions(self):
        with open("cutsell_worker/universal_clean_cut.py") as f:
            content = f.read()
        assert "build_lost_semantic_atom_diagnostics" in content
        assert "build_selection_freeze_diagnostics" in content
        assert "lost_semantic_atom_diagnostics" in content
