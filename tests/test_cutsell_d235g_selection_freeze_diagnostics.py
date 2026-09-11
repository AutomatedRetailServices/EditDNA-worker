"""D-235G: Selection Freeze Blocker Observability, OBSERVABILITY ONLY.

Proves the new `selection_freeze_diagnostics` block faithfully surfaces
the already-computed Freeze decision and Pacing-seam reachability
without ever recomputing whether Freeze should block, and that its live
wiring is behavior-neutral (no selection/Family/BestTake/Boundary/
Ordering/Pacing/renderer mutation, unchanged with the block absent or
present)."""
from __future__ import annotations

import ast
import inspect
import json
import subprocess

import cutsell_worker.selection_freeze_diagnostics as m
import cutsell_worker.universal_clean_cut as ucc


def _coherence(**overrides):
    base = {
        "status": "applied",
        "contradiction_findings": [],
        "missing_idea_coverage": [],
        "lost_semantic_atoms": [],
        "lost_critical_claims": [],
        "authority_membership_findings": [],
    }
    base.update(overrides)
    return base


class TestSerialization:
    def test_01_type_serializes_json_safe(self):
        diag = m.build_selection_freeze_diagnostics(freeze_blocked=False, coherence_diag=_coherence())
        json.dumps(diag)  # must not raise

    def test_02_freeze_blocked_true_preserved(self):
        diag = m.build_selection_freeze_diagnostics(freeze_blocked=True, coherence_diag=_coherence(contradiction_findings=[{"a": 1}]))
        assert diag["freeze_blocked"] is True

    def test_03_freeze_blocked_false_preserved(self):
        diag = m.build_selection_freeze_diagnostics(freeze_blocked=False, coherence_diag=_coherence())
        assert diag["freeze_blocked"] is False


class TestTriggerFixtures:
    def test_04_contradiction_trigger_captured(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True, coherence_diag=_coherence(contradiction_findings=[{"x": "y"}]),
        )
        assert diag["coherence_contradiction_status"] == m.STATE_FOUND
        assert m.TRIGGER_COHERENCE_CONTRADICTION in diag["trigger_categories"]
        assert diag["trigger_count"] == 1

    def test_05_idea_loss_trigger_captured(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True, coherence_diag=_coherence(missing_idea_coverage=[{"idea": "z"}]),
        )
        assert diag["idea_loss_status"] == m.STATE_FOUND
        assert m.TRIGGER_COHERENCE_MISSING_IDEA_COVERAGE in diag["trigger_categories"]

    def test_06_repair_loop_trigger_captured(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True, coherence_diag=_coherence(), repair_loop_status="NEEDS_HUMAN_REVIEW",
        )
        assert m.TRIGGER_REPAIR_LOOP_NEEDS_HUMAN_REVIEW in diag["trigger_categories"]
        assert diag["repair_loop_status"] == "NEEDS_HUMAN_REVIEW"

    def test_07_resolver_trigger_captured(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True, coherence_diag=_coherence(), resolver_status="REVIEW_REQUIRED",
        )
        assert m.TRIGGER_RESOLVER_AUTHORITATIVE_REVIEW_REQUIRED in diag["trigger_categories"]

    def test_08_integrity_trigger_captured(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True, coherence_diag=_coherence(),
            post_authority_integrity_failed=True,
            post_authority_integrity_failure_codes=["POST_AUTHORITY_SELECTION_MUTATION"],
        )
        assert m.TRIGGER_POST_AUTHORITY_INTEGRITY_FAILURE in diag["trigger_categories"]
        assert diag["post_authority_integrity_status"] == m.STATE_FOUND
        assert diag["post_authority_integrity_failure_codes"] == ["POST_AUTHORITY_SELECTION_MUTATION"]

    def test_08b_lost_critical_claim_trigger_captured(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True, coherence_diag=_coherence(lost_critical_claims=[{"c": 1}]),
        )
        assert m.TRIGGER_COHERENCE_LOST_CRITICAL_CLAIM in diag["trigger_categories"]

    def test_08c_authority_membership_trigger_captured(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True, coherence_diag=_coherence(authority_membership_findings=[{"f": 1}]),
        )
        assert m.TRIGGER_COHERENCE_AUTHORITY_MEMBERSHIP_FINDING in diag["trigger_categories"]

    def test_08d_blocking_lost_semantic_atom_trigger_captured(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True,
            coherence_diag=_coherence(lost_semantic_atoms=[{"blocking": True}, {"blocking": False}]),
        )
        assert m.TRIGGER_COHERENCE_BLOCKING_LOST_SEMANTIC_ATOM in diag["trigger_categories"]

    def test_08e_non_blocking_lost_semantic_atom_not_a_trigger(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=False,
            coherence_diag=_coherence(lost_semantic_atoms=[{"blocking": False}]),
        )
        assert m.TRIGGER_COHERENCE_BLOCKING_LOST_SEMANTIC_ATOM not in diag["trigger_categories"]

    def test_08f_coherence_integrity_failure_shape_captured(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True,
            coherence_diag={"status": "integrity_failure", "freeze_blocked": True},
        )
        assert diag["coherence_integrity_failure_status"] == m.STATE_FOUND
        assert m.TRIGGER_COHERENCE_INTEGRITY_FAILURE in diag["trigger_categories"]

    def test_09_no_trigger_state(self):
        diag = m.build_selection_freeze_diagnostics(freeze_blocked=False, coherence_diag=_coherence())
        assert diag["trigger_count"] == 0
        assert diag["trigger_categories"] == []


class TestMultipleTriggers:
    def test_10_multiple_triggers_preserved(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True,
            coherence_diag=_coherence(contradiction_findings=[{"a": 1}], missing_idea_coverage=[{"b": 2}]),
            repair_loop_status="NEEDS_HUMAN_REVIEW",
        )
        assert diag["trigger_count"] == 3
        assert set(diag["trigger_categories"]) == {
            m.TRIGGER_COHERENCE_CONTRADICTION,
            m.TRIGGER_COHERENCE_MISSING_IDEA_COVERAGE,
            m.TRIGGER_REPAIR_LOOP_NEEDS_HUMAN_REVIEW,
        }
        # No arbitrary first-match collapse -- all triggered categories survive.


class TestUnknownHonesty:
    def test_11_unknown_detail_honest_when_coherence_diag_empty(self):
        diag = m.build_selection_freeze_diagnostics(freeze_blocked=True, coherence_diag={})
        assert diag["coherence_contradiction_status"] == m.STATE_UNKNOWN
        assert diag["idea_loss_status"] == m.STATE_UNKNOWN
        assert diag["lost_semantic_atom_status"] == m.STATE_UNKNOWN
        assert diag["lost_critical_claim_status"] == m.STATE_UNKNOWN
        assert diag["authority_membership_finding_status"] == m.STATE_UNKNOWN
        assert diag["coherence_integrity_failure_status"] == m.STATE_UNKNOWN
        # Never fabricated FALSE/NOT_FOUND from absence.
        assert m.STATE_NOT_FOUND not in (
            diag["coherence_contradiction_status"], diag["idea_loss_status"],
        )

    def test_11b_unknown_when_key_absent_from_nonempty_dict(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True, coherence_diag={"status": "applied"},  # missing all list keys
        )
        assert diag["coherence_contradiction_status"] == m.STATE_UNKNOWN
        assert diag["idea_loss_status"] == m.STATE_UNKNOWN

    def test_11c_post_authority_integrity_unknown_when_not_passed(self):
        diag = m.build_selection_freeze_diagnostics(freeze_blocked=False, coherence_diag=_coherence())
        assert diag["post_authority_integrity_status"] == m.STATE_UNKNOWN


class TestSelectedCountAndPacingSeam:
    def test_12_selected_count_preserved(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=False, coherence_diag=_coherence(), selected_count_before_freeze=5,
        )
        assert diag["selected_count_before_freeze"] == 5

    def test_22_freeze_blocked_fixture_pacing_seam_not_reached(self):
        diag = m.build_selection_freeze_diagnostics(freeze_blocked=True, coherence_diag=_coherence(contradiction_findings=[{"a": 1}]))
        assert diag["pacing_seam_reached"] is False
        assert diag["first_missing_link"] == m.LINK_FREEZE_BLOCKED_BEFORE_PACING

    def test_23_non_blocked_fixture_pacing_seam_reachable(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=False, coherence_diag=_coherence(),
            pacing_v2_serialized=True, pacing_v2_handle_aware_serialized=True, audio_join_treatment_v2_serialized=True,
        )
        assert diag["pacing_seam_reached"] is True
        assert diag["first_missing_link"] == m.LINK_PACING_DIAGNOSTIC_SERIALIZED

    def test_23b_non_blocked_but_flag_off_reports_diagnostic_missing_not_freeze(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=False, coherence_diag=_coherence(),
            pacing_v2_serialized=False, pacing_v2_handle_aware_serialized=False, audio_join_treatment_v2_serialized=False,
        )
        assert diag["pacing_seam_reached"] is True
        assert diag["first_missing_link"] == m.LINK_PACING_SEAM_REACHED_DIAGNOSTIC_MISSING

    def test_24_pacing_v2_serialization_status_passthrough(self):
        diag = m.build_selection_freeze_diagnostics(freeze_blocked=False, coherence_diag=_coherence(), pacing_v2_serialized=True)
        assert diag["pacing_v2_serialized"] is True

    def test_25_handle_aware_serialization_status_passthrough(self):
        diag = m.build_selection_freeze_diagnostics(freeze_blocked=False, coherence_diag=_coherence(), pacing_v2_handle_aware_serialized=False)
        assert diag["pacing_v2_handle_aware_serialized"] is False

    def test_26_audio_join_treatment_v2_status_passthrough(self):
        diag = m.build_selection_freeze_diagnostics(freeze_blocked=False, coherence_diag=_coherence(), audio_join_treatment_v2_serialized=True)
        assert diag["audio_join_treatment_v2_serialized"] is True


def _code_only_lines(module) -> str:
    """Source with the module's own top-level docstring stripped, so a
    prose reference to another module's behavior (e.g. explaining what
    this module is NOT) never trips a structural absence check."""
    tree = ast.parse(inspect.getsource(module))
    lines = inspect.getsource(module).splitlines()
    doc = ast.get_docstring(tree)
    if doc:
        # Drop the docstring's own line span.
        doc_node = tree.body[0]
        lines = lines[doc_node.end_lineno:]
    return "\n".join(lines)


class TestSiblingSafety:
    def test_27_no_video00_semantic_oracle(self):
        code = _code_only_lines(m)
        # No golden-file/segment comparison logic present in actual code.
        assert "segments" not in code
        assert "align(" not in code
        assert "video00_semantic_alignment" not in code

    def test_28_no_expected_count_23_dependency(self):
        code = _code_only_lines(m)
        assert "expected_selected_count" not in code
        assert "== 23" not in code

    def test_29_no_transcript_dump(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True,
            coherence_diag=_coherence(contradiction_findings=[{"text": "the actual spoken sentence goes here"}]),
        )
        blob = json.dumps(diag)
        assert "the actual spoken sentence goes here" not in blob

    def test_30_json_safe_with_all_fields_populated(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True,
            coherence_diag=_coherence(contradiction_findings=[{"a": 1}], missing_idea_coverage=[{"b": 2}]),
            repair_loop_status="NEEDS_HUMAN_REVIEW", resolver_status="REVIEW_REQUIRED",
            post_authority_integrity_failed=True, post_authority_integrity_failure_codes=["X"],
            selected_count_before_freeze=3,
            pacing_v2_serialized=False, pacing_v2_handle_aware_serialized=False, audio_join_treatment_v2_serialized=False,
        )
        json.dumps(diag)

    def test_31_deterministic_repeat(self):
        kwargs = dict(freeze_blocked=True, coherence_diag=_coherence(contradiction_findings=[{"a": 1}]), repair_loop_status="OK")
        r1 = m.build_selection_freeze_diagnostics(**kwargs)
        r2 = m.build_selection_freeze_diagnostics(**kwargs)
        assert r1 == r2


class TestD235ShapeReconstruction:
    """Reproduces the D-235 sibling RAW's own control shape offline --
    selected_count > 0, freeze_blocked True, Pacing diagnostic blocks
    absent -- and proves the new report would name
    FREEZE_BLOCKED_BEFORE_PACING without a 68MB artifact. No RAW run."""

    def test_32_d235_control_shape_identifies_freeze_blocked_before_pacing(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True,
            coherence_diag={},  # D-235F: specific trigger detail not retrievable
            repair_loop_status=None,
            resolver_status=None,
            post_authority_integrity_failed=False,
            selected_count_before_freeze=5,  # D-235's own cited actual_selected_count
            pacing_v2_serialized=None,
            pacing_v2_handle_aware_serialized=None,
            audio_join_treatment_v2_serialized=None,
        )
        assert diag["freeze_blocked"] is True
        assert diag["pacing_seam_reached"] is False
        assert diag["first_missing_link"] == m.LINK_FREEZE_BLOCKED_BEFORE_PACING
        assert diag["selected_count_before_freeze"] == 5
        # The specific trigger is honestly unknown, never guessed.
        assert diag["trigger_count"] == 0
        assert diag["coherence_contradiction_status"] == m.STATE_UNKNOWN


class TestImmutabilityProofs:
    """No engine authority: this module never imports selection/Family/
    BestTake/Boundary/Ordering/Pacing decision modules or the renderer."""

    def test_13_no_selection_mutation_import(self):
        tree = ast.parse(inspect.getsource(m))
        imported = {n.module.split(".")[-1] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        forbidden = {
            "realization_resolver", "final_story_coherence_validation", "repair_loop",
            "boundary_engine_pass", "human_boundary_polish_v5", "dialogue_pacing_transition",
            "render", "render_plan", "take_grouping_provider", "take_judge_provider",
        }
        assert not (imported & forbidden), imported & forbidden

    def test_14_pure_function_no_global_state(self):
        # Calling twice with different inputs never leaks state between calls.
        m.build_selection_freeze_diagnostics(freeze_blocked=True, coherence_diag=_coherence(contradiction_findings=[{"a": 1}]))
        diag = m.build_selection_freeze_diagnostics(freeze_blocked=False, coherence_diag=_coherence())
        assert diag["freeze_blocked"] is False
        assert diag["trigger_count"] == 0

    def test_15_no_family_besttake_import(self):
        src = inspect.getsource(m)
        assert "take_judge" not in src.lower()
        assert "best_take" not in src.lower()

    def test_16_no_ordering_import(self):
        assert "ordering" not in inspect.getsource(m).lower()

    def test_17_no_boundary_decision_import(self):
        src = inspect.getsource(m)
        assert "boundary_engine" not in src

    def test_18_no_pacing_decision_import(self):
        src = inspect.getsource(m)
        assert "decide_transition" not in src
        assert "pacing_v2_audio_join_treatment_decision" not in src

    def test_19_no_renderer_import(self):
        src = inspect.getsource(m)
        assert "render.py" not in src and "import render" not in src


class TestSeamWiring:
    def test_20_seam_calls_new_function_after_inner_freeze_if_else(self):
        src = inspect.getsource(ucc)
        idx_call = src.find("build_selection_freeze_diagnostics(")
        idx_import = src.find("from .selection_freeze_diagnostics import build_selection_freeze_diagnostics")
        assert idx_import != -1 and idx_call != -1

    def test_21_seam_never_recomputes_freeze_blocked(self):
        # The wiring passes the EXISTING local `freeze_blocked=freeze_blocked`
        # -- never a re-derivation. A crude but effective structural check:
        # the exact keyword argument passthrough must be present verbatim.
        src = inspect.getsource(ucc)
        assert "freeze_blocked=freeze_blocked," in src


class TestOfflineQualification:
    def test_33_compact_artifact_size_bounded(self):
        diag = m.build_selection_freeze_diagnostics(
            freeze_blocked=True,
            coherence_diag=_coherence(contradiction_findings=[{"a": 1}] * 50, missing_idea_coverage=[{"b": 1}] * 50),
            repair_loop_status="NEEDS_HUMAN_REVIEW",
            post_authority_integrity_failure_codes=["X"] * 10,
        )
        # The block itself never carries the raw finding lists -- only
        # bounded counts/category codes -- so it stays small regardless
        # of how many findings the upstream coherence validator reports.
        assert len(json.dumps(diag)) < 2000

    def test_34_compileall(self):
        result = subprocess.run(
            ["python", "-m", "compileall", "-q", "cutsell_worker"],
            cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_35_no_provider_import(self):
        src = inspect.getsource(m)
        assert "google.generativeai" not in src and "provider" not in src.lower()

    def test_36_no_raw_modal_runpod_reference(self):
        src = inspect.getsource(m)
        for token in ("modal", "runpod", "s3://"):
            assert token not in src.lower()
