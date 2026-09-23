"""D-235M: Lost Semantic Atom EDITORIAL-REQUIREMENT EVIDENCE FOUNDATION,
OFFLINE ONLY.

Tests `cutsell_worker.lost_atom_editorial_requirement_evidence.
assess_editorial_requirement_evidence` -- a pure function that classifies
editorial-requirement status from OPTIONAL, explicitly caller-supplied,
already-existing structured evidence (P1 EditorialMoment roles/audience-
delivery status, family-level idea-coverage status, an honestly-labelled
PropositionCandidate.editorial_slot_evidence, causal-dependency evidence).
This module discovers none of that evidence itself and makes no Freeze/
repair/resolver decision -- every test below asserts a CLASSIFICATION,
never a change to `row["blocking"]` or any live engine behavior.
"""
import ast
import json

from cutsell_worker.lost_atom_editorial_requirement_evidence import (
    IDENTITY_MAPPING_AMBIGUOUS,
    IDENTITY_MAPPING_EXACT,
    IDENTITY_MAPPING_HEURISTIC_OVERLAP,
    IDENTITY_MAPPING_NONE,
    REQUIREMENT_CONFLICTED,
    REQUIREMENT_INSUFFICIENT_EVIDENCE,
    REQUIREMENT_NOT_REQUIRED,
    REQUIREMENT_REDUNDANT_REQUIRED_FUNCTION_PRESERVED,
    REQUIREMENT_REQUIRED,
    SCHEMA_VERSION,
    STATE_FOUND,
    STATE_UNKNOWN,
    assess_editorial_requirement_evidence,
)
from cutsell_worker.editorial_moment_sequence import (
    AUDIENCE_DELIVERY_NOT_SUPPORTED,
    AUDIENCE_DELIVERY_SUPPORTED,
    MOMENT_ROLE_ABANDONED_ATTEMPT,
    MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
    MOMENT_ROLE_FALSE_START,
    MOMENT_ROLE_RECORDING_PROCESS,
    MOMENT_ROLE_RETRY,
)
from cutsell_worker.language_proposition_relation import SLOT_CONCLUSION, SLOT_CTA, SLOT_HOOK, SLOT_OTHER

MODULE_PATH = "cutsell_worker/lost_atom_editorial_requirement_evidence.py"


def _content_only_lines() -> str:
    with open(MODULE_PATH) as f:
        text = f.read()
    tree = ast.parse(text)
    doc = ast.get_docstring(tree)
    if doc and doc in text:
        text = text.replace(doc, "", 1)
    return text


def _row(**overrides) -> dict:
    row = {"clip_id": "clip_1", "text": "generic discarded content", "blocking": True, "classification": "REAL_CONTENT_LOSS"}
    row.update(overrides)
    return row


class TestNoIdentityMapping:
    def test_12_no_identity_mapping_insufficient_evidence(self):
        out = assess_editorial_requirement_evidence(_row())
        assert out.editorial_requirement_status == REQUIREMENT_INSUFFICIENT_EVIDENCE

    def test_25_d235k_shape_fixture_no_evidence_insufficient(self):
        # Structural D-235K shape: REAL_CONTENT_LOSS, no context supplied.
        # Per directive: never REQUIRED, never NOT_REQUIRED -- INSUFFICIENT.
        row = _row(classification="REAL_CONTENT_LOSS", missing_critical_atoms=[], preserved_claim_ids=[])
        out = assess_editorial_requirement_evidence(row)
        assert out.editorial_requirement_status == REQUIREMENT_INSUFFICIENT_EVIDENCE
        assert out.editorial_requirement_status not in (REQUIREMENT_REQUIRED, REQUIREMENT_NOT_REQUIRED)


class TestAmbiguousMapping:
    def test_13_ambiguous_mapping_conflicted(self):
        out = assess_editorial_requirement_evidence(_row(), identity_mapping_status=IDENTITY_MAPPING_AMBIGUOUS)
        assert out.editorial_requirement_status == REQUIREMENT_CONFLICTED


class TestRetryAndProcessFirewall:
    """Fixtures 9, 10, 11 (retry/process never promoted to REQUIRED)."""

    def test_09_retry_residue_never_promoted(self):
        out = assess_editorial_requirement_evidence(
            _row(), recording_process_status=MOMENT_ROLE_RETRY,
            idea_coverage_status=True,  # even with a "required" signal present
        )
        assert out.editorial_requirement_status == REQUIREMENT_NOT_REQUIRED

    def test_10_creator_process_speech_never_promoted(self):
        out = assess_editorial_requirement_evidence(
            _row(), recording_process_status=MOMENT_ROLE_RECORDING_PROCESS,
            downstream_dependency_present=True,
        )
        assert out.editorial_requirement_status == REQUIREMENT_NOT_REQUIRED

    def test_false_start_never_promoted(self):
        out = assess_editorial_requirement_evidence(_row(), recording_process_status=MOMENT_ROLE_FALSE_START)
        assert out.editorial_requirement_status == REQUIREMENT_NOT_REQUIRED

    def test_abandoned_attempt_never_promoted(self):
        out = assess_editorial_requirement_evidence(_row(), recording_process_status=MOMENT_ROLE_ABANDONED_ATTEMPT)
        assert out.editorial_requirement_status == REQUIREMENT_NOT_REQUIRED

    def test_not_supported_audience_delivery_never_required(self):
        out = assess_editorial_requirement_evidence(
            _row(), audience_delivery_status=AUDIENCE_DELIVERY_NOT_SUPPORTED,
            idea_coverage_status=True,
        )
        assert out.editorial_requirement_status == REQUIREMENT_NOT_REQUIRED


class TestChronologyFirewall:
    def test_16_no_chronology_parameter_exists(self):
        # Structurally impossible to pass raw ordering/timing as evidence --
        # the function signature has no such parameter at all.
        import inspect
        sig = inspect.signature(assess_editorial_requirement_evidence)
        for forbidden in ("order", "position", "chronolog", "sequence_index", "timestamp"):
            assert not any(forbidden in name.lower() for name in sig.parameters)


class TestRequiredFixtures:
    """Fixtures 1-5: REQUIRED only reachable via EXACT identity + a real
    reused signal (idea coverage, downstream dependency, or an EXACT
    story-function slot) -- never via HEURISTIC_OVERLAP evidence alone."""

    def test_01_unique_necessary_setup_via_idea_coverage_required(self):
        out = assess_editorial_requirement_evidence(_row(), idea_coverage_status=True)
        assert out.editorial_requirement_status == REQUIREMENT_REQUIRED

    def test_02_unique_necessary_consequence_via_downstream_dependency_required(self):
        out = assess_editorial_requirement_evidence(_row(), downstream_dependency_present=True)
        assert out.editorial_requirement_status == REQUIREMENT_REQUIRED

    def test_03_required_causal_bridge_via_downstream_dependency_required(self):
        out = assess_editorial_requirement_evidence(
            _row(), downstream_dependency_present=True, recording_process_status=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
        )
        assert out.editorial_requirement_status == REQUIREMENT_REQUIRED

    def test_04_sole_required_cta_meaning_via_exact_slot_required(self):
        out = assess_editorial_requirement_evidence(
            _row(), editorial_slot_evidence=SLOT_CTA, slot_evidence_source=IDENTITY_MAPPING_EXACT,
        )
        assert out.editorial_requirement_status == REQUIREMENT_REQUIRED
        assert out.slot_evidence_status == STATE_FOUND

    def test_05_required_conclusion_via_exact_slot_required(self):
        out = assess_editorial_requirement_evidence(
            _row(), editorial_slot_evidence=SLOT_CONCLUSION, slot_evidence_source=IDENTITY_MAPPING_EXACT,
        )
        assert out.editorial_requirement_status == REQUIREMENT_REQUIRED

    def test_hook_slot_also_required(self):
        out = assess_editorial_requirement_evidence(
            _row(), editorial_slot_evidence=SLOT_HOOK, slot_evidence_source=IDENTITY_MAPPING_EXACT,
        )
        assert out.editorial_requirement_status == REQUIREMENT_REQUIRED

    def test_slot_other_never_required_even_if_exact(self):
        out = assess_editorial_requirement_evidence(
            _row(), editorial_slot_evidence=SLOT_OTHER, slot_evidence_source=IDENTITY_MAPPING_EXACT,
        )
        assert out.editorial_requirement_status != REQUIREMENT_REQUIRED

    def test_heuristic_overlap_slot_never_authoritative_for_required(self):
        # The REAL current engine can only ever supply HEURISTIC_OVERLAP
        # for editorial_slot_evidence (per this module's own documented
        # identity-seam finding) -- this must NEVER produce REQUIRED.
        out = assess_editorial_requirement_evidence(
            _row(), editorial_slot_evidence=SLOT_CTA, slot_evidence_source=IDENTITY_MAPPING_HEURISTIC_OVERLAP,
        )
        assert out.editorial_requirement_status != REQUIREMENT_REQUIRED

    def test_missing_slot_evidence_source_defaults_to_non_authoritative(self):
        out = assess_editorial_requirement_evidence(_row(), editorial_slot_evidence=SLOT_CTA)
        assert out.editorial_requirement_status != REQUIREMENT_REQUIRED


class TestOptionalDetailAndClearance:
    """Fixture 6: optional elaboration -> NOT_REQUIRED."""

    def test_06_optional_elaboration_not_required(self):
        out = assess_editorial_requirement_evidence(
            _row(), idea_coverage_status=False, downstream_dependency_present=False,
            recording_process_status=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
            audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,
        )
        assert out.editorial_requirement_status == REQUIREMENT_NOT_REQUIRED

    def test_19_story_completeness_remains_complete_not_required(self):
        out = assess_editorial_requirement_evidence(
            _row(), idea_coverage_status=False, downstream_dependency_present=False,
            audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,
        )
        assert out.editorial_requirement_status == REQUIREMENT_NOT_REQUIRED

    def test_20_story_completeness_becomes_incomplete_required(self):
        out = assess_editorial_requirement_evidence(_row(), idea_coverage_status=True)
        assert out.editorial_requirement_status == REQUIREMENT_REQUIRED


class TestRedundantFunctionPreserved:
    """Fixtures 7, 8."""

    def test_07_redundant_setup_preserved_elsewhere(self):
        out = assess_editorial_requirement_evidence(_row(), replacement_function_preserved=True)
        assert out.editorial_requirement_status == REQUIREMENT_REDUNDANT_REQUIRED_FUNCTION_PRESERVED

    def test_08_redundant_conclusion_preserved_elsewhere(self):
        out = assess_editorial_requirement_evidence(
            _row(), editorial_slot_evidence=SLOT_CONCLUSION, slot_evidence_source=IDENTITY_MAPPING_HEURISTIC_OVERLAP,
            replacement_function_preserved=True,
        )
        assert out.editorial_requirement_status == REQUIREMENT_REDUNDANT_REQUIRED_FUNCTION_PRESERVED

    def test_redundant_via_row_level_suppression_proxy(self):
        row = _row(content_loss_suppressed_by="same_idea_semantic_equivalence")
        out = assess_editorial_requirement_evidence(row)
        assert out.editorial_requirement_status == REQUIREMENT_REDUNDANT_REQUIRED_FUNCTION_PRESERVED

    def test_redundant_via_preserving_realization_id_proxy(self):
        row = _row(preserving_realization_id="realization_9")
        out = assess_editorial_requirement_evidence(row)
        assert out.editorial_requirement_status == REQUIREMENT_REDUNDANT_REQUIRED_FUNCTION_PRESERVED


class TestConflictedSignals:
    def test_required_and_redundant_together_conflicted(self):
        out = assess_editorial_requirement_evidence(
            _row(), idea_coverage_status=True, replacement_function_preserved=True,
        )
        assert out.editorial_requirement_status == REQUIREMENT_CONFLICTED


class TestCriticalityFirewall:
    def test_module_never_reads_or_reports_meaning_critical(self):
        # This module has no parameter or field named for meaning-
        # criticality at all -- it structurally cannot override D-235L's
        # own MEANING_CRITICAL verdict.
        out = assess_editorial_requirement_evidence(_row())
        d = out.as_dict()
        assert "meaning_critical" not in json.dumps(d).lower()

    def test_module_never_imports_materiality_module(self):
        content = _content_only_lines()
        assert "lost_semantic_atom_materiality" not in content


class TestNoLiveWiringOrMutation:
    """Fixtures 27, 28 equivalents + no Freeze/repair/resolver import."""

    def test_29_no_freeze_repair_resolver_import(self):
        content = _content_only_lines()
        forbidden = [
            "final_story_coherence_validation", "repair_loop", "realization_resolver",
            "universal_clean_cut", "final_edit_reviewer", "canonical_edit_plan",
            "pacing_v2", "audio_join_treatment", "boundary_engine",
        ]
        import_lines = [l for l in content.splitlines() if l.strip().startswith(("import ", "from "))]
        import_block = "\n".join(import_lines)
        for name in forbidden:
            assert name not in import_block, f"unexpected import of {name!r}"

    def test_30_no_provider_or_arbiter_reference(self):
        content = _content_only_lines()
        for token in ("Arbiter(", "provider=", "openai", "google.generativeai", "gemini", "CausalOrderArbiter"):
            assert token.lower() not in content.lower() or token == "CausalOrderArbiter" and "class CausalOrderArbiter" not in content

    def test_31_no_raw_modal_runpod_reference(self):
        content = _content_only_lines()
        for token in ("modal.App", "runpod", "RunPod", "dispatch_workflow", "boto3", "s3://"):
            assert token not in content

    def test_never_mutates_input_row(self):
        row = _row()
        original = json.loads(json.dumps(row))
        assess_editorial_requirement_evidence(row, idea_coverage_status=True)
        assert row == original

    def test_output_never_contains_blocking_key(self):
        out = assess_editorial_requirement_evidence(_row())
        assert not hasattr(out, "blocking")
        assert "blocking" not in out.as_dict()

    def test_no_p2_construction_function_called(self):
        # This module never imports the P2 region-BUILDING function --
        # only the SLOT_*/MOMENT_ROLE_* constants (read-only vocabulary).
        content = _content_only_lines()
        assert "build_whole_video_editorial_regions" not in content
        assert "build_editorial_moments_for_source" not in content
        assert "build_proposition_candidates" not in content


class TestDeterminismAndDiagnostics:
    def test_22_deterministic_repeat(self):
        row = _row()
        out1 = assess_editorial_requirement_evidence(row, idea_coverage_status=True)
        out2 = assess_editorial_requirement_evidence(row, idea_coverage_status=True)
        assert out1 == out2
        assert out1.as_dict() == out2.as_dict()

    def test_28_diagnostics_json_safe(self):
        out = assess_editorial_requirement_evidence(_row(), idea_coverage_status=True)
        json.dumps(out.as_dict())

    def test_schema_version_present(self):
        out = assess_editorial_requirement_evidence(_row())
        assert out.provenance[0] == SCHEMA_VERSION

    def test_reason_codes_never_empty(self):
        out = assess_editorial_requirement_evidence(_row())
        assert len(out.reason_codes) >= 1


class TestModuleQualification:
    def test_29_module_compiles(self):
        import py_compile
        py_compile.compile(MODULE_PATH, doraise=True)

    def test_reuses_real_editorial_moment_and_slot_constants(self):
        content = _content_only_lines()
        assert "from .editorial_moment_sequence import" in content
        assert "from .language_proposition_relation import" in content
        # Never redefines the constants it imports.
        assert 'MOMENT_ROLE_RETRY = "' not in content
        assert 'SLOT_CTA = "' not in content
