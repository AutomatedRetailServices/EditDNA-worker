"""D-235L: Lost Semantic Atom MATERIALITY DISCRIMINATOR, OFFLINE ONLY.

Tests `cutsell_worker.lost_semantic_atom_materiality.
assess_lost_semantic_atom_materiality` -- a pure function that classifies
ONE already-computed `_lost_semantic_atoms()` row (plus two optional,
explicitly-supplied plan-level context flags) into a bounded materiality
vocabulary and an advisory blocking recommendation. This module makes NO
Freeze/repair/resolver/Pacing/Audio-Join decision and is not wired into any
live authority -- every test below asserts a CLASSIFICATION, never a
change to `row["blocking"]` or any live engine behavior.

Per the directive's own explicit instruction, the real D-235K blocking
atom's literal text is NEVER hardcoded into the discriminator itself (see
`lost_semantic_atom_materiality.py`'s own source, which contains no
literal transcript string) -- the D-235K-shape fixture below reproduces
only the STRUCTURAL shape (REAL_CONTENT_LOSS, zero missing critical atoms,
zero preserved claims), not the phrase.
"""
import ast
import json

from cutsell_worker.lost_semantic_atom_materiality import (
    MATERIALITY_CONFLICTED,
    MATERIALITY_INSUFFICIENT_EVIDENCE,
    MATERIALITY_MEANING_CRITICAL,
    MATERIALITY_NON_MATERIAL_REAL_CONTENT,
    MATERIALITY_REDUNDANT_EQUIVALENT,
    MATERIALITY_RETRY_OR_RECORDING_RESIDUE,
    RECOMMEND_ABSTAIN,
    RECOMMEND_BLOCK,
    RECOMMEND_DO_NOT_BLOCK,
    SCHEMA_VERSION,
    STATE_FOUND,
    STATE_NOT_FOUND,
    STATE_UNKNOWN,
    assess_lost_semantic_atom_materiality,
    assess_many,
)

MODULE_PATH = "cutsell_worker/lost_semantic_atom_materiality.py"


def _content_only_lines() -> str:
    """Strips the module's own top-level docstring (same D-230/D-235G
    false-positive-avoidance pattern) before substring-checking."""
    with open(MODULE_PATH) as f:
        text = f.read()
    tree = ast.parse(text)
    doc = ast.get_docstring(tree)
    if doc and doc in text:
        text = text.replace(doc, "", 1)
    return text


def _base_row(**overrides) -> dict:
    row = {
        "clip_id": "clip_1",
        "text": "generic discarded content with no special marker",
        "missing_critical_atoms": [],
        "atom_classifications": [],
        "missing_content_token_count": 5,
        "own_content_token_count": 6,
        "coverage_against_final_keep": 0.1667,
        "blocking": True,
        "classification": "REAL_CONTENT_LOSS",
    }
    row.update(overrides)
    return row


def _critical_atom_row(atom_type="number", evidence="price", **overrides):
    return _base_row(
        missing_critical_atoms=["some_atom"],
        atom_classifications=[
            {"atom": "some_atom", "atom_type": atom_type, "importance": "CRITICAL",
             "evidence": evidence, "resolved_by": "deterministic"},
        ],
        **overrides,
    )


class TestCriticalSafetyFloor:
    """Fixtures 1-7, 17, 18, 21: every meaning-critical shape must BLOCK,
    and never DO_NOT_BLOCK regardless of any other signal present."""

    def test_01_lost_negation_blocks(self):
        row = _base_row(
            missing_critical_atoms=["not"],
            atom_classifications=[{"atom": "not", "atom_type": "negation", "importance": "CRITICAL",
                                    "evidence": "negation_changes_truth_value", "resolved_by": "deterministic"}],
        )
        out = assess_lost_semantic_atom_materiality(row)
        assert out.materiality_status == MATERIALITY_MEANING_CRITICAL
        assert out.blocking_recommendation == RECOMMEND_BLOCK

    def test_02_lost_number_blocks(self):
        row = _critical_atom_row(atom_type="number", evidence="percentage")
        out = assess_lost_semantic_atom_materiality(row)
        assert out.materiality_status == MATERIALITY_MEANING_CRITICAL
        assert out.blocking_recommendation == RECOMMEND_BLOCK

    def test_03_lost_product_fact_blocks(self):
        # Modeled with the real "price" evidence marker -- the closest real
        # atom_type/evidence pair the current engine actually produces for
        # a product-fact-shaped number.
        row = _critical_atom_row(atom_type="number", evidence="price")
        out = assess_lost_semantic_atom_materiality(row)
        assert out.blocking_recommendation == RECOMMEND_BLOCK

    def test_04_lost_qualifier_blocks(self):
        row = _critical_atom_row(atom_type="number", evidence="measurement")
        out = assess_lost_semantic_atom_materiality(row)
        assert out.blocking_recommendation == RECOMMEND_BLOCK

    def test_05_lost_cta_meaning_blocks_via_negation(self):
        # A negation flipping a CTA's own truth value ("don't buy" vs "buy")
        # is real-schema-modeled as a negation atom -- always CRITICAL.
        row = _base_row(
            missing_critical_atoms=["dont"],
            atom_classifications=[{"atom": "dont", "atom_type": "negation", "importance": "CRITICAL",
                                    "evidence": "negation_changes_truth_value", "resolved_by": "deterministic"}],
        )
        out = assess_lost_semantic_atom_materiality(row)
        assert out.blocking_recommendation == RECOMMEND_BLOCK

    def test_06_lost_causal_statement_blocks(self):
        row = _critical_atom_row(atom_type="number", evidence="chronology_relation_language_present")
        out = assess_lost_semantic_atom_materiality(row)
        assert out.blocking_recommendation == RECOMMEND_BLOCK

    def test_07_lost_correction_outcome_blocks(self):
        row = _critical_atom_row(atom_type="number", evidence="correction_language_present")
        out = assess_lost_semantic_atom_materiality(row)
        assert out.blocking_recommendation == RECOMMEND_BLOCK

    def test_17_real_content_loss_with_critical_atom_blocks(self):
        row = _critical_atom_row(atom_type="number", evidence="dose_or_quantity")
        out = assess_lost_semantic_atom_materiality(row)
        assert out.materiality_status == MATERIALITY_MEANING_CRITICAL
        assert out.critical_atom_count == 1

    def test_18_short_fragment_but_critical_meaning_still_blocks(self):
        row = _critical_atom_row(own_content_token_count=1, coverage_against_final_keep=0.0)
        out = assess_lost_semantic_atom_materiality(row)
        assert out.blocking_recommendation == RECOMMEND_BLOCK
        assert out.materiality_status == MATERIALITY_MEANING_CRITICAL

    def test_21_retry_plus_critical_fact_mixed_still_blocks(self):
        row = _critical_atom_row(
            pre_group_restart_consultations=[{"neighbour_clip_id": "c2", "same_idea": False, "confidence": 0.1}],
        )
        out = assess_lost_semantic_atom_materiality(row)
        assert out.blocking_recommendation == RECOMMEND_BLOCK
        assert out.materiality_status == MATERIALITY_MEANING_CRITICAL

    def test_critical_claim_conflict_true_always_blocks_even_with_no_atom(self):
        row = _base_row()
        out = assess_lost_semantic_atom_materiality(row, critical_claim_conflict=True)
        assert out.blocking_recommendation == RECOMMEND_BLOCK
        assert out.materiality_status == MATERIALITY_MEANING_CRITICAL

    def test_critical_claim_conflict_overrides_retry_evidence(self):
        row = _base_row(pre_group_restart_consultations=[{"same_idea": False}])
        out = assess_lost_semantic_atom_materiality(row, critical_claim_conflict=True, recording_process_evidence=True)
        assert out.blocking_recommendation == RECOMMEND_BLOCK


class TestUncertainAndMalformed:
    def test_14_unknown_coverage_malformed_atom_data_abstains(self):
        row = _base_row(missing_critical_atoms=["something"], atom_classifications=[])
        out = assess_lost_semantic_atom_materiality(row)
        assert out.materiality_status == MATERIALITY_INSUFFICIENT_EVIDENCE
        assert out.blocking_recommendation == RECOMMEND_ABSTAIN

    def test_uncertain_atom_abstains_never_do_not_block(self):
        row = _base_row(
            missing_critical_atoms=["qty"],
            atom_classifications=[{"atom": "qty", "atom_type": "number", "importance": "UNCERTAIN",
                                    "evidence": "no_deterministic_signal_found", "resolved_by": "deterministic"}],
        )
        out = assess_lost_semantic_atom_materiality(row)
        assert out.blocking_recommendation == RECOMMEND_ABSTAIN
        assert out.materiality_status == MATERIALITY_INSUFFICIENT_EVIDENCE


class TestRetryAndRecordingProcess:
    """Fixtures 8, 9, 10, 20."""

    def test_08_retry_residue_do_not_block(self):
        row = _base_row(pre_group_restart_consultations=[
            {"neighbour_clip_id": "c2", "same_idea": False, "confidence": 0.2, "reason": "not_same_idea"},
        ])
        out = assess_lost_semantic_atom_materiality(row)
        assert out.materiality_status == MATERIALITY_RETRY_OR_RECORDING_RESIDUE
        assert out.blocking_recommendation == RECOMMEND_DO_NOT_BLOCK
        assert out.retry_or_process_status == STATE_FOUND

    def test_09_false_start_do_not_block(self):
        row = _base_row()
        out = assess_lost_semantic_atom_materiality(row, recording_process_evidence=True)
        assert out.materiality_status == MATERIALITY_RETRY_OR_RECORDING_RESIDUE
        assert out.blocking_recommendation == RECOMMEND_DO_NOT_BLOCK

    def test_10_creator_process_speech_do_not_block(self):
        row = _base_row(text="how am i supposed to say pull it yeah there you go i just needed a pep talk")
        out = assess_lost_semantic_atom_materiality(row, recording_process_evidence=True)
        assert out.blocking_recommendation == RECOMMEND_DO_NOT_BLOCK

    def test_20_same_topic_no_proven_equivalence_never_invents_equivalence(self):
        # No restart_consultations, no preserving_id, no suppressed_by --
        # nothing structurally supports "same topic" as equivalence.
        row = _base_row()
        out = assess_lost_semantic_atom_materiality(row)
        assert out.materiality_status != MATERIALITY_REDUNDANT_EQUIVALENT
        assert out.materiality_status == MATERIALITY_INSUFFICIENT_EVIDENCE


class TestRedundantEquivalent:
    """Fixture 12."""

    def test_12_redundant_equivalent_preserved_do_not_block(self):
        row = _base_row(preserving_realization_id="realization_9", preserved_claim_ids=["c1"])
        out = assess_lost_semantic_atom_materiality(row)
        assert out.materiality_status == MATERIALITY_REDUNDANT_EQUIVALENT
        assert out.blocking_recommendation == RECOMMEND_DO_NOT_BLOCK
        assert out.redundancy_status == STATE_FOUND
        assert out.preserved_claim_count == 1

    def test_redundant_via_suppressed_by_alone(self):
        row = _base_row(content_loss_suppressed_by="same_idea_semantic_equivalence")
        out = assess_lost_semantic_atom_materiality(row)
        assert out.materiality_status == MATERIALITY_REDUNDANT_EQUIVALENT


class TestNonMaterialAndOptionalDetail:
    """Fixtures 13, 16, 19, 25 (D-235K shape)."""

    def test_13_optional_detail_do_not_block(self):
        row = _base_row()
        out = assess_lost_semantic_atom_materiality(row, critical_claim_conflict=False)
        assert out.materiality_status == MATERIALITY_NON_MATERIAL_REAL_CONTENT
        assert out.blocking_recommendation == RECOMMEND_DO_NOT_BLOCK

    def test_16_real_content_loss_zero_critical_atoms_no_context_abstains(self):
        row = _base_row()
        out = assess_lost_semantic_atom_materiality(row)
        assert out.materiality_status == MATERIALITY_INSUFFICIENT_EVIDENCE
        assert out.blocking_recommendation == RECOMMEND_ABSTAIN

    def test_19_long_fragment_but_non_material_length_is_not_the_driver(self):
        short_row = _base_row(own_content_token_count=3, coverage_against_final_keep=0.05)
        long_row = _base_row(own_content_token_count=200, coverage_against_final_keep=0.4)
        short_out = assess_lost_semantic_atom_materiality(short_row, critical_claim_conflict=False)
        long_out = assess_lost_semantic_atom_materiality(long_row, critical_claim_conflict=False)
        # Same explicit clearance signal drives BOTH to the same
        # classification regardless of length -- proving length is never
        # the deciding factor by itself.
        assert short_out.materiality_status == long_out.materiality_status == MATERIALITY_NON_MATERIAL_REAL_CONTENT

    def test_25_d235k_shape_fixture_never_hardcodes_the_real_phrase(self):
        # Reproduces the STRUCTURAL shape only: REAL_CONTENT_LOSS,
        # zero missing critical atoms, zero preserved claims, no context
        # supplied (matching what this offline gate can honestly know
        # about the real run without inventing linkage evidence it never
        # had). Per the directive: expected outcome is NON_MATERIAL or
        # INSUFFICIENT_EVIDENCE, never MEANING_CRITICAL.
        row = _base_row(
            classification="REAL_CONTENT_LOSS",
            missing_critical_atoms=[], atom_classifications=[],
            preserved_claim_ids=[], own_content_token_count=6,
            missing_content_token_count=5, coverage_against_final_keep=0.1667,
            blocking=True,
        )
        out = assess_lost_semantic_atom_materiality(row)
        assert out.materiality_status in (MATERIALITY_NON_MATERIAL_REAL_CONTENT, MATERIALITY_INSUFFICIENT_EVIDENCE)
        assert out.materiality_status != MATERIALITY_MEANING_CRITICAL
        assert out.blocking_recommendation != RECOMMEND_BLOCK

    def test_module_source_never_contains_the_real_raw_phrase(self):
        content = _content_only_lines()
        assert "too many people ready set" not in content.lower()
        assert "pep talk" not in content.lower()
        assert "rhino is running a crate" not in content.lower()


class TestAbandonedFragment:
    """Fixture 11: abandoned fragment alone (no explicit recording-process
    signal) supports non-material or abstain, never a confident BLOCK, and
    is NOT enough alone to become RETRY_OR_RECORDING_RESIDUE."""

    def test_11_abandoned_fragment_without_signal_abstains(self):
        row = _base_row(own_content_token_count=2, coverage_against_final_keep=0.0)
        out = assess_lost_semantic_atom_materiality(row)
        assert out.materiality_status in (MATERIALITY_NON_MATERIAL_REAL_CONTENT, MATERIALITY_INSUFFICIENT_EVIDENCE)
        assert out.blocking_recommendation != RECOMMEND_BLOCK
        assert out.materiality_status != MATERIALITY_RETRY_OR_RECORDING_RESIDUE


class TestConflictingEvidence:
    """Fixture 15."""

    def test_15_conflicting_retry_consultations_abstains(self):
        row = _base_row(pre_group_restart_consultations=[
            {"neighbour_clip_id": "c2", "same_idea": True, "confidence": 0.5},
            {"neighbour_clip_id": "c3", "same_idea": False, "confidence": 0.3},
        ])
        out = assess_lost_semantic_atom_materiality(row)
        assert out.materiality_status == MATERIALITY_CONFLICTED
        assert out.blocking_recommendation == RECOMMEND_ABSTAIN


class TestMultipleAndDeterminism:
    """Fixtures 22, 23."""

    def test_22_mixed_materiality_batch_each_classified_independently(self):
        rows = [
            _critical_atom_row(),  # -> MEANING_CRITICAL
            _base_row(clip_id="c2", pre_group_restart_consultations=[{"same_idea": False}]),  # -> RETRY
            _base_row(clip_id="c3", preserving_realization_id="r1"),  # -> REDUNDANT
        ]
        out = assess_many(rows)
        assert len(out) == 3
        assert out[0].materiality_status == MATERIALITY_MEANING_CRITICAL
        assert out[1].materiality_status == MATERIALITY_RETRY_OR_RECORDING_RESIDUE
        assert out[2].materiality_status == MATERIALITY_REDUNDANT_EQUIVALENT

    def test_22b_batch_context_maps_scoped_per_clip(self):
        rows = [_base_row(clip_id="a"), _base_row(clip_id="b")]
        out = assess_many(rows, critical_claim_conflict_by_clip_id={"a": True})
        assert out[0].materiality_status == MATERIALITY_MEANING_CRITICAL
        assert out[1].materiality_status != MATERIALITY_MEANING_CRITICAL

    def test_23_deterministic_repeat(self):
        row = _base_row(pre_group_restart_consultations=[{"same_idea": False}])
        out1 = assess_lost_semantic_atom_materiality(row)
        out2 = assess_lost_semantic_atom_materiality(row)
        assert out1 == out2
        assert out1.as_dict() == out2.as_dict()


class TestNoUsableRealizationShape:
    def test_no_usable_realization_row_shape_handled(self):
        row = {
            "clip_id": "clip_x", "text": "some text", "kind": "LOST_IN_NO_USABLE_REALIZATION_FAMILY",
            "basis": "corroborated_bts_singleton", "blocking": False,
        }
        out = assess_lost_semantic_atom_materiality(row)
        assert out.existing_loss_classification == "LOST_IN_NO_USABLE_REALIZATION_FAMILY"


class TestDiagnosticsAndBounding:
    def test_28_as_dict_is_json_safe_no_transcript_dump(self):
        row = _base_row(text="word " * 500)
        out = assess_lost_semantic_atom_materiality(row)
        d = out.as_dict()
        json.dumps(d)  # must not raise
        assert len(d["bounded_excerpt"]) < 200

    def test_schema_version_present(self):
        out = assess_lost_semantic_atom_materiality(_base_row())
        assert out.provenance[0] == SCHEMA_VERSION

    def test_reason_codes_never_empty(self):
        out = assess_lost_semantic_atom_materiality(_base_row())
        assert len(out.reason_codes) >= 1


class TestBehaviorNeutrality:
    """Fixtures 24, 25 (no provider/no RAW), 26-29 (no Freeze/Pacing/
    Audio-Join/threshold mutation)."""

    def test_24_no_provider_or_arbiter_reference(self):
        content = _content_only_lines()
        for token in ("Arbiter(", "provider=", "SemanticEquivalenceArbiter", "openai", "google.generativeai", "gemini"):
            assert token.lower() not in content.lower()

    def test_no_raw_modal_runpod_reference(self):
        content = _content_only_lines()
        for token in ("modal.App", "runpod", "RunPod", "dispatch_workflow", "boto3", "s3://"):
            assert token not in content

    def test_26_no_freeze_or_repair_or_resolver_or_pacing_import(self):
        content = _content_only_lines()
        forbidden_imports = [
            "final_story_coherence_validation", "repair_loop", "realization_resolver",
            "universal_clean_cut", "final_edit_reviewer", "canonical_edit_plan",
            "pacing_v2", "audio_join_treatment", "boundary_engine", "ordering",
        ]
        import_lines = [line for line in content.splitlines() if line.strip().startswith(("import ", "from "))]
        import_block = "\n".join(import_lines)
        for name in forbidden_imports:
            assert name not in import_block, f"unexpected import of {name!r}"

    def test_never_mutates_input_row(self):
        row = _critical_atom_row()
        original = json.loads(json.dumps(row))
        assess_lost_semantic_atom_materiality(row)
        assert row == original

    def test_output_never_contains_blocking_key(self):
        # The output type has no field named "blocking" -- proves this
        # module cannot be mistaken for setting the real Freeze flag.
        out = assess_lost_semantic_atom_materiality(_base_row())
        assert not hasattr(out, "blocking")
        assert "blocking" not in out.as_dict()

    def test_editorially_required_has_no_reachable_path_honestly(self):
        # This module's own documented gap: no combination of currently-
        # supported inputs reaches MATERIALITY_EDITORIALLY_REQUIRED. This
        # test enumerates a broad input sweep and asserts the category
        # never appears -- if it ever did without a corresponding module
        # change, this test would (correctly) start failing.
        from cutsell_worker.lost_semantic_atom_materiality import MATERIALITY_EDITORIALLY_REQUIRED
        seen = set()
        for cc in (None, True, False):
            for rp in (None, True, False):
                for extra in (
                    {},
                    {"pre_group_restart_consultations": [{"same_idea": False}]},
                    {"preserving_realization_id": "r1"},
                    {"content_loss_suppressed_by": "x"},
                    {"missing_critical_atoms": ["a"], "atom_classifications": [
                        {"atom": "a", "atom_type": "number", "importance": "CRITICAL", "evidence": "price", "resolved_by": "deterministic"}]},
                ):
                    row = _base_row(**extra)
                    out = assess_lost_semantic_atom_materiality(row, critical_claim_conflict=cc, recording_process_evidence=rp)
                    seen.add(out.materiality_status)
        assert MATERIALITY_EDITORIALLY_REQUIRED not in seen


class TestModuleQualification:
    def test_30_module_compiles(self):
        import py_compile
        py_compile.compile(MODULE_PATH, doraise=True)

    def test_reuses_real_semantic_atom_importance_constants(self):
        content = _content_only_lines()
        assert "from .semantic_atom_importance import" in content
        assert "CRITICAL" in content and "UNCERTAIN" in content
