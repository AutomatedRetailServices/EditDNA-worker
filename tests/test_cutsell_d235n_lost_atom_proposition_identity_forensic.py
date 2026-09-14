"""D-235N: EXACT PROPOSITION IDENTITY BRIDGE FORENSIC, OFFLINE ONLY.

Tests `cutsell_worker.lost_atom_proposition_identity_forensic` -- a pure
classifier over already-computed, caller-supplied values (P1 moment
source_span_id, D-199's own language-evidence-source provenance,
proposition candidate ids/lookup/object) plus a set of SOURCE-CODE-TRUTH
tests that verify this task's own forensic claims are grounded in the
ACTUAL, current, unmodified `editorial_moment_sequence_integration.py`/
`language_spine_live_integration.py`/`language_utterance_attempt.py`
source -- never asserted without direct code evidence.

This module makes NO Freeze/repair/resolver/materiality decision, mints
NO new id, and is not imported by any live path.
"""
import ast
import inspect
import json

from cutsell_worker.lost_atom_proposition_identity_forensic import (
    SEAM_AMBIGUOUS,
    SEAM_EXACT,
    SEAM_HEURISTIC_ONLY,
    SEAM_MISSING,
    SEAM_ONE_TO_MANY_EXACT,
    SCHEMA_VERSION,
    classify_proposition_identity_bridge,
    classify_seam2_clip_to_p1_moment,
    classify_seam3_moment_to_proposition_ids,
    classify_seam4_proposition_id_to_candidate,
    classify_seam5_candidate_to_slot_evidence,
)
from cutsell_worker.language_spine_live_integration import (
    LANGUAGE_EVIDENCE_CANONICAL,
    LANGUAGE_EVIDENCE_D157_FALLBACK,
)

FORENSIC_MODULE_PATH = "cutsell_worker/lost_atom_proposition_identity_forensic.py"
EDITORIAL_MOMENT_INTEGRATION_PATH = "cutsell_worker/editorial_moment_sequence_integration.py"
LANGUAGE_SPINE_LIVE_PATH = "cutsell_worker/language_spine_live_integration.py"
LANGUAGE_UTTERANCE_ATTEMPT_PATH = "cutsell_worker/language_utterance_attempt.py"
CANONICAL_IDENTITY_PATH = "cutsell_worker/canonical_identity.py"
ATTEMPT_RECONSTRUCTION_PATH = "cutsell_worker/attempt_reconstruction.py"
PIPELINE_PATH = "cutsell_worker/pipeline.py"


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _content_only_lines(path: str) -> str:
    text = _read(path)
    tree = ast.parse(text)
    doc = ast.get_docstring(tree)
    if doc and doc in text:
        text = text.replace(doc, "", 1)
    return text


class FakeCandidate:
    def __init__(self, editorial_slot_evidence="CTA"):
        self.editorial_slot_evidence = editorial_slot_evidence


# --- Seam 2: clip_id -> P1 EditorialMoment ---------------------------------
class TestSeam2ClipToMoment:
    def test_01_clip_id_exact_match(self):
        assert classify_seam2_clip_to_p1_moment(clip_id="clip_1", editorial_moment_source_span_id="clip_1") == SEAM_EXACT

    def test_02_no_moment_found_missing(self):
        assert classify_seam2_clip_to_p1_moment(clip_id="clip_1", editorial_moment_source_span_id=None) == SEAM_MISSING

    def test_mismatched_span_id_ambiguous(self):
        # Should never happen in real code (source_span_id is a verbatim
        # copy of clip_id) -- defensive classification only.
        assert classify_seam2_clip_to_p1_moment(clip_id="clip_1", editorial_moment_source_span_id="clip_2") == SEAM_AMBIGUOUS


# --- Seam 3: the decisive seam ----------------------------------------------
class TestSeam3MomentToPropositionIds:
    def test_03_canonical_populated_is_heuristic_only_never_exact(self):
        status = classify_seam3_moment_to_proposition_ids(
            language_evidence_source=LANGUAGE_EVIDENCE_CANONICAL, proposition_candidate_ids=("prop_abc",),
        )
        assert status == SEAM_HEURISTIC_ONLY
        assert status != SEAM_EXACT

    def test_04_canonical_but_empty_ids_missing(self):
        status = classify_seam3_moment_to_proposition_ids(
            language_evidence_source=LANGUAGE_EVIDENCE_CANONICAL, proposition_candidate_ids=(),
        )
        assert status == SEAM_MISSING

    def test_05_canonical_language_spine_case(self):
        # The "canonical path" fixture: real Language Spine evidence,
        # ids present -- still HEURISTIC_ONLY by design.
        status = classify_seam3_moment_to_proposition_ids(
            language_evidence_source=LANGUAGE_EVIDENCE_CANONICAL, proposition_candidate_ids=("prop_x",),
        )
        assert status == SEAM_HEURISTIC_ONLY

    def test_06_fallback_case_always_missing(self):
        status = classify_seam3_moment_to_proposition_ids(
            language_evidence_source=LANGUAGE_EVIDENCE_D157_FALLBACK, proposition_candidate_ids=(),
        )
        assert status == SEAM_MISSING

    def test_06b_fallback_case_missing_even_if_ids_somehow_present(self):
        # Defensive: even if a caller mistakenly supplied ids alongside a
        # fallback source, this function never upgrades it -- fallback
        # always reports MISSING (matches the real code's own guarantee
        # that a fallback attempt_id never appears in a real
        # PropositionCandidate's attempt_ids).
        status = classify_seam3_moment_to_proposition_ids(
            language_evidence_source=LANGUAGE_EVIDENCE_D157_FALLBACK, proposition_candidate_ids=("should_not_happen",),
        )
        assert status == SEAM_MISSING

    def test_07_one_proposition_per_moment_still_heuristic(self):
        status = classify_seam3_moment_to_proposition_ids(
            language_evidence_source=LANGUAGE_EVIDENCE_CANONICAL, proposition_candidate_ids=("prop_only_one",),
        )
        assert status == SEAM_HEURISTIC_ONLY

    def test_08_multiple_propositions_per_moment_never_promoted_to_exact(self):
        # Structural robustness: even if the data model were ever widened
        # to allow >1 id per moment, this function must never call that
        # ONE_TO_MANY_EXACT -- the source itself is still the overlap
        # match, so it stays HEURISTIC_ONLY.
        status = classify_seam3_moment_to_proposition_ids(
            language_evidence_source=LANGUAGE_EVIDENCE_CANONICAL, proposition_candidate_ids=("prop_a", "prop_b"),
        )
        assert status == SEAM_HEURISTIC_ONLY
        assert status != SEAM_ONE_TO_MANY_EXACT

    def test_unknown_evidence_source_missing(self):
        assert classify_seam3_moment_to_proposition_ids(language_evidence_source=None, proposition_candidate_ids=("x",)) == SEAM_MISSING


# --- Seam 4: proposition_candidate_id -> PropositionCandidate --------------
class TestSeam4PropositionIdToCandidate:
    def test_09_exact_lookup_found(self):
        lookup = {"prop_abc": object()}
        assert classify_seam4_proposition_id_to_candidate(proposition_candidate_id="prop_abc", proposition_candidate_lookup=lookup) == SEAM_EXACT

    def test_missing_id_not_in_lookup(self):
        assert classify_seam4_proposition_id_to_candidate(proposition_candidate_id="prop_x", proposition_candidate_lookup={}) == SEAM_MISSING

    def test_no_id_at_all_missing(self):
        assert classify_seam4_proposition_id_to_candidate(proposition_candidate_id=None, proposition_candidate_lookup={"a": 1}) == SEAM_MISSING


# --- Seam 5: PropositionCandidate -> editorial_slot_evidence ----------------
class TestSeam5CandidateToSlotEvidence:
    def test_10_slot_evidence_retained(self):
        assert classify_seam5_candidate_to_slot_evidence(FakeCandidate("CTA")) == SEAM_EXACT

    def test_no_candidate_missing(self):
        assert classify_seam5_candidate_to_slot_evidence(None) == SEAM_MISSING


# --- End-to-end chain --------------------------------------------------------
class TestEndToEndChain:
    def test_11_exact_clip_to_moment_but_chain_never_exact_overall(self):
        out = classify_proposition_identity_bridge(
            clip_id="clip_1", editorial_moment_source_span_id="clip_1",
            language_evidence_source=LANGUAGE_EVIDENCE_CANONICAL,
            proposition_candidate_ids=("prop_abc",),
            proposition_candidate_lookup={"prop_abc": FakeCandidate("CTA")},
            proposition_candidate=FakeCandidate("CTA"),
        )
        assert out.seam2_clip_to_p1_moment == SEAM_EXACT
        assert out.seam3_moment_to_proposition_ids == SEAM_HEURISTIC_ONLY
        assert out.seam4_proposition_id_to_candidate == SEAM_EXACT
        assert out.seam5_candidate_to_slot_evidence == SEAM_EXACT
        # The weakest link (Seam 3) determines the whole chain.
        assert out.end_to_end_status == SEAM_HEURISTIC_ONLY

    def test_no_moment_at_all_end_to_end_missing(self):
        out = classify_proposition_identity_bridge(
            clip_id="clip_1", editorial_moment_source_span_id=None, language_evidence_source=None,
        )
        assert out.end_to_end_status == SEAM_MISSING

    def test_fallback_source_end_to_end_missing(self):
        out = classify_proposition_identity_bridge(
            clip_id="clip_1", editorial_moment_source_span_id="clip_1",
            language_evidence_source=LANGUAGE_EVIDENCE_D157_FALLBACK,
        )
        assert out.end_to_end_status == SEAM_MISSING

    def test_16_deterministic_repeat(self):
        kwargs = dict(
            clip_id="clip_1", editorial_moment_source_span_id="clip_1",
            language_evidence_source=LANGUAGE_EVIDENCE_CANONICAL,
            proposition_candidate_ids=("prop_abc",),
            proposition_candidate_lookup={"prop_abc": FakeCandidate("CTA")},
            proposition_candidate=FakeCandidate("CTA"),
        )
        out1 = classify_proposition_identity_bridge(**kwargs)
        out2 = classify_proposition_identity_bridge(**kwargs)
        assert out1 == out2

    def test_15_multi_source_isolation(self):
        # Two different clip_ids classified independently -- no leakage.
        out_a = classify_proposition_identity_bridge(
            clip_id="clip_a", editorial_moment_source_span_id="clip_a",
            language_evidence_source=LANGUAGE_EVIDENCE_CANONICAL, proposition_candidate_ids=("prop_a",),
            proposition_candidate_lookup={"prop_a": FakeCandidate("CTA")}, proposition_candidate=FakeCandidate("CTA"),
        )
        out_b = classify_proposition_identity_bridge(
            clip_id="clip_b", editorial_moment_source_span_id=None, language_evidence_source=None,
        )
        assert out_a.clip_id == "clip_a"
        assert out_b.clip_id == "clip_b"
        assert out_a.end_to_end_status == SEAM_HEURISTIC_ONLY
        assert out_b.end_to_end_status == SEAM_MISSING
        assert out_a.end_to_end_status != out_b.end_to_end_status

    def test_28_diagnostics_json_safe(self):
        out = classify_proposition_identity_bridge(
            clip_id="clip_1", editorial_moment_source_span_id="clip_1", language_evidence_source=LANGUAGE_EVIDENCE_CANONICAL,
        )
        json.dumps(out.as_dict())

    def test_schema_version_present(self):
        out = classify_proposition_identity_bridge(clip_id="c", editorial_moment_source_span_id=None, language_evidence_source=None)
        assert out.provenance[0] == SCHEMA_VERSION


# --- Source-code-truth tests: ground this task's forensic claims in the
# ACTUAL current code, not merely asserted. -------------------------------
class TestSourceCodeTruth:
    def test_seam1_clip_id_verbatim_copy_in_pipeline(self):
        content = _read(PIPELINE_PATH)
        assert "clip_id=take.clip_id" in content

    def test_seam2_source_span_id_verbatim_copy(self):
        content = _read(EDITORIAL_MOMENT_INTEGRATION_PATH)
        assert "source_span_id=take.clip_id" in content

    def test_seam3_overlap_bridge_is_the_only_canonical_source(self):
        content = _read(EDITORIAL_MOMENT_INTEGRATION_PATH)
        assert "language_attempts_by_span_id.get(take.clip_id)" in content
        assert "LANGUAGE_EVIDENCE_CANONICAL" in content
        assert "LANGUAGE_EVIDENCE_D157_FALLBACK" in content

    def test_seam3_language_spine_docstring_admits_non_exact_alignment(self):
        content = _read(LANGUAGE_SPINE_LIVE_PATH)
        assert "will not, in general, align" in content
        assert "MAXIMUM-OVERLAP" in content

    def test_seam3_gated_behind_offdefault_diagnostics_flag(self):
        content = _read(PIPELINE_PATH)
        assert "live_language_spine_diagnostics_enabled()" in content

    def test_two_independent_attempt_id_minting_functions_confirmed(self):
        # DraftClip/CandidateTake's own attempt_id minting (attempt_
        # reconstruction.py -> canonical_identity.mint_attempt_id) is a
        # DIFFERENT function from the Language Spine's own LanguageAttempt
        # attempt_id minting (language_utterance_attempt.py::_attempt_id)
        # -- structurally distinct id spaces, not merely observed-distinct.
        canonical_identity = _read(CANONICAL_IDENTITY_PATH)
        attempt_reconstruction = _read(ATTEMPT_RECONSTRUCTION_PATH)
        language_utterance_attempt = _read(LANGUAGE_UTTERANCE_ATTEMPT_PATH)
        assert "def mint_attempt_id(" in canonical_identity
        assert "mint_attempt_id" in attempt_reconstruction
        assert "def _attempt_id(" in language_utterance_attempt
        # The two minting functions are not the same function object.
        assert "def mint_attempt_id(" not in language_utterance_attempt
        assert "def _attempt_id(" not in canonical_identity

    def test_build_proposition_candidates_is_strict_one_to_one(self):
        from cutsell_worker.language_proposition_relation import build_proposition_candidates
        source = inspect.getsource(build_proposition_candidates)
        assert "for attempt in ordered:" in source
        assert "attempt_ids=(attempt.attempt_id,)" in source

    def test_p1_moment_language_evidence_source_provenance_exposed(self):
        content = _read(EDITORIAL_MOMENT_INTEGRATION_PATH)
        assert "moment_language_evidence_source" in content

    def test_13_no_text_fuzzy_authority_in_this_forensic_module(self):
        content = _content_only_lines(FORENSIC_MODULE_PATH)
        for token in ("difflib", "SequenceMatcher", "token_overlap", "similarity", "fuzz"):
            assert token.lower() not in content.lower()

    def test_14_no_time_overlap_authority_in_this_forensic_module(self):
        # This module may NAME the overlap match in a reason code (it must
        # be able to explain why a seam is HEURISTIC_ONLY) but must never
        # IMPLEMENT a time-overlap computation itself -- no interval
        # arithmetic, no start/end comparison, no duration calculation.
        content = _content_only_lines(FORENSIC_MODULE_PATH)
        for token in ("overlap_duration", "def _overlap", "source_start", "source_end", "min(", "max("):
            assert token not in content


class TestNoNewIdentityOrLiveWiring:
    def test_17_no_new_id_minting_function(self):
        content = _content_only_lines(FORENSIC_MODULE_PATH)
        for token in ("def mint_", "hashlib", "uuid"):
            assert token not in content

    def test_18_no_provider_reference(self):
        content = _content_only_lines(FORENSIC_MODULE_PATH)
        for token in ("openai", "google.generativeai", "gemini", "Arbiter("):
            assert token.lower() not in content.lower()

    def test_19_no_raw_modal_runpod_reference(self):
        content = _content_only_lines(FORENSIC_MODULE_PATH)
        for token in ("modal.App", "runpod", "RunPod", "s3://", "boto3"):
            assert token not in content

    def test_20_no_freeze_repair_resolver_materiality_import(self):
        content = _content_only_lines(FORENSIC_MODULE_PATH)
        forbidden = [
            "final_story_coherence_validation", "repair_loop", "realization_resolver",
            "universal_clean_cut", "final_edit_reviewer", "canonical_edit_plan",
            "lost_semantic_atom_materiality", "lost_atom_editorial_requirement_evidence",
            "pacing_v2", "audio_join_treatment",
        ]
        import_lines = [l for l in content.splitlines() if l.strip().startswith(("import ", "from "))]
        import_block = "\n".join(import_lines)
        for name in forbidden:
            assert name not in import_block, f"unexpected import of {name!r}"

    def test_not_imported_by_any_live_module(self):
        for path in (
            "cutsell_worker/universal_clean_cut.py",
            "cutsell_worker/final_story_coherence_validation.py",
            "cutsell_worker/final_edit_reviewer.py",
            "cutsell_worker/repair_loop.py",
            "cutsell_worker/pipeline.py",
        ):
            content = _read(path)
            assert "lost_atom_proposition_identity_forensic" not in content

    def test_output_never_contains_blocking_key(self):
        out = classify_proposition_identity_bridge(clip_id="c", editorial_moment_source_span_id=None, language_evidence_source=None)
        assert not hasattr(out, "blocking")
        assert "blocking" not in out.as_dict()


class TestModuleQualification:
    def test_21_module_compiles(self):
        import py_compile
        py_compile.compile(FORENSIC_MODULE_PATH, doraise=True)
