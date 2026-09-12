"""D-237I workflow extraction -- Exact Identity Observability, sibling-safe
(workflow-only).

Mirrors D-235G/D-235J's own `test_cutsell_d235g_workflow_extraction.py` /
`test_cutsell_d235j_workflow_extraction.py` exactly: extracts the new
"D-237I Exact Identity Observability -> sibling-safe extraction" step's
ACTUAL embedded Python via `yaml.safe_load` (the same dedent GitHub Actions
itself performs on a `run: |` block scalar, never a raw-text slice) and
executes it against synthetic `artifact/video00-modal.json` fixtures --
proving a present `diagnostics["final_story_coherence_validation"]
["lost_atom_identity_observability"]` list is preserved byte-for-byte into
a small, bounded, sibling-safe artifact, and that an ABSENT/MALFORMED
subtree is reported OBSERVATIONALLY (source_status MISSING/MALFORMED,
exit 0) rather than failing the workflow -- the opposite convention from
D-235G/D-235J's own FAIL LOUDLY posture, per this task's own explicit
directive ("Do NOT fail the whole workflow merely because this diagnostic
is absent"). No paid compute; no cutsell_worker file touched by this
step or this test.
"""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import yaml

WORKFLOW_PATH = Path(".github/workflows/cutsell-video00-modal-raw.yml")
STEP_NAME = "D-237I Exact Identity Observability -> sibling-safe extraction"
OUT_FILE = "exact-identity-observability.json"


def _load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def _step_script() -> str:
    doc = _load_workflow()
    for step in doc["jobs"]["video00-modal-raw"]["steps"]:
        if step.get("name") == STEP_NAME:
            return step["run"]
    raise AssertionError(f"step {STEP_NAME!r} not found in {WORKFLOW_PATH}")


def _run_step(tmp_path: Path, engine_json) -> subprocess.CompletedProcess:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if engine_json is not None:
        (artifact_dir / "video00-modal.json").write_text(json.dumps(engine_json))
    return subprocess.run(
        ["bash", "-c", _step_script()], cwd=str(tmp_path), capture_output=True, text=True, timeout=30,
    )


def _read_out(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "artifact" / OUT_FILE).read_text())


# ---------------------------------------------------------------------------
# One representative D-237G row -- same shape `exact_identity_observability.
# py`'s own `lost_atom_identity_correlation`/`identity_match_diagnostic_row`
# produce; every field here mirrors a real field name on those functions.
# ---------------------------------------------------------------------------
_ROW_ONE = {
    "lost_atom_provenance_id": "prov_abc123",
    "clip_id": "clip_800d8d16b6b9e88325b7",
    "identity": {
        "schema_version": "cutsell.exact_identity_observability.v1",
        "clip_id": "clip_800d8d16b6b9e88325b7",
        "source_asset_id": "src_c18babee4a999e7a0b0f",
        "candidate_take": {
            "clip_id": "clip_800d8d16b6b9e88325b7",
            "source_asset_id": "src_c18babee4a999e7a0b0f",
            "source_start": 8.27,
            "source_end": 12.19,
            "word_index_count": 9,
            "word_index_min": 40,
            "word_index_max": 48,
            "word_indices": [40, 41, 42, 43, 44, 45, 46, 47, 48],
            "identity_status": "RECONSTRUCTED",
        },
        "language_attempts": [
            {
                "attempt_id": "lang_att_1",
                "source_asset_id": "src_c18babee4a999e7a0b0f",
                "source_start": 6.0,
                "source_end": 20.0,
                "word_index_count": 30,
                "word_index_min": 20,
                "word_index_max": 49,
                "word_indices": list(range(20, 50)),
                "identity_status": "LANGUAGE",
                "proposition_candidate_ids": ["prop_1"],
                "attempt_state": "CLEAN_ATTEMPT",
            },
            {
                "attempt_id": "lang_att_2",
                "source_asset_id": "src_c18babee4a999e7a0b0f",
                "source_start": 60.0,
                "source_end": 70.0,
                "word_index_count": 4,
                "word_index_min": 200,
                "word_index_max": 203,
                "word_indices": [200, 201, 202, 203],
                "identity_status": "LANGUAGE",
                "proposition_candidate_ids": ["prop_2", "prop_3"],
                "attempt_state": "ABANDONED_ATTEMPT",
            },
        ],
        "relationship_status": "RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED",
        "relationship_is_authoritative": False,
        "reconstructed_only_word_index_count": 0,
        "reconstructed_only_word_indices": [],
        "language_only_word_index_count": 25,
        "language_only_word_indices": list(range(20, 40)) + [49, 200, 201, 202, 203],
        "intersection_count": 9,
        "exact_match_by_clip_id_present": False,
        "exact_match_attempt_ids": [],
        "exact_match_proposition_candidate_ids": [],
    },
}

_ROW_TWO = {
    "lost_atom_provenance_id": "prov_def456",
    "clip_id": "clip_second",
    "identity": {
        "schema_version": "cutsell.exact_identity_observability.v1",
        "clip_id": "clip_second",
        "source_asset_id": "src_c18babee4a999e7a0b0f",
        "candidate_take": {
            "clip_id": "clip_second",
            "source_asset_id": "src_c18babee4a999e7a0b0f",
            "source_start": 50.0,
            "source_end": 55.0,
            "word_index_count": 5,
            "word_index_min": 100,
            "word_index_max": 104,
            "word_indices": [100, 101, 102, 103, 104],
            "identity_status": "RECONSTRUCTED",
        },
        "language_attempts": [
            {
                "attempt_id": "lang_att_3",
                "source_asset_id": "src_c18babee4a999e7a0b0f",
                "source_start": 50.0,
                "source_end": 55.0,
                "word_index_count": 5,
                "word_index_min": 100,
                "word_index_max": 104,
                "word_indices": [100, 101, 102, 103, 104],
                "identity_status": "LANGUAGE",
                "proposition_candidate_ids": ["prop_9"],
                "attempt_state": "CLEAN_ATTEMPT",
            },
        ],
        "relationship_status": "RELATIONSHIP_EXACT_SAME_MEMBERSHIP",
        "relationship_is_authoritative": True,
        "reconstructed_only_word_index_count": 0,
        "reconstructed_only_word_indices": [],
        "language_only_word_index_count": 0,
        "language_only_word_indices": [],
        "intersection_count": 5,
        "exact_match_by_clip_id_present": True,
        "exact_match_attempt_ids": ["lang_att_3"],
        "exact_match_proposition_candidate_ids": ["prop_9"],
    },
}


def _engine_with_rows(rows) -> dict:
    return {"diagnostics": {"final_story_coherence_validation": {"lost_atom_identity_observability": rows}}}


class TestStepFound:
    def test_01_step_exists_in_workflow(self):
        assert _step_script()


class TestPresentDiagnosticSubtreeExtraction:
    def test_02_subtree_extracted_present_status(self, tmp_path):
        result = _run_step(tmp_path, _engine_with_rows([_ROW_ONE]))
        assert result.returncode == 0, result.stderr
        out = _read_out(tmp_path)
        assert out["source_status"] == "PRESENT"
        assert out["lost_atom_identity_observability_row_count"] == 1

    def test_03_candidate_take_indices_preserved(self, tmp_path):
        result = _run_step(tmp_path, _engine_with_rows([_ROW_ONE]))
        out = _read_out(tmp_path)
        ct = out["lost_atom_identity_observability"][0]["identity"]["candidate_take"]
        assert ct["word_indices"] == [40, 41, 42, 43, 44, 45, 46, 47, 48]
        assert ct["word_index_count"] == 9
        assert ct["word_index_min"] == 40
        assert ct["word_index_max"] == 48
        assert ct["source_start"] == 8.27
        assert ct["source_end"] == 12.19

    def test_04_language_attempt_indices_preserved(self, tmp_path):
        result = _run_step(tmp_path, _engine_with_rows([_ROW_ONE]))
        out = _read_out(tmp_path)
        attempts = out["lost_atom_identity_observability"][0]["identity"]["language_attempts"]
        assert len(attempts) == 2
        assert attempts[0]["word_indices"] == list(range(20, 50))
        assert attempts[1]["word_indices"] == [200, 201, 202, 203]

    def test_05_proposition_ids_preserved(self, tmp_path):
        result = _run_step(tmp_path, _engine_with_rows([_ROW_ONE]))
        out = _read_out(tmp_path)
        attempts = out["lost_atom_identity_observability"][0]["identity"]["language_attempts"]
        assert attempts[0]["proposition_candidate_ids"] == ["prop_1"]
        assert attempts[1]["proposition_candidate_ids"] == ["prop_2", "prop_3"]
        assert attempts[0]["attempt_state"] == "CLEAN_ATTEMPT"
        assert attempts[1]["attempt_state"] == "ABANDONED_ATTEMPT"

    def test_06_relationship_status_preserved(self, tmp_path):
        result = _run_step(tmp_path, _engine_with_rows([_ROW_ONE, _ROW_TWO]))
        out = _read_out(tmp_path)
        rows = out["lost_atom_identity_observability"]
        assert rows[0]["identity"]["relationship_status"] == "RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED"
        assert rows[1]["identity"]["relationship_status"] == "RELATIONSHIP_EXACT_SAME_MEMBERSHIP"

    def test_07_authoritative_boolean_preserved(self, tmp_path):
        result = _run_step(tmp_path, _engine_with_rows([_ROW_ONE, _ROW_TWO]))
        out = _read_out(tmp_path)
        rows = out["lost_atom_identity_observability"]
        assert rows[0]["identity"]["relationship_is_authoritative"] is False
        assert rows[1]["identity"]["relationship_is_authoritative"] is True

    def test_08_reconstructed_only_indices_preserved(self, tmp_path):
        result = _run_step(tmp_path, _engine_with_rows([_ROW_ONE]))
        out = _read_out(tmp_path)
        row = out["lost_atom_identity_observability"][0]["identity"]
        assert row["reconstructed_only_word_indices"] == []
        assert row["reconstructed_only_word_index_count"] == 0

    def test_09_language_only_indices_preserved(self, tmp_path):
        result = _run_step(tmp_path, _engine_with_rows([_ROW_ONE]))
        out = _read_out(tmp_path)
        row = out["lost_atom_identity_observability"][0]["identity"]
        assert row["language_only_word_indices"] == list(range(20, 40)) + [49, 200, 201, 202, 203]
        assert row["language_only_word_index_count"] == 25

    def test_10_exact_match_status_preserved(self, tmp_path):
        result = _run_step(tmp_path, _engine_with_rows([_ROW_ONE, _ROW_TWO]))
        out = _read_out(tmp_path)
        rows = out["lost_atom_identity_observability"]
        assert rows[0]["identity"]["exact_match_by_clip_id_present"] is False
        assert rows[1]["identity"]["exact_match_by_clip_id_present"] is True
        assert rows[1]["identity"]["exact_match_attempt_ids"] == ["lang_att_3"]
        assert rows[1]["identity"]["exact_match_proposition_candidate_ids"] == ["prop_9"]

    def test_11_provenance_id_preserved(self, tmp_path):
        result = _run_step(tmp_path, _engine_with_rows([_ROW_ONE, _ROW_TWO]))
        out = _read_out(tmp_path)
        rows = out["lost_atom_identity_observability"]
        assert rows[0]["lost_atom_provenance_id"] == "prov_abc123"
        assert rows[1]["lost_atom_provenance_id"] == "prov_def456"
        assert rows[0]["clip_id"] == "clip_800d8d16b6b9e88325b7"
        assert rows[1]["clip_id"] == "clip_second"

    def test_12_multiple_atoms_and_attempts_preserved(self, tmp_path):
        result = _run_step(tmp_path, _engine_with_rows([_ROW_ONE, _ROW_TWO]))
        out = _read_out(tmp_path)
        assert out["lost_atom_identity_observability_row_count"] == 2
        assert len(out["lost_atom_identity_observability"][0]["identity"]["language_attempts"]) == 2
        assert len(out["lost_atom_identity_observability"][1]["identity"]["language_attempts"]) == 1

    def test_13_pure_reprojection_no_recompute(self, tmp_path):
        # A deliberately "wrong-looking" but internally consistent row must
        # pass through byte-identical -- the step never re-derives a
        # relationship/intersection/ownership value of its own.
        row = copy.deepcopy(_ROW_ONE)
        row["identity"]["intersection_count"] = 999
        row["identity"]["relationship_status"] = "SOMETHING_UNEXPECTED"
        result = _run_step(tmp_path, _engine_with_rows([row]))
        out = _read_out(tmp_path)
        got = out["lost_atom_identity_observability"][0]["identity"]
        assert got["intersection_count"] == 999
        assert got["relationship_status"] == "SOMETHING_UNEXPECTED"

    def test_14_empty_list_is_present_not_missing(self, tmp_path):
        # D-237G's own fail-open posture: an empty list is a LEGITIMATE
        # "no rows built" outcome (flag off / no live evidence upstream),
        # never conflated with the key being entirely absent.
        result = _run_step(tmp_path, _engine_with_rows([]))
        assert result.returncode == 0, result.stderr
        out = _read_out(tmp_path)
        assert out["source_status"] == "PRESENT"
        assert out["lost_atom_identity_observability_row_count"] == 0
        assert out["lost_atom_identity_observability"] == []


class TestNoTranscriptLeakage:
    def test_15_no_transcript_text_field_names_in_output(self, tmp_path):
        result = _run_step(tmp_path, _engine_with_rows([_ROW_ONE, _ROW_TWO]))
        blob = (tmp_path / "artifact" / OUT_FILE).read_text()
        for banned in ("transcript", "asr_text", "text_excerpt", "full_transcript"):
            assert banned not in blob.lower()

    def test_16_only_bounded_integer_word_indices_present(self, tmp_path):
        result = _run_step(tmp_path, _engine_with_rows([_ROW_ONE]))
        out = _read_out(tmp_path)
        ct = out["lost_atom_identity_observability"][0]["identity"]["candidate_take"]
        assert all(isinstance(i, int) for i in ct["word_indices"])

    def test_17_does_not_dump_whole_engine_json(self, tmp_path):
        engine = _engine_with_rows([_ROW_ONE])
        engine["some_unrelated_top_level_key_with_secret_looking_value"] = "SHOULD_NOT_APPEAR"
        result = _run_step(tmp_path, engine)
        blob = (tmp_path / "artifact" / OUT_FILE).read_text()
        assert "SHOULD_NOT_APPEAR" not in blob


class TestMissingAndMalformedAreObservationalNotFatal:
    def test_18_key_absent_reports_missing_exit_zero(self, tmp_path):
        engine = {"diagnostics": {"final_story_coherence_validation": {"other_key": 1}}}
        result = _run_step(tmp_path, engine)
        assert result.returncode == 0, result.stderr
        out = _read_out(tmp_path)
        assert out["source_status"] == "MISSING"
        assert out["reason"] == "lost_atom_identity_observability_key_absent"

    def test_19_parent_block_absent_reports_missing_exit_zero(self, tmp_path):
        engine = {"diagnostics": {"other_block": {}}}
        result = _run_step(tmp_path, engine)
        assert result.returncode == 0, result.stderr
        out = _read_out(tmp_path)
        assert out["source_status"] == "MISSING"
        assert out["reason"] == "final_story_coherence_validation_block_absent"

    def test_20_no_diagnostics_key_at_all_reports_missing_exit_zero(self, tmp_path):
        result = _run_step(tmp_path, {"selected": []})
        assert result.returncode == 0, result.stderr
        out = _read_out(tmp_path)
        assert out["source_status"] == "MISSING"

    def test_21_missing_engine_json_reports_missing_exit_zero(self, tmp_path):
        result = _run_step(tmp_path, None)
        assert result.returncode == 0, result.stderr
        out = _read_out(tmp_path)
        assert out["source_status"] == "MISSING"
        assert out["reason"] == "engine_json_missing_or_unparseable"

    def test_22_unparseable_engine_json_reports_missing_exit_zero(self, tmp_path):
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "video00-modal.json").write_text("{not valid json")
        result = subprocess.run(["bash", "-c", _step_script()], cwd=str(tmp_path), capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        out = _read_out(tmp_path)
        assert out["source_status"] == "MISSING"

    def test_23_malformed_not_a_list_fails_observationally_not_behaviorally(self, tmp_path):
        engine = {"diagnostics": {"final_story_coherence_validation": {"lost_atom_identity_observability": {"not": "a list"}}}}
        result = _run_step(tmp_path, engine)
        # "fails observationally, not behaviorally": the extractor DETECTS
        # and REPORTS the malformed shape, but exits 0 -- it never causes
        # the surrounding workflow to fail.
        assert result.returncode == 0, result.stderr
        out = _read_out(tmp_path)
        assert out["source_status"] == "MALFORMED"
        assert out["reason"] == "lost_atom_identity_observability_not_a_list"

    def test_24_malformed_string_value_also_observational(self, tmp_path):
        engine = {"diagnostics": {"final_story_coherence_validation": {"lost_atom_identity_observability": "not-a-list-either"}}}
        result = _run_step(tmp_path, engine)
        assert result.returncode == 0, result.stderr
        out = _read_out(tmp_path)
        assert out["source_status"] == "MALFORMED"


class TestNoRecomputationInExtractor:
    def test_25_no_set_arithmetic_in_step_script(self):
        # The extractor must be pure serialization/extraction -- it must
        # never itself compute an intersection/difference/relationship
        # classification. (The engine module `exact_identity_observability.
        # py` does that; this step only copies its already-computed output.)
        script = _step_script()
        for banned in ("frozenset(", ".intersection(", "AUTHORITATIVE_RELATIONSHIP_STATUSES", "classify_word_membership_relationship"):
            assert banned not in script

    def test_26_no_provider_or_raw_terms_in_step_script(self):
        script = _step_script().lower()
        for banned in ("import modal", "runpod", "requests.post", "requests.get", "boto3.client(\"s3\")", "s3.upload"):
            assert banned not in script


class TestValidatorArtifactInclusion:
    def test_27_artifact_path_registered_in_upload_step(self):
        doc = _load_workflow()
        upload_step = None
        for step in doc["jobs"]["video00-modal-raw"]["steps"]:
            if step.get("name") == "Upload validator reports":
                upload_step = step
        assert upload_step is not None
        assert f"artifact/{OUT_FILE}" in upload_step["with"]["path"]
        # existing sibling files must remain registered too
        assert "artifact/selection-freeze-diagnostics.json" in upload_step["with"]["path"]
        assert "artifact/lost-semantic-atom-diagnostics.json" in upload_step["with"]["path"]

    def test_28_step_runs_regardless_of_freeze_result(self):
        doc = _load_workflow()
        for step in doc["jobs"]["video00-modal-raw"]["steps"]:
            if step.get("name") == STEP_NAME:
                assert step.get("if") == "always()"

    def test_29_step_positioned_after_d235j_step_before_selection_lock(self):
        doc = _load_workflow()
        names = [s.get("name") for s in doc["jobs"]["video00-modal-raw"]["steps"]]
        d235j_idx = names.index("D-235J Lost Semantic Atom diagnostics -> sibling-safe extraction")
        d237i_idx = names.index(STEP_NAME)
        lock_idx = names.index("Verify frozen Selection lock")
        assert d235j_idx < d237i_idx < lock_idx


class TestDeterminism:
    def test_30_deterministic_output_across_runs(self, tmp_path_factory):
        engine = _engine_with_rows([_ROW_ONE, _ROW_TWO])
        p1 = tmp_path_factory.mktemp("run1")
        p2 = tmp_path_factory.mktemp("run2")
        r1 = _run_step(p1, engine)
        r2 = _run_step(p2, engine)
        assert r1.returncode == 0 and r2.returncode == 0
        assert _read_out(p1) == _read_out(p2)


class TestWorkflowYamlValidity:
    def test_31_workflow_yaml_parses(self):
        doc = _load_workflow()
        assert "video00-modal-raw" in doc["jobs"]

    def test_32_step_script_compiles_as_python(self):
        import ast

        script = _step_script()
        start = script.index("<<'PY'") + len("<<'PY'\n")
        end = script.rindex("\nPY")
        code = script[start:end]
        ast.parse(code)  # raises SyntaxError on failure
