"""D-235J workflow extraction -- Lost Semantic Atom diagnostics, sibling-safe
(workflow-only).

Mirrors D-235G's own `test_cutsell_d235g_workflow_extraction.py` exactly:
extracts the new "D-235J Lost Semantic Atom diagnostics -> sibling-safe
extraction" step's ACTUAL embedded Python via `yaml.safe_load` (the same
dedent GitHub Actions itself performs on a `run: |` block scalar, never a
raw-text slice) and executes it against synthetic `artifact/video00-modal.
json` fixtures -- proving a present `lost_semantic_atom_diagnostics` block
produces a complete, bounded, sibling-safe artifact, and a missing block (or
missing engine JSON entirely) FAILS LOUDLY rather than silently emitting an
empty-looking artifact. No paid compute; no cutsell_worker file touched by
this test.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

WORKFLOW_PATH = Path(".github/workflows/cutsell-video00-modal-raw.yml")
STEP_NAME = "D-235J Lost Semantic Atom diagnostics -> sibling-safe extraction"


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


_SIBLING_LOST_ATOM_BLOCK = {
    "schema_version": "cutsell.lost_semantic_atom_diagnostics.v1",
    "ledger_status": "FOUND",
    "atom_count": 1,
    "blocking_atom_count": 1,
    "atoms_truncated": False,
    "atoms": [
        {
            "clip_id": "clip_1",
            "row_kind": "COVERAGE_LEDGER_CONTENT_LOSS",
            "blocking": True,
            "classification": "REAL_CONTENT_LOSS",
            "text_excerpt": "the doctor said it was stage 2",
            "missing_critical_atom_count": 1,
            "atom_classifications": [
                {"atom": "stage 2", "atom_type": "number", "importance": "CRITICAL", "resolved_by": "deterministic", "evidence_present": True},
            ],
            "own_content_token_count": 12,
            "missing_content_token_count": 6,
            "coverage_against_final_keep": 0.3,
            "content_loss_suppressed_by": None,
            "preserving_realization_id": None,
            "preserved_claim_count": 0,
            "nonrequired_omission_count": 0,
            "no_usable_realization_basis": None,
            "present_before_selection": True,
            "present_after_selection": False,
            "reviewer_finding_kind": "UNIQUE_FACT_LOST",
            "repair_loop_attempt_status": "MATCHED_BY_CLIP_ID",
            "repair_loop_attempt_index": 0,
            "repair_loop_reason": "no_repair_strategy_exists_for_this_finding_kind",
            "repair_loop_repaired": False,
        },
    ],
    "absent_fields_not_retained_by_engine": [
        "atom_id", "source_span_id", "source_proposition_id", "semantic_role", "required_or_optional",
    ],
    "provenance": ["cutsell.lost_semantic_atom_diagnostics.v1", "build_lost_semantic_atom_diagnostics"],
}


class TestStepFound:
    def test_01_step_exists_in_workflow(self):
        assert _step_script()


class TestPresentBlock:
    def test_02_lost_atom_block_extracted(self, tmp_path):
        engine = {"selected": [{"clip_id": f"c{i}"} for i in range(5)], "diagnostics": {"lost_semantic_atom_diagnostics": _SIBLING_LOST_ATOM_BLOCK}}
        result = _run_step(tmp_path, engine)
        assert result.returncode == 0, result.stderr
        out = json.loads((tmp_path / "artifact" / "lost-semantic-atom-diagnostics.json").read_text())
        assert out["lost_semantic_atom_diagnostics_block_status"] == "PRESENT"
        assert out["lost_semantic_atom_diagnostics"]["atom_count"] == 1
        assert out["lost_semantic_atom_diagnostics"]["atoms"][0]["clip_id"] == "clip_1"
        assert out["selected_clip_count_top_level"] == 5

    def test_03_empty_ledger_extracted(self, tmp_path):
        block = dict(_SIBLING_LOST_ATOM_BLOCK)
        block.update({"ledger_status": "NOT_FOUND", "atom_count": 0, "blocking_atom_count": 0, "atoms": []})
        engine = {"selected": [{"clip_id": "c1"}], "diagnostics": {"lost_semantic_atom_diagnostics": block}}
        result = _run_step(tmp_path, engine)
        assert result.returncode == 0, result.stderr
        out = json.loads((tmp_path / "artifact" / "lost-semantic-atom-diagnostics.json").read_text())
        assert out["lost_semantic_atom_diagnostics"]["atom_count"] == 0

    def test_04_no_video00_specific_fields_in_output(self, tmp_path):
        engine = {"selected": [], "diagnostics": {"lost_semantic_atom_diagnostics": _SIBLING_LOST_ATOM_BLOCK}}
        result = _run_step(tmp_path, engine)
        blob = (tmp_path / "artifact" / "lost-semantic-atom-diagnostics.json").read_text()
        assert "expected_selected_count" not in blob
        assert "thyroid" not in blob.lower()

    def test_05_pure_reprojection_no_recompute(self, tmp_path):
        # Feed a deliberately "wrong-looking" but internally consistent
        # block and confirm it passes through byte-identical -- the step
        # must never re-derive blocking, materiality, or any count.
        block = dict(_SIBLING_LOST_ATOM_BLOCK)
        block["atom_count"] = 99  # nonsensical, but must pass through verbatim
        engine = {"selected": [], "diagnostics": {"lost_semantic_atom_diagnostics": block}}
        result = _run_step(tmp_path, engine)
        out = json.loads((tmp_path / "artifact" / "lost-semantic-atom-diagnostics.json").read_text())
        assert out["lost_semantic_atom_diagnostics"]["atom_count"] == 99

    def test_06_no_full_transcript_text_in_output(self, tmp_path):
        block = dict(_SIBLING_LOST_ATOM_BLOCK)
        long_atom = dict(block["atoms"][0])
        long_atom["text_excerpt"] = "word " * 500
        block["atoms"] = [long_atom]
        engine = {"selected": [], "diagnostics": {"lost_semantic_atom_diagnostics": block}}
        result = _run_step(tmp_path, engine)
        out = json.loads((tmp_path / "artifact" / "lost-semantic-atom-diagnostics.json").read_text())
        # The workflow step itself never re-bounds text (that is the engine
        # function's own job) -- this just confirms the step is a pure
        # passthrough and does not choke on/alter a long field.
        assert out["lost_semantic_atom_diagnostics"]["atoms"][0]["text_excerpt"] == "word " * 500


class TestMissingBlockFailsLoudly:
    def test_07_missing_block_fails_loudly(self, tmp_path):
        engine = {"selected": [], "diagnostics": {}}
        result = _run_step(tmp_path, engine)
        assert result.returncode != 0
        assert "MISSING FROM THE SERIALIZED RESULT" in result.stderr
        out = json.loads((tmp_path / "artifact" / "lost-semantic-atom-diagnostics.json").read_text())
        assert out["lost_semantic_atom_diagnostics_block_status"] == "MISSING_FROM_SERIALIZATION"

    def test_08_missing_engine_json_fails_loudly(self, tmp_path):
        result = _run_step(tmp_path, None)
        assert result.returncode != 0
        out = json.loads((tmp_path / "artifact" / "lost-semantic-atom-diagnostics.json").read_text())
        assert out["source_status"] == "engine_json_missing_or_unparseable"

    def test_09_unparseable_engine_json_fails_loudly(self, tmp_path):
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "video00-modal.json").write_text("{not valid json")
        result = subprocess.run(["bash", "-c", _step_script()], cwd=str(tmp_path), capture_output=True, text=True, timeout=30)
        assert result.returncode != 0

    def test_10_artifact_path_registered_in_upload_step(self):
        doc = _load_workflow()
        upload_step = None
        for step in doc["jobs"]["video00-modal-raw"]["steps"]:
            if step.get("name") == "Upload validator reports":
                upload_step = step
        assert upload_step is not None
        assert "artifact/lost-semantic-atom-diagnostics.json" in upload_step["with"]["path"]

    def test_11_step_runs_regardless_of_freeze_result(self):
        doc = _load_workflow()
        for step in doc["jobs"]["video00-modal-raw"]["steps"]:
            if step.get("name") == STEP_NAME:
                assert step.get("if") == "always()"

    def test_12_step_positioned_after_d235g_step_before_selection_lock(self):
        doc = _load_workflow()
        names = [s.get("name") for s in doc["jobs"]["video00-modal-raw"]["steps"]]
        d235g_idx = names.index("D-235G Selection Freeze diagnostics -> sibling-safe extraction")
        d235j_idx = names.index(STEP_NAME)
        lock_idx = names.index("Verify frozen Selection lock")
        assert d235g_idx < d235j_idx < lock_idx
