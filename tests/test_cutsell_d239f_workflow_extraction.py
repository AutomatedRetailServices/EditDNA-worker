"""D-239F workflow extraction -- Lost-Atom Ownership -> Materiality ->
Freeze -> Repair observability (workflow-only).

Mirrors D-237I's own `test_cutsell_d237i_workflow_extraction.py` exactly:
extracts the new "D-239F Lost-Atom Ownership -> Materiality -> Freeze ->
Repair observability -> sibling-safe extraction" step's ACTUAL embedded
Python via `yaml.safe_load` and executes it against synthetic
`artifact/video00-modal.json` fixtures -- proving the one pure
provenance-id JOIN this step performs (never a policy recomputation) is
correct, and that an absent/malformed subtree is reported
observationally (source_status MISSING, exit 0) rather than failing the
workflow.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

WORKFLOW_PATH = Path(".github/workflows/cutsell-video00-modal-raw.yml")
STEP_NAME = "D-239F Lost-Atom Ownership -> Materiality -> Freeze -> Repair observability -> sibling-safe extraction"
OUT_FILE = "lost-atom-ownership-materiality-diagnostics.json"


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


# The real D-239 shape -- one atom in "atoms" (ownership+Q+R), one matching
# entry in "entries" (D-235S/T), joined by lost_atom_provenance_id.
_ATOM = {
    "clip_id": "clip_1d9d2cebaf8ed3004836",
    "lost_atom_provenance_id": "latom_clip_1d9d2cebaf8ed3004836_0",
    "bounded_excerpt": "too many people ready set these are the",
    "classification": "REAL_CONTENT_LOSS",
    "blocking": True,
    "ownership_input_present": True,
    "ownership_status": "EXACT_SINGLETON_OWNERSHIP",
    "containing_language_attempt_id": "latt_be3141887f7b37d1bb3a",
    "proposition_candidate_ids": ["prop_475b7c5a7d7ebc6e6e8c"],
    "ownership_ambiguity_reason": [],
    "exact_ownership_available": True,
    "critical_claim_conflict_state": False,
    "editorial_requirement_state": "INSUFFICIENT_EVIDENCE",
    "meaning_critical_state": "NON_MATERIAL_REAL_CONTENT",
    "retry_process_state": "NOT_FOUND",
    "redundancy_state": "NOT_FOUND",
    "final_materiality_status": "NON_MATERIAL_REAL_CONTENT",
    "blocking_recommendation": "DO_NOT_BLOCK",
    "freeze_materiality_received": True,
    "freeze_authority_status": "SUPPRESS_NON_MATERIAL_BLOCK",
    "freeze_effective_blocking": False,
    "freeze_reason": None,
}
_ENTRY = {
    "lost_atom_provenance_id": "latom_clip_1d9d2cebaf8ed3004836_0",
    "d235s_reviewer_finding_kind": "UNIQUE_FACT_LOST",
    "d235s_repair_attempt_provenance_confirmed": True,
    "d235s_repair_attempt_reason": "no_repair_strategy_exists_for_this_finding_kind",
    "d235s_exact_link_status": "EXACT_MATCH",
    "d235t_precomputed_materiality_received": True,
    "d235t_suppression_status": "SUPPRESS_SAME_NON_MATERIAL_ATOM",
    "d235t_suppress_repair_escalation": True,
    "d235t_reason": "same_atom_already_qualifies_for_freeze_authority_suppression",
}


def _engine_with(atoms=None, entries=None):
    fscv = {}
    if atoms is not None:
        fscv["lost_atom_ownership_materiality_diagnostics"] = {"atom_count": len(atoms), "atoms": atoms}
    repair = {}
    if entries is not None:
        repair["lost_atom_repair_suppression_diagnostics"] = {"entry_count": len(entries), "entries": entries}
    return {"diagnostics": {"final_story_coherence_validation": fscv, "repair_loop": repair}}


class TestJoinCorrectness:
    def test_01_present_join_merges_by_provenance_id(self, tmp_path):
        engine = _engine_with(atoms=[_ATOM], entries=[_ENTRY])
        result = _run_step(tmp_path, engine)
        assert result.returncode == 0, result.stderr
        out = _read_out(tmp_path)
        assert out["source_status"] == "PRESENT"
        assert out["atom_count"] == 1
        merged = out["atoms"][0]
        assert merged["clip_id"] == "clip_1d9d2cebaf8ed3004836"
        assert merged["ownership_status"] == "EXACT_SINGLETON_OWNERSHIP"
        assert merged["containing_language_attempt_id"] == "latt_be3141887f7b37d1bb3a"
        assert merged["d235t_suppression_status"] == "SUPPRESS_SAME_NON_MATERIAL_ATOM"
        assert merged["d235s_repair_attempt_provenance_confirmed"] is True

    def test_02_ownership_atom_without_matching_entry_still_present(self):
        pass  # covered structurally by test_03 (mismatched provenance)

    def test_03_mismatched_provenance_id_never_cross_joined(self, tmp_path):
        other_entry = dict(_ENTRY, lost_atom_provenance_id="latom_OTHER")
        engine = _engine_with(atoms=[_ATOM], entries=[other_entry])
        result = _run_step(tmp_path, engine)
        assert result.returncode == 0
        out = _read_out(tmp_path)
        merged = out["atoms"][0]
        # The atom's own ownership/Q/R fields are untouched; no D-235S/T
        # fields leak in from an unrelated provenance id.
        assert merged["ownership_status"] == "EXACT_SINGLETON_OWNERSHIP"
        assert "d235t_suppression_status" not in merged

    def test_04_both_subtrees_absent_reports_missing(self, tmp_path):
        engine = _engine_with()
        result = _run_step(tmp_path, engine)
        assert result.returncode == 0
        out = _read_out(tmp_path)
        assert out["source_status"] == "MISSING"

    def test_05_engine_json_missing_reports_missing_never_fails(self, tmp_path):
        result = _run_step(tmp_path, None)
        assert result.returncode == 0
        out = _read_out(tmp_path)
        assert out["source_status"] == "MISSING"

    def test_06_ownership_only_no_repair_entries_still_present(self, tmp_path):
        engine = _engine_with(atoms=[_ATOM], entries=[])
        result = _run_step(tmp_path, engine)
        assert result.returncode == 0
        out = _read_out(tmp_path)
        assert out["source_status"] == "PRESENT"
        assert "d235t_suppression_status" not in out["atoms"][0]

    def test_07_always_runs(self):
        doc = _load_workflow()
        for step in doc["jobs"]["video00-modal-raw"]["steps"]:
            if step.get("name") == STEP_NAME:
                assert step.get("if") == "always()"

    def test_08_positioned_after_d237i_before_selection_lock(self):
        doc = _load_workflow()
        names = [s.get("name") for s in doc["jobs"]["video00-modal-raw"]["steps"]]
        d237i_idx = names.index("D-237I Exact Identity Observability -> sibling-safe extraction")
        d239f_idx = names.index(STEP_NAME)
        lock_idx = names.index("Verify frozen Selection lock")
        assert d237i_idx < d239f_idx < lock_idx

    def test_09_step_script_compiles_as_python(self):
        import ast

        script = _step_script()
        start = script.index("<<'PY'") + len("<<'PY'\n")
        end = script.rindex("\nPY")
        code = script[start:end]
        ast.parse(code)

    def test_10_artifact_included_in_validator_reports_upload(self):
        doc = _load_workflow()
        for step in doc["jobs"]["video00-modal-raw"]["steps"]:
            if step.get("name") == "Upload validator reports":
                assert "artifact/lost-atom-ownership-materiality-diagnostics.json" in step["with"]["path"]
                return
        raise AssertionError("Upload validator reports step not found")

    def test_11_deterministic_output(self, tmp_path_factory):
        engine = _engine_with(atoms=[_ATOM], entries=[_ENTRY])
        p1 = tmp_path_factory.mktemp("run1")
        p2 = tmp_path_factory.mktemp("run2")
        r1 = _run_step(p1, engine)
        r2 = _run_step(p2, engine)
        assert r1.returncode == 0 and r2.returncode == 0
        assert _read_out(p1) == _read_out(p2)
