"""D-235G workflow extraction -- Selection Freeze diagnostics, sibling-safe
(workflow-only).

Mirrors the D-218R precedent exactly (`test_cutsell_d218r_pacing_v2_real_
media_observability_repair.py`): extracts the new "D-235G Selection Freeze
diagnostics -> sibling-safe extraction" step's ACTUAL embedded Python via
`yaml.safe_load` (the same dedent GitHub Actions itself performs on a
`run: |` block scalar, never a raw-text slice) and executes it against
synthetic `artifact/video00-modal.json` fixtures -- proving a present
`selection_freeze_diagnostics` block produces a complete, bounded,
sibling-safe artifact, and a missing block (or missing engine JSON
entirely) FAILS LOUDLY rather than silently emitting an empty-looking
artifact. No paid compute; no cutsell_worker file touched by this test.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

WORKFLOW_PATH = Path(".github/workflows/cutsell-video00-modal-raw.yml")
STEP_NAME = "D-235G Selection Freeze diagnostics -> sibling-safe extraction"


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


_SIBLING_FREEZE_BLOCKED = {
    "schema_version": "cutsell.selection_freeze_diagnostics.v1",
    "freeze_blocked": True,
    "trigger_count": 0,
    "trigger_categories": [],
    "coherence_status": None,
    "coherence_contradiction_status": "UNKNOWN",
    "idea_loss_status": "UNKNOWN",
    "lost_semantic_atom_status": "UNKNOWN",
    "lost_critical_claim_status": "UNKNOWN",
    "authority_membership_finding_status": "UNKNOWN",
    "coherence_integrity_failure_status": "UNKNOWN",
    "repair_loop_status": None,
    "resolver_status": None,
    "post_authority_integrity_status": "UNKNOWN",
    "post_authority_integrity_failure_codes": [],
    "selected_count_before_freeze": 5,
    "pacing_seam_reached": False,
    "pacing_v2_serialized": None,
    "pacing_v2_handle_aware_serialized": None,
    "audio_join_treatment_v2_serialized": None,
    "first_missing_link": "FREEZE_BLOCKED_BEFORE_PACING",
    "provenance": ["cutsell.selection_freeze_diagnostics.v1", "build_selection_freeze_diagnostics"],
}


class TestStepFound:
    def test_01_step_exists_in_workflow(self):
        assert _step_script()


class TestPresentBlock:
    def test_02_freeze_blocked_sibling_extracted(self, tmp_path):
        engine = {"selected": [{"clip_id": f"c{i}"} for i in range(5)], "diagnostics": {"selection_freeze_diagnostics": _SIBLING_FREEZE_BLOCKED}}
        result = _run_step(tmp_path, engine)
        assert result.returncode == 0, result.stderr
        out = json.loads((tmp_path / "artifact" / "selection-freeze-diagnostics.json").read_text())
        assert out["selection_freeze_diagnostics_block_status"] == "PRESENT"
        assert out["selection_freeze_diagnostics"]["freeze_blocked"] is True
        assert out["selection_freeze_diagnostics"]["first_missing_link"] == "FREEZE_BLOCKED_BEFORE_PACING"
        assert out["selected_clip_count_top_level"] == 5

    def test_03_non_blocked_block_extracted(self, tmp_path):
        block = dict(_SIBLING_FREEZE_BLOCKED)
        block.update({
            "freeze_blocked": False, "pacing_seam_reached": True,
            "pacing_v2_serialized": True, "pacing_v2_handle_aware_serialized": True,
            "audio_join_treatment_v2_serialized": True, "first_missing_link": "PACING_DIAGNOSTIC_SERIALIZED",
        })
        engine = {"selected": [{"clip_id": "c1"}, {"clip_id": "c2"}], "diagnostics": {"selection_freeze_diagnostics": block}}
        result = _run_step(tmp_path, engine)
        assert result.returncode == 0, result.stderr
        out = json.loads((tmp_path / "artifact" / "selection-freeze-diagnostics.json").read_text())
        assert out["selection_freeze_diagnostics"]["pacing_seam_reached"] is True

    def test_04_no_video00_specific_fields_in_output(self, tmp_path):
        engine = {"selected": [], "diagnostics": {"selection_freeze_diagnostics": _SIBLING_FREEZE_BLOCKED}}
        result = _run_step(tmp_path, engine)
        blob = (tmp_path / "artifact" / "selection-freeze-diagnostics.json").read_text()
        assert "expected_selected_count" not in blob
        assert "thyroid" not in blob.lower()

    def test_05_pure_reprojection_no_recompute(self, tmp_path):
        # Feed a deliberately "wrong-looking" but internally consistent
        # block and confirm it passes through byte-identical -- the step
        # must never re-derive freeze_blocked or any sub-status.
        block = dict(_SIBLING_FREEZE_BLOCKED)
        block["trigger_count"] = 99  # nonsensical, but must pass through verbatim
        engine = {"selected": [], "diagnostics": {"selection_freeze_diagnostics": block}}
        result = _run_step(tmp_path, engine)
        out = json.loads((tmp_path / "artifact" / "selection-freeze-diagnostics.json").read_text())
        assert out["selection_freeze_diagnostics"]["trigger_count"] == 99


class TestMissingBlockFailsLoudly:
    def test_06_missing_block_fails_loudly(self, tmp_path):
        engine = {"selected": [], "diagnostics": {}}
        result = _run_step(tmp_path, engine)
        assert result.returncode != 0
        assert "MISSING FROM THE SERIALIZED RESULT" in result.stderr
        out = json.loads((tmp_path / "artifact" / "selection-freeze-diagnostics.json").read_text())
        assert out["selection_freeze_diagnostics_block_status"] == "MISSING_FROM_SERIALIZATION"

    def test_07_missing_engine_json_fails_loudly(self, tmp_path):
        result = _run_step(tmp_path, None)
        assert result.returncode != 0
        out = json.loads((tmp_path / "artifact" / "selection-freeze-diagnostics.json").read_text())
        assert out["source_status"] == "engine_json_missing_or_unparseable"

    def test_08_unparseable_engine_json_fails_loudly(self, tmp_path):
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "video00-modal.json").write_text("{not valid json")
        result = subprocess.run(["bash", "-c", _step_script()], cwd=str(tmp_path), capture_output=True, text=True, timeout=30)
        assert result.returncode != 0

    def test_09_artifact_path_registered_in_upload_step(self):
        doc = _load_workflow()
        upload_step = None
        for step in doc["jobs"]["video00-modal-raw"]["steps"]:
            if step.get("name") == "Upload validator reports":
                upload_step = step
        assert upload_step is not None
        assert "artifact/selection-freeze-diagnostics.json" in upload_step["with"]["path"]

    def test_10_step_runs_regardless_of_freeze_result(self):
        doc = _load_workflow()
        for step in doc["jobs"]["video00-modal-raw"]["steps"]:
            if step.get("name") == STEP_NAME:
                assert step.get("if") == "always()"
