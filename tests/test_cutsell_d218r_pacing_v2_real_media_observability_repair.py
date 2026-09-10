"""D-218R -- Pacing V2 real-media observability repair (workflow-only).

D-218's own finding: the engine already fully serializes `diagnostics
["pacing_v2"]` (D-216/D-217), but the Video00 Modal RAW workflow never
extracted it into a small, always-retrievable artifact -- it was buried
inside the multi-hundred-MB `cutsell-video00-modal-human-review` artifact,
which this environment's own egress policy cannot download. This adds a
new, dedicated workflow step ("D-218R Pacing V2 -> Video00 real-media
diagnostic qualification") that PURE-PROJECTS the already-computed
`diagnostics["pacing_v2"]` block into `artifact/d218_pacing_v2_real_
media_qualification.json` -- no transition, relationship hint, Prosodic
mapping, or candidate timing window is recomputed.

These tests extract the step's actual embedded Python (via `yaml.safe_
load`, exactly how GitHub Actions itself dedents a `run: |` block scalar
before executing it -- never a raw substring slice) and execute it
against synthetic `artifact/video00-modal.json` fixtures, proving:
  - a representative, present `pacing_v2` block produces a complete,
    bounded artifact with every field D-218's own audit requires;
  - a missing block (or missing engine JSON entirely) FAILS LOUDLY
    (non-zero exit, a clear stderr banner) rather than silently emitting
    an empty-looking artifact.

No cutsell_worker file is touched by this task -- workflow YAML text and
this test file only.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW_PATH = Path(".github/workflows/cutsell-video00-modal-raw.yml")
STEP_NAME = "D-218R Pacing V2 -> Video00 real-media diagnostic qualification"


def _load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def _step_script() -> str:
    """Returns the ACTUAL dedented script GitHub Actions would execute for
    this step -- `yaml.safe_load` strips a `run: |` block scalar's own
    common leading indentation exactly the way the real runner does, so
    this is a functional extraction, never a raw-text slice that could
    silently test something syntactically different from what CI runs."""
    doc = _load_workflow()
    for step in doc["jobs"]["video00-modal-raw"]["steps"]:
        if step.get("name") == STEP_NAME:
            return step["run"]
    raise AssertionError(f"step {STEP_NAME!r} not found in {WORKFLOW_PATH}")


def _run_step(tmp_path: Path, engine_json: dict | None) -> subprocess.CompletedProcess:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if engine_json is not None:
        (artifact_dir / "video00-modal.json").write_text(json.dumps(engine_json))
    return subprocess.run(
        ["bash", "-c", _step_script()], cwd=str(tmp_path), capture_output=True, text=True, timeout=30,
    )


def _representative_transition(left: str, right: str, mode: str = "HARD_CUT") -> dict:
    return {
        "left_clip_id": left, "right_clip_id": right,
        "left_source_asset_id": "src", "right_source_asset_id": "src",
        "selected_mode": mode, "live_mode": mode, "live_vs_v2_comparison": "AGREEMENT",
        "pacing_gap_decision": "TIGHTEN", "speech_overlap_status": "NO_OVERLAP",
        "meaning_safety_status": "SAFE", "word_safety_status": "SAFE",
        "double_speech_status": "SAFE_NO_OVERLAP", "decision_status": "SUPPORTED",
        "fallback_reason": None, "relationship_hint": None, "relationship_source": "NONE",
        "relationship_mapping_status": "NO_UNDERSTANDING_SUPPLIED",
        "left_prosodic_status": "UNAVAILABLE", "right_prosodic_status": "UNAVAILABLE",
        "prosodic_mapping_status": "UNAVAILABLE",
        "candidate_audio_lead": None, "candidate_audio_tail": None, "candidate_micro_overlap": None,
        "candidate_timing_status": "UNKNOWN_MISSING_WORD_TIMING",
        "firewall_violation": False, "conflict_flags": [], "provenance": ["pacing_transition_decision_v1"],
    }


def _representative_pacing_v2_block(n_transitions: int = 4) -> dict:
    transitions = [_representative_transition(f"c{i}", f"c{i + 1}") for i in range(n_transitions)]
    return {
        "schema_version": "cutsell.pacing_v2_live_diagnostics_integration.v1",
        "capability_status": "AVAILABLE",
        "missing_evidence": [],
        "transition_count": n_transitions,
        "transitions": transitions,
        "run_summary": {
            "transition_count": n_transitions, "hard_cut_count": n_transitions,
            "tight_cut_count": 0, "j_cut_count": 0, "l_cut_count": 0, "micro_overlap_count": 0,
            "relationship_hint_available_count": 0, "relationship_hint_unknown_count": n_transitions,
            "prosodic_pair_available_count": 0, "prosodic_pair_partial_count": 0,
            "prosodic_pair_unavailable_count": n_transitions,
            "candidate_j_lead_available_count": 0, "candidate_l_tail_available_count": 0,
            "candidate_micro_overlap_available_count": 0,
            "advanced_mode_eligible_count": 0, "j_cut_eligible_count": 0,
            "l_cut_eligible_count": 0, "micro_overlap_eligible_count": 0,
            "comparison_agreement_count": n_transitions,
        },
        "sequence_consistency": {"inconsistent_pair_count": 0, "inconsistent_pairs": []},
        "firewall_violation_count": 0,
        # D-142 comparison aggregates live at the TOP LEVEL of the real
        # D-216 block (sibling to run_summary, never nested inside it).
        "comparison_agreement_count": n_transitions,
        "comparison_more_conservative_count": 0,
        "comparison_advanced_mode_count": 0,
        "comparison_incomparable_count": 0,
    }


def _engine_json(pacing_v2_block, selected_count: int) -> dict:
    diagnostics = {}
    if pacing_v2_block is not None:
        diagnostics["pacing_v2"] = pacing_v2_block
    return {
        "selected": [{"clip_id": f"c{i}"} for i in range(selected_count)],
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# 1. Step exists, is textually inside the workflow, script is valid syntax.
# ---------------------------------------------------------------------------

def test_step_exists_and_is_valid_python():
    script = _step_script()
    assert "diagnostics['pacing_v2']" in script or 'diagnostics.get("pacing_v2")' in script or "'pacing_v2'" in script
    # The heredoc python body must compile once dedented by yaml.safe_load
    # (proven functionally below by actually running it, not just here).
    assert "python3 - <<'PY'" in script


def test_new_artifact_path_registered_for_upload():
    text = WORKFLOW_PATH.read_text()
    idx = text.index("Upload validator reports")
    upload_block = text[idx:idx + 2000]
    assert "artifact/d218_pacing_v2_real_media_qualification.json" in upload_block


# ---------------------------------------------------------------------------
# 2. Present block -> complete, bounded artifact with every required field.
# ---------------------------------------------------------------------------

def test_representative_block_produces_complete_artifact(tmp_path):
    block = _representative_pacing_v2_block(n_transitions=4)
    engine = _engine_json(block, selected_count=5)
    proc = _run_step(tmp_path, engine)
    assert proc.returncode == 0, proc.stderr

    out = json.loads((tmp_path / "artifact" / "d218_pacing_v2_real_media_qualification.json").read_text())
    assert out["pacing_v2_block_status"] == "PRESENT"
    assert out["capability_status"] == "AVAILABLE"
    assert out["transition_count"] == 4
    assert out["selected_clip_count"] == 5
    assert out["transition_count_matches_selected_minus_one"] is True
    assert len(out["transitions"]) == 4
    # Every per-pair field D-218's own audit requires is present.
    row = out["transitions"][0]
    for field in (
        "left_clip_id", "right_clip_id", "relationship_hint", "relationship_source",
        "relationship_mapping_status", "left_prosodic_status", "right_prosodic_status",
        "prosodic_mapping_status", "candidate_audio_lead", "candidate_audio_tail",
        "candidate_micro_overlap", "candidate_timing_status", "selected_mode", "live_mode",
        "live_vs_v2_comparison", "pacing_gap_decision", "speech_overlap_status",
        "meaning_safety_status", "word_safety_status", "double_speech_status",
        "decision_status", "fallback_reason", "firewall_violation",
    ):
        assert field in row, f"missing required per-pair field: {field}"
    assert "run_summary" in out and out["run_summary"]["transition_count"] == 4
    assert "sequence_consistency" in out
    assert out["firewall_violation_count"] == 0
    assert out["advanced_recommendation_count"] == 0
    assert out["advanced_execution_count"] == 0  # structural fact, always 0
    assert out["live_independent_audio_window_count"] == 0  # structural fact, always 0
    assert out["current_d142_live_modes"] == ["HARD_CUT"] * 4
    assert out["selected_d215_modes"] == ["HARD_CUT"] * 4
    assert out["pacing_v2_usefulness_classification"] in (
        "USEFUL_AND_BOUNDED", "USEFUL_BUT_LIMITED", "SAFE_BUT_NOT_YET_USEFUL", "MATERIALLY_WRONG",
    )
    # D-142 comparison aggregates (top-level on the real block, never
    # nested inside run_summary) must be pure-projected, not dropped.
    assert out["d142_comparison"] == {
        "comparison_agreement_count": 4,
        "comparison_more_conservative_count": 0,
        "comparison_advanced_mode_count": 0,
        "comparison_incomparable_count": 0,
    }


def test_d142_comparison_survives_when_block_lacks_it(tmp_path):
    # A block that (for any reason) never populated the comparison_*
    # aggregates must still produce a complete artifact -- None, never a
    # KeyError or a silently dropped key.
    block = _representative_pacing_v2_block(n_transitions=2)
    for key in (
        "comparison_agreement_count", "comparison_more_conservative_count",
        "comparison_advanced_mode_count", "comparison_incomparable_count",
    ):
        block.pop(key, None)
    engine = _engine_json(block, selected_count=3)
    proc = _run_step(tmp_path, engine)
    assert proc.returncode == 0, proc.stderr
    out = json.loads((tmp_path / "artifact" / "d218_pacing_v2_real_media_qualification.json").read_text())
    assert out["d142_comparison"] == {
        "comparison_agreement_count": None,
        "comparison_more_conservative_count": None,
        "comparison_advanced_mode_count": None,
        "comparison_incomparable_count": None,
    }


def test_does_not_dump_unrelated_full_result_json(tmp_path):
    block = _representative_pacing_v2_block(n_transitions=2)
    engine = _engine_json(block, selected_count=3)
    engine["unrelated_huge_field"] = "x" * 500_000
    proc = _run_step(tmp_path, engine)
    assert proc.returncode == 0, proc.stderr
    out_path = tmp_path / "artifact" / "d218_pacing_v2_real_media_qualification.json"
    assert out_path.stat().st_size < 100_000  # bounded, never the unrelated huge field
    out = json.loads(out_path.read_text())
    assert "unrelated_huge_field" not in json.dumps(out)


def test_advanced_mode_transition_surfaces_correctly(tmp_path):
    block = _representative_pacing_v2_block(n_transitions=1)
    block["transitions"][0]["selected_mode"] = "J_CUT"
    block["transitions"][0]["candidate_audio_lead"] = 0.3
    block["run_summary"]["j_cut_count"] = 1
    block["run_summary"]["hard_cut_count"] = 0
    block["run_summary"]["advanced_mode_eligible_count"] = 1
    block["run_summary"]["j_cut_eligible_count"] = 1
    engine = _engine_json(block, selected_count=2)
    proc = _run_step(tmp_path, engine)
    assert proc.returncode == 0, proc.stderr
    out = json.loads((tmp_path / "artifact" / "d218_pacing_v2_real_media_qualification.json").read_text())
    assert out["advanced_recommendation_count"] == 1
    assert out["advanced_execution_count"] == 0  # eligible != executed, always
    assert out["pacing_v2_usefulness_classification"] == "USEFUL_AND_BOUNDED"


# ---------------------------------------------------------------------------
# 3. Missing pacing_v2 -> FAILS LOUDLY, never a silent empty artifact.
# ---------------------------------------------------------------------------

def test_missing_pacing_v2_block_fails_loudly_not_silently(tmp_path):
    engine = _engine_json(None, selected_count=2)  # diagnostics present, pacing_v2 absent
    proc = _run_step(tmp_path, engine)
    assert proc.returncode != 0
    assert "MISSING FROM THE SERIALIZED RESULT" in proc.stderr
    out_path = tmp_path / "artifact" / "d218_pacing_v2_real_media_qualification.json"
    assert out_path.exists()  # honest marker, never nothing at all
    out = json.loads(out_path.read_text())
    assert out["pacing_v2_block_status"] == "MISSING_FROM_SERIALIZATION"
    assert "transitions" not in out  # never a fabricated empty-looking success shape


def test_missing_engine_json_entirely_fails_loudly(tmp_path):
    proc = _run_step(tmp_path, engine_json=None)  # no video00-modal.json written at all
    assert proc.returncode != 0
    assert "missing or unparseable" in proc.stderr
    out_path = tmp_path / "artifact" / "d218_pacing_v2_real_media_qualification.json"
    assert out_path.exists()
    out = json.loads(out_path.read_text())
    assert out["source_status"] == "engine_json_missing_or_unparseable"


def test_unparseable_engine_json_fails_loudly(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "video00-modal.json").write_text("{not valid json")
    proc = subprocess.run(["bash", "-c", _step_script()], cwd=str(tmp_path), capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0


# ---------------------------------------------------------------------------
# 4. Determinism / idempotence.
# ---------------------------------------------------------------------------

def test_deterministic_repeat(tmp_path):
    block = _representative_pacing_v2_block(n_transitions=3)
    engine = _engine_json(block, selected_count=4)
    proc1 = _run_step(tmp_path, engine)
    out1 = (tmp_path / "artifact" / "d218_pacing_v2_real_media_qualification.json").read_text()
    proc2 = _run_step(tmp_path, engine)
    out2 = (tmp_path / "artifact" / "d218_pacing_v2_real_media_qualification.json").read_text()
    assert proc1.returncode == proc2.returncode == 0
    assert out1 == out2


# ---------------------------------------------------------------------------
# 5. No cutsell_worker reference, no re-derivation, no provider.
# ---------------------------------------------------------------------------

def test_no_cutsell_worker_import_in_step():
    # Comments legitimately name "cutsell_worker" in prose (e.g. "never fed
    # back into cutsell_worker"); the real property is no actual import.
    script = _step_script()
    assert "import cutsell_worker" not in script
    assert "from cutsell_worker" not in script
    assert "from .cutsell_worker" not in script


def test_no_recomputation_only_projection():
    script = _step_script()
    for banned in ("decide_transition", "build_pacing_v2_live_diagnostics", "analyze_prosodic_delivery"):
        assert banned not in script


def test_no_provider_or_network_call():
    script = _step_script()
    for banned in ("requests.", "urllib", "openai", "genai."):
        assert banned not in script.lower()


def test_full_workflow_yaml_still_parses():
    doc = _load_workflow()
    assert "video00-modal-raw" in doc["jobs"]
