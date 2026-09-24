from pathlib import Path

import pytest
import yaml

from benchmarks import run_video00_gpt_whisperx_stability as batch


def test_sixth_and_zero_trials_cannot_claim_paid_work(tmp_path):
    for index in (0, -1, 6, 100):
        with pytest.raises(ValueError):
            batch.claim_trial(tmp_path, index)
    assert not list(tmp_path.iterdir())


def test_claim_is_consumed_even_without_a_successful_result(tmp_path):
    batch.claim_trial(tmp_path, 1)
    with pytest.raises(FileExistsError):
        batch.claim_trial(tmp_path, 1)
    with pytest.raises(RuntimeError, match="sequentially"):
        batch.claim_trial(tmp_path, 2)


def test_all_five_must_run_sequentially_and_only_once(tmp_path):
    for index in range(1, 6):
        batch.claim_trial(tmp_path, index)
        (tmp_path / f"trial-{index}.terminal").touch()
    assert len(list(tmp_path.glob("*.claimed"))) == 5
    for index in range(1, 6):
        with pytest.raises(FileExistsError):
            batch.claim_trial(tmp_path, index)


def test_uncertain_terminal_state_blocks_next_dispatch(tmp_path):
    (tmp_path / "trial-1.terminal").touch()
    (tmp_path / "STOP").touch()
    with pytest.raises(RuntimeError, match="uncertain terminal state"):
        batch.claim_trial(tmp_path, 2)


def test_changed_configuration_fails_before_claim_or_dispatch(tmp_path, monkeypatch):
    private = tmp_path / "cutsell-five-private"
    private.mkdir()
    batch.write_json(private / "environment.json", {"OPENAI_API_KEY": "ordinary-test-secret"})
    batch.write_json(private / "lock.json", {"config_sha256": "different"})
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    with pytest.raises(RuntimeError, match="Configuration changed"):
        batch.run_trial(1)
    assert not list(private.glob("*.claimed"))


def test_workflow_never_reruns_or_runs_trials_in_parallel():
    path = batch.ROOT / ".github/workflows/cutsell-gpt-whisperx-five.yml"
    workflow = yaml.safe_load(path.read_text())
    assert list(workflow["jobs"]) == ["five-trials"]
    job = workflow["jobs"]["five-trials"]
    assert "github.run_attempt == 1" in job["if"]
    assert "strategy" not in job
    calls = [s for s in job["steps"] if "stability.py trial " in s.get("run", "")]
    assert [s["run"].split()[-1] for s in calls] == ["1", "2", "3", "4", "5"]
    assert all(s["continue-on-error"] for s in calls)
    artifacts = [s["with"]["path"] for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact")]
    assert all("private" not in path for path in artifacts)


def test_final_report_never_hides_failed_or_missing_trials(tmp_path, monkeypatch):
    output = tmp_path / "stability-artifacts"
    (output / "trial-1").mkdir(parents=True)
    batch.write_json(output / "trial-1/summary.json", {"trial": 1, "status": "failed"})
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    assert batch.finalize() == 1
    import json
    result = json.loads((output / "five-trial-summary.json").read_text())
    assert len(result["trials"]) == 5
    assert result["trials"][0]["status"] == "failed"
    assert all(r["status"] == "not_run" for r in result["trials"][1:])
    assert result["all_engine_runs_completed"] is False
