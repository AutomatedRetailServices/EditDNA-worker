"""Yaskira01 Modal RAW preflight -- the "Pin Yaskira01 Medium WhisperX and
Gemini 3.5 audiovisual review" step must emit an env the worker's own AV
provider accepts.

Run 36587713782 dispatched a paid L4 container that died immediately with
``ValueError: AV budget and conservative multimodal prices must be
explicitly configured`` because the step set the per-edit budget but not
the two per-million token prices ``whole_video_av`` requires. This test
executes the step's ACTUAL embedded Python (extracted via ``yaml.safe_load``,
same dedent GitHub Actions performs) against a synthetic env file, then
feeds the produced env to the real ``build_av_provider`` so the exact
runtime validation runs on the free CPU path. No paid compute.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from cutsell_worker.hybrid_provider_settings import load_hybrid_provider_settings
from cutsell_worker.whole_video_av import GeminiWholeVideoAVProvider, build_av_provider

WORKFLOW_PATH = Path(".github/workflows/cutsell-video00-modal-raw.yml")
STEP_NAME = "Pin Yaskira01 Medium WhisperX and Gemini 3.5 audiovisual review"
AV_KEYS = (
    "CUTSELL_WATCH_LISTEN_AV_MAX_EDIT_USD",
    "CUTSELL_WATCH_LISTEN_AV_INPUT_USD_PER_MILLION",
    "CUTSELL_WATCH_LISTEN_AV_OUTPUT_USD_PER_MILLION",
)


def _step_script() -> str:
    doc = yaml.safe_load(WORKFLOW_PATH.read_text())
    for step in doc["jobs"]["video00-modal-raw"]["steps"]:
        if step.get("name") == STEP_NAME:
            return step["run"]
    raise AssertionError(f"step {STEP_NAME!r} not found in {WORKFLOW_PATH}")


def _run_step(tmp_path: Path, env_json: dict, *, source_key: str = "Yaskira/01.mp4") -> tuple[subprocess.CompletedProcess, Path]:
    # The step hardcodes /tmp/cutsell-env.json; redirect it into tmp_path so
    # the test never touches the real path.
    script = _step_script().replace("/tmp/cutsell-env.json", str(tmp_path / "cutsell-env.json"))
    env_file = tmp_path / "cutsell-env.json"
    env_file.write_text(json.dumps(env_json))
    proc = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "SOURCE_KEY": source_key},
        capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
    )
    return proc, env_file


def _base_env() -> dict:
    return {
        "CUTSELL_VALIDATION_ASR_PROVIDER": "faster-whisper-medium-whisperx",
        "GEMINI_API_KEY": "test-key-not-real",
    }


def test_step_emits_env_the_worker_av_provider_accepts(tmp_path):
    proc, env_file = _run_step(tmp_path, _base_env())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    env = json.loads(env_file.read_text())
    for key in AV_KEYS:
        assert float(env[key]) > 0, key
    # The exact runtime check that killed run 36587713782 must now pass.
    provider = build_av_provider(load_hybrid_provider_settings(env), env)
    assert isinstance(provider, GeminiWholeVideoAVProvider)
    assert provider.model == "gemini-3.5-flash-lite"
    assert provider.ledger.max_usd == pytest.approx(0.10)
    assert provider.input_usd_per_million == pytest.approx(0.30)
    assert provider.output_usd_per_million == pytest.approx(2.50)
    assert provider.retry_generation_timeout is True
    assert env["CUTSELL_ASR_MODEL"] == "medium"
    assert env["CUTSELL_VALIDATION_ASR_PROVIDER"] == "faster-whisper-medium-whisperx"


def test_step_script_names_every_av_key_the_worker_requires():
    script = _step_script()
    for key in AV_KEYS:
        assert key in script, f"{key} missing from the step -> worker raises before any editing"


@pytest.mark.parametrize("missing", AV_KEYS)
def test_worker_rejects_env_missing_any_av_price(missing):
    """Documents the failure mode: dropping any one of the three keys is
    exactly the ValueError observed in run 36587713782."""
    env = {**_base_env(), "CUTSELL_HYBRID_LLM_ENABLED": "1", "CUTSELL_HYBRID_PROVIDER": "google",
           "CUTSELL_HYBRID_PRIMARY_MODEL": "gemini-3.5-flash-lite", "CUTSELL_WATCH_LISTEN_AV_ENABLED": "1",
           "CUTSELL_WATCH_LISTEN_AV_MAX_EDIT_USD": "0.10",
           "CUTSELL_WATCH_LISTEN_AV_INPUT_USD_PER_MILLION": "0.30",
           "CUTSELL_WATCH_LISTEN_AV_OUTPUT_USD_PER_MILLION": "2.50"}
    env.pop(missing)
    with pytest.raises(ValueError, match="explicitly configured"):
        build_av_provider(load_hybrid_provider_settings(env), env)


def test_step_refuses_wrong_provider_before_paid_dispatch(tmp_path):
    proc, _ = _run_step(tmp_path, {**_base_env(), "CUTSELL_VALIDATION_ASR_PROVIDER": "faster-whisper"})
    assert proc.returncode != 0
    assert "Medium + WhisperX" in proc.stderr + proc.stdout


def test_step_is_a_no_op_for_other_sources(tmp_path):
    proc, env_file = _run_step(tmp_path, _base_env(), source_key="Yaskira/02.mp4")
    assert proc.returncode == 0
    assert json.loads(env_file.read_text()) == _base_env()
