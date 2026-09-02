"""D-044 fix: regression coverage proving cutsell-video00-modal-raw.yml
applies the same canonical CUTSELL_HYBRID_LLM_ENABLED / CUTSELL_HYBRID_PROVIDER
overlay the canonical RunPod Serverless RAW workflow applies before every
live dispatch -- the exact gap that left
diagnostics.semantic_idea_equivalence.status == "not_requested" for the
entire D-043 live benchmark (run 33636255124), because
brain_runtime.build_brain_runtime only constructs the bounded
semantic-equivalence arbiter when CUTSELL_HYBRID_LLM_ENABLED is truthy.

Scope, per the D-044 fix directive: modify only the Modal RAW workflow and
these tests. No cutsell_worker editorial module is touched -- these tests
assert on workflow YAML text (and, for the "base template cannot silently
disable this" claim, actually invoke the real `jq` filter the workflow
runs, so this is a functional check of the merge semantics, not just a
string match) plus a guard that no editorial module references this fix at
all.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

MODAL_WORKFLOW_PATH = Path(".github/workflows/cutsell-video00-modal-raw.yml")
RUNPOD_WORKFLOW_PATH = Path(".github/workflows/cutsell-video00-raw-v5-auto-microtrim.yml")


def modal_workflow_text() -> str:
    return MODAL_WORKFLOW_PATH.read_text()


def runpod_workflow_text() -> str:
    return RUNPOD_WORKFLOW_PATH.read_text()


def _runpod_overlay_value(key: str) -> str:
    """Extracts the literal value RunPod's own workflow overlays for `key`
    from its `env:(($base.env // {}) + {...})` jq object -- the canonical
    source of truth this fix must mirror, not reinvent."""
    text = runpod_workflow_text()
    match = re.search(rf'{re.escape(key)}\s*:\s*"([^"]+)"', text)
    assert match, f"{key} overlay not found in the canonical RunPod workflow -- source of truth missing"
    return match.group(1)


def _modal_env_secret_step_text() -> str:
    text = modal_workflow_text()
    idx = text.index("Build masked Modal env-secret file")
    return text[idx:text.index("- name:", idx + 1)]


def test_runpod_workflow_itself_still_overlays_both_canonical_vars():
    # Sanity-checks the source of truth this fix mirrors hasn't silently
    # changed shape underneath this test.
    assert _runpod_overlay_value("CUTSELL_HYBRID_LLM_ENABLED") == "1"
    assert _runpod_overlay_value("CUTSELL_HYBRID_PROVIDER") == "google"


def test_modal_workflow_forces_cutsell_hybrid_llm_enabled():
    step_text = _modal_env_secret_step_text()
    assert 'CUTSELL_HYBRID_LLM_ENABLED:"1"' in step_text


def test_modal_workflow_forces_cutsell_hybrid_provider_google():
    step_text = _modal_env_secret_step_text()
    assert 'CUTSELL_HYBRID_PROVIDER:"google"' in step_text


def test_modal_overlay_values_stay_in_parity_with_the_runpod_workflows_own_values():
    # If the canonical RunPod workflow's own overlay value ever changes,
    # this test forces the Modal workflow to be updated in lockstep rather
    # than silently drifting back out of parity.
    step_text = _modal_env_secret_step_text()
    hybrid_enabled = _runpod_overlay_value("CUTSELL_HYBRID_LLM_ENABLED")
    hybrid_provider = _runpod_overlay_value("CUTSELL_HYBRID_PROVIDER")
    assert f'CUTSELL_HYBRID_LLM_ENABLED:"{hybrid_enabled}"' in step_text
    assert f'CUTSELL_HYBRID_PROVIDER:"{hybrid_provider}"' in step_text


def _extract_jq_filter(step_text: str) -> str:
    match = re.search(r"jq -c '([^']+)'", step_text)
    assert match, "jq merge filter not found in the Modal env-secret step"
    return match.group(1)


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed in this environment")
def test_base_template_values_cannot_silently_disable_the_canonical_overrides():
    # Functional proof, not just a string match: runs the REAL jq filter the
    # workflow executes against a synthetic base template that explicitly
    # sets both vars to values that would disable the semantic-equivalence
    # arbiter (matching the D-043 live failure's actual root cause), and
    # asserts the overlay wins regardless -- `+` in jq's object-merge
    # syntax takes the RIGHT-hand operand on key conflicts, which is why
    # the overlay must be the second (right-hand) operand of the `+`, not
    # the first.
    step_text = _modal_env_secret_step_text()
    jq_filter = _extract_jq_filter(step_text)
    hostile_base_template = {
        "env": {
            "CUTSELL_HYBRID_LLM_ENABLED": "0",
            "CUTSELL_HYBRID_PROVIDER": "openai",
            "S3_BUCKET": "some-bucket",
        }
    }
    proc = subprocess.run(
        ["jq", "-c", jq_filter],
        input=json.dumps(hostile_base_template),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    merged = json.loads(proc.stdout)
    assert merged["CUTSELL_HYBRID_LLM_ENABLED"] == "1"
    assert merged["CUTSELL_HYBRID_PROVIDER"] == "google"
    # Every other base-template key must survive the merge unmodified --
    # this overlay must never behave like a full env replacement.
    assert merged["S3_BUCKET"] == "some-bucket"


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed in this environment")
def test_base_template_missing_the_two_keys_entirely_still_gets_them_forced():
    step_text = _modal_env_secret_step_text()
    jq_filter = _extract_jq_filter(step_text)
    base_template_without_hybrid_keys = {"env": {"S3_BUCKET": "some-bucket"}}
    proc = subprocess.run(
        ["jq", "-c", jq_filter],
        input=json.dumps(base_template_without_hybrid_keys),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    merged = json.loads(proc.stdout)
    assert merged["CUTSELL_HYBRID_LLM_ENABLED"] == "1"
    assert merged["CUTSELL_HYBRID_PROVIDER"] == "google"


def test_the_two_overlay_values_are_never_masked_in_ci_logs():
    # D-043's own live report was corrupted by blanket-masking every
    # template env value; this fix's own values must stay visible in the
    # CI log (and any future parity/diagnostic printing) so a reviewer can
    # actually see they took effect, not just trust a comment.
    step_text = _modal_env_secret_step_text()
    assert "key not in ('CUTSELL_HYBRID_LLM_ENABLED', 'CUTSELL_HYBRID_PROVIDER')" in step_text


def test_no_cutsell_worker_editorial_module_references_this_workflow_fix():
    # This is an infrastructure/workflow-config fix only -- no editorial
    # module (IdeaClusterer, BestTake, D-038/039/040 claim-coverage code,
    # StoryValidator) should ever need to know this overlay exists.
    forbidden = ("D-044", "cutsell-video00-modal-raw.yml", "CUTSELL_HYBRID_LLM_ENABLED\":\"1\"")
    for path in Path("cutsell_worker").rglob("*.py"):
        text = path.read_text()
        for needle in forbidden:
            assert needle not in text, f"{needle!r} leaked into production code: {path}"


def test_modal_workflow_does_not_touch_cutsell_unified_selection_reasoner():
    # Explicitly out of scope for this fix (dormant/rollback-only in the
    # active Clean Cut Core V1 path; universal_clean_cut.py never reaches
    # the selection_reasoner branch while clean_cut_core_v1_enabled is
    # True, its own default) -- "do not invent additional env changes."
    step_text = _modal_env_secret_step_text()
    assert "CUTSELL_UNIFIED_SELECTION_REASONER" not in step_text


def test_modal_workflow_does_not_touch_asr_model_or_brain_backend_or_editorial_mode():
    # These three RunPod overlay keys are out of scope for this fix: two
    # are already confirmed at parity from the D-043 live run's own
    # diagnostics (brain_backend/editorial_mode both already read
    # "runpod_local"/"clean_cut" without any override), and the third
    # (CUTSELL_ASR_MODEL) defaults to "medium" in cutsell_worker's own code
    # either way. "Do not invent additional env changes."
    step_text = _modal_env_secret_step_text()
    assert "CUTSELL_BRAIN_BACKEND" not in step_text
    assert "CUTSELL_EDITORIAL_MODE" not in step_text
    assert "CUTSELL_ASR_MODEL" not in step_text
