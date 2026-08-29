import pytest

from cutsell_worker.brain_runtime import build_brain_runtime
from cutsell_worker.config import load_runtime_config


def base_env():
    return {
        "CUTSELL_BRAIN_BACKEND": "runpod_local",
        "CUTSELL_HYBRID_LLM_ENABLED": "1",
        "CUTSELL_HYBRID_PROVIDER": "google",
        "GEMINI_API_KEY": "test-key-not-used-during-construction",
    }


def test_unified_selection_disables_legacy_bounded_editorial_judge():
    env = {**base_env(), "CUTSELL_UNIFIED_SELECTION_REASONER": "1"}
    brain = build_brain_runtime(load_runtime_config(env), env)

    assert brain.selection_reasoner is not None
    assert brain.editorial_judge is None
    assert brain.external_calls_enabled is True


def test_legacy_hybrid_remains_available_when_unified_flag_is_off():
    env = base_env()
    brain = build_brain_runtime(load_runtime_config(env), env)

    assert brain.selection_reasoner is None
    assert brain.editorial_judge is not None
    assert brain.external_calls_enabled is True


def test_unified_selection_requires_explicit_hybrid_paid_gate():
    env = {
        "CUTSELL_BRAIN_BACKEND": "runpod_local",
        "CUTSELL_UNIFIED_SELECTION_REASONER": "1",
        "CUTSELL_HYBRID_LLM_ENABLED": "0",
        "CUTSELL_HYBRID_PROVIDER": "google",
        "GEMINI_API_KEY": "test-key",
    }

    with pytest.raises(RuntimeError, match="requires CUTSELL_HYBRID_LLM_ENABLED=1"):
        build_brain_runtime(load_runtime_config(env), env)
