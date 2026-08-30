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


def test_unified_selection_reasoner_ledger_uses_its_own_ceiling_not_the_legacy_edit_ceiling():
    # RAW run 33319393884 (head 4c0ccc9): both reasoners' ledgers were built
    # from the SAME settings.max_cost_per_edit_usd -- sized for the legacy
    # per-group Hybrid judge's many small calls, not Unified Selection's one
    # whole-video call. Once the output token reserve was corrected to be
    # non-truncating, a single real Unified Selection call cost more than
    # that shared ceiling and failed open with "budget exhausted" before
    # ever making an HTTP call. This pins the fix: the two ledgers must be
    # sized from two independent settings fields.
    env = {**base_env(), "CUTSELL_UNIFIED_SELECTION_REASONER": "1"}
    brain = build_brain_runtime(load_runtime_config(env), env)

    settings = brain.selection_reasoner.settings
    assert brain.selection_reasoner.ledger.max_usd == settings.max_cost_per_unified_selection_call_usd
    assert brain.selection_reasoner.ledger.max_usd != settings.max_cost_per_edit_usd


def test_legacy_editorial_judge_ledger_still_uses_the_legacy_edit_ceiling():
    env = base_env()
    brain = build_brain_runtime(load_runtime_config(env), env)

    settings = brain.editorial_judge.transport.settings
    assert brain.editorial_judge.transport.ledger.max_usd == settings.max_cost_per_edit_usd


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
