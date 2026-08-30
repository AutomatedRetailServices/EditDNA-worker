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


def test_deterministic_best_take_authority_defaults_on():
    env = {**base_env(), "CUTSELL_UNIFIED_SELECTION_REASONER": "1"}
    brain = build_brain_runtime(load_runtime_config(env), env)

    assert brain.deterministic_best_take_authority_enabled is True


def test_deterministic_best_take_authority_rollback_env_flag_disables_it():
    env = {
        **base_env(),
        "CUTSELL_UNIFIED_SELECTION_REASONER": "1",
        "CUTSELL_DETERMINISTIC_BEST_TAKE_AUTHORITY": "0",
    }
    brain = build_brain_runtime(load_runtime_config(env), env)

    assert brain.deterministic_best_take_authority_enabled is False


def test_semantic_equivalence_arbiter_builds_when_hybrid_enabled_without_unified_selection():
    # Phase 2: take grouping runs upstream of the Unified Selection/legacy
    # branch, so the arbiter must be available even when
    # CUTSELL_UNIFIED_SELECTION_REASONER is off -- gated on requested_hybrid
    # alone, not on the whole-video reasoner flag.
    env = base_env()
    brain = build_brain_runtime(load_runtime_config(env), env)

    assert brain.semantic_equivalence_arbiter is not None
    assert brain.selection_reasoner is None


def test_semantic_equivalence_arbiter_ledger_uses_its_own_dedicated_ceiling():
    env = base_env()
    brain = build_brain_runtime(load_runtime_config(env), env)

    settings = brain.semantic_equivalence_arbiter.settings
    ledger = brain.semantic_equivalence_arbiter.ledger
    assert ledger.max_usd == settings.max_cost_per_semantic_equivalence_call_usd
    assert ledger.max_usd != settings.max_cost_per_edit_usd
    assert ledger.max_usd != settings.max_cost_per_unified_selection_call_usd


def test_semantic_equivalence_arbiter_none_when_hybrid_paid_inference_disabled():
    env = {
        "CUTSELL_BRAIN_BACKEND": "runpod_local",
        "CUTSELL_HYBRID_LLM_ENABLED": "0",
    }
    brain = build_brain_runtime(load_runtime_config(env), env)

    assert brain.semantic_equivalence_arbiter is None


def test_semantic_equivalence_arbiter_rollback_env_flag_disables_it():
    env = {**base_env(), "CUTSELL_SEMANTIC_EQUIVALENCE_ARBITER": "0"}
    brain = build_brain_runtime(load_runtime_config(env), env)

    assert brain.semantic_equivalence_arbiter is None
    # Rollback is scoped to this arbiter only -- the legacy judge is untouched.
    assert brain.editorial_judge is not None


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
