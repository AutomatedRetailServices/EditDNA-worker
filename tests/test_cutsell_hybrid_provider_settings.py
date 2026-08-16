from cutsell_worker.hybrid_provider_settings import (
    HybridProviderSettings,
    load_hybrid_provider_settings,
)


def test_hybrid_provider_is_disabled_by_default():
    settings = load_hybrid_provider_settings({})
    assert settings.enabled is False
    assert settings.provider == "google"
    assert settings.primary_model == "gemini-3.5-flash-lite"
    assert settings.escalation_model == "gemini-3.6-flash"
    assert settings.max_test_budget_usd == 0.50
    assert settings.max_cost_per_edit_usd == 0.0075


def test_key_or_provider_name_alone_cannot_enable_paid_calls():
    settings = load_hybrid_provider_settings({
        "GEMINI_API_KEY": "present-but-not-sufficient",
        "CUTSELL_HYBRID_PROVIDER": "google",
    })
    assert settings.enabled is False


def test_explicit_enable_requires_supported_provider():
    settings = load_hybrid_provider_settings({
        "CUTSELL_HYBRID_LLM_ENABLED": "1",
        "CUTSELL_HYBRID_PROVIDER": "unknown-vendor",
    })
    assert settings.enabled is False
    assert settings.provider == "none"


def test_explicit_google_enable_is_possible_but_still_only_configuration():
    settings = load_hybrid_provider_settings({
        "CUTSELL_HYBRID_LLM_ENABLED": "1",
        "CUTSELL_HYBRID_PROVIDER": "google",
    })
    assert settings.enabled is True


def test_estimated_cost_stays_below_two_cents_for_compact_primary_and_escalation_calls():
    settings = HybridProviderSettings()
    primary = settings.estimate_cost_usd(input_tokens=4000, output_tokens=500)
    escalation = settings.estimate_cost_usd(input_tokens=4000, output_tokens=500, escalation=True)
    assert primary < 0.003
    assert escalation < 0.01
    assert settings.allows_estimated_session_cost(primary)
    assert settings.allows_estimated_session_cost(escalation)


def test_escalation_is_exceptional_after_bakeoff():
    settings = HybridProviderSettings()
    assert settings.should_escalate(local_confidence=0.30, conflict_score=0.10)
    assert settings.should_escalate(local_confidence=0.80, conflict_score=0.90)
    assert not settings.should_escalate(local_confidence=0.50, conflict_score=0.70)


def test_budget_caps_can_only_be_tightened_to_non_negative_values():
    settings = load_hybrid_provider_settings({
        "CUTSELL_HYBRID_MAX_SESSION_USD": "-1",
        "CUTSELL_HYBRID_MAX_EDIT_USD": "-2",
        "CUTSELL_HYBRID_TEST_BUDGET_USD": "-5",
        "CUTSELL_HYBRID_DAILY_BUDGET_USD": "-9",
    })
    assert settings.max_cost_per_session_usd == 0.0
    assert settings.max_cost_per_edit_usd == 0.0
    assert settings.max_test_budget_usd == 0.0
    assert settings.max_daily_budget_usd == 0.0
