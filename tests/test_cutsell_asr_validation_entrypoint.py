"""The real RAW entry point must honor the ASR decode policy selector."""
from types import SimpleNamespace

from cutsell_worker.universal_clean_cut_validation import _validation_asr
from cutsell_worker.asr import DEFAULT_TEMPERATURE_LADDER, DETERMINISTIC_TEMPERATURE


def test_validation_asr_preserves_legacy_configuration_when_flag_off():
    provider = _validation_asr(
        SimpleNamespace(asr_model="medium"),
        env={"CUTSELL_ASR_DETERMINISTIC_CONFIG": "0"},
    )
    assert provider.model_name == "medium"
    assert provider.temperature_ladder == DEFAULT_TEMPERATURE_LADDER
    assert provider.sampling_fallback_enabled


def test_validation_asr_honors_deterministic_flag_and_selected_model():
    provider = _validation_asr(
        SimpleNamespace(asr_model="large-v3"),
        env={"CUTSELL_ASR_DETERMINISTIC_CONFIG": "1"},
    )
    assert provider.model_name == "large-v3"
    assert provider.temperature_ladder == DETERMINISTIC_TEMPERATURE
    assert not provider.sampling_fallback_enabled
