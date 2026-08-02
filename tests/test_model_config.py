import importlib.util
import sys
import types

import pytest

from worker.models import (
    ASRProvider,
    ModelConfigurationError,
    SemanticClassifierProvider,
    TakeJudgeProvider,
    VisionProvider,
    load_model_config,
)


ACTIVE_ENV = {
    "ASR_ENABLED", "WHISPER_MODEL_NAME", "WHISPER_MODEL", "WHISPER_DEVICE", "ASR_DEVICE",
    "EDITDNA_USE_LLM", "EDITDNA_LLM_MODEL", "VISION_ENABLED", "VISION_INTERVAL_SEC",
    "VISION_MAX_SAMPLES", "W_VISION", "BAD_TAKES_ENABLED", "BOUNDARY_REFINER_ENABLED",
    "BOUNDARY_REFINER_MIN_DURATION_SEC", "BOUNDARY_REFINER_HEAD_STEP_SEC",
    "BOUNDARY_REFINER_TAIL_STEP_SEC", "TAKE_JUDGE_ENABLED", "TAKE_JUDGE_MODEL",
    "TAKE_JUDGE_MAX_GROUPS", "TAKE_JUDGE_MAX_TAKES", "TAKE_JUDGE_FRAMES", "TAKE_JUDGE_V2_MIN_CONFIDENCE",
    "COMPOSER_MIN_SEMANTIC", "COMPOSER_MAX_PER_SLOT", "OPENAI_API_KEY",
    "SEMANTIC_V2_MIN_CONFIDENCE",
}


@pytest.fixture
def clean_model_env(monkeypatch):
    for name in ACTIVE_ENV:
        monkeypatch.delenv(name, raising=False)


def test_every_model_default(clean_model_env):
    config = load_model_config()
    assert (config.asr.enabled, config.asr.model_name, config.asr.device) == (True, "medium", "auto")
    assert (config.semantic_llm.enabled, config.semantic_llm.model_name) == (False, "gpt-5.1")
    assert config.semantic_llm.min_confidence == 0.70
    assert (config.vision.enabled, config.vision.model_name) == (False, "ViT-B/32")
    assert (config.vision.interval_seconds, config.vision.max_samples, config.vision.weight) == (2.0, 50, 0.7)
    assert (config.visual_bad_take.enabled, config.visual_bad_take.model_name) == (False, "gpt-4o")
    assert (config.boundary_refiner.enabled, config.boundary_refiner.model_name) == (False, "gpt-5.1")
    assert (config.boundary_refiner.min_duration_seconds, config.boundary_refiner.head_step_seconds, config.boundary_refiner.tail_step_seconds, config.boundary_refiner.frame_count) == (3.0, 1.5, 1.5, 3)
    assert (config.take_judge.enabled, config.take_judge.model_name) == (False, "gpt-4o-mini")
    assert (config.take_judge.max_groups, config.take_judge.max_takes, config.take_judge.frame_count) == (6, 3, 1)
    assert (config.global_composer.enabled, config.global_composer.model_name, config.global_composer.min_semantic_score, config.global_composer.max_per_slot) == (False, None, 0.75, 7)
    assert all(item.retry_count == 0 and item.timeout_seconds is None for item in (
        config.asr, config.semantic_llm, config.vision, config.visual_bad_take,
        config.boundary_refiner, config.take_judge, config.global_composer,
    ))


def test_every_active_environment_override(clean_model_env, monkeypatch):
    values = {
        "ASR_ENABLED": "0", "WHISPER_MODEL_NAME": "large", "WHISPER_DEVICE": "cuda",
        "EDITDNA_USE_LLM": "1", "EDITDNA_LLM_MODEL": "semantic-override",
        "VISION_ENABLED": "true", "VISION_INTERVAL_SEC": "4.5", "VISION_MAX_SAMPLES": "12", "W_VISION": "0.25",
        "BAD_TAKES_ENABLED": "yes", "BOUNDARY_REFINER_ENABLED": "on",
        "BOUNDARY_REFINER_MIN_DURATION_SEC": "4", "BOUNDARY_REFINER_HEAD_STEP_SEC": "0.5", "BOUNDARY_REFINER_TAIL_STEP_SEC": "0.75",
        "TAKE_JUDGE_ENABLED": "1", "TAKE_JUDGE_MODEL": "judge-override", "TAKE_JUDGE_MAX_GROUPS": "8", "TAKE_JUDGE_MAX_TAKES": "4", "TAKE_JUDGE_FRAMES": "2",
        "COMPOSER_MIN_SEMANTIC": "0.9", "COMPOSER_MAX_PER_SLOT": "9",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    config = load_model_config()
    assert (config.asr.enabled, config.asr.model_name, config.asr.device) == (False, "large", "cuda")
    assert (config.semantic_llm.enabled, config.semantic_llm.model_name) == (True, "semantic-override")
    assert (config.vision.enabled, config.vision.interval_seconds, config.vision.max_samples, config.vision.weight) == (True, 4.5, 12, 0.25)
    assert config.visual_bad_take.enabled is True
    assert (config.boundary_refiner.enabled, config.boundary_refiner.min_duration_seconds, config.boundary_refiner.head_step_seconds, config.boundary_refiner.tail_step_seconds) == (True, 4.0, 0.5, 0.75)
    assert (config.take_judge.enabled, config.take_judge.model_name, config.take_judge.max_groups, config.take_judge.max_takes, config.take_judge.frame_count) == (True, "judge-override", 8, 4, 2)
    assert (config.global_composer.min_semantic_score, config.global_composer.max_per_slot) == (0.9, 9)


def test_asr_model_and_device_precedence(clean_model_env, monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "fallback-model")
    monkeypatch.setenv("WHISPER_MODEL_NAME", "preferred-model")
    monkeypatch.setenv("ASR_DEVICE", "fallback-device")
    monkeypatch.setenv("WHISPER_DEVICE", "preferred-device")
    assert load_model_config().asr.model_name == "preferred-model"
    assert load_model_config().asr.device == "preferred-device"
    monkeypatch.delenv("WHISPER_MODEL_NAME")
    monkeypatch.delenv("WHISPER_DEVICE")
    assert load_model_config().asr.model_name == "fallback-model"
    assert load_model_config().asr.device == "fallback-device"


@pytest.mark.parametrize("value, expected", [("1", True), ("true", True), ("yes", True), ("on", True), ("0", False), ("false", False), ("no", False), ("off", False)])
def test_boolean_parsing(clean_model_env, monkeypatch, value, expected):
    monkeypatch.setenv("VISION_ENABLED", value)
    assert load_model_config().vision.enabled is expected


@pytest.mark.parametrize("name,value", [("VISION_MAX_SAMPLES", "many"), ("TAKE_JUDGE_MAX_GROUPS", "2.5")])
def test_invalid_integer(clean_model_env, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ModelConfigurationError, match=name):
        load_model_config()


def test_invalid_float(clean_model_env, monkeypatch):
    monkeypatch.setenv("W_VISION", "heavy")
    with pytest.raises(ModelConfigurationError, match="W_VISION"):
        load_model_config()


def test_invalid_boolean_does_not_echo_value(clean_model_env, monkeypatch):
    secretish_value = "not-a-bool-secret-value"
    monkeypatch.setenv("TAKE_JUDGE_ENABLED", secretish_value)
    with pytest.raises(ModelConfigurationError) as caught:
        load_model_config()
    assert secretish_value not in str(caught.value)


def test_disabled_openai_features_need_no_api_key(clean_model_env):
    config = load_model_config()
    assert not any((config.semantic_llm.enabled, config.visual_bad_take.enabled, config.boundary_refiner.enabled, config.take_judge.enabled))


def test_configuration_reloads_after_environment_change(clean_model_env, monkeypatch):
    assert load_model_config().vision.max_samples == 50
    monkeypatch.setenv("VISION_MAX_SAMPLES", "17")
    assert load_model_config().vision.max_samples == 17


def test_legacy_variables_do_not_change_active_behavior(clean_model_env, monkeypatch):
    baseline = load_model_config()
    for name in ("EDITDNA_TAKE_JUDGE_ENABLED", "TAKEJUDGE_MODEL_URI", "TAKEJUDGE_MIN_CONF", "ASR_MODEL_SIZE", "ASR_LANG"):
        monkeypatch.setenv(name, "legacy-value")
    assert load_model_config() == baseline


def test_provider_protocols_import_without_heavy_ml_dependencies():
    assert all(protocol.__module__ == "worker.models.providers" for protocol in (
        ASRProvider, SemanticClassifierProvider, VisionProvider, TakeJudgeProvider,
    ))


def test_current_pipeline_compatibility_constants(clean_model_env):
    for name, attrs in (("requests", {}), ("boto3", {}), ("clip", {}), ("faster_whisper", {"WhisperModel": object})):
        if name not in sys.modules and importlib.util.find_spec(name) is None:
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            sys.modules[name] = module
    sys.modules.pop("worker.pipeline", None)
    from worker import pipeline
    assert (pipeline.WHISPER_MODEL_NAME, pipeline.WHISPER_DEVICE, pipeline.ASR_ENABLED) == ("medium", "auto", True)
    assert (pipeline.EDITDNA_USE_LLM, pipeline.EDITDNA_LLM_MODEL) == (False, "gpt-5.1")
    assert (pipeline.VISION_ENABLED, pipeline.VISION_INTERVAL_SEC, pipeline.VISION_MAX_SAMPLES, pipeline.W_VISION) == (False, 2.0, 50, 0.7)
    assert (pipeline.VISION_MODEL, pipeline.BAD_TAKES_MODEL, pipeline.BOUNDARY_REFINER_MODEL) == ("ViT-B/32", "gpt-4o", "gpt-5.1")
    assert (pipeline.BAD_TAKES_ENABLED, pipeline.BOUNDARY_REFINER_ENABLED, pipeline.TAKE_JUDGE_ENABLED) == (False, False, False)
    assert (pipeline.TAKE_JUDGE_MODEL, pipeline.TAKE_JUDGE_MAX_GROUPS, pipeline.TAKE_JUDGE_MAX_TAKES, pipeline.TAKE_JUDGE_FRAMES) == ("gpt-4o-mini", 6, 3, 1)
