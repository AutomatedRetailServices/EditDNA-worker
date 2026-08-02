import logging
import sys
import types

import pytest

from worker.models.config import ModelConfigurationError, load_model_config
from worker.models.openai_client import OpenAIProviderError, OpenAIResponseValidationError, create_openai_client
from worker.models import openai_provider as provider


class Response:
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


def install_response(monkeypatch, content):
    completion = types.SimpleNamespace(create=lambda **kwargs: Response(content))
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completion))
    monkeypatch.setattr(provider, "create_openai_client", lambda: client)
    return client


def test_client_factory_is_lazy_and_applies_transport_settings(monkeypatch):
    calls = []
    module = types.ModuleType("openai")
    module.OpenAI = lambda **kwargs: calls.append(kwargs) or object()
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "2")
    assert calls == []
    create_openai_client()
    assert calls == [{"api_key": "test-key", "timeout": 12.5, "max_retries": 2}]


@pytest.mark.parametrize("override,expected", [(None, 2), ("0", 0), ("5", 5)])
def test_client_factory_preserves_and_overrides_sdk_retry_behavior(monkeypatch, override, expected):
    calls = []
    module = types.ModuleType("openai")
    module.OpenAI = lambda **kwargs: calls.append(kwargs) or object()
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    if override is None:
        monkeypatch.delenv("OPENAI_MAX_RETRIES", raising=False)
    else:
        monkeypatch.setenv("OPENAI_MAX_RETRIES", override)

    create_openai_client()

    assert calls[0]["max_retries"] == expected


def test_missing_key_is_safe_and_only_fails_when_factory_called(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    secret = "secret-fixture"
    with pytest.raises(OpenAIProviderError) as caught:
        create_openai_client()
    assert secret not in str(caught.value)


@pytest.mark.parametrize("name,value", [("OPENAI_TIMEOUT_SECONDS", "0"), ("OPENAI_MAX_RETRIES", "-1"), ("OPENAI_TIMEOUT_SECONDS", "secret-fixture")])
def test_transport_config_validation_is_safe(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ModelConfigurationError) as caught:
        load_model_config()
    assert value not in str(caught.value)


def test_semantic_validation(monkeypatch):
    install_response(monkeypatch, '{"clips":[{"id":"a","slot":"HOOK","keep":true,"semantic_score":0.8,"reason":"short"},{"id":"unknown","slot":"CTA","keep":true,"semantic_score":1,"reason":"x"}]}')
    result = provider.classify_semantic("model", [], ["a"])
    assert result["a"].semantic_score == 0.8 and "unknown" not in result


def test_one_invalid_known_semantic_item_abstains_entire_response(monkeypatch):
    install_response(monkeypatch, '{"clips":[{"id":"a","slot":"HOOK","keep":true,"semantic_score":0.8,"reason":"short"},{"id":"b","slot":"UNKNOWN","keep":true,"semantic_score":0.7,"reason":"bad"}]}')
    with pytest.raises(OpenAIResponseValidationError):
        provider.classify_semantic("model", [], ["a", "b"])


@pytest.mark.parametrize("content", [
    '{"clips":[{"id":"a","slot":"OTHER","keep":true,"semantic_score":0.5,"reason":"x"}]}',
    '{"clips":[{"id":"a","slot":"HOOK","keep":"yes","semantic_score":0.5,"reason":"x"}]}',
    '{"clips":[{"id":"a","slot":"HOOK","keep":true,"semantic_score":2,"reason":"x"}]}',
    "not json",
])
def test_invalid_semantic_output_abstains(monkeypatch, content):
    install_response(monkeypatch, content)
    with pytest.raises(OpenAIResponseValidationError):
        provider.classify_semantic("model", [], ["a"])


def test_take_judge_validates_membership_and_scores(monkeypatch):
    install_response(monkeypatch, '{"winner_id":"a","scores":[{"id":"a","score":0.9},{"id":"b","score":0.2}]}')
    assert provider.judge_takes("model", [], ["a", "b"]).winner_id == "a"


@pytest.mark.parametrize("content", [
    '{"winner_id":"unknown","scores":[]}',
    '{"winner_id":"a","scores":[{"id":"unknown","score":0.2}]}',
    '{"winner_id":"a","scores":[{"id":"a","score":1.2}]}',
    "malformed",
])
def test_invalid_take_judge_abstains(monkeypatch, content):
    install_response(monkeypatch, content)
    with pytest.raises(OpenAIResponseValidationError):
        provider.judge_takes("model", [], ["a", "b"])


def test_boundary_requires_exact_complete_labels(monkeypatch):
    install_response(monkeypatch, '{"frames":[{"label":"head","verdict":"GOOD"},{"label":"mid","verdict":"BAD"},{"label":"tail","verdict":"GOOD"}]}')
    assert provider.refine_boundaries("model", []).MID is provider.Verdict.BAD


@pytest.mark.parametrize("content", [
    '{"frames":[{"label":"head","verdict":"GOOD"},{"label":"mid","verdict":"BAD"}]}',
    '{"frames":[{"label":"head","verdict":"MAYBE"},{"label":"mid","verdict":"BAD"},{"label":"tail","verdict":"GOOD"}]}',
    "bad-json",
])
def test_invalid_boundary_abstains(monkeypatch, content):
    install_response(monkeypatch, content)
    with pytest.raises(OpenAIResponseValidationError):
        provider.refine_boundaries("model", [])


@pytest.mark.parametrize("raw,expected", [("GOOD", provider.Verdict.GOOD), ("BAD", provider.Verdict.BAD), ("  good\n", provider.Verdict.GOOD)])
def test_bad_take_trivial_normalization(monkeypatch, raw, expected):
    install_response(monkeypatch, raw)
    assert provider.detect_bad_take("model", []) is expected


def test_bad_take_rejects_explanation(monkeypatch):
    install_response(monkeypatch, "BAD because the actor looked away")
    with pytest.raises(OpenAIResponseValidationError):
        provider.detect_bad_take("model", [])


def test_provider_errors_and_logs_do_not_leak_content(monkeypatch, caplog):
    transcript, image, key, raw = "TRANSCRIPT_FIXTURE", "BASE64_FIXTURE", "sk-fixture", "RAW_RESPONSE_FIXTURE"
    monkeypatch.setattr(provider, "create_openai_client", lambda: (_ for _ in ()).throw(RuntimeError(raw + transcript + image + key)))
    with caplog.at_level(logging.WARNING), pytest.raises(OpenAIProviderError) as caught:
        provider.detect_bad_take("safe-model", [{"content": transcript + image}])
    combined = caplog.text + str(caught.value)
    assert all(value not in combined for value in (transcript, image, key, raw))
