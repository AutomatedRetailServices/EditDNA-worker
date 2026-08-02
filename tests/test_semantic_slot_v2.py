import json
import logging
import types
from dataclasses import FrozenInstanceError

import pytest

from worker.models import openai_provider as provider
from worker.models.openai_client import OpenAIProviderError, OpenAIResponseValidationError
from worker.semantic_slot_v2 import (
    CanonicalSlot, EvidenceTag, SlotClassificationResult, build_clause_inputs, derive_signals,
)


def response_for(**overrides):
    item = {
        "id": "a", "primary_slot": "FEATURES", "secondary_slot": "BENEFIT",
        "confidence": .9, "secondary_confidence": .7, "completeness": .9,
        "sales_relevance": .95, "standalone_quality": .8, "abstain": False,
        "reason": "Feature creates an outcome", "evidence_tags": ["product_attribute", "practical_benefit"],
    }
    item.update(overrides)
    return json.dumps({"results": [item]})


def install(monkeypatch, content):
    reply = types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: reply)))
    monkeypatch.setattr(provider, "create_openai_client", lambda: client)


def clause(text="It has a ceramic plate so your hair stays smoother."):
    return build_clause_inputs([{"id": "a", "text": text, "start": 1, "end": 4, "slot": "FEATURES", "semantic_score": .6, "meta": {}}])[0]


def test_taxonomy_and_evidence_tags_are_exact_and_immutable():
    assert {x.value for x in CanonicalSlot} == {"HOOK", "PROBLEM", "BENEFIT", "FEATURES", "PROOF", "STORY", "CTA", "OTHER"}
    assert "direct_action" in {x.value for x in EvidenceTag}
    value = SlotClassificationResult(CanonicalSlot.HOOK, None, .8, None, 1, 1, 1, False, "x", ())
    with pytest.raises(FrozenInstanceError):
        value.confidence = .1


@pytest.mark.parametrize("overrides", [
    {"primary_slot": "UNKNOWN"},
    {"secondary_slot": "FEATURES"},
    {"secondary_confidence": .95},
    {"evidence_tags": ["provider_invented"]},
    {"reason": "x" * 161},
    {"confidence": 2},
])
def test_strict_provider_validation(monkeypatch, overrides):
    install(monkeypatch, response_for(**overrides))
    with pytest.raises(OpenAIResponseValidationError):
        provider.classify_semantic_v2("gpt-5.1", [clause()])


def test_valid_primary_secondary_result(monkeypatch):
    install(monkeypatch, response_for())
    result = provider.classify_semantic_v2("gpt-5.1", [clause()])["a"]
    assert result.primary_slot is CanonicalSlot.FEATURES
    assert result.secondary_slot is CanonicalSlot.BENEFIT
    assert result.evidence_tags == (EvidenceTag.PRODUCT_ATTRIBUTE, EvidenceTag.PRACTICAL_BENEFIT)


def test_prompt_enumerates_only_enum_evidence_tags(monkeypatch):
    captured = {}

    def fake_chat(operation, model, messages, **kwargs):
        captured["system"] = messages[0]["content"][0]["text"]
        return response_for(evidence_tags=[])

    monkeypatch.setattr(provider, "_chat", fake_chat)
    result = provider.classify_semantic_v2("gpt-5.1", [clause()])["a"]
    prompt = captured["system"]
    assert all(tag.value in prompt for tag in EvidenceTag)
    assert "only values from this exact allowed list" in prompt
    assert "never return arbitrary or invented tags" in prompt
    assert "question_mark" not in prompt and "imperative_verb" not in prompt
    assert result.evidence_tags == ()


@pytest.mark.parametrize("tags", [[], ["question_hook"], [tag.value for tag in EvidenceTag]])
def test_empty_and_allowed_evidence_tags_validate(monkeypatch, tags):
    install(monkeypatch, response_for(evidence_tags=tags))
    result = provider.classify_semantic_v2("gpt-5.1", [clause()])["a"]
    assert [tag.value for tag in result.evidence_tags] == tags


def test_context_is_adjacent_and_never_crosses_source():
    clips = [
        {"id": "a", "text": "previous", "source_index": 0, "slot": "STORY", "meta": {}},
        {"id": "b", "text": "target", "source_index": 0, "slot": "STORY", "meta": {}},
        {"id": "c", "text": "another file", "source_index": 1, "slot": "STORY", "meta": {}},
        {"id": "d", "text": "following", "source_index": 1, "slot": "STORY", "meta": {}},
    ]
    values = build_clause_inputs(clips)
    assert (values[0].preceding_transcript, values[0].following_transcript) == (None, "target")
    assert (values[1].preceding_transcript, values[1].following_transcript) == ("previous", None)
    assert (values[2].preceding_transcript, values[2].following_transcript) == (None, "following")
    assert values[3].following_transcript is None


@pytest.mark.parametrize("text,field", [
    ("Does your hair always frizz?", "question_mark"),
    ("Click the link now", "direct_action_phrase"),
    ("I tried this for 7 days", "first_person_narrative"),
    ("It contains a ceramic plate", "product_attribute"),
    ("Take two, start over", "production_talk"),
    ("And because", "incomplete_phrase"),
])
def test_deterministic_signals(text, field):
    assert getattr(derive_signals(text, "STORY"), field) is True


def test_provider_failure_logs_no_sensitive_content(monkeypatch, caplog):
    secrets = ("private transcript", "/tmp/creator.mov", "sk-secret", "RAW_RESPONSE")
    monkeypatch.setattr(provider, "create_openai_client", lambda: (_ for _ in ()).throw(RuntimeError(" ".join(secrets))))
    with caplog.at_level(logging.WARNING), pytest.raises(OpenAIProviderError) as caught:
        provider.classify_semantic_v2("gpt-5.1", [clause(secrets[0])])
    combined = caplog.text + str(caught.value)
    assert all(secret not in combined for secret in secrets)
