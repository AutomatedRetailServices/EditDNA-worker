import importlib.util
import sys
import types

import pytest

for name, attrs in (("requests", {}), ("boto3", {}), ("clip", {}), ("faster_whisper", {"WhisperModel": object})):
    if name not in sys.modules and importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module

from worker import pipeline
from worker.semantic_slot_v2 import CanonicalSlot, EvidenceTag, SlotClassificationResult


@pytest.mark.parametrize("text,expected", [
    ("Does your hair always frizz?", "HOOK"),
    ("I'm tired of wasting money on products that fail", "PROBLEM"),
    ("It has a ceramic plate", "FEATURES"),
    ("It helps you feel confident", "BENEFITS"),
    ("I get so many compliments", "PROOF"),
    ("The first time I found it, I was traveling", "STORY"),
    ("Click the link and add to cart", "CTA"),
    ("Take two, wait, start over", "OTHER"),
])
def test_strict_definition_examples(text, expected):
    assert pipeline.classify_slot(text) == expected


def result(slot=CanonicalSlot.PROOF, confidence=.9, completeness=.9, abstain=False):
    return SlotClassificationResult(
        slot, CanonicalSlot.STORY, confidence, min(confidence, .7), completeness, .9, .8,
        abstain, "Observed measurable outcome", (EvidenceTag.MEASURABLE_RESULT,),
    )


def clips():
    values = [
        pipeline.make_base_clip("a", 1, 3, "I used it for seven days and lost two inches."),
        pipeline.make_base_clip("b", 4, 6, "This remains untouched."),
    ]
    for item in values:
        item["source_index"] = 0
        item["source_local"] = "/tmp/private.mov"
    return values


@pytest.mark.parametrize("provider_value", [None, result(confidence=.4), result(abstain=True), result(completeness=.2)])
def test_fallback_preserves_heuristic(monkeypatch, provider_value):
    values = clips()
    monkeypatch.setattr(pipeline, "EDITDNA_USE_LLM", True)
    monkeypatch.setattr(pipeline, "llm_classify_clips", lambda _: None if provider_value is None else {"a": provider_value})
    heuristic = pipeline.classify_slot(values[0]["text"])
    pipeline.enrich_clips_semantic(values)
    assert values[0]["slot"] == heuristic


def test_valid_result_updates_only_primary_and_stores_secondary(monkeypatch):
    values = clips()
    untouched_before = dict(values[1])
    protected = {key: values[0][key] for key in ("id", "text", "start", "end", "source_index", "source_local")}
    monkeypatch.setattr(pipeline, "EDITDNA_USE_LLM", True)
    monkeypatch.setattr(pipeline, "llm_classify_clips", lambda _: {"a": result()})
    assert pipeline.enrich_clips_semantic(values) is True
    assert values[0]["slot"] == "PROOF"
    assert values[0]["meta"]["semantic_v2"]["secondary_slot"] == "STORY"
    assert values[0]["meta"]["semantic_v2"]["applied"] is True
    assert {key: values[0][key] for key in protected} == protected
    # The unprocessed clause receives only the normal deterministic heuristic pass.
    assert values[1]["id"] == untouched_before["id"] and values[1]["text"] == untouched_before["text"]
    assert values[1]["start"] == untouched_before["start"] and values[1]["end"] == untouched_before["end"]
    assert "semantic_v2" not in values[1]["meta"]


def test_internal_benefit_applies_historical_public_slot(monkeypatch):
    values = clips()[:1]
    monkeypatch.setattr(pipeline, "EDITDNA_USE_LLM", True)
    monkeypatch.setattr(pipeline, "llm_classify_clips", lambda _: {"a": result(CanonicalSlot.BENEFIT)})
    pipeline.enrich_clips_semantic(values)
    assert values[0]["slot"] == "BENEFITS"
    assert values[0]["meta"]["slot"] == "BENEFITS"
    # semantic_v2 is an explicitly internal, canonical metadata contract.
    assert values[0]["meta"]["semantic_v2"]["primary_slot"] == "BENEFIT"


def test_public_benefits_is_canonicalized_for_provider_input():
    value = pipeline.make_base_clip("benefit", 0, 1, "It helps you save time.")
    value["slot"] = "BENEFITS"
    clause = pipeline.build_clause_inputs([value])[0]
    assert clause.heuristic_slot == "BENEFIT"
    assert clause.signals.heuristic_slot == "BENEFIT"


def test_public_slot_collections_and_composer_preserve_benefits():
    benefit = pipeline.make_base_clip("benefit", 0, 2, "It helps you save time.")
    benefit.update(slot="BENEFITS", keep=True, score=.9, semantic_score=.9)
    slots = pipeline.build_slots_dict([benefit])
    assert "BENEFITS" in slots and slots["BENEFITS"] == [benefit]
    assert "BENEFIT" not in slots
    assert pipeline.build_composer([benefit])["benefit_ids"] == ["benefit"]


@pytest.mark.parametrize("slot", ["HOOK", "PROBLEM", "FEATURES", "PROOF", "STORY", "CTA", "OTHER"])
def test_non_benefit_public_slots_are_not_renamed(slot):
    value = pipeline.make_base_clip("x", 0, 1, "Some text.")
    value["slot"] = slot
    assert pipeline.build_clause_inputs([value])[0].heuristic_slot == slot


def other_result(confidence=.95, abstain=False):
    return SlotClassificationResult(
        CanonicalSlot.OTHER, None, confidence, None, .95, .1, .8,
        abstain, "Not meaningful sales content", (EvidenceTag.NON_SALES_CONTENT,),
    )


@pytest.mark.parametrize("text", ["Hello everyone.", "Camera rolling, take two.", "The weather is cloudy today."])
def test_confident_other_is_excluded_without_changing_keep(monkeypatch, text):
    target = pipeline.make_base_clip("target", 0, 2, text)
    target.update(slot="STORY", keep=True, score=.9, semantic_score=.9)
    target["meta"].update(keep=True, slot="STORY", semantic_score=.9, score=.9)
    unrelated = pipeline.make_base_clip("unrelated", 3, 5, "Click the link now.")
    unrelated.update(slot="CTA", keep=True, score=.9, semantic_score=.9)
    unrelated["meta"].update(keep=True, slot="CTA", semantic_score=.9, score=.9)
    unrelated_before = {key: unrelated[key] for key in ("id", "text", "start", "end", "slot", "keep")}
    monkeypatch.setattr(pipeline, "EDITDNA_USE_LLM", True)
    monkeypatch.setattr(pipeline, "llm_classify_clips", lambda _: {"target": other_result()})

    pipeline.enrich_clips_semantic([target, unrelated])

    semantic = target["meta"]["semantic_v2"]
    assert target["slot"] == pipeline.classify_slot(text)
    assert target["meta"]["keep"] is True and target["keep"] is True
    assert semantic["applied"] is False
    assert semantic["application_status"] == "excluded_other"
    assert semantic["application_reason"] == "validated_other_excluded_from_sales_composer"
    assert semantic["excluded_from_composer"] is True
    composer = pipeline.build_composer([target, unrelated])
    assert "target" not in composer["used_clip_ids"]
    assert {key: unrelated[key] for key in unrelated_before} == unrelated_before
    assert "semantic_v2" not in unrelated["meta"]


@pytest.mark.parametrize("semantic", [other_result(abstain=True), other_result(confidence=.4)])
def test_fallback_other_preserves_heuristic_and_composer_eligibility(monkeypatch, semantic):
    target = pipeline.make_base_clip("target", 0, 2, "I discovered this last year.")
    target.update(slot="STORY", keep=True, semantic_score=.9)
    target["meta"].update(keep=True, slot="STORY", semantic_score=.9)
    monkeypatch.setattr(pipeline, "EDITDNA_USE_LLM", True)
    monkeypatch.setattr(pipeline, "llm_classify_clips", lambda _: {"target": semantic})
    pipeline.enrich_clips_semantic([target])
    assert target["slot"] == "STORY"
    assert target["meta"]["keep"] is True
    assert "excluded_from_composer" not in target["meta"]["semantic_v2"]


def test_disabled_by_default_and_model_unchanged():
    assert pipeline.EDITDNA_USE_LLM is False
    assert pipeline.EDITDNA_LLM_MODEL == "gpt-5.1"
    assert pipeline.SEMANTIC_V2_MIN_CONFIDENCE == .70
