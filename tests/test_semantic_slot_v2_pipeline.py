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
    ("It helps you feel confident", "BENEFIT"),
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


def test_disabled_by_default_and_model_unchanged():
    assert pipeline.EDITDNA_USE_LLM is False
    assert pipeline.EDITDNA_LLM_MODEL == "gpt-5.1"
    assert pipeline.SEMANTIC_V2_MIN_CONFIDENCE == .70
