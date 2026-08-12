import json
from types import SimpleNamespace

from cutsell_worker.clean_cut_openai import OpenAICleanCutProvider
from cutsell_worker.clean_cut_provider import apply_provider_judgements, safe_clean_cut_judge
from cutsell_worker.contracts import CandidateTake, Word


def _take(clip_id, start, end, text):
    tokens = text.split()
    duration = max(0.001, end - start)
    step = duration / max(1, len(tokens))
    words = tuple(
        Word(token, start + index * step, min(end, start + (index + 1) * step), 0.9)
        for index, token in enumerate(tokens)
    )
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src-1",
        source_order=0,
        start=start,
        end=end,
        text=text,
        words=words,
        complete_idea=len(tokens) >= 6,
    )


class FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(self.payload))


class FakeClient:
    def __init__(self, payload):
        self.responses = FakeResponses(payload)


def test_openai_provider_sends_only_microtake_and_keeps_neighbors_read_only():
    left = _take("left", 0.0, 4.0, "this is a valid long creator sentence here")
    micro = _take("micro", 4.1, 5.0, "duffel")
    right = _take("right", 5.1, 9.1, "this is another valid long creator sentence here")
    client = FakeClient({
        "judgements": [{
            "id": "micro",
            "action": "delete",
            "confidence": 0.99,
            "reason": "abandoned fragment",
            "keep_start_word_index": None,
            "keep_end_word_index": None,
        }]
    })
    provider = OpenAICleanCutProvider(client_factory=lambda: client)

    judged = safe_clean_cut_judge(provider, (left, micro, right))

    assert len(client.responses.calls) == 1
    user_payload = json.loads(client.responses.calls[0]["input"][1]["content"])
    assert [item["id"] for item in user_payload["takes"]] == ["micro"]
    evidence = user_payload["takes"][0]
    assert evidence["previous_transcript"] == left.text
    assert evidence["next_transcript"] == right.text
    assert evidence["word_count"] == 1
    assert evidence["complete_idea"] is False
    assert evidence["asr_avg_confidence"] == 0.9
    assert evidence["asr_min_confidence"] == 0.9
    assert evidence["gap_from_previous_sec"] == 0.1
    assert evidence["gap_to_next_sec"] == 0.1
    assert "similarity_to_previous" in evidence
    assert "similarity_to_next" in evidence
    assert "exact_prefix_of_previous" in evidence
    assert "exact_prefix_of_next" in evidence
    assert evidence["words"][0]["confidence"] == 0.9

    system_prompt = client.responses.calls[0]["input"][0]["content"]
    assert "Grammatical incompleteness by itself is NOT enough to DELETE" in system_prompt
    assert "confidence >=0.95" in system_prompt

    assert [(item.clip_id, item.action) for item in judged.judgements] == [
        ("left", "keep"),
        ("micro", "delete"),
        ("right", "keep"),
    ]
    kept, deleted, diagnostics = apply_provider_judgements((left, micro, right), judged)
    assert [take.clip_id for take in kept] == ["left", "right"]
    assert [take.clip_id for take in deleted] == ["micro"]
    assert next(item for item in diagnostics if item["clip_id"] == "left")["applied_delete"] is False
    assert next(item for item in diagnostics if item["clip_id"] == "right")["applied_delete"] is False


def test_openai_provider_marks_exact_prefix_retry_evidence():
    fragment = _take("fragment", 0.0, 0.8, "people ask")
    full = _take("full", 0.9, 4.9, "people ask me all the time if I enjoy this")
    client = FakeClient({
        "judgements": [{
            "id": "fragment",
            "action": "delete",
            "confidence": 0.98,
            "reason": "abandoned exact-prefix retry",
            "keep_start_word_index": None,
            "keep_end_word_index": None,
        }]
    })
    provider = OpenAICleanCutProvider(client_factory=lambda: client)

    safe_clean_cut_judge(provider, (fragment, full))

    payload = json.loads(client.responses.calls[0]["input"][1]["content"])
    evidence = payload["takes"][0]
    assert evidence["exact_prefix_of_next"] is True
    assert evidence["gap_to_next_sec"] == 0.1
    assert evidence["similarity_to_next"] > 0.0


def test_openai_provider_skips_api_call_when_there_are_no_microtakes():
    left = _take("left", 0.0, 4.0, "this is a valid long creator sentence here")
    right = _take("right", 4.2, 8.2, "another complete creator sentence remains safely untouched")
    client = FakeClient({"judgements": []})
    provider = OpenAICleanCutProvider(client_factory=lambda: client)

    judged = safe_clean_cut_judge(provider, (left, right))

    assert client.responses.calls == []
    assert [(item.clip_id, item.action) for item in judged.judgements] == [
        ("left", "keep"),
        ("right", "keep"),
    ]
    assert judged.status.reason == "no_ambiguous_microtakes"


def test_openai_provider_can_keep_intentional_short_reaction():
    reaction = _take("reaction", 1.0, 1.6, "Yeah")
    client = FakeClient({
        "judgements": [{
            "id": "reaction",
            "action": "keep",
            "confidence": 0.99,
            "reason": "intentional reaction",
            "keep_start_word_index": None,
            "keep_end_word_index": None,
        }]
    })
    provider = OpenAICleanCutProvider(client_factory=lambda: client)

    judged = safe_clean_cut_judge(provider, (reaction,))
    kept, deleted, diagnostics = apply_provider_judgements((reaction,), judged)

    assert [take.clip_id for take in kept] == ["reaction"]
    assert deleted == ()
    assert diagnostics[0]["action"] == "keep"
