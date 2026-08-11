from types import SimpleNamespace

from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.take_judge_openai import OpenAITakeJudgeProvider
from cutsell_worker.take_judge_provider import safe_rank_takes


class FakeResponses:
    def __init__(self, output_text):
        if isinstance(output_text, (list, tuple)):
            self.output_texts = list(output_text)
        else:
            self.output_texts = [output_text]
        self.calls = 0

    def create(self, **kwargs):
        index = min(self.calls, len(self.output_texts) - 1)
        self.calls += 1
        return SimpleNamespace(output_text=self.output_texts[index])


class FakeClient:
    def __init__(self, output_text):
        self.responses = FakeResponses(output_text)


def _take(clip_id, eye_contact):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src-1",
        source_order=0,
        start=0.0,
        end=2.0,
        text="same sales idea",
        signals=MediaSignals("src-1", 0.0, 2.0, eye_contact=eye_contact),
    )


def test_openai_take_judge_can_choose_stronger_delivery():
    provider = OpenAITakeJudgeProvider(
        client_factory=lambda: FakeClient(
            '{"ranked":['
            '{"id":"b","score":0.94,"reason":"stronger delivery"},'
            '{"id":"a","score":0.71,"reason":"weaker eye contact"}'
            ']}'
        )
    )
    result = safe_rank_takes((_take("a", 0.3), _take("b", 0.9)), provider)
    assert result.status.status == "applied"
    assert [item.clip_id for item in result.ranked] == ["b", "a"]


def test_malformed_take_judge_json_is_repaired_once_before_fallback():
    client = FakeClient((
        '{"ranked":[{"id":"b","score":0.94,"reason":"stronger delivery"},{"id":"a","score":0.71,"reason":"weaker"}],}',
        '{"ranked":[{"id":"b","score":0.94,"reason":"stronger delivery"},{"id":"a","score":0.71,"reason":"weaker"}]}',
    ))
    provider = OpenAITakeJudgeProvider(client_factory=lambda: client)
    result = safe_rank_takes((_take("a", 0.3), _take("b", 0.9)), provider)
    assert result.status.status == "applied"
    assert [item.clip_id for item in result.ranked] == ["b", "a"]
    assert client.responses.calls == 2


def test_semantically_invalid_take_judge_payload_still_falls_back():
    client = FakeClient(
        '{"ranked":[{"id":"a","score":0.9,"reason":"only one returned"}]}'
    )
    provider = OpenAITakeJudgeProvider(client_factory=lambda: client)
    result = safe_rank_takes((_take("a", 0.3), _take("b", 0.9)), provider)
    assert result.status.status == "provider_error_fallback"
    assert {item.clip_id for item in result.ranked} == {"a", "b"}
    assert client.responses.calls == 1
