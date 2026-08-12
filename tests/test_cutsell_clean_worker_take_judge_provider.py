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
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
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


def test_malformed_score_uses_deterministic_score_without_group_fallback():
    client = FakeClient(
        '{"ranked":['
        '{"id":"b","score":"N/A","reason":"score formatting failed"},'
        '{"id":"a","score":0.71,"reason":"weaker"}'
        ']}'
    )
    provider = OpenAITakeJudgeProvider(client_factory=lambda: client)
    result = safe_rank_takes((_take("a", 0.3), _take("b", 0.9)), provider)

    assert result.status.status == "applied"
    assert result.status.reason == "score_fallback:1"
    b = next(item for item in result.ranked if item.clip_id == "b")
    assert 0.0 <= b.score <= 1.0
    assert "deterministic baseline used" in b.reason
    assert client.responses.calls == 1


def test_null_score_uses_deterministic_score_without_group_fallback():
    provider = OpenAITakeJudgeProvider(
        client_factory=lambda: FakeClient(
            '{"ranked":['
            '{"id":"b","score":null,"reason":"missing score"},'
            '{"id":"a","score":0.71,"reason":"weaker"}'
            ']}'
        )
    )
    result = safe_rank_takes((_take("a", 0.3), _take("b", 0.9)), provider)
    assert result.status.status == "applied"
    assert result.status.reason == "score_fallback:1"


def test_invalid_clip_id_gets_one_constrained_identity_repair():
    client = FakeClient((
        '{"ranked":['
        '{"id":"ghost","score":0.94,"reason":"stronger delivery"},'
        '{"id":"a","score":0.71,"reason":"weaker"}'
        ']}',
        '{"ranked":['
        '{"id":"b","score":0.94,"reason":"stronger delivery"},'
        '{"id":"a","score":0.71,"reason":"weaker"}'
        ']}',
    ))
    provider = OpenAITakeJudgeProvider(client_factory=lambda: client)
    result = safe_rank_takes((_take("a", 0.3), _take("b", 0.9)), provider)

    assert result.status.status == "applied"
    assert result.status.reason == "candidate_ids_repaired"
    assert [item.clip_id for item in result.ranked] == ["b", "a"]
    assert client.responses.calls == 2
    repair_payload = client.responses.kwargs[1]["input"][1]["content"]
    assert '"allowed_candidate_ids": ["a", "b"]' in repair_payload


def test_omitted_candidate_gets_one_constrained_identity_repair():
    client = FakeClient((
        '{"ranked":[{"id":"a","score":0.9,"reason":"only one returned"}]}',
        '{"ranked":['
        '{"id":"a","score":0.9,"reason":"first"},'
        '{"id":"b","score":0.8,"reason":"second"}'
        ']}',
    ))
    provider = OpenAITakeJudgeProvider(client_factory=lambda: client)
    result = safe_rank_takes((_take("a", 0.3), _take("b", 0.9)), provider)
    assert result.status.status == "applied"
    assert result.status.reason == "candidate_ids_repaired"
    assert {item.clip_id for item in result.ranked} == {"a", "b"}
    assert client.responses.calls == 2


def test_bad_identity_repair_still_falls_back_with_detail():
    client = FakeClient((
        '{"ranked":[{"id":"ghost","score":0.9,"reason":"bad id"},{"id":"a","score":0.8,"reason":"valid"}]}',
        '{"ranked":[{"id":"ghost2","score":0.9,"reason":"still bad"},{"id":"a","score":0.8,"reason":"valid"}]}',
    ))
    provider = OpenAITakeJudgeProvider(client_factory=lambda: client)
    result = safe_rank_takes((_take("a", 0.3), _take("b", 0.9)), provider)
    assert result.status.status == "provider_error_fallback"
    assert result.status.reason == "ValueError: take judge returned invalid clip id"
    assert {item.clip_id for item in result.ranked} == {"a", "b"}
    assert client.responses.calls == 2


def test_non_object_ranking_item_still_falls_back_with_detail_when_repair_fails():
    client = FakeClient(
        '{"ranked":["bad-item",{"id":"a","score":0.9,"reason":"valid"}]}'
    )
    provider = OpenAITakeJudgeProvider(client_factory=lambda: client)
    result = safe_rank_takes((_take("a", 0.3), _take("b", 0.9)), provider)
    assert result.status.status == "provider_error_fallback"
    assert "non-object ranking item" in result.status.reason
    assert client.responses.calls == 2
