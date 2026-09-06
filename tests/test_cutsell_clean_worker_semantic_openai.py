from types import SimpleNamespace

from cutsell_worker.contracts import CandidateTake, SemanticRole
from cutsell_worker.semantic_openai import OpenAISemanticProvider


class FakeResponses:
    def __init__(self, output_text):
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, output_text):
        self.responses = FakeResponses(output_text)


def _take(clip_id, text):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src_1",
        source_order=0,
        start=0.0,
        end=1.0,
        text=text,
    )


def test_openai_semantic_adapter_maps_roles_without_deletion_authority():
    client = FakeClient(
        '{"clips":['
        '{"id":"a","role":"HOOK","confidence":0.91,"reason":"attention opener"},'
        '{"id":"b","role":"CTA","confidence":0.88,"reason":"asks viewer to act"}'
        ']}'
    )
    provider = OpenAISemanticProvider(client_factory=lambda: client)
    result = provider.classify((_take("a", "Stop scrolling"), _take("b", "Tap the cart")))
    assert [label.role for label in result.labels] == [SemanticRole.HOOK, SemanticRole.CTA]
    assert result.status.status == "applied"
    assert client.responses.calls[0]["model"] == "gpt-4o-mini"


def test_openai_semantic_adapter_rejects_missing_clip():
    client = FakeClient('{"clips":[{"id":"a","role":"HOOK","confidence":0.9,"reason":"hook"}]}')
    provider = OpenAISemanticProvider(client_factory=lambda: client)
    try:
        provider.classify((_take("a", "A"), _take("b", "B")))
    except ValueError as exc:
        assert "omitted" in str(exc)
    else:
        raise AssertionError("provider must not silently omit clips")
