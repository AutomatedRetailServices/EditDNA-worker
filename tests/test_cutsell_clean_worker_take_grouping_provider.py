from types import SimpleNamespace

from cutsell_worker.contracts import CandidateTake
from cutsell_worker.take_grouping_openai import OpenAITakeGroupingProvider
from cutsell_worker.take_grouping_provider import safe_group_takes


class FakeResponses:
    def __init__(self, output_text):
        self.output_text = output_text
    def create(self, **kwargs):
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, output_text):
        self.responses = FakeResponses(output_text)


def _takes():
    return (
        CandidateTake("a", "src", 0, 0.0, 2.0, "This changed my morning routine"),
        CandidateTake("b", "src", 0, 2.2, 4.2, "My mornings are completely different because of this"),
        CandidateTake("c", "src", 0, 5.0, 7.0, "It comes with three attachments"),
    )


def test_semantic_grouping_can_join_same_idea_with_different_wording():
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a","b"],["c"]],"reason":"a and b retry the same story beat"}'
        )
    )
    result = safe_group_takes(provider, _takes())
    assert result.status.status == "applied"
    assert result.groups == (("a", "b"), ("c",))


def test_semantic_grouping_rejects_missing_candidate_and_uses_baseline():
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a","b"]],"reason":"invalid omission"}'
        )
    )
    result = safe_group_takes(provider, _takes())
    assert result.status.status == "provider_error_fallback"
    flattened = [item for group in result.groups for item in group]
    assert set(flattened) == {"a", "b", "c"}


def test_semantic_grouping_rejects_duplicate_candidate():
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a","b"],["b","c"]],"reason":"invalid duplicate"}'
        )
    )
    result = safe_group_takes(provider, _takes())
    assert result.status.status == "provider_error_fallback"
    flattened = [item for group in result.groups for item in group]
    assert len(flattened) == 3
    assert set(flattened) == {"a", "b", "c"}
