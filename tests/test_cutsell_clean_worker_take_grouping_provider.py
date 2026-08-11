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


def test_semantic_grouping_repairs_missing_candidate_as_singleton():
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a","b"]],"reason":"invalid omission"}'
        )
    )
    result = safe_group_takes(provider, _takes())
    assert result.status.status == "applied"
    assert result.groups == (("a", "b"), ("c",))
    flattened = [item for group in result.groups for item in group]
    assert set(flattened) == {"a", "b", "c"}


def test_semantic_grouping_repairs_duplicate_candidate_without_dropping_any_take():
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a","b"],["b","c"]],"reason":"invalid duplicate"}'
        )
    )
    result = safe_group_takes(provider, _takes())
    assert result.status.status == "applied"
    assert result.groups == (("a", "b"), ("c",))
    flattened = [item for group in result.groups for item in group]
    assert len(flattened) == 3
    assert set(flattened) == {"a", "b", "c"}


def test_distant_broad_topic_group_is_split_to_prevent_overcut():
    takes = (
        CandidateTake("a", "src", 0, 0.0, 3.0, "I started using this every morning because my routine was chaotic"),
        CandidateTake("b", "src", 0, 70.0, 74.0, "My morning routine also includes charging it beside the sink"),
    )
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a","b"]],"reason":"same morning routine topic"}'
        )
    )
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups == (("a",), ("b",))
    assert "provider_output_repaired" in result.reason


def test_distant_near_duplicate_retry_can_still_group():
    takes = (
        CandidateTake("a", "src", 0, 0.0, 3.0, "This is the exact serum I use every single morning"),
        CandidateTake("b", "src", 0, 65.0, 68.0, "This is the exact serum I use every single morning"),
    )
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a","b"]],"reason":"same sentence retried later"}'
        )
    )
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups == (("a", "b"),)


def test_nearby_reworded_retry_remains_groupable():
    takes = (
        CandidateTake("a", "src", 0, 10.0, 13.0, "I could not believe how easy this was to use"),
        CandidateTake("b", "src", 0, 15.0, 18.0, "Using this ended up being way easier than I expected"),
    )
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a","b"]],"reason":"immediate retry with different wording"}'
        )
    )
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups == (("a", "b"),)
