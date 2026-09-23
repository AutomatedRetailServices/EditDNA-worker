from types import SimpleNamespace

from cutsell_worker.contracts import CandidateTake
from cutsell_worker.take_grouping_openai import OpenAITakeGroupingProvider
from cutsell_worker.take_grouping_provider import safe_group_takes


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


def _takes():
    return (
        CandidateTake("a", "src", 0, 0.0, 2.0, "This changed my morning routine completely"),
        CandidateTake("b", "src", 0, 2.2, 4.2, "This completely changed my morning routine"),
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


def test_malformed_grouping_json_is_repaired_once_before_fallback():
    client = FakeClient((
        '{"groups":[["a","b"],["c"]],"reason":"broken",}',
        '{"groups":[["a","b"],["c"]],"reason":"same retry after format repair"}',
    ))
    provider = OpenAITakeGroupingProvider(client_factory=lambda: client)
    result = safe_group_takes(provider, _takes())
    assert result.status.status == "applied"
    assert result.groups == (("a", "b"), ("c",))
    assert "json_format_repaired" in result.reason
    assert client.responses.calls == 2


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


def test_nearby_reworded_retry_remains_groupable_when_lexical_evidence_is_strong():
    takes = (
        CandidateTake("a", "src", 0, 10.0, 13.0, "I could not believe how easy this was to use"),
        CandidateTake("b", "src", 0, 15.0, 18.0, "I could not believe this was so easy to use"),
    )
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a","b"]],"reason":"immediate retry with reworded sentence"}'
        )
    )
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups == (("a", "b"),)


def test_nearby_distinct_story_beats_are_split_even_when_provider_groups_them():
    takes = (
        CandidateTake("a", "src", 0, 0.0, 4.0, "This one was one of them my favorite second video is with Blaine O Connor"),
        CandidateTake("b", "src", 0, 4.0, 9.0, "We had instant chemistry the second we met and we had an amazing friendship"),
        CandidateTake("c", "src", 0, 9.0, 13.0, "We made a lot of videos but did not want to make the same video over and over"),
    )
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a","b","c"]],"reason":"same thematic storytelling block"}'
        )
    )
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups == (("a",), ("b",), ("c",))
    assert "provider_output_repaired" in result.reason


def test_provider_group_constraint_requires_every_member_not_transitive_chain():
    takes = (
        CandidateTake("a", "src", 0, 0.0, 4.0, "This serum cleared the acne on my back in the summer"),
        CandidateTake("b", "src", 0, 5.0, 9.0, "This serum cleared the acne on my back during the summer"),
        CandidateTake("c", "src", 0, 10.0, 14.0, "During the summer this serum helped the dry skin on my back"),
    )
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a","b","c"]],"reason":"provider chained similar summer story beats"}'
        )
    )
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups == (("a", "b"), ("c",))
    assert "provider_output_repaired" in result.reason


def test_provider_split_near_duplicate_retries_are_reconciled_locally():
    takes = (
        CandidateTake("a", "src", 0, 10.0, 15.0, "Por temporada me salía acné en la espalda que yo resolvía con resorcina"),
        CandidateTake("b", "src", 0, 18.0, 23.0, "Por temporada me salía acné en la espalda la cual yo resolvía con resorcina"),
        CandidateTake("c", "src", 0, 40.0, 44.0, "También se me caía mucho el pelo cuando me bañaba"),
    )
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a"],["b"],["c"]],"reason":"provider missed retry"}'
        )
    )
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups == (("a", "b"), ("c",))
    assert "local_retry_reconciled" in result.reason


def test_run12_nearby_acne_retry_is_recovered_without_broad_topic_merge():
    takes = (
        CandidateTake("a", "src", 0, 10.0, 14.0, "por temporada me salía acné en la"),
        CandidateTake("b", "src", 0, 20.9, 25.0, "por temporada me salía acné en la espalda la cual yo resorbí"),
        CandidateTake("c", "src", 0, 28.0, 32.0, "también tuve problemas de digestión en donde me"),
    )
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a"],["b"],["c"]],"reason":"provider split nearby retries"}'
        )
    )
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups == (("a", "b"), ("c",))
    assert "local_retry_reconciled" in result.reason


def test_run12_medium_distance_near_verbatim_retry_is_recovered():
    takes = (
        CandidateTake("a", "src", 0, 0.0, 5.0, "People ask me all the time do you actually have fun doing your job and the answer is yes"),
        CandidateTake("b", "src", 0, 19.3, 24.0, "People ask me all the time. Do you actually have fun in your job? And the answer is yes, obviously"),
        CandidateTake("c", "src", 0, 30.0, 34.0, "The best part is getting to create videos with my friends"),
    )
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a"],["b"],["c"]],"reason":"provider split same sentence"}'
        )
    )
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups == (("a", "b"), ("c",))
    assert "local_retry_reconciled" in result.reason


def test_nearby_distinct_story_beats_are_not_reconciled_just_for_shared_topic():
    takes = (
        CandidateTake("a", "src", 0, 10.0, 14.0, "My morning routine starts with washing my face before breakfast"),
        CandidateTake("b", "src", 0, 20.0, 25.0, "My morning routine also includes charging the device beside the sink"),
    )
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a"],["b"]],"reason":"two distinct details"}'
        )
    )
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups == (("a",), ("b",))
    assert "local_retry_reconciled" not in result.reason


def test_far_similar_but_not_near_verbatim_content_stays_separate():
    takes = (
        CandidateTake("a", "src", 0, 0.0, 4.0, "People ask me all the time whether I actually enjoy doing this job"),
        CandidateTake("b", "src", 0, 90.0, 94.0, "People ask me often if this job is something I really enjoy doing"),
    )
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a"],["b"]],"reason":"separate moments"}'
        )
    )
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups == (("a",), ("b",))


def test_moderately_similar_nearby_provider_splits_stay_separate():
    takes = (
        CandidateTake("a", "src", 0, 10.0, 14.0, "I had acne on my back during the summer and used a treatment for it"),
        CandidateTake("b", "src", 0, 18.0, 22.0, "I had acne on my back during the summer but my skin also became very dry"),
    )
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a"],["b"]],"reason":"distinct details despite shared setup"}'
        )
    )
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups == (("a",), ("b",))


def test_reconciliation_requires_every_cross_pair_not_transitive_chain():
    takes = (
        CandidateTake("a", "src", 0, 0.0, 4.0, "This serum cleared the acne on my back in the summer"),
        CandidateTake("b", "src", 0, 6.0, 10.0, "This serum cleared the acne on my back during the summer"),
        CandidateTake("c", "src", 0, 12.0, 16.0, "During the summer this serum helped the dry skin on my back"),
    )
    provider = OpenAITakeGroupingProvider(
        client_factory=lambda: FakeClient(
            '{"groups":[["a"],["b"],["c"]],"reason":"three separate groups"}'
        )
    )
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups[0] == ("a", "b")
    assert result.groups[1] == ("c",)
