from types import SimpleNamespace

from cutsell_worker.composer_openai import OpenAIComposerProvider
from cutsell_worker.composer_provider import safe_compose_order
from cutsell_worker.contracts import CandidateTake, EditStrategy, SemanticLabel, SemanticRole


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
        CandidateTake("a", "src", 0, 0.0, 2.0, "I had this problem"),
        CandidateTake("b", "src", 0, 2.0, 4.0, "Then I tried this product"),
        CandidateTake("c", "src", 0, 4.0, 6.0, "This is the result"),
    )


def _labels():
    return (
        SemanticLabel("a", SemanticRole.STORY, 0.9, "story setup"),
        SemanticLabel("b", SemanticRole.FEATURES, 0.8, "product use"),
        SemanticLabel("c", SemanticRole.PROOF, 0.9, "result"),
    )


def test_flexible_composer_can_reorder_only_existing_real_clips():
    provider = OpenAIComposerProvider(
        client_factory=lambda: FakeClient(
            '{"ordered_clip_ids":["b","a","c"],"reason":"stronger product-led opening"}'
        )
    )
    result = safe_compose_order(provider, _takes(), _labels(), EditStrategy.MIXED)
    assert result.status.status == "applied"
    assert result.ordered_clip_ids == ("b", "a", "c")


def test_flexible_composer_repairs_dropped_clip_without_losing_source_material():
    provider = OpenAIComposerProvider(
        client_factory=lambda: FakeClient(
            '{"ordered_clip_ids":["a","c"],"reason":"drop one"}'
        )
    )
    result = safe_compose_order(provider, _takes(), _labels(), EditStrategy.STORYTELLING)
    assert result.status.status == "applied"
    assert result.ordered_clip_ids == ("a", "c", "b")
    assert set(result.ordered_clip_ids) == {"a", "b", "c"}


def test_flexible_composer_repairs_duplicate_or_invented_clip_safely():
    provider = OpenAIComposerProvider(
        client_factory=lambda: FakeClient(
            '{"ordered_clip_ids":["a","a","fake"],"reason":"invalid"}'
        )
    )
    result = safe_compose_order(provider, _takes(), _labels(), EditStrategy.DIRECT_SALES)
    assert result.status.status == "applied"
    assert result.ordered_clip_ids == ("a", "b", "c")
    assert len(set(result.ordered_clip_ids)) == 3
