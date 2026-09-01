from cutsell_worker.composer_provider import ComposerProviderResult, safe_compose_order
from cutsell_worker.contracts import CandidateTake, EditStrategy
from cutsell_worker.draft_review_provider import DraftReviewResult, safe_review_draft
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.take_grouping_provider import TakeGroupingProviderResult, safe_group_takes


def _takes():
    return (
        CandidateTake("a", "src", 0, 0.0, 1.0, "alpha"),
        CandidateTake("b", "src", 0, 1.0, 2.0, "beta"),
        CandidateTake("c", "src", 0, 2.0, 3.0, "gamma"),
    )


class _GroupingProvider:
    def group(self, takes, context_text=""):
        return TakeGroupingProviderResult(
            groups=(("a", "b", "ghost"), ("b",)),
            status=ProviderStatus("openai", True, True, "applied"),
            reason="semantic retry",
        )


class _ComposerProvider:
    def order(self, takes, labels, strategy, context_text=""):
        return ComposerProviderResult(
            ordered_clip_ids=("c", "ghost", "c", "a"),
            status=ProviderStatus("openai", True, True, "applied"),
            reason="stronger opening",
        )


class _ReviewProvider:
    def review(self, takes, labels, strategy, context_text=""):
        return DraftReviewResult(
            ordered_clip_ids=("b", "ghost", "b", "a"),
            postable=False,
            issues=("redundancy",),
            reason="tighten",
            status=ProviderStatus("openai", True, True, "applied"),
        )


def test_grouping_repairs_unknown_duplicate_and_omitted_candidates():
    result = safe_group_takes(_GroupingProvider(), _takes())
    assert result.status.status == "applied"
    # Unknown and duplicate ids are repaired, but the provider may no longer force
    # lexically unrelated nearby clips into one retry group merely because they are
    # close in time. Preserve uncertain creator speech as separate candidates.
    assert result.groups == (("a",), ("b",), ("c",))
    assert "provider_output_repaired" in result.reason


def test_composer_repairs_permutation_without_dropping_creator_speech():
    result = safe_compose_order(_ComposerProvider(), _takes(), (), EditStrategy.MIXED)
    assert result.status.status == "applied"
    assert result.ordered_clip_ids == ("c", "a", "b")
    assert "provider_output_repaired" in result.reason


def test_draft_review_sanitizes_unknown_and_duplicate_ids_but_may_remove_clips():
    result = safe_review_draft(_ReviewProvider(), _takes(), (), EditStrategy.MIXED)
    assert result.status.status == "applied"
    assert result.ordered_clip_ids == ("b", "a")
    assert result.postable is False
    assert "review_output_repaired" in result.issues
