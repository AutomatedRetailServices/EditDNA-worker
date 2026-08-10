from cutsell_worker.contracts import CandidateTake, EditStrategy
from cutsell_worker.draft_review_provider import DraftReviewResult, safe_review_draft
from cutsell_worker.providers import ProviderStatus


def _take(clip_id, text, start):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(start) + 2.0,
        text=text,
    )


class Reviewer:
    def __init__(self, ordered, *, postable=True):
        self.ordered = ordered
        self.postable = postable

    def review(self, takes, labels, strategy, context_text=""):
        return DraftReviewResult(
            ordered_clip_ids=tuple(self.ordered),
            postable=self.postable,
            issues=() if self.postable else ("story jump",),
            reason="global story review",
            status=ProviderStatus("test", True, True, "applied"),
        )


def test_review_can_remove_redundant_clip_and_reorder_existing_material():
    takes = (
        _take("a", "hook", 0),
        _take("b", "repeated detail", 3),
        _take("c", "payoff", 6),
    )
    result = safe_review_draft(
        Reviewer(("a", "c")),
        takes,
        (),
        EditStrategy.STORYTELLING,
        context_text="mode=natural; story_logic=hook then payoff",
    )
    assert result.ordered_clip_ids == ("a", "c")
    assert result.postable is True


def test_review_cannot_invent_unknown_clip():
    takes = (_take("a", "hook", 0), _take("b", "payoff", 3))
    result = safe_review_draft(
        Reviewer(("a", "invented")),
        takes,
        (),
        EditStrategy.STORYTELLING,
    )
    assert result.ordered_clip_ids == ("a", "b")
    assert result.postable is False
    assert result.status.status == "provider_error_fallback"


def test_review_cannot_duplicate_clip():
    takes = (_take("a", "hook", 0), _take("b", "payoff", 3))
    result = safe_review_draft(
        Reviewer(("a", "a")),
        takes,
        (),
        EditStrategy.STORYTELLING,
    )
    assert result.ordered_clip_ids == ("a", "b")
    assert result.status.status == "provider_error_fallback"
