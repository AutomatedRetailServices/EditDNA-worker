from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.take_grouping_provider import TakeGroupingProviderResult, safe_group_takes


class StaticGroupingProvider:
    def __init__(self, groups):
        self._groups = tuple(tuple(group) for group in groups)

    def group(self, takes, context_text=""):
        return TakeGroupingProviderResult(
            groups=self._groups,
            status=ProviderStatus("openai", True, True, "applied"),
            reason="static grouping",
        )


def test_prefix_debris_inside_group_does_not_block_true_retry_reconciliation():
    takes = (
        CandidateTake(
            "prefix",
            "src",
            0,
            153.36,
            155.45,
            "People ask me all the time do",
            complete_idea=False,
        ),
        CandidateTake(
            "a",
            "src",
            0,
            158.28,
            163.36,
            "People ask me all the time do you actually have fun doing your job and the answer is yes",
            complete_idea=True,
        ),
        CandidateTake(
            "b",
            "src",
            0,
            177.63,
            183.03,
            "People ask me all the time. Do you actually have fun in your job? And the answer is yes, obviously",
            complete_idea=True,
        ),
        CandidateTake(
            "c",
            "src",
            0,
            183.03,
            189.19,
            "Sometimes you are hanging with one of your friends and having a great time",
            complete_idea=True,
        ),
    )
    provider = StaticGroupingProvider((("prefix", "a"), ("b",), ("c",)))
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups[0] == ("prefix", "a", "b")
    assert result.groups[1] == ("c",)
    assert "local_retry_reconciled" in result.reason


def test_incomplete_nonprefix_member_is_split_while_true_retries_still_merge():
    takes = (
        CandidateTake(
            "noise",
            "src",
            0,
            153.36,
            155.45,
            "That was honestly wild",
            complete_idea=False,
        ),
        CandidateTake(
            "a",
            "src",
            0,
            158.28,
            163.36,
            "People ask me all the time do you actually have fun doing your job and the answer is yes",
            complete_idea=True,
        ),
        CandidateTake(
            "b",
            "src",
            0,
            177.63,
            183.03,
            "People ask me all the time. Do you actually have fun in your job? And the answer is yes, obviously",
            complete_idea=True,
        ),
    )
    provider = StaticGroupingProvider((("noise", "a"), ("b",)))
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups[0] == ("noise",)
    assert result.groups[1] == ("a", "b")
    assert "provider_output_repaired" in result.reason
    assert "local_retry_reconciled" in result.reason
