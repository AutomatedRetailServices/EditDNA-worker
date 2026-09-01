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
            reason="static test grouping",
        )


def test_adjacent_retry_group_extends_across_short_prefix_false_start():
    takes = (
        CandidateTake("a", "src", 0, 166.5, 169.8, "Por temporada me salía acné en la espalda que resorbía, resorbía."),
        CandidateTake("b", "src", 0, 171.3, 181.1, "Por temporada me salía un acné en la espalda con la que yo resolvía con resorcina."),
        CandidateTake("prefix", "src", 0, 181.1, 184.8, "Por temporada"),
        CandidateTake("c", "src", 0, 186.2, 190.5, "me salía un acné en la espalda con la que yo resolvía con resorcina."),
        CandidateTake("d", "src", 0, 192.4, 197.5, "También me salían espinillas como una alergia."),
    )
    provider = StaticGroupingProvider((("a", "b"), ("prefix",), ("c",), ("d",)))
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups[0] == ("a", "b", "prefix", "c")
    assert result.groups[1] == ("d",)
    assert "adjacent_retry_extended" in result.reason


def test_prefix_fragment_immediately_before_validated_retry_group_is_absorbed():
    takes = (
        CandidateTake("prefix", "src", 0, 153.3, 155.4, "People ask me all the time do", complete_idea=False),
        CandidateTake("a", "src", 0, 158.2, 163.3, "People ask me all the time do you actually have fun doing your job and the answer is yes"),
        CandidateTake("b", "src", 0, 177.6, 183.0, "People ask me all the time. Do you actually have fun in your job? And the answer is yes, obviously"),
        CandidateTake("c", "src", 0, 183.0, 189.1, "Sometimes you are hanging out with someone super cool"),
    )
    provider = StaticGroupingProvider((("prefix",), ("a",), ("b",), ("c",)))
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups[0] == ("prefix", "a", "b")
    assert result.groups[1] == ("c",)
    assert "interstitial_retry_debris_absorbed" in result.reason


def test_unrelated_short_line_before_retry_group_is_preserved():
    takes = (
        CandidateTake("short", "src", 0, 153.3, 155.4, "That was wild", complete_idea=False),
        CandidateTake("a", "src", 0, 158.2, 163.3, "People ask me all the time do you actually have fun doing your job and the answer is yes"),
        CandidateTake("b", "src", 0, 177.6, 183.0, "People ask me all the time. Do you actually have fun in your job? And the answer is yes, obviously"),
    )
    provider = StaticGroupingProvider((("short",), ("a",), ("b",)))
    result = safe_group_takes(provider, takes)
    assert result.status.status == "applied"
    assert result.groups[0] == ("short",)
    assert result.groups[1] == ("a", "b")
