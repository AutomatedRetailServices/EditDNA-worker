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


def _provider(groups_json: str):
    return OpenAITakeGroupingProvider(client_factory=lambda: FakeClient(groups_json))


def test_group_level_gap_does_not_make_old_member_block_nearby_retry():
    takes = (
        CandidateTake("a", "src", 0, 166.5, 169.8, "Por temporada me salía acné en la espalda que resorbía resorbía"),
        CandidateTake("b", "src", 0, 171.3, 179.3, "Por temporada me salía acné en la espalda"),
        CandidateTake("c", "src", 0, 185.3, 191.5, "Por temporada me salía un acné en la espalda con la que yo resolvía con resorcina"),
        CandidateTake("d", "src", 0, 195.0, 201.0, "También tuve problemas digestivos durante ese tiempo"),
    )
    result = safe_group_takes(
        _provider('{"groups":[["a","b"],["c"],["d"]],"reason":"provider split third retry"}'),
        takes,
    )
    assert result.status.status == "applied"
    assert result.groups[0] == ("a", "b", "c")
    assert result.groups[1] == ("d",)
    assert "local_retry_reconciled" in result.reason


def test_medium_distance_near_verbatim_retry_reconciles_with_complete_link():
    takes = (
        CandidateTake("a", "src", 0, 98.6, 104.1, "We were laughing so much who could barely keep a straight face and the thing is that"),
        CandidateTake("b", "src", 0, 113.1, 116.7, "We were laughing so much who could barely keep a straight face"),
        CandidateTake("c", "src", 0, 123.0, 128.0, "The next part of the story is completely different"),
    )
    result = safe_group_takes(
        _provider('{"groups":[["a"],["b"],["c"]],"reason":"provider split near-verbatim retry"}'),
        takes,
    )
    assert result.groups == (("a", "b"), ("c",))
    assert "local_retry_reconciled" in result.reason


def test_incomplete_fragments_inside_retry_envelope_are_absorbed():
    takes = (
        CandidateTake("a", "src", 0, 98.6, 104.1, "We were laughing so much who could barely keep a straight face and the thing is that"),
        CandidateTake("x", "src", 0, 105.6, 106.6, "You know", complete_idea=False),
        CandidateTake("y", "src", 0, 109.7, 111.0, "Trying to say in character", complete_idea=False),
        CandidateTake("b", "src", 0, 113.1, 116.7, "We were laughing so much who could barely keep a straight face"),
        CandidateTake("c", "src", 0, 123.0, 128.0, "The next part of the story is completely different"),
    )
    result = safe_group_takes(
        _provider('{"groups":[["a"],["x"],["y"],["b"],["c"]],"reason":"provider split retry and debris"}'),
        takes,
    )
    assert result.groups[0] == ("a", "x", "y", "b")
    assert result.groups[1] == ("c",)
    assert "interstitial_retry_debris_absorbed" in result.reason


def test_short_incomplete_line_outside_retry_envelope_is_preserved():
    takes = (
        CandidateTake("a", "src", 0, 0.0, 4.0, "This is the line I want to retry exactly"),
        CandidateTake("b", "src", 0, 6.0, 10.0, "This is the line I want to retry exactly"),
        CandidateTake("x", "src", 0, 20.0, 21.0, "You know", complete_idea=False),
    )
    result = safe_group_takes(
        _provider('{"groups":[["a"],["b"],["x"]],"reason":"provider split exact retry"}'),
        takes,
    )
    assert result.groups == (("a", "b"), ("x",))
