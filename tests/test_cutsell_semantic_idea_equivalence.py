from cutsell_worker.semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalencePair,
    IdeaEquivalenceRequest,
    IdeaEquivalenceResult,
    SemanticEquivalenceGatePolicy,
    safe_check_idea_equivalence,
    same_idea_by_pair_index,
    should_request_semantic_equivalence,
    validate_idea_equivalence_result,
)


def _pair(left: str = "left text", right: str = "right text") -> IdeaEquivalencePair:
    return IdeaEquivalencePair(left_text=left, right_text=right)


def _request(count: int = 1) -> IdeaEquivalenceRequest:
    return IdeaEquivalenceRequest(pairs=tuple(_pair(f"l{i}", f"r{i}") for i in range(count)))


class GoodArbiter:
    def __init__(self, decisions):
        self.calls = 0
        self._decisions = decisions

    def check(self, request):
        self.calls += 1
        return IdeaEquivalenceResult(
            decisions=self._decisions,
            provider="fake",
            model="fake-semantic-equivalence",
            requested=True,
            available=True,
            estimated_input_tokens=200,
            estimated_output_tokens=40,
        )


class BrokenArbiter:
    def __init__(self):
        self.calls = 0

    def check(self, request):
        self.calls += 1
        raise RuntimeError("provider exploded")


class UnknownIndexArbiter:
    def check(self, request):
        return IdeaEquivalenceResult(
            decisions=(IdeaEquivalenceDecision(pair_index=99, same_idea=True, confidence=0.9),),
            provider="fake",
            model="broken",
            requested=True,
            available=True,
        )


class OverBudgetArbiter:
    def check(self, request):
        return IdeaEquivalenceResult(
            decisions=(IdeaEquivalenceDecision(pair_index=0, same_idea=True, confidence=0.9),),
            provider="fake",
            model="too-expensive",
            requested=True,
            available=True,
            estimated_input_tokens=50_000,
            estimated_output_tokens=10,
        )


def test_gate_declines_empty_request():
    assert should_request_semantic_equivalence(IdeaEquivalenceRequest(pairs=())) is False


def test_gate_declines_over_max_pairs():
    policy = SemanticEquivalenceGatePolicy(max_pairs_per_request=2)
    assert should_request_semantic_equivalence(_request(3), policy) is False
    assert should_request_semantic_equivalence(_request(2), policy) is True


def test_safe_check_returns_unavailable_without_arbiter():
    result = safe_check_idea_equivalence(None, _request(1))
    assert result.available is False
    assert result.requested is False
    assert same_idea_by_pair_index(result) == {}


def test_safe_check_declines_when_gate_rejects_request():
    policy = SemanticEquivalenceGatePolicy(max_pairs_per_request=1)
    arbiter = GoodArbiter((IdeaEquivalenceDecision(0, True, 0.9),))
    result = safe_check_idea_equivalence(arbiter, _request(2), policy)
    assert result.available is False
    assert arbiter.calls == 0


def test_safe_check_returns_decisions_when_arbiter_succeeds():
    arbiter = GoodArbiter((
        IdeaEquivalenceDecision(0, True, 0.91, "same story beat"),
    ))
    result = safe_check_idea_equivalence(arbiter, _request(1))
    assert result.available is True
    assert arbiter.calls == 1
    lookup = same_idea_by_pair_index(result)
    assert lookup[0] == (True, 0.91, "same story beat")


def test_safe_check_fails_open_on_provider_exception():
    arbiter = BrokenArbiter()
    result = safe_check_idea_equivalence(arbiter, _request(1))
    assert result.available is False
    assert arbiter.calls == 1
    assert same_idea_by_pair_index(result) == {}


def test_safe_check_fails_open_on_unknown_pair_index():
    result = safe_check_idea_equivalence(UnknownIndexArbiter(), _request(1))
    assert result.available is False
    assert same_idea_by_pair_index(result) == {}


def test_safe_check_fails_open_when_over_token_budget():
    result = safe_check_idea_equivalence(OverBudgetArbiter(), _request(1))
    assert result.available is False


def test_validate_rejects_confidence_outside_unit_interval():
    request = _request(1)
    bad = IdeaEquivalenceResult(
        decisions=(IdeaEquivalenceDecision(0, True, 1.5),),
        provider="fake",
        model="fake",
        requested=True,
        available=True,
    )
    try:
        validate_idea_equivalence_result(request, bad)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_validate_rejects_missing_pair():
    request = _request(2)
    incomplete = IdeaEquivalenceResult(
        decisions=(IdeaEquivalenceDecision(0, True, 0.9),),
        provider="fake",
        model="fake",
        requested=True,
        available=True,
    )
    try:
        validate_idea_equivalence_result(request, incomplete)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_validate_truncates_long_reason_and_normalizes_order():
    request = _request(2)
    result = IdeaEquivalenceResult(
        decisions=(
            IdeaEquivalenceDecision(1, False, 0.2, "x" * 500),
            IdeaEquivalenceDecision(0, True, 0.8, "short"),
        ),
        provider="fake",
        model="fake",
        requested=True,
        available=True,
    )
    normalized = validate_idea_equivalence_result(request, result)
    assert [d.pair_index for d in normalized.decisions] == [0, 1]
    assert len(normalized.decisions[1].reason) == 200


def test_same_idea_lookup_is_empty_when_result_unavailable():
    unavailable = IdeaEquivalenceResult((), "none", "none", False, False)
    assert same_idea_by_pair_index(unavailable) == {}
