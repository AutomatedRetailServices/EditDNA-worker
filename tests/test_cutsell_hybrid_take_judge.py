from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult
from cutsell_worker.hybrid_take_judge import HybridTakeJudgeProvider


def take(clip_id: str, text: str, start: float, end: float) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="source-a",
        source_order=0,
        start=start,
        end=end,
        text=text,
    )


class WinnerJudge:
    def __init__(self, winner_id: str, confidence: float = 0.96):
        self.winner_id = winner_id
        self.confidence = confidence
        self.calls = 0

    def judge(self, session):
        self.calls += 1
        decisions = []
        for candidate in session.candidates:
            if candidate.clip_id == self.winner_id:
                decisions.append(EditorialDecision(candidate.clip_id, "winner", self.confidence, "complete_retry"))
            else:
                decisions.append(EditorialDecision(candidate.clip_id, "failed", 0.95, "failed_attempt"))
        return EditorialJudgeResult(
            tuple(decisions),
            provider="fake",
            model="fake-editorial",
            requested=True,
            available=True,
            estimated_input_tokens=800,
            estimated_output_tokens=100,
        )


class AmbiguousJudge:
    def judge(self, session):
        return EditorialJudgeResult(
            tuple(EditorialDecision(c.clip_id, "winner", 0.92, "ambiguous") for c in session.candidates),
            provider="fake",
            model="fake-editorial",
            requested=True,
            available=True,
            estimated_input_tokens=800,
            estimated_output_tokens=100,
        )


def test_no_editorial_provider_is_zero_cost_baseline():
    takes = (
        take("a", "this is the first take", 0.0, 2.0),
        take("b", "this is the complete second take", 2.5, 5.0),
    )
    result = HybridTakeJudgeProvider(editorial_judge=None).rank(takes)
    assert {item.clip_id for item in result.ranked} == {"a", "b"}
    assert result.status.requested is False
    assert result.status.status in {"hybrid_local", "baseline_single_candidate"}


def test_model_can_promote_one_confident_winner_inside_same_group():
    takes = (
        take("a", "I love this wait", 0.0, 1.5),
        take("b", "I love this because it fits perfectly", 2.0, 5.0),
    )
    judge = WinnerJudge("b")
    result = HybridTakeJudgeProvider(editorial_judge=judge).rank(takes)
    assert judge.calls >= 1
    assert result.ranked[0].clip_id == "b"
    assert result.status.status == "applied"
    assert result.ranked[0].reason.startswith("hybrid_editorial_winner:")


def test_low_confidence_model_cannot_override_local_best_take():
    takes = (
        take("a", "first complete take with detail", 0.0, 3.0),
        take("b", "short retry", 3.2, 4.0),
    )
    baseline = HybridTakeJudgeProvider(editorial_judge=None).rank(takes)
    judge = WinnerJudge("b", confidence=0.70)
    hybrid = HybridTakeJudgeProvider(editorial_judge=judge).rank(takes)
    assert hybrid.ranked[0].clip_id == baseline.ranked[0].clip_id


def test_multiple_model_winners_fail_open_to_local_ranking():
    takes = (
        take("a", "first take", 0.0, 1.2),
        take("b", "second take", 1.4, 2.8),
    )
    baseline = HybridTakeJudgeProvider(editorial_judge=None).rank(takes)
    hybrid = HybridTakeJudgeProvider(editorial_judge=AmbiguousJudge()).rank(takes)
    assert hybrid.ranked[0].clip_id == baseline.ranked[0].clip_id
    assert hybrid.status.status == "hybrid_fallback"


def test_single_candidate_never_calls_semantic_judge():
    judge = WinnerJudge("a")
    result = HybridTakeJudgeProvider(editorial_judge=judge).rank((take("a", "valid speech", 0.0, 2.0),))
    assert judge.calls == 0
    assert result.ranked[0].clip_id == "a"
    assert result.status.status == "baseline_single_candidate"
