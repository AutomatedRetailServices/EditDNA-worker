from cutsell_worker.hybrid_editorial import (
    EditorialCandidate,
    EditorialDecision,
    EditorialJudgeResult,
    EditorialSession,
    HybridGatePolicy,
    resolve_hybrid_labels,
    safe_editorial_judge,
    should_request_editorial_judge,
)


def candidate(clip_id: str, local_label: str = "keep", confidence: float = 0.75):
    return EditorialCandidate(
        clip_id=clip_id,
        text=f"speech for {clip_id}",
        start=0.0,
        end=2.0,
        local_label=local_label,
        local_confidence=confidence,
    )


def session(*, confidence: float, conflict: float = 0.0):
    return EditorialSession(
        session_id="session-1",
        source_asset_id="source-1",
        candidates=(candidate("a", "failed"), candidate("b", "winner")),
        local_confidence=confidence,
        conflict_score=conflict,
    )


class GoodJudge:
    def __init__(self):
        self.calls = 0

    def judge(self, item):
        self.calls += 1
        return EditorialJudgeResult(
            decisions=(
                EditorialDecision("a", "bts", 0.95, "recording_process_meta"),
                EditorialDecision("b", "winner", 0.96, "complete_delivery"),
            ),
            provider="fake",
            model="fake-editorial",
            requested=True,
            available=True,
            estimated_input_tokens=900,
            estimated_output_tokens=90,
        )


class UnknownClipJudge:
    def __init__(self):
        self.calls = 0

    def judge(self, item):
        self.calls += 1
        return EditorialJudgeResult(
            decisions=(
                EditorialDecision("outside-session", "failed", 0.99, "bad"),
                EditorialDecision("b", "winner", 0.99, "good"),
            ),
            provider="fake",
            model="broken",
            requested=True,
            available=True,
        )


class OverBudgetJudge:
    def __init__(self):
        self.calls = 0

    def judge(self, item):
        self.calls += 1
        return EditorialJudgeResult(
            decisions=(
                EditorialDecision("a", "failed", 0.99, "failed_attempt"),
                EditorialDecision("b", "winner", 0.99, "complete_delivery"),
            ),
            provider="fake",
            model="too-expensive",
            requested=True,
            available=True,
            estimated_input_tokens=50_000,
            estimated_output_tokens=100,
        )


def test_high_confidence_low_conflict_session_stays_local():
    item = session(confidence=0.96, conflict=0.05)
    assert should_request_editorial_judge(item) is False
    judge = GoodJudge()
    result = safe_editorial_judge(judge, item)
    assert result.requested is False
    assert judge.calls == 0


def test_ambiguous_session_requests_semantic_judge():
    item = session(confidence=0.74)
    assert should_request_editorial_judge(item) is True
    judge = GoodJudge()
    result = safe_editorial_judge(judge, item)
    assert result.available is True
    assert judge.calls == 1
    assert [decision.label for decision in result.decisions] == ["bts", "winner"]


def test_conflicting_high_confidence_session_still_requests_judge():
    item = session(confidence=0.95, conflict=0.45)
    assert should_request_editorial_judge(item) is True


def test_provider_cannot_reference_clip_outside_mini_session():
    item = session(confidence=0.70)
    judge = UnknownClipJudge()
    result = safe_editorial_judge(judge, item)
    assert judge.calls == 2
    assert result.available is False
    assert result.requested is True
    assert resolve_hybrid_labels(item, result) == {"a": "failed", "b": "winner"}


def test_token_budget_is_hard_guard_even_for_valid_semantics():
    item = session(confidence=0.70)
    policy = HybridGatePolicy(max_estimated_input_tokens=1_000)
    judge = OverBudgetJudge()
    result = safe_editorial_judge(judge, item, policy)
    assert judge.calls == 2
    assert result.available is False
    assert resolve_hybrid_labels(item, result) == {"a": "failed", "b": "winner"}


def test_low_confidence_model_decision_does_not_override_local_brain():
    item = session(confidence=0.70)
    result = EditorialJudgeResult(
        decisions=(
            EditorialDecision("a", "keep", 0.55, "weak_model_opinion"),
            EditorialDecision("b", "failed", 0.50, "weak_model_opinion"),
        ),
        provider="fake",
        model="fake",
        requested=True,
        available=True,
    )
    assert resolve_hybrid_labels(item, result) == {"a": "failed", "b": "winner"}


def test_high_confidence_model_semantics_can_override_local_label_only():
    item = session(confidence=0.70)
    result = EditorialJudgeResult(
        decisions=(
            EditorialDecision("a", "bts", 0.95, "self_directed_recording_meta"),
            EditorialDecision("b", "winner", 0.95, "complete_product_delivery"),
        ),
        provider="fake",
        model="fake",
        requested=True,
        available=True,
    )
    resolved = resolve_hybrid_labels(item, result)
    assert resolved == {"a": "bts", "b": "winner"}
    # Timing remains owned by deterministic candidates, not the judge.
    assert item.candidates[0].start == 0.0
    assert item.candidates[0].end == 2.0
