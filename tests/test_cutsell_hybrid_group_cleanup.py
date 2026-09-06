from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult
from cutsell_worker.hybrid_group_cleanup import apply_hybrid_group_cleanup


def take(clip_id: str, text: str, start: float) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=start + 2.0,
        text=text,
    )


class Judge:
    def __init__(self, decisions):
        self.decisions = decisions
        self.calls = 0

    def judge(self, session):
        self.calls += 1
        return EditorialJudgeResult(
            decisions=tuple(self.decisions),
            provider="fake",
            model="flash-lite",
            requested=True,
            available=True,
            estimated_input_tokens=200,
            estimated_output_tokens=80,
        )


class BrokenJudge:
    def judge(self, session):
        raise RuntimeError("provider down")


def test_high_confidence_bts_is_removed_and_winner_is_preferred():
    members = (
        take("a", "why do I keep saying that it's stupid", 0.0),
        take("b", "This cardigan is so soft and comes in three colors", 3.0),
    )
    judge = Judge((
        EditorialDecision("a", "bts", 0.99, "recording_process_self_talk"),
        EditorialDecision("b", "winner", 0.97, "complete_audience_delivery"),
    ))
    result = apply_hybrid_group_cleanup(members, judge)
    assert [item.clip_id for item in result.deleted] == ["a"]
    assert [item.clip_id for item in result.kept] == ["b"]
    assert result.preferred_winner_id == "b"
    assert result.requested is True


def test_low_confidence_delete_fails_open_for_candidate():
    members = (
        take("a", "maybe self talk", 0.0),
        take("b", "valid product speech", 3.0),
    )
    judge = Judge((
        EditorialDecision("a", "bts", 0.70, "weak_bts_signal"),
        EditorialDecision("b", "winner", 0.95, "complete"),
    ))
    result = apply_hybrid_group_cleanup(members, judge)
    assert result.deleted == ()
    assert [item.clip_id for item in result.kept] == ["a", "b"]
    assert result.preferred_winner_id == "b"


def test_all_bts_group_can_be_removed_when_model_is_very_confident():
    members = (
        take("a", "what did I just say", 0.0),
        take("b", "okay what the fuck is happening", 3.0),
    )
    judge = Judge((
        EditorialDecision("a", "bts", 0.99, "self_review"),
        EditorialDecision("b", "bts", 0.99, "break_character"),
    ))
    result = apply_hybrid_group_cleanup(members, judge)
    assert result.kept == ()
    assert {item.clip_id for item in result.deleted} == {"a", "b"}
    assert result.preferred_winner_id is None


def test_missing_provider_preserves_local_group_without_call():
    members = (take("a", "valid product speech", 0.0),)
    result = apply_hybrid_group_cleanup(members, None)
    assert result.kept == members
    assert result.deleted == ()
    assert result.requested is False


def test_provider_failure_preserves_local_group():
    members = (
        take("a", "valid one", 0.0),
        take("b", "valid two", 3.0),
    )
    result = apply_hybrid_group_cleanup(members, BrokenJudge())
    assert result.kept == members
    assert result.deleted == ()
    assert result.preferred_winner_id is None
