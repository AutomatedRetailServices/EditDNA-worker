from cutsell_worker.contracts import CandidateTake, ProcessingRequest, SourceAsset
from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult
from cutsell_worker.pipeline import build_flow_b_draft


def take(clip_id: str, text: str, start: float) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=start + 2.5,
        text=text,
    )


def request() -> ProcessingRequest:
    source = SourceAsset(
        source_asset_id="src",
        project_id="project",
        user_id="user",
        original_name="raw.mp4",
        source_order=0,
        duration_sec=20.0,
        uri="s3://bucket/raw.mp4",
    )
    return ProcessingRequest(project_id="project", user_id="user", sources=(source,))


class FakeEditorialJudge:
    def __init__(self):
        self.calls = 0

    def judge(self, session):
        self.calls += 1
        decisions = []
        for candidate in session.candidates:
            if candidate.clip_id == "meta":
                decisions.append(EditorialDecision("meta", "bts", 0.99, "recording_process"))
            elif candidate.clip_id == "good":
                decisions.append(EditorialDecision("good", "winner", 0.99, "complete_take"))
            else:
                decisions.append(EditorialDecision(candidate.clip_id, "alternate", 0.95, "usable_retry"))
        return EditorialJudgeResult(
            decisions=tuple(decisions),
            provider="fake",
            model="flash-lite",
            requested=True,
            available=True,
            estimated_input_tokens=250,
            estimated_output_tokens=90,
        )


def test_hybrid_pipeline_discards_semantic_bts_and_keeps_preferred_winner():
    takes = (
        take("meta", "This cardigan is soft and comes in three colors", 0.0),
        take("good", "This cardigan is soft and comes in three colors today", 3.0),
    )
    judge = FakeEditorialJudge()
    result = build_flow_b_draft(request(), takes, editorial_judge=judge)

    assert judge.calls >= 1
    assert [clip.clip_id for clip in result.draft.selected] == ["good"]
    assert "meta" in {clip.clip_id for clip in result.draft.discarded}
    assert result.draft.diagnostics["hybrid_editorial_deleted_count"] == 1
    assert result.stage_status["hybrid_editorial"] == "provider_complete"


def test_pipeline_without_editorial_judge_stays_local_and_does_not_semantically_delete():
    takes = (
        take("a", "This cardigan is soft and comes in three colors", 0.0),
        take("b", "This cardigan is soft and comes in three colors today", 3.0),
    )
    result = build_flow_b_draft(request(), takes, editorial_judge=None)
    all_ids = {clip.clip_id for clip in (*result.draft.selected, *result.draft.alternates)}
    assert all_ids == {"a", "b"}
    assert result.draft.diagnostics["hybrid_editorial_deleted_count"] == 0
    assert result.stage_status["hybrid_editorial"] == "disabled_local_only"
