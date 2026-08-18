from cutsell_worker.contracts import CandidateTake, SourceAsset, TranscriptSegment
from cutsell_worker.hybrid_editorial import (
    EditorialCandidate,
    EditorialDecision,
    EditorialJudgeResult,
    EditorialSession,
)
from cutsell_worker.hybrid_payload import build_compact_editorial_payload
from cutsell_worker.hybrid_session_cleanup import apply_hybrid_session_cleanup
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, WholeVideoContext
from cutsell_worker.whole_video_local import RunPodLocalWholeVideoProvider


class CaptureJudge:
    def __init__(self):
        self.sessions = []

    def judge(self, session):
        self.sessions.append(session)
        return EditorialJudgeResult(
            decisions=tuple(
                EditorialDecision(candidate.clip_id, "keep", 0.99, "valid")
                for candidate in session.candidates
            ),
            provider="fake",
            model="flash-lite",
            requested=True,
            available=True,
            estimated_input_tokens=200,
            estimated_output_tokens=40,
        )


def _context():
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="full source transcript where the creator explains one idea, retries it, checks notes, then delivers it cleanly",
            dominant_style="talking_head",
            creator_intent="deliver one coherent scripted explanation",
            events=(),
            edit_mode="sales",
            sales_intent=0.92,
            main_topic="product explanation",
            product_or_subject="sleep product",
            story_logic="preserve the clean final delivery and remove abandoned retries or note consultations",
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def test_hybrid_session_receives_whole_source_context():
    takes = (
        CandidateTake("a", "src", 0, 0.0, 2.0, "first attempt of the idea"),
        CandidateTake("b", "src", 0, 3.0, 5.0, "clean final delivery of the idea"),
    )
    judge = CaptureJudge()
    apply_hybrid_session_cleanup(takes, _context(), judge)

    assert len(judge.sessions) == 1
    source_context = dict(judge.sessions[0].source_context)
    assert "full source transcript" in source_context["summary"]
    assert source_context["main_topic"] == "product explanation"
    assert "remove abandoned retries" in source_context["story_logic"]


def test_compact_payload_exposes_context_once_not_per_candidate():
    session = EditorialSession(
        session_id="s",
        source_asset_id="src",
        candidates=(
            EditorialCandidate("a", "first try", 0.0, 1.0, "keep", 0.5),
            EditorialCandidate("b", "clean retry", 2.0, 3.0, "keep", 0.5),
        ),
        local_confidence=0.5,
        conflict_score=0.5,
        task="classify_recording_process_within_single_creator_session",
        source_context=(("summary", "the same idea is attempted twice"), ("story_logic", "keep the coherent final delivery")),
    )
    payload = build_compact_editorial_payload(session)

    assert payload["source_context"]["summary"] == "the same idea is attempted twice"
    assert "full message/story" in " ".join(payload["rules"])
    assert "source_context" not in payload["candidates"][0]


def test_runpod_local_context_keeps_far_more_than_old_700_character_window():
    source = SourceAsset("src", "project", "user", "video.mp4", 0, 120.0, "s3://bucket/video.mp4")
    long_text = " ".join(f"sentence{i}" for i in range(600))
    transcript = TranscriptSegment("src", 0.0, 120.0, long_text)
    result = RunPodLocalWholeVideoProvider().analyze((source,), (transcript,), ())

    summary = result.sources[0].summary
    assert len(summary) > 700
    assert len(summary) <= 3600
    assert result.status.reason == "local_asr_full_message_context"
