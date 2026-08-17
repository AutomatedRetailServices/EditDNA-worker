from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult
from cutsell_worker.hybrid_session_cleanup import apply_hybrid_session_cleanup


def take(index: int) -> CandidateTake:
    return CandidateTake(
        clip_id=f"clip-{index}",
        source_asset_id="src",
        source_order=0,
        start=float(index * 3),
        end=float(index * 3 + 2),
        text=f"candidate speech number {index}",
    )


class BatchJudge:
    def __init__(self):
        self.sessions = []

    def judge(self, session):
        self.sessions.append(session)
        return EditorialJudgeResult(
            decisions=tuple(
                EditorialDecision(
                    candidate.clip_id,
                    "bts" if candidate.clip_id == "clip-13" else "keep",
                    0.99,
                    "recording_process" if candidate.clip_id == "clip-13" else "valid_speech",
                )
                for candidate in session.candidates
            ),
            provider="fake",
            model="flash-lite",
            requested=True,
            available=True,
            estimated_input_tokens=300,
            estimated_output_tokens=100,
        )


def test_long_creator_partition_uses_compact_default_chunks_and_covers_every_candidate_once():
    takes = tuple(take(index) for index in range(25))
    judge = BatchJudge()
    result = apply_hybrid_session_cleanup(takes, None, judge)

    assert [len(session.candidates) for session in judge.sessions] == [12, 12, 1]
    seen = [candidate.clip_id for session in judge.sessions for candidate in session.candidates]
    assert seen == [item.clip_id for item in takes]
    assert result.requested_chunk_count == 3
    assert result.available_chunk_count == 3
    assert [item.clip_id for item in result.deleted] == ["clip-13"]
    assert all(session.task == "classify_recording_process_within_single_creator_session" for session in judge.sessions)


def test_default_chunk_size_never_exceeds_hybrid_gate_hard_limit():
    takes = tuple(take(index) for index in range(30))
    judge = BatchJudge()
    apply_hybrid_session_cleanup(takes, None, judge)
    assert max(len(session.candidates) for session in judge.sessions) <= 14


def test_no_provider_is_zero_cost_and_preserves_everything():
    takes = tuple(take(index) for index in range(4))
    result = apply_hybrid_session_cleanup(takes, None, None)
    assert result.kept == takes
    assert result.deleted == ()
    assert result.requested_chunk_count == 0
