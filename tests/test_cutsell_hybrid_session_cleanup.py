from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult
from cutsell_worker.hybrid_session_cleanup import apply_hybrid_session_cleanup
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def take(index: int, *, signals: MediaSignals | None = None, text: str | None = None, duration: float = 2.0) -> CandidateTake:
    return CandidateTake(
        clip_id=f"clip-{index}",
        source_asset_id="src",
        source_order=0,
        start=float(index * 3),
        end=float(index * 3 + duration),
        text=text or f"candidate speech number {index}",
        signals=signals,
    )


def context_for(*events: TemporalEvent) -> WholeVideoContext:
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="creator records a complete product story with retries and a final clean delivery",
            dominant_style="talking_head",
            creator_intent="deliver a clean take",
            events=tuple(events),
            edit_mode="natural",
            sales_intent=0.0,
            main_topic="story",
            product_or_subject="product",
            story_logic="retry then successful delivery",
        ),),
        status=ProviderStatus("test", True, True, "applied"),
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


class FixedJudge:
    def __init__(self, label: str, confidence: float):
        self.label = label
        self.confidence = confidence

    def judge(self, session):
        return EditorialJudgeResult(
            decisions=tuple(
                EditorialDecision(candidate.clip_id, self.label, self.confidence, "test")
                for candidate in session.candidates
            ),
            provider="fake",
            model="flash-lite",
            requested=True,
            available=True,
            estimated_input_tokens=100,
            estimated_output_tokens=50,
        )


class MappingJudge:
    def __init__(self, labels):
        self.labels = labels

    def judge(self, session):
        return EditorialJudgeResult(
            decisions=tuple(
                EditorialDecision(candidate.clip_id, *self.labels[candidate.clip_id], "test")
                for candidate in session.candidates
            ),
            provider="fake",
            model="flash-lite",
            requested=True,
            available=True,
            estimated_input_tokens=100,
            estimated_output_tokens=50,
        )


def test_long_creator_partition_uses_overlapping_story_windows():
    takes = tuple(take(index) for index in range(25))
    judge = BatchJudge()
    result = apply_hybrid_session_cleanup(takes, None, judge)

    assert [len(session.candidates) for session in judge.sessions] == [10, 10, 10, 10]
    windows = [[candidate.clip_id for candidate in session.candidates] for session in judge.sessions]
    assert windows[0] == [f"clip-{index}" for index in range(10)]
    assert windows[1] == [f"clip-{index}" for index in range(5, 15)]
    assert windows[-1] == [f"clip-{index}" for index in range(15, 25)]
    assert result.requested_chunk_count == 4
    assert result.available_chunk_count == 4
    assert [item.clip_id for item in result.deleted] == ["clip-13"]
    assert all(session.task == "classify_recording_process_within_single_creator_session" for session in judge.sessions)
    # Overlap must not duplicate downstream semantic decisions.
    assert len(result.semantic_decisions) == 25


def test_default_window_size_stays_below_structured_output_failure_envelope():
    takes = tuple(take(index) for index in range(30))
    judge = BatchJudge()
    apply_hybrid_session_cleanup(takes, None, judge)
    assert max(len(session.candidates) for session in judge.sessions) == 10


def test_no_provider_is_zero_cost_and_preserves_everything():
    takes = tuple(take(index) for index in range(4))
    result = apply_hybrid_session_cleanup(takes, None, None)
    assert result.kept == takes
    assert result.deleted == ()
    assert result.requested_chunk_count == 0
    assert result.semantic_decisions == ()


def test_medium_high_failed_label_deletes_when_retry_event_corroborates_locally():
    item = take(1)
    context = context_for(TemporalEvent(
        "src", item.start + 1.5, item.end, "retry_setup", 0.86,
        "creator visibly resets after failed attempt",
    ))
    result = apply_hybrid_session_cleanup((item,), context, FixedJudge("failed", 0.85))
    assert result.kept == ()
    assert result.deleted == (item,)
    decision = result.diagnostics[0]["decisions"][0]
    assert decision["delete_basis"] == "semantic_failed_plus_local_performance"
    assert decision["local_failure_corroborated"] is True


def test_multimodal_physical_reset_candidates_corroborate_failed_take():
    item = take(1)
    context = context_for(
        TemporalEvent("src", 3.2, 3.4, "hand_motion_reset_candidate", 0.96, "mic drops after fumble"),
        TemporalEvent("src", 3.5, 3.7, "body_reset_candidate", 0.94, "creator resets posture"),
        TemporalEvent("src", 3.4, 3.8, "facial_expression_shift_candidate", 0.88, "error expression"),
    )
    result = apply_hybrid_session_cleanup((item,), context, FixedJudge("failed", 0.84))
    assert result.deleted == (item,)
    decision = result.diagnostics[0]["decisions"][0]
    assert decision["local_failure_corroborated"] is True
    assert any(reason.startswith("multimodal_reset_cluster") for reason in decision["local_failure_reasons"])


def test_medium_high_failed_label_stays_fail_open_without_local_corroboration():
    item = take(1)
    result = apply_hybrid_session_cleanup((item,), None, FixedJudge("failed", 0.85))
    assert result.kept == (item,)
    assert result.deleted == ()
    decision = result.diagnostics[0]["decisions"][0]
    assert decision["delete_basis"] == "kept_fail_open"


def test_visual_fumble_can_corroborate_medium_high_failed_label():
    signals = MediaSignals("src", 3.0, 5.0, visual_fumble=0.81)
    item = take(1, signals=signals)
    result = apply_hybrid_session_cleanup((item,), None, FixedJudge("failed", 0.86))
    assert result.deleted == (item,)


def test_medium_high_bts_deletes_when_local_performance_corroborates():
    signals = MediaSignals("src", 3.0, 5.0, visual_fumble=0.90)
    item = take(1, signals=signals)
    result = apply_hybrid_session_cleanup((item,), None, FixedJudge("bts", 0.86))
    assert result.kept == ()
    assert result.deleted == (item,)
    assert result.diagnostics[0]["decisions"][0]["delete_basis"] == "semantic_bts_plus_local_performance"


def test_medium_high_bts_still_fails_open_when_isolated_and_uncorroborated():
    item = take(1)
    result = apply_hybrid_session_cleanup((item,), None, FixedJudge("bts", 0.86))
    assert result.kept == (item,)
    assert result.deleted == ()


def test_dense_bts_cluster_can_corroborate_one_member_without_local_signal():
    signals = MediaSignals("src", 0.0, 2.0, visual_fumble=0.90)
    a = take(0, signals=signals)
    b = take(1, signals=MediaSignals("src", 3.0, 5.0, visual_fumble=0.90))
    c = take(2)
    d = take(3)
    labels = {
        "clip-0": ("bts", 0.86),
        "clip-1": ("failed", 0.85),
        "clip-2": ("bts", 0.84),
        "clip-3": ("keep", 0.90),
    }
    result = apply_hybrid_session_cleanup((a, b, c, d), None, MappingJudge(labels))
    assert {item.clip_id for item in result.deleted} == {"clip-0", "clip-1", "clip-2"}
    c_decision = next(item for item in result.diagnostics[0]["decisions"] if item["clip_id"] == "clip-2")
    assert c_decision["delete_basis"] == "semantic_bts_inside_corroborated_failure_cluster"


def test_one_word_failed_debris_can_delete_at_point_eight_with_local_fumble():
    signals = MediaSignals("src", 0.0, 0.7, visual_fumble=0.90)
    item = take(0, signals=signals, text="uh", duration=0.7)
    result = apply_hybrid_session_cleanup((item,), None, FixedJudge("failed", 0.80))
    assert result.deleted == (item,)
    assert result.diagnostics[0]["decisions"][0]["delete_basis"] == "micro_failed_plus_local_performance"
