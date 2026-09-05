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
    # D-081: "bts" at 0.99 confidence with NO local corroboration is a pure
    # semantic-judgment ("high_confidence_semantic") delete basis -- it may
    # no longer irreversibly remove the candidate. It stays kept, carrying
    # semantic_delete_recommended=True evidence for downstream authorities.
    assert result.deleted == ()
    decision_13 = next(
        item
        for diagnostic in result.diagnostics
        for item in diagnostic["decisions"]
        if item["clip_id"] == "clip-13"
    )
    assert decision_13["delete_basis"] == "high_confidence_semantic"
    assert decision_13["semantic_delete_recommended"] is True
    assert decision_13["applied_delete"] is False
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
    # D-081: "failed" + local corroboration is still a semantic-judgment
    # delete basis (D-080's exact root cause: the same corroboration signal
    # was present when the LLM instead labeled the real-world equivalent of
    # this take "winner" in a separate live run). It must remain KEPT and
    # inspectable, recorded only as semantic_delete_recommended evidence.
    item = take(1)
    context = context_for(TemporalEvent(
        "src", item.start + 1.5, item.end, "retry_setup", 0.86,
        "creator visibly resets after failed attempt",
    ))
    result = apply_hybrid_session_cleanup((item,), context, FixedJudge("failed", 0.85))
    assert result.kept == (item,)
    assert result.deleted == ()
    decision = result.diagnostics[0]["decisions"][0]
    assert decision["delete_basis"] == "semantic_failed_plus_local_performance"
    assert decision["local_failure_corroborated"] is True
    assert decision["semantic_delete_recommended"] is True
    assert decision["applied_delete"] is False


def test_multimodal_physical_reset_candidates_corroborate_failed_take():
    item = take(1)
    context = context_for(
        TemporalEvent("src", 3.2, 3.4, "hand_motion_reset_candidate", 0.96, "mic drops after fumble"),
        TemporalEvent("src", 3.5, 3.7, "body_reset_candidate", 0.94, "creator resets posture"),
        TemporalEvent("src", 3.4, 3.8, "facial_expression_shift_candidate", 0.88, "error expression"),
    )
    result = apply_hybrid_session_cleanup((item,), context, FixedJudge("failed", 0.84))
    # D-081: local corroboration remains recorded evidence, but no longer an
    # irreversible delete on its own -- see test above.
    assert result.kept == (item,)
    assert result.deleted == ()
    decision = result.diagnostics[0]["decisions"][0]
    assert decision["local_failure_corroborated"] is True
    assert decision["semantic_delete_recommended"] is True
    assert any(reason.startswith("multimodal_reset_cluster") for reason in decision["local_failure_reasons"])


def test_medium_high_failed_label_stays_fail_open_without_local_corroboration():
    item = take(1)
    result = apply_hybrid_session_cleanup((item,), None, FixedJudge("failed", 0.85))
    assert result.kept == (item,)
    assert result.deleted == ()
    decision = result.diagnostics[0]["decisions"][0]
    assert decision["delete_basis"] == "kept_fail_open"


def test_visual_fumble_can_corroborate_medium_high_failed_label():
    # D-081: corroboration is recorded, not destructive -- see above.
    signals = MediaSignals("src", 3.0, 5.0, visual_fumble=0.81)
    item = take(1, signals=signals)
    result = apply_hybrid_session_cleanup((item,), None, FixedJudge("failed", 0.86))
    assert result.kept == (item,)
    assert result.deleted == ()
    decision = result.diagnostics[0]["decisions"][0]
    assert decision["semantic_delete_recommended"] is True


def test_medium_high_bts_deletes_when_local_performance_corroborates():
    # D-081: "bts" + local corroboration is also semantic-judgment -- recorded,
    # not destructive.
    signals = MediaSignals("src", 3.0, 5.0, visual_fumble=0.90)
    item = take(1, signals=signals)
    result = apply_hybrid_session_cleanup((item,), None, FixedJudge("bts", 0.86))
    assert result.kept == (item,)
    assert result.deleted == ()
    decision = result.diagnostics[0]["decisions"][0]
    assert decision["delete_basis"] == "semantic_bts_plus_local_performance"
    assert decision["semantic_delete_recommended"] is True
    assert decision["applied_delete"] is False


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
    # D-081: the cluster corroboration bases (semantic_bts_plus_local_
    # performance, semantic_failed_plus_local_performance,
    # semantic_bts_inside_corroborated_failure_cluster) are all semantic
    # judgment -- none may delete early anymore. All four takes stay kept,
    # with the dense-cluster evidence still recorded for downstream use.
    assert result.deleted == ()
    assert {item.clip_id for item in result.kept} == {"clip-0", "clip-1", "clip-2", "clip-3"}
    c_decision = next(item for item in result.diagnostics[0]["decisions"] if item["clip_id"] == "clip-2")
    assert c_decision["delete_basis"] == "semantic_bts_inside_corroborated_failure_cluster"
    assert c_decision["semantic_delete_recommended"] is True
    assert c_decision["applied_delete"] is False
    assert c_decision["dense_semantic_failure_cluster"] is True


def test_one_word_failed_debris_can_delete_at_point_eight_with_local_fumble():
    signals = MediaSignals("src", 0.0, 0.7, visual_fumble=0.90)
    item = take(0, signals=signals, text="uh", duration=0.7)
    result = apply_hybrid_session_cleanup((item,), None, FixedJudge("failed", 0.80))
    assert result.deleted == (item,)
    assert result.diagnostics[0]["decisions"][0]["delete_basis"] == "micro_failed_plus_local_performance"


def test_d072_replacement_guard_diagnostics_are_wired_end_to_end():
    """D-072: the guard's observability fields must reach the real
    per-decision diagnostic dict apply_hybrid_session_cleanup produces,
    not just the direct unit-level _later_semantic_retry_replacement
    call. Reproduces the exact D-070 shape (a complete-idea failed take
    whose topically-overlapping later candidate fails sequence identity)
    end to end and confirms both the unchanged decision AND the new
    explanatory fields."""
    failed = take(
        0, text="Al terminar mi contrato, le pedí a mi ginecóloga.",
    )
    candidate = take(
        1,
        text=(
            "Al terminar mi contrato, cambié de ginecóloga y le pedí que me "
            "hiciera un test de todo lo que ella se pudiera imaginar y me "
            "pudiese indicar."
        ),
    )
    labels = {"clip-0": ("failed", 0.95), "clip-1": ("keep", 0.95)}
    result = apply_hybrid_session_cleanup((failed, candidate), None, MappingJudge(labels))

    decision = next(item for item in result.diagnostics[0]["decisions"] if item["clip_id"] == "clip-0")

    # Decision itself: unchanged from pre-D-072 behavior.
    assert decision["later_retry_replacement_id"] is None
    assert decision["later_retry_semantic_overlap"] == 1.0

    # New, additive-only observability.
    assert decision["replacement_candidate_clip_id_before_guard"] == "clip-1"
    assert round(decision["sequence_identity"], 4) == 0.4108
    assert decision["sequence_identity_threshold"] == 0.52
    assert decision["lexical_identity_passed"] is False
    assert decision["replacement_rejection_reason"] == "SEQUENCE_IDENTITY_BELOW_THRESHOLD"


def test_d072_diagnostics_default_to_not_applicable_for_non_failed_decisions():
    """Decisions this guard was never invoked for (label != 'failed') must
    carry the explicit NOT_APPLICABLE reason, never a stale value from a
    different clip's own guard evaluation."""
    item = take(0)
    result = apply_hybrid_session_cleanup((item,), None, FixedJudge("keep", 0.95))
    decision = result.diagnostics[0]["decisions"][0]
    assert decision["replacement_candidate_clip_id_before_guard"] is None
    assert decision["sequence_identity"] is None
    assert decision["lexical_identity_passed"] is None
    assert decision["replacement_rejection_reason"] == "NOT_APPLICABLE"
