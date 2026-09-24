from cutsell_worker.attempt_reconstruction import reconstruct_delivery_attempts
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _context(*events: TemporalEvent) -> WholeVideoContext:
    return WholeVideoContext(
        sources=(
            SourceVideoContext(
                source_asset_id="src-1",
                summary="",
                dominant_style="talking_head",
                creator_intent="deliver one explanation",
                events=tuple(events),
            ),
        ),
        status=ProviderStatus("test", True, True, "applied"),
    )


def _take(clip_id: str, start: float, end: float, text: str) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src-1",
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=True,
    )


def test_tiny_nonterminal_tail_stays_with_delivery_despite_strong_body_reset():
    left = _take("left", 10.0, 18.0, "The scan always showed that everything was functioning")
    tail = _take("tail", 19.0, 19.55, "normally.")
    reset = TemporalEvent(
        source_asset_id="src-1",
        start=18.55,
        end=19.10,
        kind="body_reset",
        confidence=0.99,
        description="brief posture reset during the pause",
    )

    attempts, diagnostics = reconstruct_delivery_attempts((left, tail), _context(reset))

    assert len(attempts) == 1
    assert attempts[0].text.endswith("functioning normally.")
    assert diagnostics["merged_fragment_count"] == 1
    assert diagnostics["boundaries"] == []


def test_tiny_tail_after_terminal_sentence_can_still_form_reset_boundary():
    left = _take("left", 10.0, 18.0, "That delivery is already complete.")
    tail = _take("tail", 19.0, 19.55, "Okay.")
    reset = TemporalEvent(
        source_asset_id="src-1",
        start=18.55,
        end=19.10,
        kind="body_reset",
        confidence=0.99,
        description="creator resets before the next try",
    )

    attempts, diagnostics = reconstruct_delivery_attempts((left, tail), _context(reset))

    assert len(attempts) == 2
    assert diagnostics["boundaries"][0]["reason"] == "pause_plus_strong_reset"


def test_mid_sentence_pause_with_a_strong_reset_does_not_split_a_continuing_sentence():
    """D-291.12 (RAW #127 shape): '... y me comenzó' | 'a hacer ruido.' with a
    0.8 s pause and a 1.0-confidence hand-motion candidate in it: the right
    side continues the sentence (non-terminal left, lower-case right, no
    restart), so this is a mid-sentence hold, not an attempt boundary."""
    left = _take("left", 54.27, 56.867, "si tuve pérdida de pelo y me comenzó")
    right = _take("right", 57.67, 58.116, "a hacer ruido.")
    reset = TemporalEvent(
        source_asset_id="src-1", start=57.605, end=57.671,
        kind="hand_motion_reset_candidate", confidence=1.0, description="mic hand moves during the hold",
    )
    attempts, diagnostics = reconstruct_delivery_attempts((left, right), _context(reset))
    assert len(attempts) == 1
    assert attempts[0].text.endswith("y me comenzó a hacer ruido.")
    assert diagnostics["boundaries"] == []


def test_a_capitalised_restart_after_the_same_pause_still_forms_the_boundary():
    left = _take("left", 54.27, 56.867, "si tuve pérdida de pelo y me comenzó")
    right = _take("right", 57.67, 60.0, "Se me caía mucho el pelo cuando me lavaba.")
    reset = TemporalEvent(
        source_asset_id="src-1", start=57.605, end=57.671,
        kind="hand_motion_reset_candidate", confidence=1.0, description="reset",
    )
    attempts, diagnostics = reconstruct_delivery_attempts((left, right), _context(reset))
    assert len(attempts) == 2
    assert diagnostics["boundaries"][0]["reason"] == "pause_plus_strong_reset"
