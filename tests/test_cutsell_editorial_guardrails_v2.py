from cutsell_worker.contracts import CandidateTake, TranscriptSegment, Word
from cutsell_worker.hybrid_retry_completion_integrity import _safe_short_alternate_debris
from cutsell_worker import speech_visual_microtrim


def _take(clip_id, start, end, text, *, complete_idea=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete_idea,
    )


def test_complete_short_alternate_is_not_deleted_as_neighbor_debris():
    previous = _take("previous", 0.0, 3.0, "mandaron sonografia tiroides examen")
    candidate = _take("candidate", 4.0, 8.0, "hacer sonografia tiroides otros", complete_idea=True)
    following = _take("following", 9.0, 12.0, "otros hacer sonografia tiroides estudios")
    semantic = {"candidate": ("alternate", 0.85)}

    assert _safe_short_alternate_debris(candidate, previous, following, semantic) is False


def test_incomplete_short_alternate_can_still_be_removed_when_fully_covered():
    previous = _take("previous", 0.0, 3.0, "mandaron sonografia tiroides examen")
    candidate = _take("candidate", 4.0, 8.0, "hacer sonografia tiroides otros", complete_idea=False)
    following = _take("following", 9.0, 12.0, "otros hacer sonografia tiroides estudios")
    semantic = {"candidate": ("alternate", 0.85)}

    assert _safe_short_alternate_debris(candidate, previous, following, semantic) is True


def test_long_silent_visual_reset_gap_is_eligible_without_touching_words(monkeypatch):
    class FakeASR:
        def __init__(self, model_name):
            self.model_name = model_name

        def transcribe(self, path, source_asset_id, language_hint=None):
            return (
                TranscriptSegment(
                    source_asset_id=source_asset_id,
                    start=0.0,
                    end=2.2,
                    text="left right",
                    words=(
                        Word("left", 0.4, 1.0),
                        Word("right", 1.9, 2.2),
                    ),
                ),
            )

    monkeypatch.setattr(speech_visual_microtrim, "FasterWhisperASR", FakeASR)
    monkeypatch.setattr(speech_visual_microtrim, "_silences", lambda path: ((1.0, 1.9),))
    monkeypatch.setattr(
        speech_visual_microtrim,
        "_visual_reset_onset",
        lambda path, left_word_end, safe_start, safe_end: (
            1.15,
            {"reason": "persistent_face_or_gesture_reset_after_word"},
        ),
    )

    cuts, diagnostics = speech_visual_microtrim.detect_speech_safe_visual_microtrims("fake.mp4")

    assert diagnostics["detector_version"] == "v2_long_visual_slack"
    assert diagnostics["long_candidate_count"] == 1
    assert diagnostics["speech_lock_ok"] is True
    assert len(cuts) == 1
    assert cuts[0]["reason"] == "auto_speech_safe_long_post_take_visual_slack"
    assert cuts[0]["start"] > 1.0
    assert cuts[0]["end"] < 1.9
