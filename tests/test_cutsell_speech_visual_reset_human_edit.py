from types import SimpleNamespace

import cutsell_worker.speech_visual_microtrim as svm


class _FakeASR:
    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, *args, **kwargs):
        left = SimpleNamespace(text="idea", start=0.20, end=1.00)
        right = SimpleNamespace(text="continua", start=2.00, end=2.50)
        return [SimpleNamespace(words=(left, right))]


def test_longer_human_style_reset_gap_is_trimmed_when_speech_safe(monkeypatch):
    monkeypatch.setattr(svm, "FasterWhisperASR", _FakeASR)
    monkeypatch.setattr(svm, "_silences", lambda path: ((1.00, 2.00),))
    monkeypatch.setattr(
        svm,
        "_visual_reset_onset",
        lambda path, left_word_end, safe_start, safe_end: (
            1.18,
            {"reason": "persistent_face_or_gesture_reset_after_word"},
        ),
    )

    cuts, diag = svm.detect_speech_safe_visual_microtrims("unused.mp4")

    assert len(cuts) == 1
    cut = cuts[0]
    assert cut["start"] == 1.18
    assert cut["end"] == 1.955
    assert cut["duration_sec"] == 0.775
    assert cut["left_word_end"] == 1.0
    assert cut["right_word_start"] == 2.0
    assert diag["speech_lock_ok"] is True
    assert diag["max_reset_gap_sec"] == 1.25


def test_long_reset_gap_fails_open_without_enough_quiet(monkeypatch):
    monkeypatch.setattr(svm, "FasterWhisperASR", _FakeASR)
    # Only a small part of the safe interval is acoustically quiet.
    monkeypatch.setattr(svm, "_silences", lambda path: ((1.05, 1.35),))
    monkeypatch.setattr(
        svm,
        "_visual_reset_onset",
        lambda *args, **kwargs: (
            1.18,
            {"reason": "persistent_face_or_gesture_reset_after_word"},
        ),
    )

    cuts, diag = svm.detect_speech_safe_visual_microtrims("unused.mp4")

    assert cuts == ()
    assert diag["speech_lock_ok"] is True


def test_gap_beyond_human_reset_window_is_never_trimmed(monkeypatch):
    class _WideGapASR(_FakeASR):
        def transcribe(self, *args, **kwargs):
            left = SimpleNamespace(text="idea", start=0.20, end=1.00)
            right = SimpleNamespace(text="continua", start=2.40, end=2.90)
            return [SimpleNamespace(words=(left, right))]

    monkeypatch.setattr(svm, "FasterWhisperASR", _WideGapASR)
    monkeypatch.setattr(svm, "_silences", lambda path: ((1.00, 2.40),))
    monkeypatch.setattr(
        svm,
        "_visual_reset_onset",
        lambda *args, **kwargs: (1.15, {"reason": "persistent_face_or_gesture_reset_after_word"}),
    )

    cuts, _ = svm.detect_speech_safe_visual_microtrims("unused.mp4")

    assert cuts == ()
