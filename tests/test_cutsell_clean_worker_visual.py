from types import SimpleNamespace

from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.frame_sampling import sample_take_frames
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.visual_analysis import (
    VisualObservation,
    VisualProviderResult,
    apply_visual_observations,
    safe_visual_analyze,
)


def _take():
    return CandidateTake(
        clip_id="clip-1",
        source_asset_id="src-1",
        source_order=0,
        start=2.0,
        end=5.0,
        text="This is my best take",
        signals=MediaSignals("src-1", 2.0, 5.0, silence_ratio=0.1, audio_quality=0.8),
    )


def test_frame_sampling_stays_inside_take_and_source(tmp_path):
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    frames = sample_take_frames("raw.mov", _take(), str(tmp_path), frame_count=3, runner=runner)
    assert len(frames) == 3
    assert all(frame.source_asset_id == "src-1" for frame in frames)
    assert all(2.0 < frame.timestamp < 5.0 for frame in frames)
    assert [round(frame.timestamp, 2) for frame in frames] == [2.75, 3.5, 4.25]


def test_visual_observations_preserve_existing_audio_signals():
    take = _take()
    enriched = apply_visual_observations((take,), (
        VisualObservation(
            clip_id="clip-1",
            face_visibility=0.9,
            eye_contact=0.8,
            product_visibility=0.7,
            visual_fumble=0.05,
        ),
    ))[0]
    assert enriched.signals.silence_ratio == 0.1
    assert enriched.signals.audio_quality == 0.8
    assert enriched.signals.face_visibility == 0.9
    assert enriched.signals.product_visibility == 0.7


def test_visual_provider_failure_is_fail_open():
    class BrokenVisual:
        def analyze(self, takes, samples):
            raise TimeoutError("vision unavailable")
    result = safe_visual_analyze(BrokenVisual(), (_take(),), ())
    assert result.observations == ()
    assert result.status.status == "provider_error"
    assert result.status.reason == "TimeoutError"
