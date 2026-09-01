from types import SimpleNamespace

from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.frame_sampling import adaptive_frame_count, sample_take_frames
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


def test_adaptive_frame_sampling_scales_and_stays_inside_take(tmp_path):
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    assert adaptive_frame_count(_take()) == 5
    frames = sample_take_frames("raw.mov", _take(), str(tmp_path), runner=runner)
    assert len(frames) == 5
    assert all(frame.source_asset_id == "src-1" for frame in frames)
    assert all(2.0 < frame.timestamp < 5.0 for frame in frames)
    assert frames[0].relative_position < 0.1
    assert frames[-1].relative_position > 0.9
    assert [frame.timestamp for frame in frames] == sorted(frame.timestamp for frame in frames)


def test_long_take_sampling_is_bounded():
    take = CandidateTake("long", "src-1", 0, 0.0, 30.0, "long take")
    assert adaptive_frame_count(take) == 12


def test_visual_observations_preserve_audio_and_add_human_delivery_signals():
    take = _take()
    enriched = apply_visual_observations((take,), (
        VisualObservation(
            clip_id="clip-1",
            face_visibility=0.9,
            eye_contact=0.8,
            product_visibility=0.7,
            visual_fumble=0.05,
            expression_naturalness=0.91,
            gesture_naturalness=0.86,
            delivery_energy=0.82,
            distraction_risk=0.04,
        ),
    ))[0]
    assert enriched.signals.silence_ratio == 0.1
    assert enriched.signals.audio_quality == 0.8
    assert enriched.signals.face_visibility == 0.9
    assert enriched.signals.product_visibility == 0.7
    assert enriched.signals.expression_naturalness == 0.91
    assert enriched.signals.gesture_naturalness == 0.86
    assert enriched.signals.delivery_energy == 0.82
    assert enriched.signals.distraction_risk == 0.04


def test_visual_provider_failure_is_fail_open():
    class BrokenVisual:
        def analyze(self, takes, samples):
            raise TimeoutError("vision unavailable")
    result = safe_visual_analyze(BrokenVisual(), (_take(),), ())
    assert result.observations == ()
    assert result.status.status == "provider_error"
    assert result.status.reason == "TimeoutError"
