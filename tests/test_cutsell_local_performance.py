from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.local_performance import (
    LocalPerformanceTimeline,
    PerformanceFrame,
    apply_local_performance_to_takes,
    detect_candidate_events,
)
from cutsell_worker.providers import ProviderStatus


def frame(t, *, eye=0.9, motion=0.01, expression=0.04, body=(0.5, 0.5), left=(0.4, 0.6), right=(0.6, 0.6)):
    return PerformanceFrame(
        source_asset_id="s1",
        timestamp=t,
        face_visible=1.0,
        pose_visible=1.0,
        eye_contact_proxy=eye,
        motion=motion,
        expression=expression,
        body_x=body[0],
        body_y=body[1],
        left_wrist_x=left[0],
        left_wrist_y=left[1],
        right_wrist_x=right[0],
        right_wrist_y=right[1],
    )


def timeline(observations):
    observations = tuple(observations)
    return LocalPerformanceTimeline(
        source_asset_id="s1",
        observations=observations,
        events=detect_candidate_events(observations),
        sampled_fps=12.0,
        source_fps=30.0,
        status=ProviderStatus("local_performance", True, True, "applied"),
    )


def test_detects_camera_body_and_hand_change_as_candidates_only():
    observations = (
        frame(1.00),
        frame(1.08, eye=0.25, motion=0.08, expression=0.17, body=(0.59, 0.53), left=(0.55, 0.62)),
    )
    kinds = {event.kind for event in detect_candidate_events(observations)}
    assert "camera_disengagement_candidate" in kinds
    assert "facial_expression_shift_candidate" in kinds
    assert "body_reset_candidate" in kinds
    assert "hand_motion_reset_candidate" in kinds
    assert all(kind.endswith("_candidate") for kind in kinds)


def test_normal_stable_performance_does_not_create_false_reset():
    observations = (
        frame(0.00),
        frame(0.08, eye=0.88, motion=0.012, expression=0.045, body=(0.503, 0.501), left=(0.405, 0.602)),
        frame(0.16, eye=0.91, motion=0.010, expression=0.043, body=(0.501, 0.500), left=(0.402, 0.601)),
    )
    assert detect_candidate_events(observations) == ()


def test_dense_local_evidence_blends_into_existing_take_signals():
    observations = (
        frame(0.00),
        frame(0.08, eye=0.24, motion=0.09, expression=0.16, body=(0.59, 0.53), left=(0.55, 0.62)),
        frame(0.16, eye=0.30, motion=0.07, expression=0.15, body=(0.60, 0.54), left=(0.56, 0.63)),
    )
    take = CandidateTake(
        clip_id="c1",
        source_asset_id="s1",
        source_order=0,
        start=0.0,
        end=0.3,
        text="complete sentence",
        signals=MediaSignals(
            source_asset_id="s1",
            start=0.0,
            end=0.3,
            face_visibility=0.6,
            eye_contact=0.8,
            visual_fumble=0.0,
            gesture_naturalness=0.8,
            distraction_risk=0.0,
        ),
    )
    fused = apply_local_performance_to_takes((take,), (timeline(observations),))[0]
    assert fused.signals is not None
    assert fused.signals.eye_contact < 0.8
    assert fused.signals.visual_fumble > 0.0
    assert fused.signals.gesture_naturalness < 0.8
    assert fused.signals.distraction_risk > 0.0


def test_local_tracking_does_not_replace_product_or_audio_signals():
    observations = (frame(0.00), frame(0.08), frame(0.16))
    take = CandidateTake(
        clip_id="c1",
        source_asset_id="s1",
        source_order=0,
        start=0.0,
        end=0.3,
        text="hello",
        signals=MediaSignals(
            source_asset_id="s1",
            start=0.0,
            end=0.3,
            audio_quality=0.91,
            product_visibility=0.83,
            delivery_energy=0.74,
        ),
    )
    fused = apply_local_performance_to_takes((take,), (timeline(observations),))[0]
    assert fused.signals.audio_quality == 0.91
    assert fused.signals.product_visibility == 0.83
    assert fused.signals.delivery_energy == 0.74
