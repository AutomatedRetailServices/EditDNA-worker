"""Visual Finishing POLICY CONTRACT + PLAN GENERATION (D-260).

Pure-Python tests -- no ffmpeg, no cv2/mediapipe, no real media. This
module is a pure derivation layer over already-computed D-258
measurements, so every fixture here is a directly-constructed
`VisualClipMeasurement`/`VisualJoinMeasurement` with exact literal
numeric fields -- matching D-249's own test philosophy exactly (only
the measurement layer needs real media; everything downstream of a
measurement is pure and deserves exact, deterministic fixtures, not
floating-point-noisy geometry reconstructions).
"""
from __future__ import annotations

import inspect

from cutsell_worker import visual_finishing_policy as vfp
from cutsell_worker.visual_finishing_measurement import (
    MEASUREMENT_STATUS_COMPLETE,
    MEASUREMENT_STATUS_DECODE_ERROR,
    MEASUREMENT_STATUS_NO_FACE,
    MEASUREMENT_STATUS_UNAVAILABLE,
    CAPTION_SAFE_NOT_ESTABLISHED,
    PRODUCT_BBOX_UNAVAILABLE,
    VisualClipMeasurement,
    VisualFrameMeasurement,
    build_frame_measurement,
    compute_visual_join_measurement,
)
from cutsell_worker.visual_finishing_policy import (
    DEFAULT_PUNCH_IN_SCALE,
    EVIDENCE_STATE_INSUFFICIENT,
    EVIDENCE_STATE_SUFFICIENT,
    FACE_CENTER_X_DISCONTINUITY_THRESHOLD,
    FACE_CENTER_Y_DISCONTINUITY_THRESHOLD,
    FACE_SCALE_RELATIVE_AREA_DISCONTINUITY_THRESHOLD,
    HEADROOM_DISCONTINUITY_THRESHOLD,
    MAX_ADDITIONAL_CROP_LOSS_NORMALIZED,
    MAX_PUNCH_IN_SCALE,
    MAX_REFRAME_TRANSLATION_NORMALIZED,
    MIN_PUNCH_IN_CLIP_DURATION_SEC,
    MIN_RELIABLE_FACE_DETECTION_RATE,
    PLAN_STATUS_ABSTAIN,
    PLAN_STATUS_BLOCKED,
    PLAN_STATUS_PARTIAL,
    PLAN_STATUS_READY_FOR_VISUAL_CORRECTION,
    PLAN_STATUS_READY_NO_CHANGE,
    POLICY_VERSION,
    VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE,
    VISUAL_ACTION_BLOCKED_FACE_SAFETY,
    VISUAL_ACTION_BLOCKED_MULTI_FACE,
    VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN,
    VISUAL_ACTION_NO_CHANGE,
    VISUAL_ACTION_POSITION_MATCH,
    VISUAL_ACTION_PUNCH_IN,
    VISUAL_ACTION_SCALE_AND_POSITION_MATCH,
    VISUAL_ACTION_SCALE_MATCH,
    VISUAL_ACTION_STATIC_REFRAME,
    VisualClipPolicyDecision,
    VisualFinishingPlan,
    VisualJoinPolicyDecision,
    canonical_thresholds_snapshot,
    compute_visual_finishing_plan_identity,
    evaluate_clip_policy,
    evaluate_join_policy,
    generate_visual_finishing_plan,
)


def _clip(
    clip_id, *, cx=0.5, cy=0.5, area=0.04, headroom=0.3, rate=1.0,
    status=MEASUREMENT_STATUS_COMPLETE, frames=(),
) -> VisualClipMeasurement:
    return VisualClipMeasurement(
        source_id="s", clip_id=clip_id, frame_width=1080, frame_height=1920,
        aspect_ratio=1080 / 1920, orientation="PORTRAIT", rotation_degrees=None,
        requested_frame_count=4, valid_frame_count=4, face_valid_frame_count=4,
        face_detection_rate=rate,
        face_center_x_median=cx, face_center_y_median=cy,
        face_area_ratio_median=area, headroom_ratio_median=headroom,
        luma_mean_median=100.0, contrast_median=10.0, luma_variability=0.0,
        color_mean_r_median=1.0, color_mean_g_median=2.0, color_mean_b_median=3.0,
        product_bbox_status=PRODUCT_BBOX_UNAVAILABLE, caption_safe_status=CAPTION_SAFE_NOT_ESTABLISHED,
        freeze_frame_evidence=None, black_frame_evidence=None, frames=frames,
        measurement_status=status,
    )


def _single_face_frame(clip_id="a", *, clipped=False, ts=1.0):
    x_min = -0.05 if clipped else 0.45
    return VisualFrameMeasurement(
        source_id="s", clip_id=clip_id, frame_timestamp_sec=ts, frame_width=1080, frame_height=1920,
        face_detected=True, face_count=1, multiple_faces_detected=False, face_confidence=None,
        face_bbox_x_min=x_min, face_bbox_y_min=0.4, face_bbox_x_max=0.55, face_bbox_y_max=0.5,
        face_center_x=0.5, face_center_y=0.45, face_bbox_width=0.1, face_bbox_height=0.1,
        face_area_ratio=0.01, headroom_ratio=0.4,
        face_bbox_clipped_left=clipped, face_bbox_clipped_right=False,
        face_bbox_clipped_top=False, face_bbox_clipped_bottom=False,
        torso_center_x=None, torso_center_y=None,
        luma_mean=100.0, luma_std=10.0, color_mean_r=1.0, color_mean_g=2.0, color_mean_b=3.0,
        measurement_status=MEASUREMENT_STATUS_COMPLETE, errors=(),
    )


def _multi_face_frame(clip_id="a", *, ts=1.0):
    return VisualFrameMeasurement(
        source_id="s", clip_id=clip_id, frame_timestamp_sec=ts, frame_width=1080, frame_height=1920,
        face_detected=True, face_count=2, multiple_faces_detected=True, face_confidence=None,
        face_bbox_x_min=0.4, face_bbox_y_min=0.4, face_bbox_x_max=0.5, face_bbox_y_max=0.5,
        face_center_x=0.45, face_center_y=0.45, face_bbox_width=0.1, face_bbox_height=0.1,
        face_area_ratio=0.01, headroom_ratio=0.4,
        face_bbox_clipped_left=False, face_bbox_clipped_right=False,
        face_bbox_clipped_top=False, face_bbox_clipped_bottom=False,
        torso_center_x=None, torso_center_y=None,
        luma_mean=100.0, luma_std=10.0, color_mean_r=1.0, color_mean_g=2.0, color_mean_b=3.0,
        measurement_status=MEASUREMENT_STATUS_COMPLETE, errors=(),
    )


# ---------------------------------------------------------------------------
# 1: the ten canonical values exact.
# ---------------------------------------------------------------------------

def test_ten_canonical_values_exact():
    assert FACE_CENTER_X_DISCONTINUITY_THRESHOLD == 0.025
    assert FACE_CENTER_Y_DISCONTINUITY_THRESHOLD == 0.025
    assert FACE_SCALE_RELATIVE_AREA_DISCONTINUITY_THRESHOLD == 0.30
    assert HEADROOM_DISCONTINUITY_THRESHOLD == 0.06
    assert DEFAULT_PUNCH_IN_SCALE == 1.10
    assert MAX_PUNCH_IN_SCALE == 1.15
    assert MAX_REFRAME_TRANSLATION_NORMALIZED == 0.10
    assert MAX_ADDITIONAL_CROP_LOSS_NORMALIZED == 0.10
    assert MIN_RELIABLE_FACE_DETECTION_RATE == 0.75
    assert MIN_PUNCH_IN_CLIP_DURATION_SEC == 1.5
    assert POLICY_VERSION == "V1"


def test_no_new_numeric_constant_beyond_the_ten():
    snapshot = canonical_thresholds_snapshot()
    assert len(snapshot) == 10


# ---------------------------------------------------------------------------
# 2-3: boundary equality / above-threshold behavior (X/Y/scale/headroom).
# ---------------------------------------------------------------------------

def test_x_discontinuity_exact_threshold_is_no_change():
    a = _clip("a", cx=0.5)
    b = _clip("b", cx=0.5 + FACE_CENTER_X_DISCONTINUITY_THRESHOLD)
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.action == VISUAL_ACTION_NO_CHANGE


def test_x_discontinuity_above_threshold_triggers_position_match():
    a = _clip("a", cx=0.5)
    b = _clip("b", cx=0.5 + FACE_CENTER_X_DISCONTINUITY_THRESHOLD + 0.001)
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.action == VISUAL_ACTION_POSITION_MATCH


def test_y_discontinuity_exact_threshold_is_no_change():
    a = _clip("a", cy=0.5)
    b = _clip("b", cy=0.5 + FACE_CENTER_Y_DISCONTINUITY_THRESHOLD)
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.action == VISUAL_ACTION_NO_CHANGE


def test_y_discontinuity_above_threshold_triggers_position_match():
    a = _clip("a", cy=0.5)
    b = _clip("b", cy=0.5 + FACE_CENTER_Y_DISCONTINUITY_THRESHOLD + 0.001)
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.action == VISUAL_ACTION_POSITION_MATCH


def test_scale_discontinuity_exact_threshold_is_no_change():
    a = _clip("a", area=0.04)
    b = _clip("b", area=0.04 * (1 + FACE_SCALE_RELATIVE_AREA_DISCONTINUITY_THRESHOLD))
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db, clip_duration_sec=0.5)
    assert jd.action == VISUAL_ACTION_NO_CHANGE


def test_scale_discontinuity_above_threshold_triggers_correction():
    a = _clip("a", area=0.04)
    b = _clip("b", area=0.04 * (1 + FACE_SCALE_RELATIVE_AREA_DISCONTINUITY_THRESHOLD) + 0.001)
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db, clip_duration_sec=0.5)
    assert jd.action in (VISUAL_ACTION_SCALE_MATCH, VISUAL_ACTION_PUNCH_IN)


def test_headroom_exact_threshold_is_no_change():
    a = _clip("a", headroom=0.3)
    b = _clip("b", headroom=0.3 + HEADROOM_DISCONTINUITY_THRESHOLD)
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.action == VISUAL_ACTION_NO_CHANGE


def test_headroom_above_threshold_triggers_static_reframe():
    a = _clip("a", headroom=0.3)
    b = _clip("b", headroom=0.3 + HEADROOM_DISCONTINUITY_THRESHOLD + 0.001)
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.action == VISUAL_ACTION_STATIC_REFRAME


# ---------------------------------------------------------------------------
# 4: NO_CHANGE bias -- identical clips.
# ---------------------------------------------------------------------------

def test_no_change_bias_identical_clips():
    a = _clip("a")
    b = _clip("b")
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.action == VISUAL_ACTION_NO_CHANGE
    assert jd.reasons == (vfp.REASON_NO_DISCONTINUITY_EVIDENCE,)


# ---------------------------------------------------------------------------
# 5: low face rate abstains.
# ---------------------------------------------------------------------------

def test_low_face_detection_rate_abstains():
    c = _clip("low", rate=0.5)
    d = evaluate_clip_policy(c, product_safety_established=True)
    assert d.action == VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE
    assert d.evidence_state == EVIDENCE_STATE_INSUFFICIENT


def test_exactly_min_reliable_rate_does_not_abstain():
    c = _clip("exact", rate=MIN_RELIABLE_FACE_DETECTION_RATE)
    d = evaluate_clip_policy(c, product_safety_established=True)
    assert d.action != VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE


def test_none_face_detection_rate_abstains():
    c = _clip("none_rate", rate=None)
    d = evaluate_clip_policy(c, product_safety_established=True)
    assert d.action == VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE


def test_no_face_measurement_status_abstains():
    c = _clip("nf", status=MEASUREMENT_STATUS_NO_FACE)
    d = evaluate_clip_policy(c, product_safety_established=True)
    assert d.action == VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE


def test_unavailable_measurement_status_abstains():
    c = _clip("unavail", status=MEASUREMENT_STATUS_UNAVAILABLE)
    d = evaluate_clip_policy(c, product_safety_established=True)
    assert d.action == VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE


def test_decode_error_measurement_status_abstains():
    c = _clip("err", status=MEASUREMENT_STATUS_DECODE_ERROR)
    d = evaluate_clip_policy(c, product_safety_established=True)
    assert d.action == VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# 6: multi-face blocks.
# ---------------------------------------------------------------------------

def test_multi_face_blocks():
    frames = (_multi_face_frame(),)
    c = _clip("multi", frames=frames)
    d = evaluate_clip_policy(c, product_safety_established=True)
    assert d.action == VISUAL_ACTION_BLOCKED_MULTI_FACE
    assert vfp.SAFETY_BLOCKER_MULTI_FACE_PRESENT in d.safety_blockers


def test_multi_face_propagates_to_join_decision():
    a = _clip("a")
    b = _clip("b", frames=(_multi_face_frame(clip_id="b"),))
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.action == VISUAL_ACTION_BLOCKED_MULTI_FACE


# ---------------------------------------------------------------------------
# 7: face clipping blocks unsafe action.
# ---------------------------------------------------------------------------

def test_face_clipped_blocks():
    frames = (_single_face_frame(clipped=True),)
    c = _clip("clipped", frames=frames)
    d = evaluate_clip_policy(c, product_safety_established=True)
    assert d.action == VISUAL_ACTION_BLOCKED_FACE_SAFETY
    assert vfp.SAFETY_BLOCKER_FACE_ALREADY_CLIPPED in d.safety_blockers


# ---------------------------------------------------------------------------
# 8-9: product uncertainty blocks; zero scalar not interpreted as absence.
# ---------------------------------------------------------------------------

def test_product_safety_unknown_blocks_by_default():
    c = _clip("a")
    d = evaluate_clip_policy(c)  # default product_safety_established=False
    assert d.action == VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN
    assert vfp.SAFETY_BLOCKER_PRODUCT_SAFETY_UNKNOWN in d.safety_blockers


def test_product_safety_established_true_unblocks():
    c = _clip("a")
    d = evaluate_clip_policy(c, product_safety_established=True)
    assert d.action != VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN


def test_product_bbox_unavailable_is_the_only_status_d258_ever_reports():
    # Structural proof this module never fabricates a location: it reads
    # `product_bbox_status` nowhere as a location, only as an absence
    # signal (which it treats correctly as fail-closed, not "safe").
    c = _clip("a")
    assert c.product_bbox_status == PRODUCT_BBOX_UNAVAILABLE
    d = evaluate_clip_policy(c)  # default: not established
    assert d.action == VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN


# ---------------------------------------------------------------------------
# 10: caption unknown does not fabricate safe area.
# ---------------------------------------------------------------------------

def test_caption_unknown_emits_warning_not_a_fabricated_region():
    c = _clip("a")
    d = evaluate_clip_policy(c, product_safety_established=True)
    assert vfp.WARNING_CAPTION_SAFE_REGION_NOT_ESTABLISHED in d.warnings
    # No field on the decision claims a caption-safe geometry.
    assert not hasattr(d, "caption_safe_region")


def test_caption_unknown_does_not_block_a_face_safe_correction():
    a = _clip("a")
    b = _clip("b", cx=0.5 + FACE_CENTER_X_DISCONTINUITY_THRESHOLD + 0.01)
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.action == VISUAL_ACTION_POSITION_MATCH  # not blocked by caption-unknown alone


# ---------------------------------------------------------------------------
# 11-14: punch-in duration gate, defaults, max, over-max not silently authorized.
# ---------------------------------------------------------------------------

def _scale_break_pair(extra=0.05):
    a = _clip("a", area=0.04)
    b = _clip("b", area=0.04 * (1 + FACE_SCALE_RELATIVE_AREA_DISCONTINUITY_THRESHOLD + extra))
    return a, b


def test_punch_in_duration_gate_short_clip_falls_back_to_scale_match():
    a, b = _scale_break_pair()
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db,
                               clip_duration_sec=MIN_PUNCH_IN_CLIP_DURATION_SEC - 0.1)
    assert jd.action == VISUAL_ACTION_SCALE_MATCH
    assert vfp.REASON_PUNCH_IN_CLIP_TOO_SHORT in jd.reasons


def test_punch_in_default_scale_exact():
    a, b = _scale_break_pair()
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db,
                               clip_duration_sec=MIN_PUNCH_IN_CLIP_DURATION_SEC)
    assert jd.action == VISUAL_ACTION_PUNCH_IN
    assert jd.authorized_scale == DEFAULT_PUNCH_IN_SCALE


def test_punch_in_max_scale_never_exceeded():
    a, b = _scale_break_pair(extra=5.0)  # a huge relative delta
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db,
                               clip_duration_sec=5.0)
    assert jd.action == VISUAL_ACTION_PUNCH_IN
    assert jd.authorized_scale <= MAX_PUNCH_IN_SCALE


def test_over_max_scale_request_not_silently_fully_authorized():
    a, b = _scale_break_pair(extra=5.0)
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db,
                               clip_duration_sec=5.0)
    assert jd.requested_scale is not None
    assert jd.requested_scale > jd.authorized_scale  # requested vs authorized separated


# ---------------------------------------------------------------------------
# 15-16: reframe max / crop-loss max exact.
# ---------------------------------------------------------------------------

def test_reframe_translation_clamped_to_max():
    a = _clip("a", cx=0.5)
    b = _clip("b", cx=0.5 + MAX_REFRAME_TRANSLATION_NORMALIZED + 0.2)  # far beyond max
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.action == VISUAL_ACTION_POSITION_MATCH
    assert abs(jd.authorized_translation_x) <= MAX_REFRAME_TRANSLATION_NORMALIZED + 1e-9


def test_crop_loss_ceiling_matches_reframe_ceiling_in_v1():
    # Documented V1 simplification: both ceilings are 0.10 today.
    assert MAX_ADDITIONAL_CROP_LOSS_NORMALIZED == MAX_REFRAME_TRANSLATION_NORMALIZED


# ---------------------------------------------------------------------------
# 17: requested vs authorized separated (already partly covered above).
# ---------------------------------------------------------------------------

def test_requested_and_authorized_translation_both_present_and_distinct_when_clamped():
    a = _clip("a", cx=0.5)
    b = _clip("b", cx=0.9)
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.requested_translation_x == compute_visual_join_measurement(a, b).face_center_dx
    assert jd.requested_translation_x != jd.authorized_translation_x


# ---------------------------------------------------------------------------
# 18: combined-action safety priority.
# ---------------------------------------------------------------------------

def test_combined_scale_and_position_when_both_break():
    a = _clip("a", cx=0.5, area=0.04)
    b = _clip("b", cx=0.5 + FACE_CENTER_X_DISCONTINUITY_THRESHOLD + 0.01,
              area=0.04 * (1 + FACE_SCALE_RELATIVE_AREA_DISCONTINUITY_THRESHOLD + 0.05))
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.action == VISUAL_ACTION_SCALE_AND_POSITION_MATCH


def test_face_safety_priority_overrides_continuity_improvement():
    a = _clip("a", cx=0.5)
    b = _clip("b", cx=0.9, frames=(_single_face_frame(clip_id="b", clipped=True),))
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.action == VISUAL_ACTION_BLOCKED_FACE_SAFETY  # not POSITION_MATCH despite the large dx


def test_product_safety_priority_overrides_continuity_improvement():
    a = _clip("a", cx=0.5)
    b = _clip("b", cx=0.9)
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b)  # product safety NOT established for b
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.action == VISUAL_ACTION_BLOCKED_PRODUCT_SAFETY_UNKNOWN


# ---------------------------------------------------------------------------
# 19-21: exposure/color/gaze excluded from P0.
# ---------------------------------------------------------------------------

def test_no_exposure_match_action_in_vocabulary():
    assert not hasattr(vfp, "VISUAL_ACTION_EXPOSURE_MATCH")


def test_no_color_match_action_in_vocabulary():
    assert not hasattr(vfp, "VISUAL_ACTION_COLOR_MATCH")


def test_no_gaze_action_or_field_anywhere():
    source = inspect.getsource(vfp)
    assert "gaze" not in source.lower()
    assert "GAZE" not in source


def test_luma_and_color_differences_stay_diagnostic_only_no_p0_action():
    # A pair with a large luma/color difference but no face/position/
    # scale/headroom discontinuity must resolve to NO_CHANGE.
    a = _clip("a")
    b = _clip("b")
    b = VisualClipMeasurement(**{**b.__dict__, "luma_mean_median": 250.0,
                                  "color_mean_r_median": 250.0, "color_mean_g_median": 5.0, "color_mean_b_median": 5.0})
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd.action == VISUAL_ACTION_NO_CHANGE


# ---------------------------------------------------------------------------
# 22: no sales vocabulary anywhere in the module.
# ---------------------------------------------------------------------------

_FORBIDDEN_SALES_WORDS = (
    "hook", "cta", "benefit", "feature", "proof", "offer",
    "sales", "selling", "conversion", "funnel",
)


def test_no_sales_vocabulary_in_closed_reason_warning_blocker_vocabulary():
    # The firewall's actual mechanism: none of the STRING VALUES in the
    # closed reasons/warnings/safety-blocker/action vocabulary may
    # express a commercial concept. (The module's own docstring
    # legitimately discusses "Smart Sales Funnel" in English prose to
    # document this very firewall -- scanning full source text would
    # false-positive on that prose, so this test checks the actual
    # vocabulary values instead, which is the real enforcement point.)
    vocabulary_values = [
        v for name, v in vars(vfp).items()
        if name.isupper() and isinstance(v, str)
    ]
    joined = " ".join(vocabulary_values).lower()
    for word in _FORBIDDEN_SALES_WORDS:
        assert word not in joined, f"forbidden commercial word {word!r} found in vocabulary value"


def test_no_sales_vocabulary_in_any_generated_reason_or_warning():
    a = _clip("a")
    b = _clip("b", cx=0.5 + FACE_CENTER_X_DISCONTINUITY_THRESHOLD + 0.01)
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    all_text = " ".join(list(da.reasons) + list(db.reasons) + list(jd.reasons)).lower()
    for word in _FORBIDDEN_SALES_WORDS:
        assert word not in all_text


# ---------------------------------------------------------------------------
# 23: no forced alternation (qualitative rule) -- deterministic replay proof.
# ---------------------------------------------------------------------------

def test_no_forced_alternation_decision_is_purely_evidence_local():
    # The SAME evidence (identical clips) must always yield NO_CHANGE
    # regardless of any "previous action" -- there is no such input to
    # this function at all, structurally preventing rhythm-based logic.
    a = _clip("a")
    b = _clip("b")
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd1 = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    jd2 = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    assert jd1.action == jd2.action == VISUAL_ACTION_NO_CHANGE


def test_evaluate_join_policy_signature_has_no_previous_action_parameter():
    import inspect as _inspect
    sig = _inspect.signature(evaluate_join_policy)
    for name in sig.parameters:
        assert "previous" not in name.lower()
        assert "alternat" not in name.lower()
        assert "rhythm" not in name.lower()


# ---------------------------------------------------------------------------
# 24-26: deterministic plan identity, filename independence, policy-version sensitivity.
# ---------------------------------------------------------------------------

def test_plan_identity_deterministic():
    a = _clip("a")
    b = _clip("b")
    id1 = compute_visual_finishing_plan_identity("s", (evaluate_clip_policy(a, product_safety_established=True),
                                                        evaluate_clip_policy(b, product_safety_established=True)), ())
    id2 = compute_visual_finishing_plan_identity("s", (evaluate_clip_policy(a, product_safety_established=True),
                                                        evaluate_clip_policy(b, product_safety_established=True)), ())
    assert id1 == id2


def test_plan_identity_filename_independent():
    # Two VisualClipMeasurement objects with the same clip_id/content but
    # different source_id (a filename-like distinguishing field NOT used
    # by clip decisions' identity contribution) should not change the
    # plan identity, since identity is keyed on clip_id/action/authorized
    # values, never on any path-like field.
    a1 = _clip("a")
    a2 = VisualClipMeasurement(**{**a1.__dict__, "source_id": "different_source_name"})
    d1 = evaluate_clip_policy(a1, product_safety_established=True)
    d2 = evaluate_clip_policy(a2, product_safety_established=True)
    id1 = compute_visual_finishing_plan_identity("s", (d1,), ())
    id2 = compute_visual_finishing_plan_identity("s", (d2,), ())
    assert id1 == id2


def test_plan_identity_sensitive_to_decision_content():
    a = _clip("a")
    b = _clip("b", cx=0.5 + FACE_CENTER_X_DISCONTINUITY_THRESHOLD + 0.01)
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    id_no_join = compute_visual_finishing_plan_identity("s", (da, db), ())
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    id_with_join = compute_visual_finishing_plan_identity("s", (da, db), (jd,))
    assert id_no_join != id_with_join


# ---------------------------------------------------------------------------
# 27-28: plan frozen/immutable; no renderer strings.
# ---------------------------------------------------------------------------

def test_plan_and_decisions_frozen():
    import dataclasses

    a = _clip("a")
    b = _clip("b")
    plan = generate_visual_finishing_plan((a, b), (compute_visual_join_measurement(a, b),), source_id="s",
                                           product_safety_established={"a": True, "b": True})
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        plan.plan_status = PLAN_STATUS_BLOCKED
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        plan.clip_decisions[0].action = VISUAL_ACTION_NO_CHANGE


_FORBIDDEN_RENDERER_TOKENS = ("crop=", "zoompan", "-vf", "filter_complex", "subprocess.run(")


def test_no_renderer_strings_anywhere_in_module():
    source = inspect.getsource(vfp)
    for token in _FORBIDDEN_RENDERER_TOKENS:
        assert token not in source


def test_plan_fields_are_symbolic_floats_never_strings_for_corrections():
    a = _clip("a", cx=0.5)
    b = _clip("b", cx=0.9)
    da = evaluate_clip_policy(a, product_safety_established=True)
    db = evaluate_clip_policy(b, product_safety_established=True)
    jd = evaluate_join_policy(a, b, compute_visual_join_measurement(a, b), left_decision=da, right_decision=db)
    if jd.authorized_translation_x is not None:
        assert isinstance(jd.authorized_translation_x, float)


# ---------------------------------------------------------------------------
# 29-31: no crop/reframe/punch-in EXECUTION.
# ---------------------------------------------------------------------------

def test_module_never_imports_render_or_subprocess():
    source = inspect.getsource(vfp)
    assert "import subprocess" not in source
    assert "from .render" not in source


def test_module_has_no_execute_function():
    assert not hasattr(vfp, "execute_visual_finishing_plan")
    assert not any(name.startswith("execute_") for name in dir(vfp))


# ---------------------------------------------------------------------------
# 32-38: render.py / local_performance / speech_visual_microtrim / Pacing /
# Boundary / Freeze / Audio Finishing unchanged (structural, no coupling).
# ---------------------------------------------------------------------------

def test_render_module_not_imported():
    source = inspect.getsource(vfp)
    assert "from .render import" not in source
    assert "from . import render" not in source


def test_local_performance_not_imported():
    source = inspect.getsource(vfp)
    assert "local_performance" not in source


def test_speech_visual_microtrim_not_imported():
    source = inspect.getsource(vfp)
    assert "speech_visual_microtrim" not in source


def test_no_pacing_boundary_freeze_audio_finishing_imports():
    source = inspect.getsource(vfp)
    for forbidden in (
        "from .pacing", "from .boundary_engine_pass", "from .selection_freeze",
        "from .audio_finishing_policy", "from .audio_finishing_executor",
        "from .audio_finishing_outcome",
    ):
        assert forbidden not in source


def test_existing_measurement_module_unaffected():
    from cutsell_worker import visual_finishing_measurement  # noqa: F401


# ---------------------------------------------------------------------------
# 39-40: no provider, no RAW.
# ---------------------------------------------------------------------------

def test_no_provider_or_raw_reference():
    source = inspect.getsource(vfp)
    for forbidden in ("runpod", "RunPod", "modal.", "openai", "anthropic", "boto3"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Plan-level status derivation matrix.
# ---------------------------------------------------------------------------

def test_plan_status_ready_no_change_when_all_no_change():
    a = _clip("a")
    b = _clip("b")
    plan = generate_visual_finishing_plan((a, b), (compute_visual_join_measurement(a, b),), source_id="s",
                                           product_safety_established={"a": True, "b": True})
    assert plan.plan_status == PLAN_STATUS_READY_NO_CHANGE


def test_plan_status_ready_for_correction():
    a = _clip("a", cx=0.5)
    b = _clip("b", cx=0.9)
    plan = generate_visual_finishing_plan((a, b), (compute_visual_join_measurement(a, b),), source_id="s",
                                           product_safety_established={"a": True, "b": True})
    assert plan.plan_status == PLAN_STATUS_READY_FOR_VISUAL_CORRECTION


def test_plan_status_blocked_when_any_decision_blocked():
    a = _clip("a")
    b = _clip("b")
    plan = generate_visual_finishing_plan((a, b), (compute_visual_join_measurement(a, b),), source_id="s")
    assert plan.plan_status == PLAN_STATUS_BLOCKED


def test_plan_status_abstain_when_all_abstain():
    a = _clip("a", status=MEASUREMENT_STATUS_NO_FACE)
    b = _clip("b", status=MEASUREMENT_STATUS_NO_FACE)
    plan = generate_visual_finishing_plan((a, b), (), source_id="s")
    assert plan.plan_status == PLAN_STATUS_ABSTAIN


def test_plan_status_partial_mix_of_correction_and_abstain():
    a = _clip("a", cx=0.5)
    b = _clip("b", cx=0.9)
    c = _clip("c", status=MEASUREMENT_STATUS_NO_FACE)
    d = _clip("d", status=MEASUREMENT_STATUS_NO_FACE)
    plan = generate_visual_finishing_plan(
        (a, b, c, d),
        (compute_visual_join_measurement(a, b), compute_visual_join_measurement(c, d)),
        source_id="s", product_safety_established={"a": True, "b": True},
    )
    assert plan.plan_status == PLAN_STATUS_PARTIAL


def test_plan_status_unknown_when_no_decisions():
    plan = generate_visual_finishing_plan((), (), source_id="s")
    assert plan.plan_status == vfp.PLAN_STATUS_UNKNOWN


# ---------------------------------------------------------------------------
# Field-list completeness (Stage 5/6/7 minimum contracts).
# ---------------------------------------------------------------------------

def test_clip_decision_field_list_matches_stage5_minimum():
    names = set(VisualClipPolicyDecision.__dataclass_fields__)
    expected = {
        "clip_id", "action", "evidence_state", "face_detection_rate", "face_bbox",
        "face_center", "face_area_ratio", "headroom_ratio", "reframe_authorized",
        "punch_in_authorized", "requested_translation_x", "requested_translation_y",
        "authorized_translation_x", "authorized_translation_y", "requested_scale",
        "authorized_scale", "safety_blockers", "warnings", "reasons", "provenance",
    }
    assert expected <= names


def test_join_decision_field_list_matches_stage6_minimum():
    names = set(VisualJoinPolicyDecision.__dataclass_fields__)
    expected = {
        "left_clip_id", "right_clip_id", "action", "evidence_state",
        "face_center_dx", "face_center_dy", "face_scale_delta", "headroom_delta",
        "thresholds_used", "position_match_authorized", "scale_match_authorized",
        "punch_in_authorized", "target_position_x", "target_position_y", "target_scale",
        "safety_blockers", "warnings", "reasons", "provenance",
    }
    assert expected <= names


def test_plan_field_list_matches_stage7_minimum():
    names = set(VisualFinishingPlan.__dataclass_fields__)
    expected = {
        "policy_version", "source_id", "clip_measurement_references", "join_measurement_references",
        "clip_decisions", "join_decisions", "plan_status", "canonical_thresholds_snapshot",
        "abstentions", "warnings", "reasons", "provenance", "visual_finishing_identity",
    }
    assert expected <= names


# ---------------------------------------------------------------------------
# Unit-semantics regression: relative vs absolute scale delta.
# ---------------------------------------------------------------------------

def test_relative_scale_delta_not_absolute_delta():
    a = _clip("a", area=0.04)
    b = _clip("b", area=0.04 * 1.5)  # +50% relative change
    relative = vfp._relative_scale_delta(a, b)
    assert relative == __import__("pytest").approx(0.5)
    join = compute_visual_join_measurement(a, b)
    assert relative != join.face_area_ratio_delta  # the absolute D-258 delta differs
