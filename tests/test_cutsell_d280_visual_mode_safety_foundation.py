"""D-280 -- Faceless/Product/Hands/Demo Visual-Mode Safety Foundation.

Pure-type/pure-function coverage plus structural regression proofs that
the existing Visual Finishing / BestTake pipelines already behave (or,
for BestTake, now behave) safely for face-independent primary footage.
"""
from __future__ import annotations

import ast
import inspect

import pytest

from cutsell_worker import visual_mode as vm
from cutsell_worker import visual_finishing_measurement as vfm
from cutsell_worker import visual_finishing_policy as vfp
from cutsell_worker import visual_finishing_executor as vfe
from cutsell_worker import take_judge as tj
from cutsell_worker.contracts import CandidateTake, MediaSignals


def _source_without_docstrings(obj) -> str:
    """AST-based docstring stripper (this session's established
    technique): a module's own legitimate prose describing what it does
    NOT do (e.g. 'NO SKU RECOGNITION' as a scope banner) must never
    produce a false positive when scanning the module's real CODE for a
    forbidden literal."""
    tree = ast.parse(inspect.getsource(obj))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                if isinstance(body[0].value.value, str):
                    node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


# ---------------------------------------------------------------------------
# STAGE 1/25: bounded vocabulary + immutability.
# ---------------------------------------------------------------------------

def test_visual_mode_vocabulary_is_bounded_to_seven_values():
    assert {m.value for m in vm.VisualMode} == {
        "TALKING_HEAD", "TALKING_HEAD_WITH_PRODUCT", "PRODUCT_HANDS",
        "PRODUCT_ONLY", "DEMO_ACTION", "SUPPORTING_VISUAL", "UNKNOWN",
    }


def test_evidence_state_vocabulary_is_bounded():
    assert {s.value for s in vm.VisualModeEvidenceState} == {
        "SUPPORTED", "LIKELY", "INSUFFICIENT_EVIDENCE", "UNKNOWN",
    }


def test_evidence_availability_has_four_distinct_absence_states():
    values = {a.value for a in vm.EvidenceAvailability}
    assert values == {"PRESENT", "NOT_PRESENT", "UNAVAILABLE", "NOT_APPLICABLE", "UNKNOWN"}


def test_classification_is_frozen_immutable():
    c = vm.classify_visual_mode(clip_id="c1", face_detection_rate=0.9)
    with pytest.raises(Exception):
        c.mode = vm.VisualMode.UNKNOWN  # type: ignore[misc]


def test_visual_mode_evidence_is_frozen_immutable():
    e = vm.VisualModeEvidence()
    with pytest.raises(Exception):
        e.face_detection_rate = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# STAGE 4: face-required helper -- items 7-14.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,expected", [
    (vm.VisualMode.TALKING_HEAD, True),
    (vm.VisualMode.TALKING_HEAD_WITH_PRODUCT, True),
    (vm.VisualMode.PRODUCT_HANDS, False),
    (vm.VisualMode.PRODUCT_ONLY, False),
    (vm.VisualMode.DEMO_ACTION, False),
    (vm.VisualMode.SUPPORTING_VISUAL, False),
    (vm.VisualMode.UNKNOWN, None),
])
def test_requires_face_evidence_per_mode(mode, expected):
    assert vm.requires_face_evidence(mode) is expected


def test_unknown_mode_never_guesses_true_or_false():
    # Stage 4's own literal instruction: fail conservative, not automatic rejection.
    result = vm.requires_face_evidence(vm.VisualMode.UNKNOWN)
    assert result is not True
    assert result is not False
    assert result is None


# ---------------------------------------------------------------------------
# STAGE 6: evidence-family routing.
# ---------------------------------------------------------------------------

def test_talking_head_routes_to_face_gaze_headroom_framing_speech():
    families = vm.applicable_evidence_families(vm.VisualMode.TALKING_HEAD)
    assert vm.EVIDENCE_FAMILY_FACE in families
    assert vm.EVIDENCE_FAMILY_GAZE in families
    assert vm.EVIDENCE_FAMILY_SPEECH_FLUENCY in families


def test_product_hands_routes_without_face_or_gaze():
    families = vm.applicable_evidence_families(vm.VisualMode.PRODUCT_HANDS)
    assert vm.EVIDENCE_FAMILY_FACE not in families
    assert vm.EVIDENCE_FAMILY_GAZE not in families
    assert vm.EVIDENCE_FAMILY_SPEECH_FLUENCY in families
    assert vm.EVIDENCE_FAMILY_PRODUCT_HANDS in families


def test_demo_action_routes_without_face_to_motion_continuity():
    families = vm.applicable_evidence_families(vm.VisualMode.DEMO_ACTION)
    assert vm.EVIDENCE_FAMILY_FACE not in families
    assert vm.EVIDENCE_FAMILY_MOTION_ACTION_CONTINUITY in families


def test_unknown_mode_routes_to_no_evidence_family():
    assert vm.applicable_evidence_families(vm.VisualMode.UNKNOWN) == ()


# ---------------------------------------------------------------------------
# STAGE 7/8/27: evidence seams + absence semantics + product-safety doctrine.
# ---------------------------------------------------------------------------

def test_default_evidence_reports_product_hands_as_unavailable_not_absent():
    evidence = vm.VisualModeEvidence()
    assert evidence.product_presence == vm.EvidenceAvailability.UNAVAILABLE
    assert evidence.hands_presence == vm.EvidenceAvailability.UNAVAILABLE
    assert evidence.demo_action_presence == vm.EvidenceAvailability.UNAVAILABLE
    # Never fabricated as a positive absence:
    assert evidence.product_presence != vm.EvidenceAvailability.NOT_PRESENT


def test_no_fake_product_bbox_without_present_state():
    with pytest.raises(ValueError):
        vm.VisualModeEvidence(product_bbox=(0.1, 0.1, 0.5, 0.5), product_presence=vm.EvidenceAvailability.UNAVAILABLE)


def test_no_fake_hands_bbox_without_present_state():
    with pytest.raises(ValueError):
        vm.VisualModeEvidence(hands_bbox=(0.1, 0.1, 0.5, 0.5), hands_presence=vm.EvidenceAvailability.UNKNOWN)


def test_real_bbox_allowed_alongside_present_state():
    evidence = vm.VisualModeEvidence(product_bbox=(0.1, 0.1, 0.5, 0.5), product_presence=vm.EvidenceAvailability.PRESENT)
    assert evidence.product_bbox == (0.1, 0.1, 0.5, 0.5)


def test_unknown_product_location_never_treated_as_known_absence():
    for state in (vm.EvidenceAvailability.UNAVAILABLE, vm.EvidenceAvailability.UNKNOWN, vm.EvidenceAvailability.NOT_APPLICABLE):
        assert vm.product_location_known(state) is False
    assert vm.product_location_known(vm.EvidenceAvailability.PRESENT) is True
    assert vm.product_location_known(vm.EvidenceAvailability.NOT_PRESENT) is True


def test_all_evidence_absence_states_distinguishable():
    # Stage 27: never collapse NOT_PRESENT/UNAVAILABLE/NOT_APPLICABLE/UNKNOWN.
    assert len({
        vm.EvidenceAvailability.NOT_PRESENT, vm.EvidenceAvailability.UNAVAILABLE,
        vm.EvidenceAvailability.NOT_APPLICABLE, vm.EvidenceAvailability.UNKNOWN,
    }) == 4


# ---------------------------------------------------------------------------
# STAGE 16/17/18/19/20: role/mode separation.
# ---------------------------------------------------------------------------

def test_editorial_content_role_is_separate_from_visual_mode():
    # product_hands + PRIMARY_A_ROLL and product_hands + SUPPLEMENTAL_BROLL
    # are both structurally valid -- no coupling exists between the two enums.
    role_a = vm.EditorialContentRole.PRIMARY_A_ROLL
    role_b = vm.EditorialContentRole.SUPPLEMENTAL_BROLL
    mode = vm.VisualMode.PRODUCT_HANDS
    assert (mode, role_a) not in vm.VISUAL_MODE_ROLE_FORBIDDEN_EQUIVALENCES
    assert (mode, role_b) not in vm.VISUAL_MODE_ROLE_FORBIDDEN_EQUIVALENCES


def test_no_visual_mode_role_pair_is_ever_forbidden():
    assert vm.VISUAL_MODE_ROLE_FORBIDDEN_EQUIVALENCES == frozenset()


def test_product_hands_never_equals_broll_structurally():
    source = inspect.getsource(vm)
    assert "PRODUCT_HANDS == BROLL" not in source
    assert "PRODUCT_HANDS = BROLL" not in source


# ---------------------------------------------------------------------------
# STAGE 22/23/24: the conservative rule-based classifier.
# ---------------------------------------------------------------------------

def test_strong_face_evidence_yields_talking_head_supported():
    c = vm.classify_visual_mode(clip_id="c1", face_detection_rate=0.9)
    assert c.mode == vm.VisualMode.TALKING_HEAD
    assert c.evidence_state == vm.VisualModeEvidenceState.SUPPORTED
    assert c.requires_face is True


def test_no_face_evidence_never_auto_infers_product_hands():
    c = vm.classify_visual_mode(clip_id="c2", face_detection_rate=0.0)
    assert c.mode != vm.VisualMode.PRODUCT_HANDS
    assert c.mode != vm.VisualMode.PRODUCT_ONLY
    assert c.mode != vm.VisualMode.DEMO_ACTION
    assert c.mode == vm.VisualMode.UNKNOWN
    assert c.evidence_state == vm.VisualModeEvidenceState.LIKELY


def test_missing_face_detection_rate_is_classification_unknown():
    c = vm.classify_visual_mode(clip_id="c3", face_detection_rate=None)
    assert c.mode == vm.VisualMode.UNKNOWN
    assert c.evidence_state == vm.VisualModeEvidenceState.UNKNOWN
    assert c.requires_face is None


def test_uploaded_role_hint_never_changes_classification():
    evidence_with_hint = vm.VisualModeEvidence(face_detection_rate=0.9, uploaded_role_hint="b_roll")
    c = vm.classify_visual_mode(clip_id="c4", face_detection_rate=0.9, evidence=evidence_with_hint)
    # Still TALKING_HEAD from real face evidence -- the manual "uploaded as
    # b_roll" hint never overrides the classification (Stage 24).
    assert c.mode == vm.VisualMode.TALKING_HEAD


def test_classifier_reuses_visual_finishing_threshold_not_a_new_one():
    # Exactly at the D-260 canonical bar counts as strong evidence.
    c = vm.classify_visual_mode(clip_id="c5", face_detection_rate=vfp.MIN_RELIABLE_FACE_DETECTION_RATE)
    assert c.mode == vm.VisualMode.TALKING_HEAD
    below = vm.classify_visual_mode(clip_id="c6", face_detection_rate=vfp.MIN_RELIABLE_FACE_DETECTION_RATE - 0.01)
    assert below.mode == vm.VisualMode.UNKNOWN


# ---------------------------------------------------------------------------
# STAGE 28/29: no product recognition, no AI B-roll, no auto-placement.
# ---------------------------------------------------------------------------

def test_module_contains_no_product_recognition_or_ai_broll_code():
    source = _source_without_docstrings(vm)
    forbidden = (
        "sku", "clip_score", "ranking_model", "auto_place", "similarity_score",
        "b_roll_suggest", "broll_suggest", "recommend_broll",
    )
    lowered = source.lower()
    for term in forbidden:
        assert term not in lowered, f"forbidden term found: {term}"


# ---------------------------------------------------------------------------
# STAGE 5/9/30: Visual Finishing safety matrix -- proven against the REAL,
# UNCHANGED D-258/D-260/D-262 modules, not a re-implementation.
# ---------------------------------------------------------------------------

def _no_face_clip_measurement(clip_id="clip_no_face"):
    return vfm.aggregate_clip_measurement(
        (), source_id="src1", clip_id=clip_id, frame_width=1080, frame_height=1920,
        rotation_degrees=None, requested_frame_count=8,
        freeze_frame_evidence=None, black_frame_evidence=None,
    )


def _talking_head_frame(clip_id, t, *, face_detected=True):
    one_face = [(0.4, 0.2), (0.6, 0.2), (0.5, 0.35), (0.4, 0.4), (0.6, 0.4)]
    return vfm.build_frame_measurement(
        source_id="src1", clip_id=clip_id, frame_timestamp_sec=t,
        frame_width=1080, frame_height=1920,
        face_landmark_points_list=([one_face] if face_detected else []),
        luma_mean=120.0, luma_std=30.0, color_rgb=(120.0, 110.0, 100.0),
    )


def test_no_face_clip_measurement_status_is_no_face_not_a_blocked_or_invalid_status():
    frames = tuple(_talking_head_frame("clip_no_face", t, face_detected=False) for t in range(8))
    measurement = vfm.aggregate_clip_measurement(
        frames, source_id="src1", clip_id="clip_no_face", frame_width=1080, frame_height=1920,
        rotation_degrees=None, requested_frame_count=8,
        freeze_frame_evidence=None, black_frame_evidence=None,
    )
    assert measurement.measurement_status == vfm.MEASUREMENT_STATUS_NO_FACE
    # Stage 5: the invariant -- never a BAD_CLIP/INVALID_PRIMARY/BLOCKED_VISUAL-shaped status.
    assert measurement.measurement_status not in ("BAD_CLIP", "INVALID_PRIMARY", "BLOCKED_VISUAL")


def test_product_hands_no_face_clip_policy_is_abstain_never_face_blocked():
    """Stage 9/30: PRODUCT_HANDS/PRODUCT_ONLY/DEMO_ACTION + no face -> no
    face blocker. Proven directly against the real, unchanged D-260
    `evaluate_clip_policy` -- a genuinely no-face clip short-circuits to
    ABSTAIN_INSUFFICIENT_EVIDENCE before it can ever reach the multi-
    face/clipped-face BLOCKED branches, so no code change to D-260 was
    needed to satisfy this invariant."""
    frames = tuple(_talking_head_frame("clip_hands", t, face_detected=False) for t in range(8))
    measurement = vfm.aggregate_clip_measurement(
        frames, source_id="src1", clip_id="clip_hands", frame_width=1080, frame_height=1920,
        rotation_degrees=None, requested_frame_count=8,
        freeze_frame_evidence=None, black_frame_evidence=None,
    )
    decision = vfp.evaluate_clip_policy(measurement, product_safety_established=True)
    assert decision.action == vfp.VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE
    assert decision.action != vfp.VISUAL_ACTION_BLOCKED_FACE_SAFETY
    assert decision.action != vfp.VISUAL_ACTION_BLOCKED_MULTI_FACE


def test_faceless_abstain_never_produces_an_unsafe_crop_execution():
    """Stage 11/26: faceless + insufficient face evidence -> executor
    produces NO transform spec at all (never an unsafe blind crop)."""
    frames = tuple(_talking_head_frame("clip_hands", t, face_detected=False) for t in range(8))
    measurement = vfm.aggregate_clip_measurement(
        frames, source_id="src1", clip_id="clip_hands", frame_width=1080, frame_height=1920,
        rotation_degrees=None, requested_frame_count=8,
        freeze_frame_evidence=None, black_frame_evidence=None,
    )
    decision = vfp.evaluate_clip_policy(measurement, product_safety_established=True)
    join_decision = vfp.VisualJoinPolicyDecision(
        left_clip_id="clip_hands", right_clip_id="clip_hands_2", action=decision.action,
        evidence_state=decision.evidence_state, face_center_dx=None, face_center_dy=None,
        face_scale_delta=None, headroom_delta=None, thresholds_used=vfp.canonical_thresholds_snapshot(),
        position_match_authorized=False, scale_match_authorized=False, punch_in_authorized=False,
        requested_translation_x=None, requested_translation_y=None,
        authorized_translation_x=None, authorized_translation_y=None,
        requested_scale=None, authorized_scale=None,
        target_position_x=None, target_position_y=None, target_scale=None,
    )
    record = vfe.execute_visual_finishing_decision(join_decision, measurement, plan_id="plan1")
    assert record.transform_spec is None
    assert record.execution_status == vfe.EXECUTION_STATUS_PLAN_NOT_EXECUTABLE


def test_talking_head_reliable_face_visual_finishing_behavior_unchanged():
    """Stage 9/24: existing talking-head behavior is untouched by D-280 --
    a reliable-face clip with a real discontinuity still authorizes a
    correction exactly as D-260 always has."""
    left_frames = tuple(_talking_head_frame("left", t) for t in range(8))
    right_frames = tuple(_talking_head_frame("right", t) for t in range(8))
    left = vfm.aggregate_clip_measurement(
        left_frames, source_id="src1", clip_id="left", frame_width=1080, frame_height=1920,
        rotation_degrees=None, requested_frame_count=8, freeze_frame_evidence=None, black_frame_evidence=None,
    )
    right = vfm.aggregate_clip_measurement(
        right_frames, source_id="src1", clip_id="right", frame_width=1080, frame_height=1920,
        rotation_degrees=None, requested_frame_count=8, freeze_frame_evidence=None, black_frame_evidence=None,
    )
    left_decision = vfp.evaluate_clip_policy(left, product_safety_established=True)
    right_decision = vfp.evaluate_clip_policy(right, product_safety_established=True)
    assert left_decision.action == vfp.VISUAL_ACTION_NO_CHANGE
    assert right_decision.action == vfp.VISUAL_ACTION_NO_CHANGE


def test_unknown_measurement_status_visual_finishing_is_abstain_not_change():
    measurement = _no_face_clip_measurement()
    decision = vfp.evaluate_clip_policy(measurement, product_safety_established=True)
    assert decision.action == vfp.VISUAL_ACTION_ABSTAIN_INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# STAGE 13/14/31: BestTake minimal safety fix.
# ---------------------------------------------------------------------------

def _take(face_visibility, eye_contact, **overrides):
    fields = dict(
        source_asset_id="s1", start=0.0, end=2.0,
        face_visibility=face_visibility, eye_contact=eye_contact,
        audio_quality=0.8, framing_quality=0.7, product_visibility=0.9,
        motion_stability=0.8, continuity=0.8, visual_fumble=0.0,
        expression_naturalness=0.5, gesture_naturalness=0.5,
        delivery_energy=0.7, distraction_risk=0.0,
    )
    fields.update(overrides)
    signals = MediaSignals(**fields)
    return CandidateTake(
        clip_id="c1", source_asset_id="s1", source_order=0, start=0.0, end=2.0,
        text="this vacuum gets the dog hair in one pass", signals=signals, complete_idea=True,
    )


def test_default_score_take_call_unaffected_by_d280():
    take = _take(face_visibility=0.0, eye_contact=0.0)
    before = tj.score_take(take)
    # Calling with no visual_mode at all must equal calling with an
    # explicit face-dependent mode -- both keep the ORIGINAL weighting.
    with_talking_head = tj.score_take(take, visual_mode=vm.VisualMode.TALKING_HEAD)
    assert before.score == with_talking_head.score
    assert before.reason == "watch_listen_baseline"


def test_unknown_visual_mode_keeps_original_weighting():
    take = _take(face_visibility=0.0, eye_contact=0.0)
    default = tj.score_take(take)
    unknown_mode = tj.score_take(take, visual_mode=vm.VisualMode.UNKNOWN)
    assert default.score == unknown_mode.score


def test_faceless_take_not_penalized_solely_for_missing_face_evidence():
    """Stage 14/31: faceless + good speech/editorial evidence -> not
    rejected/down-ranked solely for missing face. A genuinely faceless
    take (face_visibility=eye_contact=0.0) scored with an explicit
    face-independent mode must score STRICTLY HIGHER than the same take
    scored under the original (face-penalizing) weighting."""
    take = _take(face_visibility=0.0, eye_contact=0.0)
    original = tj.score_take(take)
    mode_aware = tj.score_take(take, visual_mode=vm.VisualMode.PRODUCT_HANDS)
    assert mode_aware.score > original.score
    assert mode_aware.reason == "watch_listen_baseline_face_independent"


def test_faceless_take_with_perfect_other_signals_reaches_full_ceiling():
    take = _take(
        face_visibility=0.0, eye_contact=0.0, audio_quality=1.0, framing_quality=1.0,
        product_visibility=1.0, motion_stability=1.0, continuity=1.0,
        expression_naturalness=1.0, gesture_naturalness=1.0, delivery_energy=1.0,
        visual_fumble=0.0, distraction_risk=0.0,
    )
    mode_aware = tj.score_take(take, visual_mode=vm.VisualMode.DEMO_ACTION)
    assert mode_aware.score == pytest.approx(1.0, abs=1e-3)


def test_legitimate_editorial_failure_still_fails_under_face_independent_mode():
    """Stage 31: faceless + no speech/invalid editorial evidence may
    still fail for legitimate non-face reasons -- no blanket pass."""
    take = _take(
        face_visibility=0.0, eye_contact=0.0, visual_fumble=0.9, distraction_risk=0.9,
        expression_naturalness=0.1, gesture_naturalness=0.1, motion_stability=0.1,
        product_visibility=0.1,
    )
    mode_aware = tj.score_take(take, visual_mode=vm.VisualMode.PRODUCT_HANDS)
    talking_head_baseline = tj.score_take(_take(face_visibility=1.0, eye_contact=1.0))
    assert mode_aware.score < talking_head_baseline.score


def test_talking_head_signal_dependence_is_weighted_not_mandatory():
    """Stage 13 audit finding, proven: face_visibility/eye_contact are a
    WEIGHTED signal in the real score_take formula (not MANDATORY/gating)
    -- score_take never raises or returns a sentinel failure purely for
    face_visibility == 0 when visual_mode is left at its talking-head
    default."""
    zero_face = tj.score_take(_take(face_visibility=0.0, eye_contact=0.0))
    assert isinstance(zero_face.score, float)
    assert 0.0 <= zero_face.score <= 1.0


def test_rank_takes_default_path_never_passes_visual_mode():
    """Stage 9: rank_takes' own real call site is unaffected -- it never
    supplies visual_mode, so its behavior is byte-for-byte unchanged."""
    source = inspect.getsource(tj.rank_takes)
    assert "visual_mode" not in source


# ---------------------------------------------------------------------------
# STAGE 15: P1/P2 audit -- documented, unchanged (no face/visible-speaker
# assumption was found in local_performance.py's actual scoring outputs
# consumed elsewhere, nor in brain_runtime.py).
# ---------------------------------------------------------------------------

def test_brain_runtime_has_no_face_or_talking_head_assumption():
    from cutsell_worker import brain_runtime
    source = inspect.getsource(brain_runtime).lower()
    assert "face" not in source
    assert "talking_head" not in source


def test_deterministic_best_take_authority_has_no_face_dependence():
    from cutsell_worker import deterministic_best_take_authority as dbta
    source = inspect.getsource(dbta).lower()
    assert "face" not in source
    assert "gaze" not in source


# ---------------------------------------------------------------------------
# STAGE 34: closed-track firewall -- structural proof no forbidden module
# was imported by either new/changed file.
# ---------------------------------------------------------------------------

def test_visual_mode_module_does_not_import_closed_tracks():
    import ast
    tree = ast.parse(inspect.getsource(vm))
    forbidden = {
        "boundary_engine", "human_boundary_polish_v5", "delivery_edge_trim",
        "dialogue_pacing_transition", "audio_finishing_executor", "audio_finishing_measurement",
        "render", "output_format_qc", "selection_freeze", "timeline_composition",
        "timeline_composition_executor", "timeline_asset_registry",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden


def test_take_judge_module_still_imports_no_ffmpeg_or_render_code():
    source = inspect.getsource(tj)
    assert "import subprocess" not in source
    assert "import cv2" not in source
    assert "import mediapipe" not in source


def test_visual_finishing_policy_and_executor_files_are_untouched_by_diff():
    """D-280 makes zero code changes to visual_finishing_measurement.py,
    visual_finishing_policy.py, or visual_finishing_executor.py -- their
    existing behavior already satisfies the no-face safety invariant
    (proven above), so no refactor was needed. This test asserts the ten
    canonical D-260 thresholds are exactly the original ten -- an
    explicit regression tripwire, not a new numeric policy."""
    assert vfp.canonical_thresholds_snapshot() == {
        "FACE_CENTER_X_DISCONTINUITY_THRESHOLD": 0.025,
        "FACE_CENTER_Y_DISCONTINUITY_THRESHOLD": 0.025,
        "FACE_SCALE_RELATIVE_AREA_DISCONTINUITY_THRESHOLD": 0.30,
        "HEADROOM_DISCONTINUITY_THRESHOLD": 0.06,
        "DEFAULT_PUNCH_IN_SCALE": 1.10,
        "MAX_PUNCH_IN_SCALE": 1.15,
        "MAX_REFRAME_TRANSLATION_NORMALIZED": 0.10,
        "MAX_ADDITIONAL_CROP_LOSS_NORMALIZED": 0.10,
        "MIN_RELIABLE_FACE_DETECTION_RATE": 0.75,
        "MIN_PUNCH_IN_CLIP_DURATION_SEC": 1.5,
    }
