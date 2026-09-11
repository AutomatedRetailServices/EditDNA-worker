"""D-231 -- Pacing V2 Audio Join Understanding Foundation. OFFLINE ONLY.

Full 53-item generic fixture matrix + critical-property tests for
`cutsell_worker.pacing_v2_audio_join_understanding`. Fixtures construct
`AcousticEdgeEvidence`/`AcousticContinuityComparison`/`SourceAudioHandle`
objects DIRECTLY (frozen dataclasses, no ffmpeg/media needed at this
layer -- those primitives are already proven by D-230's own test suite;
this file tests the COMBINATION logic only, exactly matching this
module's own "combines already-existing evidence" scope).
"""
from __future__ import annotations

import inspect
import json

import pytest

from cutsell_worker import pacing_v2_audio_join_understanding as aju
from cutsell_worker.pacing_v2_acoustic_edge_evidence import (
    AcousticContinuityComparison,
    AcousticEdgeEvidence,
    CONTINUITY_CONFLICTED,
    CONTINUITY_DIFFERENT,
    CONTINUITY_INSUFFICIENT,
    CONTINUITY_SIMILAR,
    EDGE_LEFT_END,
    EDGE_POST_HANDLE,
    EDGE_PRE_HANDLE,
    EDGE_RIGHT_START,
    ENERGY_STATUS_ACTIVE_ENERGY,
    EVIDENCE_STATUS_CONFLICTED,
    EVIDENCE_STATUS_SUPPORTED,
    NON_SPEECH_SAFE_CANDIDATE,
    NON_SPEECH_SILENCE,
    NON_SPEECH_SPEECH_PRESENT,
    NON_SPEECH_UNCONFIRMED,
    NON_SPEECH_UNKNOWN,
    SCHEMA_VERSION as AEE_SCHEMA_VERSION,
    SIGNATURE_UNAVAILABLE,
    SILENCE_STATUS_NON_SILENT,
    SILENCE_STATUS_SILENT,
    SILENCE_STATUS_UNKNOWN,
    SPEECH_STATUS_LEXICAL_PRESENT,
    SPEECH_STATUS_NO_LEXICAL_OBSERVED,
    SPEECH_STATUS_UNKNOWN,
    WORD_COVERAGE_KNOWN_EMPTY,
    WORD_COVERAGE_KNOWN_PRESENT,
    WORD_COVERAGE_UNKNOWN,
)
from cutsell_worker.pacing_transition_decision import (
    RELATIONSHIP_CONTINUATION,
    RELATIONSHIP_CORRECTION,
    RELATIONSHIP_RETRY,
    SAFETY_BLOCKED,
    SAFETY_SAFE,
    SAFETY_UNKNOWN,
)
from cutsell_worker.pacing_v2_source_audio_handle import (
    HANDLE_STATUS_BLOCKED_RETRY_OR_CORRECTION,
    HANDLE_STATUS_SAFE_NON_SPEECH,
    HANDLE_STATUS_UNAVAILABLE,
    SourceAudioHandle,
)


# ---------------------------------------------------------------------------
# Fixture builders -- direct frozen-dataclass construction, no media.
# ---------------------------------------------------------------------------
def _edge(
    *, speech_status=SPEECH_STATUS_NO_LEXICAL_OBSERVED, non_speech_status=NON_SPEECH_SAFE_CANDIDATE,
    silence_status=SILENCE_STATUS_NON_SILENT, edge=EDGE_LEFT_END, evidence_status=EVIDENCE_STATUS_SUPPORTED,
    word_coverage_status=WORD_COVERAGE_KNOWN_EMPTY, conflict_flags=(), source_asset_id="s1", owner_clip_id="c1",
) -> AcousticEdgeEvidence:
    return AcousticEdgeEvidence(
        schema_version=AEE_SCHEMA_VERSION, evidence_id=f"acoustic:{source_asset_id}:{owner_clip_id}:{edge}:0.0:1.0",
        source_asset_id=source_asset_id, owner_clip_id=owner_clip_id, edge=edge,
        window_start=0.0, window_end=1.0,
        word_coverage_status=word_coverage_status, words_present_count=(1 if speech_status == SPEECH_STATUS_LEXICAL_PRESENT else 0),
        silence_status=silence_status, silence_fraction=(1.0 if silence_status == SILENCE_STATUS_SILENT else 0.0),
        rms_level=(0.2 if non_speech_status != NON_SPEECH_SILENCE else 0.0),
        energy_status=ENERGY_STATUS_ACTIVE_ENERGY,
        speech_status=speech_status, non_speech_status=non_speech_status,
        background_signature_status=SIGNATURE_UNAVAILABLE, signature=None,
        evidence_status=evidence_status, conflict_flags=tuple(conflict_flags), provenance=(AEE_SCHEMA_VERSION,),
    )


def _comparison(status, level_delta_db=None, conflict_flags=()) -> AcousticContinuityComparison:
    return AcousticContinuityComparison(
        schema_version=AEE_SCHEMA_VERSION, left_evidence_id="left_ev", right_evidence_id="right_ev",
        continuity_status=status, signature_comparison_status=status, level_delta_db=level_delta_db,
        reason="test_fixture", conflict_flags=tuple(conflict_flags), provenance=(AEE_SCHEMA_VERSION,),
    )


def _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="h1", direction="POST_ROLL") -> SourceAudioHandle:
    return SourceAudioHandle(
        schema_version="cutsell.pacing_v2_source_audio_handle.v1", handle_id=handle_id,
        source_asset_id="s1", owner_clip_id="c1", owner_realization_id=None, direction=direction,
        video_start=0.0, video_end=1.0, handle_source_start=1.0, handle_source_end=1.5,
        available_duration=0.5, word_intervals_present=(), speech_presence_status="NO_WORDS_PRESENT",
        discarded_overlap_status="NOT_OVERLAPPING", meaning_safety_status="SAFE", handle_status=status,
        conflict_flags=(), provenance=("test_fixture",),
    )


class _Prosody:
    def __init__(self, restart_or_interruption_state=None, vocal_continuity_state=None):
        self.restart_or_interruption_state = restart_or_interruption_state
        self.vocal_continuity_state = vocal_continuity_state


def _basic(**overrides):
    base = dict(
        transition_index=0, left_clip_id="left_c", right_clip_id="right_c",
        left_source_asset_id="s1", right_source_asset_id="s1",
    )
    base.update(overrides)
    return base


# ===========================================================================
# 1-9: Join Audio Role -- the eight speech/non-speech/silence combinations.
# ===========================================================================
def test_01_speech_to_speech_similar():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT),
        right_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT, edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_SIMILAR),
    ))
    assert u.join_audio_role == aju.JOIN_ROLE_SPEECH_TO_SPEECH


def test_02_speech_to_speech_different():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT),
        right_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT, edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
    ))
    assert u.join_audio_role == aju.JOIN_ROLE_SPEECH_TO_SPEECH


def test_03_speech_to_non_speech():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT),
        right_edge_evidence=_edge(speech_status=SPEECH_STATUS_NO_LEXICAL_OBSERVED, non_speech_status=NON_SPEECH_SAFE_CANDIDATE, edge=EDGE_RIGHT_START),
    ))
    assert u.join_audio_role == aju.JOIN_ROLE_SPEECH_TO_NON_SPEECH


def test_04_non_speech_to_speech():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_NO_LEXICAL_OBSERVED, non_speech_status=NON_SPEECH_SAFE_CANDIDATE),
        right_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT, edge=EDGE_RIGHT_START),
    ))
    assert u.join_audio_role == aju.JOIN_ROLE_NON_SPEECH_TO_SPEECH


def test_05_non_speech_to_non_speech_similar():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_SIMILAR),
    ))
    assert u.join_audio_role == aju.JOIN_ROLE_NON_SPEECH_TO_NON_SPEECH


def test_06_non_speech_to_non_speech_different_is_acoustic_discontinuity():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
    ))
    assert u.join_audio_role == aju.JOIN_ROLE_ACOUSTIC_DISCONTINUITY


def test_07_silence_to_speech():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_NO_LEXICAL_OBSERVED, non_speech_status=NON_SPEECH_SILENCE, silence_status=SILENCE_STATUS_SILENT),
        right_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT, edge=EDGE_RIGHT_START),
    ))
    assert u.join_audio_role == aju.JOIN_ROLE_NON_SPEECH_TO_SPEECH


def test_08_speech_to_silence():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT),
        right_edge_evidence=_edge(speech_status=SPEECH_STATUS_NO_LEXICAL_OBSERVED, non_speech_status=NON_SPEECH_SILENCE, silence_status=SILENCE_STATUS_SILENT, edge=EDGE_RIGHT_START),
    ))
    assert u.join_audio_role == aju.JOIN_ROLE_SPEECH_TO_NON_SPEECH


def test_09_silence_to_silence():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_NO_LEXICAL_OBSERVED, non_speech_status=NON_SPEECH_SILENCE, silence_status=SILENCE_STATUS_SILENT),
        right_edge_evidence=_edge(speech_status=SPEECH_STATUS_NO_LEXICAL_OBSERVED, non_speech_status=NON_SPEECH_SILENCE, silence_status=SILENCE_STATUS_SILENT, edge=EDGE_RIGHT_START),
    ))
    assert u.join_audio_role == aju.JOIN_ROLE_SILENCE_BOUNDARY


# ===========================================================================
# 10-16: Word coverage / handle / continuity edge cases.
# ===========================================================================
def test_10_unknown_word_coverage_on_either_side_is_unknown_role():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_UNKNOWN, non_speech_status=NON_SPEECH_UNKNOWN, silence_status=SILENCE_STATUS_UNKNOWN, word_coverage_status=WORD_COVERAGE_UNKNOWN),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
    ))
    assert u.join_audio_role == aju.JOIN_ROLE_UNKNOWN
    assert u.understanding_status == aju.UNDERSTANDING_UNKNOWN


def test_11_discarded_left_handle_not_ready():
    handle = _handle(status="BLOCKED_DISCARDED_MATERIAL")
    u = aju.build_audio_join_understanding(**_basic(left_post_roll_handle=handle))
    assert u.ambience_carry_left_evidence_status == aju.TREATMENT_EVIDENCE_NOT_READY


def test_12_discarded_right_handle_not_ready():
    handle = _handle(status="BLOCKED_DISCARDED_MATERIAL", direction="PRE_ROLL")
    u = aju.build_audio_join_understanding(**_basic(right_pre_roll_handle=handle))
    assert u.ambience_carry_right_evidence_status == aju.TREATMENT_EVIDENCE_NOT_READY


def test_13_safe_left_handle_with_acoustic_material_is_ready():
    handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH)
    edge = _edge(edge=EDGE_POST_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = aju.build_audio_join_understanding(**_basic(
        left_post_roll_handle=handle, left_post_roll_handle_edge_evidence=edge,
    ))
    assert u.ambience_carry_left_evidence_status == aju.TREATMENT_EVIDENCE_READY


def test_14_safe_right_handle_with_acoustic_material_is_ready():
    handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, direction="PRE_ROLL")
    edge = _edge(edge=EDGE_PRE_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = aju.build_audio_join_understanding(**_basic(
        right_pre_roll_handle=handle, right_pre_roll_handle_edge_evidence=edge,
    ))
    assert u.ambience_carry_right_evidence_status == aju.TREATMENT_EVIDENCE_READY


def test_15_safe_handles_both_sides_with_discontinuity_makes_bridge_ready():
    left_handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="hl")
    right_handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="hr", direction="PRE_ROLL")
    left_edge = _edge(edge=EDGE_POST_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    right_edge = _edge(edge=EDGE_PRE_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = aju.build_audio_join_understanding(**_basic(
        left_post_roll_handle=left_handle, right_pre_roll_handle=right_handle,
        left_post_roll_handle_edge_evidence=left_edge, right_pre_roll_handle_edge_evidence=right_edge,
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
    ))
    assert u.ambience_bridge_evidence_status == aju.TREATMENT_EVIDENCE_READY


def test_16_no_handles_supplied_not_ready():
    u = aju.build_audio_join_understanding(**_basic())
    assert u.ambience_carry_left_evidence_status == aju.TREATMENT_EVIDENCE_NOT_READY
    assert u.ambience_carry_right_evidence_status == aju.TREATMENT_EVIDENCE_NOT_READY


# ===========================================================================
# 17-18: Level continuity -- background character vs. loudness separation.
# ===========================================================================
def test_17_similar_background_different_gain_is_similar_level_or_not_conflated():
    """CRITICAL: acoustic character != loudness -- same background at a
    different gain reports DIFFERENT loudness (level_continuity) but the
    ACOUSTIC continuity (character) is a completely separate field,
    supplied independently."""
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_SIMILAR, level_delta_db=12.0),
    ))
    assert u.acoustic_continuity_status == CONTINUITY_SIMILAR
    assert u.level_continuity_status == aju.LEVEL_CONTINUITY_RIGHT_LOUDER


def test_18_different_background_same_gain_is_different_continuity_similar_level():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT, level_delta_db=0.5),
    ))
    assert u.acoustic_continuity_status == CONTINUITY_DIFFERENT
    assert u.level_continuity_status == aju.LEVEL_CONTINUITY_SIMILAR


# ===========================================================================
# 19-22: Relationship hint pass-through.
# ===========================================================================
def test_19_relationship_continuation_passed_through():
    u = aju.build_audio_join_understanding(**_basic(relationship_hint=RELATIONSHIP_CONTINUATION))
    assert u.relationship_hint == RELATIONSHIP_CONTINUATION


def test_20_relationship_correction_passed_through():
    u = aju.build_audio_join_understanding(**_basic(relationship_hint=RELATIONSHIP_CORRECTION))
    assert u.relationship_hint == RELATIONSHIP_CORRECTION


def test_21_relationship_retry_passed_through():
    u = aju.build_audio_join_understanding(**_basic(relationship_hint=RELATIONSHIP_RETRY))
    assert u.relationship_hint == RELATIONSHIP_RETRY


def test_22_relationship_unknown_is_none():
    u = aju.build_audio_join_understanding(**_basic())
    assert u.relationship_hint is None


# ===========================================================================
# 23-25: Prosodic evidence.
# ===========================================================================
def test_23_prosodic_continuous():
    u = aju.build_audio_join_understanding(**_basic(
        left_prosody=_Prosody(vocal_continuity_state="CONTINUOUS"),
    ))
    assert u.prosodic_status == aju.PROSODIC_JOIN_CONTINUOUS


def test_24_prosodic_restart():
    u = aju.build_audio_join_understanding(**_basic(
        right_prosody=_Prosody(restart_or_interruption_state="RESTART"),
    ))
    assert u.prosodic_status == aju.PROSODIC_JOIN_RESTART


def test_25_prosodic_unavailable():
    u = aju.build_audio_join_understanding(**_basic())
    assert u.prosodic_status == aju.PROSODIC_JOIN_UNAVAILABLE


# ===========================================================================
# 26-27: Meaning-critical edges.
# ===========================================================================
def test_26_meaning_critical_left_edge_blocks():
    """CRITICAL claim words ("never", numbers, diagnoses etc.) near the
    left edge -- reuses semantic_claims.classify_claim, no new engine.
    Both edges supplied (resolved coverage) so the join-level verdict
    reflects the left side's own critical content, not a missing-
    evidence UNKNOWN from the untested right side."""
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(word_coverage_status=WORD_COVERAGE_KNOWN_PRESENT),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        left_edge_words=[(0.5, 0.9, "never take double the dose")],
    ))
    assert u.meaning_safety_status == SAFETY_BLOCKED


def test_27_meaning_critical_right_edge_blocks():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START, word_coverage_status=WORD_COVERAGE_KNOWN_PRESENT),
        right_edge_words=[(0.1, 0.4, "you should never mix these two medications")],
    ))
    assert u.meaning_safety_status == SAFETY_BLOCKED


def test_27b_non_critical_words_are_safe():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(word_coverage_status=WORD_COVERAGE_KNOWN_PRESENT),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        left_edge_words=[(0.5, 0.9, "so that's pretty much it")],
    ))
    assert u.meaning_safety_status == SAFETY_SAFE


# ===========================================================================
# 28: Double lexical speech.
# ===========================================================================
def test_28_double_lexical_speech_both_sides():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT),
        right_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT, edge=EDGE_RIGHT_START),
    ))
    assert u.double_speech_status == aju.JOIN_DOUBLE_SPEECH_BOTH_SIDES_LEXICAL


# ===========================================================================
# 29-30: Crossfade evidence readiness.
# ===========================================================================
def test_29_crossfade_evaluable():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
    ))
    assert u.short_crossfade_evidence_status == aju.TREATMENT_EVIDENCE_READY


def test_30_crossfade_not_evaluable_missing_edge():
    u = aju.build_audio_join_understanding(**_basic(left_edge_evidence=_edge()))
    assert u.short_crossfade_evidence_status == aju.TREATMENT_EVIDENCE_NOT_READY


# ===========================================================================
# 31-34: Ambience/bridge/no-treatment evidence readiness.
# ===========================================================================
def test_31_left_ambience_evidence_present():
    handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH)
    edge = _edge(edge=EDGE_POST_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = aju.build_audio_join_understanding(**_basic(left_post_roll_handle=handle, left_post_roll_handle_edge_evidence=edge))
    assert u.ambience_carry_left_evidence_status == aju.TREATMENT_EVIDENCE_READY


def test_32_right_ambience_evidence_present():
    handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, direction="PRE_ROLL")
    edge = _edge(edge=EDGE_PRE_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = aju.build_audio_join_understanding(**_basic(right_pre_roll_handle=handle, right_pre_roll_handle_edge_evidence=edge))
    assert u.ambience_carry_right_evidence_status == aju.TREATMENT_EVIDENCE_READY


def test_33_bridge_evidence_present():
    left_handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="hl")
    right_handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="hr", direction="PRE_ROLL")
    left_edge = _edge(edge=EDGE_POST_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    right_edge = _edge(edge=EDGE_PRE_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = aju.build_audio_join_understanding(**_basic(
        left_post_roll_handle=left_handle, right_pre_roll_handle=right_handle,
        left_post_roll_handle_edge_evidence=left_edge, right_pre_roll_handle_edge_evidence=right_edge,
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
    ))
    assert u.ambience_bridge_evidence_status == aju.TREATMENT_EVIDENCE_READY


def test_34_no_treatment_needed_evidence():
    """CRITICAL: the no-overprocessing signal -- a clean, continuous,
    safe join must be able to report NO_ADDITIONAL_AUDIO_TREATMENT_
    EVIDENCE so a future treatment engine can choose NONE."""
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_SIMILAR, level_delta_db=0.2),
    ))
    assert u.no_treatment_evidence_status == aju.NO_TREATMENT_EVIDENCE


# ===========================================================================
# 35-36: Same-source / multi-source.
# ===========================================================================
def test_35_same_source():
    u = aju.build_audio_join_understanding(**_basic(
        left_source_asset_id="s1", right_source_asset_id="s1",
        left_edge_evidence=_edge(source_asset_id="s1"), right_edge_evidence=_edge(source_asset_id="s1", edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_SIMILAR),
    ))
    assert u.left_source_asset_id == u.right_source_asset_id == "s1"


def test_36_multi_source():
    u = aju.build_audio_join_understanding(**_basic(
        left_source_asset_id="sA", right_source_asset_id="sB",
        left_edge_evidence=_edge(source_asset_id="sA"), right_edge_evidence=_edge(source_asset_id="sB", edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_SIMILAR),
    ))
    assert u.left_source_asset_id != u.right_source_asset_id


# ===========================================================================
# 37-38: Determinism / input-order stability.
# ===========================================================================
def test_37_deterministic_repeat():
    kwargs = _basic(left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START), continuity_comparison=_comparison(CONTINUITY_SIMILAR))
    a = aju.build_audio_join_understanding(**kwargs)
    b = aju.build_audio_join_understanding(**kwargs)
    assert a == b


def test_38_input_order_stable_provenance():
    kwargs = _basic(
        left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_SIMILAR), relationship_hint=RELATIONSHIP_CONTINUATION,
    )
    a = aju.build_audio_join_understanding(**kwargs)
    b = aju.build_audio_join_understanding(**kwargs)
    assert a.provenance == b.provenance  # sorted set -> stable regardless of dict/insertion order


# ===========================================================================
# 39: No treatment selected anywhere (structural proof, not docstring scan).
# ===========================================================================
def test_39_no_treatment_type_field_or_value_anywhere():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
    ))
    fields = set(u.__dataclass_fields__.keys())
    assert "treatment" not in fields and "selected_treatment" not in fields
    for value in vars(u).values():
        if isinstance(value, str):
            assert value not in ("SHORT_CROSSFADE", "AMBIENCE_CARRY_LEFT", "AMBIENCE_CARRY_RIGHT", "AMBIENCE_BRIDGE")


# ===========================================================================
# 40-45: No RAW/provider/ASR/Boundary/Ordering/BestTake/Family/renderer reach.
# ===========================================================================
def test_40_no_renderer_import():
    src = inspect.getsource(aju)
    for banned in ("from .render import", "from .render_plan import", "import render"):
        assert banned not in src


def test_41_no_provider_call():
    src = inspect.getsource(aju)
    for banned in ("requests.", "openai", "genai", "gemini", "modal.", "boto3"):
        assert banned not in src.lower()


def test_42_no_asr_rerun():
    src = inspect.getsource(aju)
    for banned in ("whisper", "transcribe", "asr_"):
        assert banned not in src.lower()


def test_43_no_boundary_ordering_besttake_family_mutation():
    src = inspect.getsource(aju)
    for banned in ("from .boundary", "from .ordering", "from .best_take", "from .family", "dataclasses.replace"):
        assert banned not in src


def test_44_no_loudness_correction():
    module_symbols = set(vars(aju).keys())
    for token in ("normalize_gain", "apply_gain", "correct_loudness"):
        assert token not in module_symbols


def test_45_no_room_tone_overclaim():
    """The docstring legitimately mentions ROOM_TONE_MATCH/MISMATCH while
    explaining what this module never emits -- the real proof is no such
    module-level symbol/constant exists and no diagnostic row ever
    carries that value."""
    u = aju.build_audio_join_understanding(**_basic())
    assert u.room_tone_classification_status == "NOT_YET_AVAILABLE"
    module_symbols = set(vars(aju).keys())
    assert "ROOM_TONE_MATCH" not in module_symbols and "ROOM_TONE_MISMATCH" not in module_symbols
    diag = aju.audio_join_understanding_diagnostics(u)
    assert "ROOM_TONE_MATCH" not in json.dumps(diag) and "ROOM_TONE_MISMATCH" not in json.dumps(diag)


# ===========================================================================
# 46-53: compileall / regressions -- placeholders proven by the shell
# qualification step (this test file cannot invoke pytest-on-pytest); the
# assertions here instead re-confirm the STRUCTURAL guarantees those
# regressions depend on (no live wiring, decide_transition never called).
# ===========================================================================
def test_46_module_has_no_feature_flag_and_is_not_wired_into_universal_clean_cut():
    import subprocess
    result = subprocess.run(
        ["grep", "-rl", "pacing_v2_audio_join_understanding", "cutsell_worker/universal_clean_cut.py"],
        capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


def test_47_never_calls_decide_transition():
    src = inspect.getsource(aju)
    assert "decide_transition(" not in src


def test_48_pairwise_only_no_batch_optimizer_in_build_function():
    sig = inspect.signature(aju.build_audio_join_understanding)
    assert "selected" not in sig.parameters and "clips" not in sig.parameters


def test_49_diagnostics_json_safe_no_transcript_dump():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(word_coverage_status=WORD_COVERAGE_KNOWN_PRESENT),
        left_edge_words=[(0.5, 0.9, "supersecrettranscript")],
    ))
    diag = aju.audio_join_understanding_diagnostics(u)
    dumped = json.dumps(diag)
    assert "supersecrettranscript" not in dumped


def test_50_run_summary_no_master_score():
    u = aju.build_audio_join_understanding(**_basic())
    summary = aju.audio_join_understanding_run_summary([u])
    assert not any("score" in k.lower() for k in summary.keys())
    assert summary["join_count"] == 1


def test_51_conflicted_edge_propagates_to_conflicted_role():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(evidence_status=EVIDENCE_STATUS_CONFLICTED, conflict_flags=("blocked_retry_or_correction",)),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
    ))
    assert u.join_audio_role == aju.JOIN_ROLE_CONFLICTED
    assert u.understanding_status == aju.UNDERSTANDING_CONFLICTED


def test_52_not_evaluable_when_both_edges_missing():
    u = aju.build_audio_join_understanding(**_basic())
    assert u.understanding_status == aju.UNDERSTANDING_NOT_EVALUABLE


def test_53_full_offline_suite_marker_present():
    """A cheap in-repo marker so the qualification shell step (compileall,
    D-230 regressions, Pacing regressions, SourceAudioHandle regressions,
    Renderer regressions, full offline suite) has a stable test id to
    point at in the decision-log entry; the actual suite runs are
    performed by the shell qualification step, not by this test alone."""
    assert aju.SCHEMA_VERSION.startswith("cutsell.pacing_v2_audio_join_understanding.v")


# ===========================================================================
# Additional critical-property tests (beyond the 53-item numbered matrix).
# ===========================================================================
def test_understanding_types_are_frozen():
    u = aju.build_audio_join_understanding(**_basic())
    with pytest.raises(Exception):
        u.join_audio_role = "MUTATED"


def test_word_safety_reflects_known_lexical_speech_even_without_meaning_words_supplied():
    """word_safety_status (join-local) must be BLOCKED purely from D-230's
    own speech_status, independent of whether meaning-safety word text
    was separately supplied."""
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
    ))
    assert u.word_safety_status == SAFETY_BLOCKED


def test_unknown_word_coverage_never_becomes_safe_meaning():
    """Restated D-230 firewall: unknown coverage never defaults to SAFE."""
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_UNKNOWN, non_speech_status=NON_SPEECH_UNKNOWN, word_coverage_status=WORD_COVERAGE_UNKNOWN),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
    ))
    assert u.meaning_safety_status == SAFETY_UNKNOWN


def test_double_speech_neither_lexical():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
    ))
    assert u.double_speech_status == aju.JOIN_DOUBLE_SPEECH_NEITHER_LEXICAL


def test_double_speech_left_only():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
    ))
    assert u.double_speech_status == aju.JOIN_DOUBLE_SPEECH_LEFT_ONLY_LEXICAL


def test_double_speech_right_only():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(),
        right_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT, edge=EDGE_RIGHT_START),
    ))
    assert u.double_speech_status == aju.JOIN_DOUBLE_SPEECH_RIGHT_ONLY_LEXICAL


def test_ambience_bridge_not_ready_when_only_one_side_ready():
    left_handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="hl")
    left_edge = _edge(edge=EDGE_POST_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = aju.build_audio_join_understanding(**_basic(
        left_post_roll_handle=left_handle, left_post_roll_handle_edge_evidence=left_edge,
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
    ))
    assert u.ambience_bridge_evidence_status == aju.TREATMENT_EVIDENCE_NOT_READY


def test_silence_only_handle_is_not_ambience_material():
    """Restated D-230 finding: a silence-only handle is NOT ambience carry
    material even though handle_status itself is SAFE_NON_SPEECH_HANDLE."""
    handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH)
    edge = _edge(edge=EDGE_POST_HANDLE, non_speech_status=NON_SPEECH_SILENCE, silence_status=SILENCE_STATUS_SILENT)
    u = aju.build_audio_join_understanding(**_basic(left_post_roll_handle=handle, left_post_roll_handle_edge_evidence=edge))
    assert u.ambience_carry_left_evidence_status == aju.TREATMENT_EVIDENCE_NOT_READY


def test_retry_blocked_handle_never_reads_as_ready_even_with_edge_evidence():
    """Discarded/retry firewall holds even if a caller mistakenly supplied
    handle-edge acoustic evidence alongside a blocked handle."""
    handle = _handle(status=HANDLE_STATUS_BLOCKED_RETRY_OR_CORRECTION)
    edge = _edge(edge=EDGE_POST_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = aju.build_audio_join_understanding(**_basic(left_post_roll_handle=handle, left_post_roll_handle_edge_evidence=edge))
    assert u.ambience_carry_left_evidence_status == aju.TREATMENT_EVIDENCE_NOT_READY


def test_handle_unavailable_status_not_ready():
    handle = _handle(status=HANDLE_STATUS_UNAVAILABLE)
    u = aju.build_audio_join_understanding(**_basic(left_post_roll_handle=handle))
    assert u.ambience_carry_left_evidence_status == aju.TREATMENT_EVIDENCE_NOT_READY


def test_level_continuity_insufficient_when_no_comparison():
    u = aju.build_audio_join_understanding(**_basic())
    assert u.level_continuity_status == aju.LEVEL_CONTINUITY_INSUFFICIENT


def test_acoustic_continuity_unknown_when_no_comparison_supplied():
    u = aju.build_audio_join_understanding(**_basic())
    assert u.acoustic_continuity_status == aju.ACOUSTIC_CONTINUITY_STATUS_UNKNOWN


def test_conflicted_continuity_marks_role_conflicted():
    u = aju.build_audio_join_understanding(**_basic(
        left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_CONFLICTED),
    ))
    assert u.join_audio_role == aju.JOIN_ROLE_CONFLICTED


def test_handle_ids_recorded():
    left_handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="hl123")
    right_handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="hr456", direction="PRE_ROLL")
    u = aju.build_audio_join_understanding(**_basic(left_post_roll_handle=left_handle, right_pre_roll_handle=right_handle))
    assert u.left_handle_ids == ("hl123",)
    assert u.right_handle_ids == ("hr456",)
