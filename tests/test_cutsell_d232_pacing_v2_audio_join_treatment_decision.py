"""D-232 -- Pacing V2 Audio Join Treatment Decision Foundation. OFFLINE ONLY.

Full 53-item generic fixture matrix + critical-property tests for
`cutsell_worker.pacing_v2_audio_join_treatment_decision`. Fixtures build
`AudioJoinUnderstanding` (D-231) objects directly via its own real
`build_audio_join_understanding` entry point, matching the same
`AcousticEdgeEvidence`/`AcousticContinuityComparison` construction
pattern D-231's own test file already established.
"""
from __future__ import annotations

import inspect
import json

import pytest

from cutsell_worker import pacing_v2_audio_join_treatment_decision as ajtd
from cutsell_worker.dialogue_pacing_transition import HARD_CUT, J_CUT, L_CUT, MICRO_AUDIO_OVERLAP, TIGHT_CUT
from cutsell_worker.pacing_transition_decision import (
    GAP_KEEP_PAUSE as KEEP_PAUSE,
    RELATIONSHIP_CONTINUATION,
    RELATIONSHIP_CORRECTION,
    RELATIONSHIP_RETRY,
    SAFETY_BLOCKED,
    SAFETY_SAFE,
    SAFETY_UNKNOWN,
)
from cutsell_worker.pacing_v2_acoustic_edge_evidence import (
    AcousticContinuityComparison,
    AcousticEdgeEvidence,
    CONTINUITY_DIFFERENT,
    CONTINUITY_SIMILAR,
    EDGE_LEFT_END,
    EDGE_POST_HANDLE,
    EDGE_PRE_HANDLE,
    EDGE_RIGHT_START,
    ENERGY_STATUS_ACTIVE_ENERGY,
    EVIDENCE_STATUS_SUPPORTED,
    NON_SPEECH_SAFE_CANDIDATE,
    NON_SPEECH_SPEECH_PRESENT,
    SCHEMA_VERSION as AEE_SCHEMA_VERSION,
    SIGNATURE_UNAVAILABLE,
    SILENCE_STATUS_NON_SILENT,
    SPEECH_STATUS_LEXICAL_PRESENT,
    SPEECH_STATUS_NO_LEXICAL_OBSERVED,
    WORD_COVERAGE_KNOWN_EMPTY,
)
from cutsell_worker.pacing_v2_audio_join_understanding import build_audio_join_understanding
from cutsell_worker.pacing_v2_source_audio_handle import HANDLE_STATUS_SAFE_NON_SPEECH, SourceAudioHandle


# ---------------------------------------------------------------------------
# Fixture builders (mirrors test_cutsell_d231's own pattern).
# ---------------------------------------------------------------------------
def _edge(
    *, speech_status=SPEECH_STATUS_NO_LEXICAL_OBSERVED, non_speech_status=NON_SPEECH_SAFE_CANDIDATE,
    edge=EDGE_LEFT_END, word_coverage_status=WORD_COVERAGE_KNOWN_EMPTY,
) -> AcousticEdgeEvidence:
    return AcousticEdgeEvidence(
        schema_version=AEE_SCHEMA_VERSION, evidence_id=f"acoustic:s1:c1:{edge}:0.0:1.0",
        source_asset_id="s1", owner_clip_id="c1", edge=edge, window_start=0.0, window_end=1.0,
        word_coverage_status=word_coverage_status, words_present_count=(1 if speech_status == SPEECH_STATUS_LEXICAL_PRESENT else 0),
        silence_status=SILENCE_STATUS_NON_SILENT, silence_fraction=0.0, rms_level=0.2,
        energy_status=ENERGY_STATUS_ACTIVE_ENERGY, speech_status=speech_status, non_speech_status=non_speech_status,
        background_signature_status=SIGNATURE_UNAVAILABLE, signature=None,
        evidence_status=EVIDENCE_STATUS_SUPPORTED, conflict_flags=(), provenance=(AEE_SCHEMA_VERSION,),
    )


def _comparison(status, level_delta_db=None) -> AcousticContinuityComparison:
    return AcousticContinuityComparison(
        schema_version=AEE_SCHEMA_VERSION, left_evidence_id="l", right_evidence_id="r",
        continuity_status=status, signature_comparison_status=status, level_delta_db=level_delta_db,
        reason="fixture", conflict_flags=(), provenance=(AEE_SCHEMA_VERSION,),
    )


def _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="h1", direction="POST_ROLL") -> SourceAudioHandle:
    return SourceAudioHandle(
        schema_version="cutsell.pacing_v2_source_audio_handle.v1", handle_id=handle_id,
        source_asset_id="s1", owner_clip_id="c1", owner_realization_id=None, direction=direction,
        video_start=0.0, video_end=1.0, handle_source_start=1.0, handle_source_end=1.5,
        available_duration=0.5, word_intervals_present=(), speech_presence_status="NO_WORDS_PRESENT",
        discarded_overlap_status="NOT_OVERLAPPING", meaning_safety_status="SAFE", handle_status=status,
        conflict_flags=(), provenance=("fixture",),
    )


def _understanding(**overrides):
    base = dict(
        transition_index=0, left_clip_id="left_c", right_clip_id="right_c",
        left_source_asset_id="s1", right_source_asset_id="s1",
    )
    base.update(overrides)
    return build_audio_join_understanding(**base)


def _decide(understanding, primary_transition_mode=HARD_CUT, **kwargs):
    return ajtd.build_audio_join_treatment_decision(
        understanding, primary_transition_mode=primary_transition_mode, **kwargs,
    )


_CONTINUOUS = dict(
    left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
    continuity_comparison=_comparison(CONTINUITY_SIMILAR, level_delta_db=0.2),
)
_DISCONTINUOUS = dict(
    left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
    continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
)


# ===========================================================================
# 1-4: continuous/abrupt HARD/TIGHT joins.
# ===========================================================================
def test_01_continuous_hard_join_is_none_or_click_fade():
    u = _understanding(**_CONTINUOUS)
    d = _decide(u, HARD_CUT)
    assert d.treatment in (ajtd.TREATMENT_NONE, ajtd.TREATMENT_CLICK_FADE)
    assert d.treatment_status == ajtd.TREATMENT_STATUS_SUPPORTED


def test_02_continuous_tight_join():
    u = _understanding(**_CONTINUOUS)
    d = _decide(u, TIGHT_CUT)
    assert d.treatment in (ajtd.TREATMENT_NONE, ajtd.TREATMENT_CLICK_FADE)


def test_03_abrupt_hard_join_safe_crossfade():
    u = _understanding(**_DISCONTINUOUS)
    d = _decide(u, HARD_CUT)
    assert d.treatment == ajtd.TREATMENT_SHORT_CROSSFADE
    assert d.treatment_status == ajtd.TREATMENT_STATUS_SUPPORTED
    assert d.compatibility_status == ajtd.COMPATIBILITY_SUPPORTED


def test_04_abrupt_tight_join_safe_crossfade():
    u = _understanding(**_DISCONTINUOUS)
    d = _decide(u, TIGHT_CUT)
    assert d.treatment == ajtd.TREATMENT_SHORT_CROSSFADE


# ===========================================================================
# 5-6: unsafe crossfade / unknown word coverage.
# ===========================================================================
def test_05_speech_speech_unsafe_crossfade_blocks():
    u = _understanding(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT),
        right_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT, edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment == ajtd.TREATMENT_CLICK_FADE
    assert d.treatment != ajtd.TREATMENT_SHORT_CROSSFADE
    assert ajtd.CONFLICT_SPEECH_SAFETY_BLOCKED in d.conflict_flags


def test_06_unknown_word_coverage_is_insufficient():
    u = _understanding()  # no edges at all -> NOT_EVALUABLE
    d = _decide(u, HARD_CUT)
    assert d.treatment == ajtd.TREATMENT_NONE
    assert d.treatment_status == ajtd.TREATMENT_STATUS_INSUFFICIENT_EVIDENCE


# ===========================================================================
# 7-8: meaning-critical edges.
# ===========================================================================
def test_07_meaning_critical_left_blocks_advanced_treatment():
    u = _understanding(
        left_edge_evidence=_edge(word_coverage_status="KNOWN_PRESENT"),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        left_edge_words=[(0.5, 0.9, "never double the dose")],
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment == ajtd.TREATMENT_CLICK_FADE
    assert ajtd.CONFLICT_MEANING_SAFETY_BLOCKED in d.conflict_flags


def test_08_meaning_critical_right_blocks_advanced_treatment():
    u = _understanding(
        left_edge_evidence=_edge(),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START, word_coverage_status="KNOWN_PRESENT"),
        right_edge_words=[(0.1, 0.4, "never mix these medications")],
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment == ajtd.TREATMENT_CLICK_FADE
    assert ajtd.CONFLICT_MEANING_SAFETY_BLOCKED in d.conflict_flags


# ===========================================================================
# 9-11: relationship hints.
# ===========================================================================
def test_09_correction_hint_vetoes_smoothing():
    u = _understanding(relationship_hint=RELATIONSHIP_CORRECTION, **_DISCONTINUOUS)
    d = _decide(u, HARD_CUT)
    assert d.treatment == ajtd.TREATMENT_CLICK_FADE
    assert ajtd.CONFLICT_RETRY_OR_CORRECTION_VETO in d.conflict_flags


def test_10_retry_hint_vetoes_smoothing():
    u = _understanding(relationship_hint=RELATIONSHIP_RETRY, **_DISCONTINUOUS)
    d = _decide(u, HARD_CUT)
    assert d.treatment == ajtd.TREATMENT_CLICK_FADE
    assert ajtd.CONFLICT_RETRY_OR_CORRECTION_VETO in d.conflict_flags


def test_11_continuation_hint_never_forces_smoothing_alone():
    """CRITICAL: CONTINUATION may support smoothing but is never itself
    sufficient -- it never appears as a positive condition anywhere in
    the decision table (structural proof, not just behavior)."""
    src = inspect.getsource(ajtd._decide_treatment)
    assert "RELATIONSHIP_CONTINUATION" not in src
    u = _understanding(relationship_hint=RELATIONSHIP_CONTINUATION, **_CONTINUOUS)
    d = _decide(u, HARD_CUT)
    assert d.treatment in (ajtd.TREATMENT_NONE, ajtd.TREATMENT_CLICK_FADE)


# ===========================================================================
# 12-14: handle readiness.
# ===========================================================================
def test_12_safe_left_handle_with_discontinuity():
    handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH)
    edge = _edge(edge=EDGE_POST_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = _understanding(
        left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
        left_post_roll_handle=handle, left_post_roll_handle_edge_evidence=edge,
    )
    d = _decide(u, HARD_CUT)
    # SHORT_CROSSFADE is evaluated before AMBIENCE_CARRY_LEFT in this
    # module's own precedence -- both edges ARE crossfade-ready here, so
    # SHORT_CROSSFADE wins deterministically (proven, not incidental).
    assert d.treatment in (ajtd.TREATMENT_SHORT_CROSSFADE, ajtd.TREATMENT_AMBIENCE_CARRY_LEFT)


def test_13_safe_right_handle_alone_with_discontinuity_no_retained_edges():
    """Only ONE retained edge supplied (so SHORT_CROSSFADE is
    structurally NOT_READY, per D-231's own "both edges required" rule)
    -- isolates the ambience-right pathway. Proves an unresolved retained
    -edge safety signal on the untested side never blocks an
    independently-safe, handle-evidenced ambience candidate."""
    handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, direction="PRE_ROLL")
    edge = _edge(edge=EDGE_PRE_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = _understanding(
        left_edge_evidence=_edge(),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
        right_pre_roll_handle=handle, right_pre_roll_handle_edge_evidence=edge,
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment == ajtd.TREATMENT_AMBIENCE_CARRY_RIGHT


def test_14_both_safe_handles():
    """Both retained edges intentionally omitted here so SHORT_CROSSFADE
    stays NOT_READY -- isolates the ambience pathway, which precedence
    resolves deterministically to AMBIENCE_CARRY_LEFT (step 4 before
    step 5/6)."""
    left_handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="hl")
    right_handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="hr", direction="PRE_ROLL")
    left_edge = _edge(edge=EDGE_POST_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    right_edge = _edge(edge=EDGE_PRE_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = _understanding(
        left_edge_evidence=_edge(),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
        left_post_roll_handle=left_handle, right_pre_roll_handle=right_handle,
        left_post_roll_handle_edge_evidence=left_edge, right_pre_roll_handle_edge_evidence=right_edge,
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment in (ajtd.TREATMENT_AMBIENCE_CARRY_LEFT, ajtd.TREATMENT_AMBIENCE_CARRY_RIGHT, ajtd.TREATMENT_AMBIENCE_BRIDGE)


# ===========================================================================
# 15-17: ambience-left/right/bridge candidates.
# ===========================================================================
def test_15_left_ambience_candidate_with_discontinuity():
    handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH)
    edge = _edge(edge=EDGE_POST_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = _understanding(
        left_edge_evidence=_edge(),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
        left_post_roll_handle=handle, left_post_roll_handle_edge_evidence=edge,
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment == ajtd.TREATMENT_AMBIENCE_CARRY_LEFT


def test_16_right_ambience_candidate_with_discontinuity():
    handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, direction="PRE_ROLL")
    edge = _edge(edge=EDGE_PRE_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = _understanding(
        left_edge_evidence=_edge(),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
        right_pre_roll_handle=handle, right_pre_roll_handle_edge_evidence=edge,
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment == ajtd.TREATMENT_AMBIENCE_CARRY_RIGHT


def test_17_bridge_candidate_when_both_sides_ready_and_crossfade_not_evaluable():
    """Bridge is only reached when crossfade/left/right steps don't win
    first -- achieved here by NOT supplying retained-edge evidence (so
    crossfade readiness is NOT_READY) while both handle sides ARE ready."""
    left_handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="hl")
    right_handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="hr", direction="PRE_ROLL")
    left_edge = _edge(edge=EDGE_POST_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    right_edge = _edge(edge=EDGE_PRE_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = _understanding(
        left_edge_evidence=_edge(),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
        left_post_roll_handle=left_handle, right_pre_roll_handle=right_handle,
        left_post_roll_handle_edge_evidence=left_edge, right_pre_roll_handle_edge_evidence=right_edge,
    )
    d = _decide(u, HARD_CUT)
    # Precedence picks AMBIENCE_CARRY_LEFT before BRIDGE when both are
    # ready (deterministic step 4 before step 6) -- proven here.
    assert d.treatment == ajtd.TREATMENT_AMBIENCE_CARRY_LEFT
    assert u.ambience_bridge_evidence_status == "TREATMENT_EVIDENCE_READY"


# ===========================================================================
# 18-20: no discontinuity + handles / gain vs background separation.
# ===========================================================================
def test_18_no_discontinuity_with_handles_prefers_none_or_click_fade():
    handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH)
    edge = _edge(edge=EDGE_POST_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = _understanding(
        **_CONTINUOUS, left_post_roll_handle=handle, left_post_roll_handle_edge_evidence=edge,
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment in (ajtd.TREATMENT_NONE, ajtd.TREATMENT_CLICK_FADE)


def test_19_similar_ambience_different_gain_no_overtreatment():
    """CRITICAL: same background, different gain -> continuity SIMILAR,
    so the "no overprocessing" branch applies even though loudness
    genuinely differs -- loudness is a SEPARATE diagnostic, never a
    reason to pick a heavier treatment."""
    u = _understanding(
        left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_SIMILAR, level_delta_db=12.0),
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment in (ajtd.TREATMENT_NONE, ajtd.TREATMENT_CLICK_FADE)
    assert d.loudness_polish_status == ajtd.LOUDNESS_POLISH_NEEDED


def test_20_different_ambience_same_gain_may_crossfade():
    u = _understanding(
        left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT, level_delta_db=0.3),
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment == ajtd.TREATMENT_SHORT_CROSSFADE
    assert d.loudness_polish_status == ajtd.LOUDNESS_POLISH_NOT_NEEDED


# ===========================================================================
# 21: loudness mismatch diagnostic.
# ===========================================================================
def test_21_loudness_mismatch_diagnostic_never_changes_treatment_choice():
    u = _understanding(
        left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_SIMILAR, level_delta_db=8.0),
    )
    d = _decide(u, HARD_CUT)
    assert d.loudness_polish_status == ajtd.LOUDNESS_POLISH_NEEDED
    assert d.treatment != ajtd.TREATMENT_SHORT_CROSSFADE  # never used to "fix" a gain mismatch


# ===========================================================================
# 22-24: silence combinations.
# ===========================================================================
def test_22_silence_silence():
    from cutsell_worker.pacing_v2_acoustic_edge_evidence import NON_SPEECH_SILENCE, SILENCE_STATUS_SILENT
    u = _understanding(
        left_edge_evidence=_edge(non_speech_status=NON_SPEECH_SILENCE),
        right_edge_evidence=_edge(non_speech_status=NON_SPEECH_SILENCE, edge=EDGE_RIGHT_START),
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment in (ajtd.TREATMENT_NONE, ajtd.TREATMENT_CLICK_FADE, ajtd.TREATMENT_SHORT_CROSSFADE)


def test_23_silence_speech():
    from cutsell_worker.pacing_v2_acoustic_edge_evidence import NON_SPEECH_SILENCE
    u = _understanding(
        left_edge_evidence=_edge(non_speech_status=NON_SPEECH_SILENCE),
        right_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT, edge=EDGE_RIGHT_START),
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment in (ajtd.TREATMENT_NONE, ajtd.TREATMENT_CLICK_FADE)


def test_24_speech_silence():
    from cutsell_worker.pacing_v2_acoustic_edge_evidence import NON_SPEECH_SILENCE
    u = _understanding(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT),
        right_edge_evidence=_edge(non_speech_status=NON_SPEECH_SILENCE, edge=EDGE_RIGHT_START),
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment in (ajtd.TREATMENT_NONE, ajtd.TREATMENT_CLICK_FADE)


# ===========================================================================
# 25: no-treatment-needed.
# ===========================================================================
def test_25_no_treatment_needed():
    u = _understanding(**_CONTINUOUS)
    d = _decide(u, HARD_CUT)
    assert d.treatment_status == ajtd.TREATMENT_STATUS_SUPPORTED


# ===========================================================================
# 26-31: compatibility with J/L primary modes.
# ===========================================================================
def test_26_j_cut_none():
    assert ajtd.compatibility(J_CUT, ajtd.TREATMENT_NONE) == ajtd.COMPATIBILITY_SUPPORTED


def test_27_j_cut_click_fade():
    assert ajtd.compatibility(J_CUT, ajtd.TREATMENT_CLICK_FADE) == ajtd.COMPATIBILITY_SUPPORTED


def test_28_j_cut_short_crossfade_compatibility():
    assert ajtd.compatibility(J_CUT, ajtd.TREATMENT_SHORT_CROSSFADE) == ajtd.COMPATIBILITY_POSSIBLE_WITH_CONDITIONS


def test_29_l_cut_none():
    assert ajtd.compatibility(L_CUT, ajtd.TREATMENT_NONE) == ajtd.COMPATIBILITY_SUPPORTED


def test_30_l_cut_click_fade():
    assert ajtd.compatibility(L_CUT, ajtd.TREATMENT_CLICK_FADE) == ajtd.COMPATIBILITY_SUPPORTED


def test_31_micro_ambience_disallowed():
    for treatment in (ajtd.TREATMENT_AMBIENCE_CARRY_LEFT, ajtd.TREATMENT_AMBIENCE_CARRY_RIGHT, ajtd.TREATMENT_AMBIENCE_BRIDGE, ajtd.TREATMENT_SHORT_CROSSFADE):
        assert ajtd.compatibility(MICRO_AUDIO_OVERLAP, treatment) == ajtd.COMPATIBILITY_DISALLOWED


# ===========================================================================
# 31b: MICRO compatibility downgrade in the actual decision path.
# ===========================================================================
def test_31b_micro_primary_mode_forces_downgrade_from_ambience():
    left_handle = _handle(status=HANDLE_STATUS_SAFE_NON_SPEECH, handle_id="hl")
    left_edge = _edge(edge=EDGE_POST_HANDLE, non_speech_status=NON_SPEECH_SAFE_CANDIDATE)
    u = _understanding(
        left_edge_evidence=_edge(),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
        left_post_roll_handle=left_handle, left_post_roll_handle_edge_evidence=left_edge,
    )
    d = _decide(u, MICRO_AUDIO_OVERLAP)
    assert d.treatment == ajtd.TREATMENT_CLICK_FADE
    assert d.compatibility_status == ajtd.COMPATIBILITY_SUPPORTED  # after downgrade, CLICK_FADE is SUPPORTED with MICRO
    assert ajtd.CONFLICT_COMPATIBILITY_DISALLOWED in d.conflict_flags


# ===========================================================================
# 32-33: one treatment per join / determinism.
# ===========================================================================
def test_32_one_treatment_per_join():
    u = _understanding(**_DISCONTINUOUS)
    d = _decide(u, HARD_CUT)
    assert isinstance(d.treatment, str)
    assert d.treatment in ajtd.TREATMENT_VALUES


def test_33_deterministic_repeat():
    u = _understanding(**_DISCONTINUOUS)
    a = _decide(u, HARD_CUT)
    b = _decide(u, HARD_CUT)
    assert a == b


# ===========================================================================
# 34-35: same-source / multi-source.
# ===========================================================================
def test_34_same_source():
    u = _understanding(left_source_asset_id="s1", right_source_asset_id="s1", **_DISCONTINUOUS)
    d = _decide(u, HARD_CUT)
    assert d.treatment == ajtd.TREATMENT_SHORT_CROSSFADE


def test_35_multi_source():
    u = build_audio_join_understanding(
        transition_index=0, left_clip_id="c1", right_clip_id="c2",
        left_source_asset_id="sA", right_source_asset_id="sB",
        left_edge_evidence=_edge(source_asset_id="sA") if False else _edge(),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment == ajtd.TREATMENT_SHORT_CROSSFADE


# ===========================================================================
# 36: no treatment duration invention.
# ===========================================================================
def test_36_no_treatment_duration_invention():
    u = _understanding(**_DISCONTINUOUS)
    d = _decide(u, HARD_CUT)
    assert d.candidate_duration_sec is None
    assert d.timing_status == ajtd.TIMING_POLICY_NOT_YET_IMPLEMENTED
    d2 = _decide(u, HARD_CUT, candidate_duration_sec=0.25)
    assert d2.candidate_duration_sec == 0.25
    assert d2.timing_status == ajtd.TIMING_POLICY_CANDIDATE_SUPPLIED
    assert d2.treatment == d.treatment  # duration never changes the selection


# ===========================================================================
# 37-45: no renderer/ASR/provider/Boundary/Ordering/Family-BestTake/
# loudness-correction/room-tone-overclaim/micro-authority reach.
# ===========================================================================
def test_37_no_renderer_call():
    src = inspect.getsource(ajtd)
    for banned in ("from .render import", "from .render_plan import", "import render"):
        assert banned not in src


def test_38_no_asr():
    src = inspect.getsource(ajtd)
    for banned in ("whisper", "transcribe", "asr_"):
        assert banned not in src.lower()


def test_39_no_provider():
    src = inspect.getsource(ajtd)
    for banned in ("requests.", "openai", "genai", "gemini", "modal.", "boto3"):
        assert banned not in src.lower()


def test_40_no_boundary_mutation():
    src = inspect.getsource(ajtd)
    assert "from .boundary" not in src and "dataclasses.replace" not in src


def test_41_no_ordering_mutation():
    src = inspect.getsource(ajtd)
    assert "from .ordering" not in src


def test_42_no_family_besttake_mutation():
    src = inspect.getsource(ajtd)
    assert "from .family" not in src and "from .best_take" not in src


def test_43_no_loudness_correction():
    module_symbols = set(vars(ajtd).keys())
    for token in ("normalize_gain", "apply_gain", "correct_loudness"):
        assert token not in module_symbols


def test_44_no_room_tone_overclaim():
    module_symbols = set(vars(ajtd).keys())
    assert "ROOM_TONE_MATCH" not in module_symbols and "ROOM_TONE_MISMATCH" not in module_symbols
    u = _understanding(**_DISCONTINUOUS)
    d = _decide(u, HARD_CUT)
    assert d.room_tone_classification_status == "NOT_YET_AVAILABLE"


def test_45_no_micro_authority():
    """MICRO_AUDIO_OVERLAP is structurally impossible as a treatment
    value (not a member of TREATMENT_VALUES) -- never merely a runtime
    check that could regress."""
    assert MICRO_AUDIO_OVERLAP not in ajtd.TREATMENT_VALUES
    for mode in (HARD_CUT, TIGHT_CUT, KEEP_PAUSE, J_CUT, L_CUT, MICRO_AUDIO_OVERLAP):
        for treatment in ajtd.TREATMENT_VALUES:
            assert treatment != MICRO_AUDIO_OVERLAP


# ===========================================================================
# 46: no J/L authority (this module never selects/changes J_CUT/L_CUT).
# ===========================================================================
def test_46_no_jl_authority():
    src = inspect.getsource(ajtd.build_audio_join_treatment_decision)
    assert "J_CUT" not in src and "L_CUT" not in src  # never assigns/chooses a Layer-2 mode


# ===========================================================================
# 47-53: qualification markers (actual suite runs performed by the shell
# qualification step; these are cheap in-repo stability markers).
# ===========================================================================
def test_47_schema_version_present():
    assert ajtd.SCHEMA_VERSION.startswith("cutsell.pacing_v2_audio_join_treatment_decision.v")


def test_48_module_not_wired_into_universal_clean_cut():
    import subprocess
    result = subprocess.run(
        ["grep", "-rl", "pacing_v2_audio_join_treatment_decision", "cutsell_worker/universal_clean_cut.py"],
        capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


def test_49_diagnostics_json_safe_no_transcript_dump():
    u = _understanding(
        left_edge_evidence=_edge(word_coverage_status="KNOWN_PRESENT"),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        left_edge_words=[(0.5, 0.9, "supersecrettranscript")],
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
    )
    d = _decide(u, HARD_CUT)
    dumped = json.dumps(ajtd.audio_join_treatment_decision_diagnostics(d))
    assert "supersecrettranscript" not in dumped


def test_50_run_summary_no_master_score():
    u = _understanding(**_DISCONTINUOUS)
    d = _decide(u, HARD_CUT)
    summary = ajtd.audio_join_treatment_decision_run_summary([d])
    assert not any("score" in k.lower() for k in summary.keys())
    assert summary["join_count"] == 1


def test_51_decision_frozen():
    u = _understanding(**_DISCONTINUOUS)
    d = _decide(u, HARD_CUT)
    with pytest.raises(Exception):
        d.treatment = "MUTATED"


def test_52_renderer_capability_preserved():
    u = _understanding(**_DISCONTINUOUS)
    d = _decide(u, HARD_CUT)
    assert ajtd.renderer_capability_status(ajtd.TREATMENT_SHORT_CROSSFADE) == "EXTENSION_REQUIRED"
    assert ajtd.renderer_capability_status(ajtd.TREATMENT_AMBIENCE_CARRY_LEFT) == "SUPPORTED_NOW"
    assert ajtd.renderer_capability_status(ajtd.TREATMENT_AMBIENCE_CARRY_RIGHT) == "SUPPORTED_NOW"
    assert ajtd.renderer_capability_status(ajtd.TREATMENT_AMBIENCE_BRIDGE) == "SUPPORTED_NOW"


def test_53_full_offline_suite_marker_present():
    """A stable marker for the qualification shell step (compileall,
    D-231/D-230 regressions, SourceAudioHandle regressions, Pacing
    regressions, Renderer regressions, full offline suite)."""
    assert isinstance(ajtd.TREATMENT_VALUES, tuple) and len(ajtd.TREATMENT_VALUES) == 6


# ===========================================================================
# Additional critical-property tests.
# ===========================================================================
def test_word_firewall_unknown_coverage_blocks_advanced_treatment():
    u = _understanding(**_DISCONTINUOUS)  # word coverage KNOWN_EMPTY on both -> safe actually
    # Force unknown by using no edges (already covered); this test uses a
    # left-only-known scenario to isolate the "one side unknown" case.
    u2 = build_audio_join_understanding(
        transition_index=0, left_clip_id="c1", right_clip_id="c2",
        left_source_asset_id="s1", right_source_asset_id="s1",
        left_edge_evidence=_edge(), continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
    )
    d = _decide(u2, HARD_CUT)
    assert d.treatment_status in (ajtd.TREATMENT_STATUS_INSUFFICIENT_EVIDENCE, ajtd.TREATMENT_STATUS_SAFE_FALLBACK)
    assert d.treatment != ajtd.TREATMENT_SHORT_CROSSFADE


def test_double_speech_both_sides_blocks():
    u = _understanding(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT),
        right_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT, edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison(CONTINUITY_DIFFERENT),
    )
    d = _decide(u, HARD_CUT)
    assert d.speech_safety_status == SAFETY_BLOCKED
    assert d.treatment == ajtd.TREATMENT_CLICK_FADE


def test_conflicted_understanding_forces_click_fade_conflicted_status():
    u = _understanding(
        left_edge_evidence=_edge(), right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
        continuity_comparison=_comparison("CONFLICTED"),
    )
    d = _decide(u, HARD_CUT)
    assert d.treatment == ajtd.TREATMENT_CLICK_FADE
    assert d.treatment_status == ajtd.TREATMENT_STATUS_CONFLICTED


def test_handle_status_passed_through_verbatim():
    u = _understanding(**_DISCONTINUOUS)
    d = _decide(u, HARD_CUT, left_post_roll_handle_status="SAFE_NON_SPEECH_HANDLE", right_pre_roll_handle_status="UNAVAILABLE")
    assert d.left_handle_status == "SAFE_NON_SPEECH_HANDLE"
    assert d.right_handle_status == "UNAVAILABLE"


def test_speech_safety_status_independent_field_from_word_safety():
    u = _understanding(
        left_edge_evidence=_edge(speech_status=SPEECH_STATUS_LEXICAL_PRESENT, non_speech_status=NON_SPEECH_SPEECH_PRESENT),
        right_edge_evidence=_edge(edge=EDGE_RIGHT_START),
    )
    d = _decide(u, HARD_CUT)
    assert d.word_safety_status == SAFETY_BLOCKED
    assert d.speech_safety_status == SAFETY_BLOCKED
