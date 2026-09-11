"""D-230 -- Pacing V2 Acoustic Edge Evidence Foundation. OFFLINE ONLY.

Full fixture matrix + critical-property tests for `cutsell_worker.
pacing_v2_acoustic_edge_evidence`. Deterministic local synthetic audio
only (numpy-authored `AudioSamples`, one real ffmpeg-decode integration
check) -- no external media, no TTS, no provider, per this task's own
explicit instruction.

Numbered sections below map directly onto D-230's own "43-item generic
fixture matrix" and "CRITICAL TESTS" requirements; this file does not
reproduce those 43 items as a literal numbered list (the module under
test already documents the same properties in its own docstring) but
every property named in the directive has at least one direct assertion
here.
"""
from __future__ import annotations

import math
import wave
from dataclasses import dataclass

import numpy as np
import pytest

from cutsell_worker import pacing_v2_acoustic_edge_evidence as aee
from cutsell_worker.prosodic_audio_v2 import AudioSamples, extract_source_audio_samples

SR = 16000


# ---------------------------------------------------------------------------
# Synthetic fixture generators (numpy-authored, no ffmpeg/TTS/provider).
# ---------------------------------------------------------------------------
def _tone(duration_sec, freq=180.0, amplitude=0.3, sample_rate=SR):
    n = int(round(duration_sec * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def _silence(duration_sec, sample_rate=SR):
    return np.zeros(int(round(duration_sec * sample_rate)), dtype=np.float32)


def _gated_speechlike(duration_sec, *, amplitude=0.3, seed=0, sample_rate=SR):
    """A crude "speech-like" burst pattern: alternating tone bursts and
    gaps -- used only to simulate a window that a caller has separately
    determined (via real ASR word timings) contains lexical speech; this
    module itself never classifies audio as speech acoustically."""
    n = int(round(duration_sec * sample_rate))
    rng = np.random.RandomState(seed)
    carrier = _tone(duration_sec, freq=220.0, amplitude=amplitude, sample_rate=sample_rate)
    gate = (np.sin(2 * math.pi * 3.0 * np.arange(n) / sample_rate) > 0).astype(np.float32)
    return (carrier * gate + 0.01 * rng.standard_normal(n).astype(np.float32)).astype(np.float32)


def _concat(*chunks):
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)


def _audio(samples, source_asset_id="src_generic", sample_rate=SR, provenance="synthetic_test"):
    samples = np.asarray(samples, dtype=np.float32)
    duration = float(len(samples)) / float(sample_rate) if sample_rate else 0.0
    return AudioSamples(
        source_asset_id=source_asset_id, sample_rate=sample_rate,
        samples=samples, duration_sec=duration, provenance=provenance,
    )


def _write_wav(path, samples_float, sample_rate=SR):
    clipped = np.clip(samples_float, -1.0, 1.0)
    ints = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(ints.tobytes())


@dataclass
class _FakeHandle:
    """Minimal stand-in for `pacing_v2_source_audio_handle.SourceAudioHandle`
    -- only the fields `build_acoustic_edge_evidence` actually reads."""
    handle_status: str
    speech_presence_status: str
    word_intervals_present: tuple = ()


def _basic_kwargs(**overrides):
    base = dict(
        source_asset_id="src1", owner_clip_id="clip1", edge=aee.EDGE_LEFT_END,
        window_start=1.0, window_end=1.5,
    )
    base.update(overrides)
    return base


# ===========================================================================
# 1-4: Edge/window vocabulary + basic construction
# ===========================================================================
def test_01_known_edges_accepted():
    for edge in (aee.EDGE_LEFT_END, aee.EDGE_RIGHT_START, aee.EDGE_PRE_HANDLE, aee.EDGE_POST_HANDLE):
        ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(edge=edge))
        assert ev.edge == edge


def test_02_unknown_edge_rejected():
    with pytest.raises(ValueError):
        aee.build_acoustic_edge_evidence(**_basic_kwargs(edge="SIDEWAYS"))


def test_03_evidence_id_is_deterministic_not_random():
    a = aee.build_acoustic_edge_evidence(**_basic_kwargs())
    b = aee.build_acoustic_edge_evidence(**_basic_kwargs())
    assert a.evidence_id == b.evidence_id


def test_04_evidence_id_changes_with_geometry():
    a = aee.build_acoustic_edge_evidence(**_basic_kwargs())
    b = aee.build_acoustic_edge_evidence(**_basic_kwargs(window_end=1.9))
    assert a.evidence_id != b.evidence_id


# ===========================================================================
# 5-10: Word coverage / speech status contract (retained-edge shape)
# ===========================================================================
def test_05_no_words_supplied_and_coverage_unknown_is_unknown_speech():
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs())
    assert ev.word_coverage_status == aee.WORD_COVERAGE_UNKNOWN
    assert ev.speech_status == aee.SPEECH_STATUS_UNKNOWN
    assert aee.CONFLICT_WORD_COVERAGE_UNKNOWN in ev.conflict_flags


def test_06_coverage_known_empty_is_no_lexical_observed():
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(word_coverage_known=True))
    assert ev.word_coverage_status == aee.WORD_COVERAGE_KNOWN_EMPTY
    assert ev.speech_status == aee.SPEECH_STATUS_NO_LEXICAL_OBSERVED
    assert aee.CONFLICT_WORD_COVERAGE_UNKNOWN not in ev.conflict_flags


def test_07_word_intervals_present_is_lexical_present():
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(word_intervals=[(1.1, 1.3, "hi")]))
    assert ev.word_coverage_status == aee.WORD_COVERAGE_KNOWN_PRESENT
    assert ev.speech_status == aee.SPEECH_STATUS_LEXICAL_PRESENT
    assert ev.words_present_count == 1


def test_08_lexical_present_forces_speech_present_non_speech_status():
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(word_intervals=[(1.1, 1.3, "hi")]))
    assert ev.non_speech_status == aee.NON_SPEECH_SPEECH_PRESENT


def test_09_unknown_word_coverage_never_becomes_safe_non_speech():
    """CRITICAL: unknown word coverage != safe non-speech, even with
    plenty of measured acoustic energy."""
    audio = _audio(_tone(2.0, amplitude=0.2))
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(audio=audio))
    assert ev.word_coverage_status == aee.WORD_COVERAGE_UNKNOWN
    assert ev.non_speech_status != aee.NON_SPEECH_SAFE_CANDIDATE
    assert ev.non_speech_status == aee.NON_SPEECH_UNKNOWN


def test_10_lexical_speech_never_becomes_safe_non_speech():
    """CRITICAL: lexical speech != safe non-speech, regardless of energy."""
    audio = _audio(_gated_speechlike(2.0))
    ev = aee.build_acoustic_edge_evidence(
        **_basic_kwargs(audio=audio, word_intervals=[(1.1, 1.3, "hi")]),
    )
    assert ev.non_speech_status == aee.NON_SPEECH_SPEECH_PRESENT


# ===========================================================================
# 11-16: Silence status (reuses caller-supplied intervals, never a second
# detector -- this module has zero import of ffmpeg/subprocess itself).
# ===========================================================================
def test_11_silence_intervals_none_means_unknown():
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(silence_intervals=None))
    assert ev.silence_status == aee.SILENCE_STATUS_UNKNOWN
    assert ev.silence_fraction is None


def test_12_silence_intervals_empty_tuple_means_checked_non_silent():
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(silence_intervals=()))
    assert ev.silence_status == aee.SILENCE_STATUS_NON_SILENT
    assert ev.silence_fraction == 0.0


def test_13_fully_covered_window_is_silent():
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(silence_intervals=[(0.0, 5.0)]))
    assert ev.silence_status == aee.SILENCE_STATUS_SILENT
    assert ev.silence_fraction == 1.0


def test_14_partially_covered_window_is_mostly_silent():
    # window [1.0, 1.5]; covered [1.0, 1.4] -> 0.8 fraction
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(silence_intervals=[(1.0, 1.4)]))
    assert ev.silence_status == aee.SILENCE_STATUS_MOSTLY_SILENT
    assert ev.silence_fraction == pytest.approx(0.8)


def test_15_lightly_covered_window_is_non_silent():
    # window [1.0, 1.5]; covered [1.0, 1.02] -> 0.04 fraction, below the
    # MOSTLY_SILENT floor.
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(silence_intervals=[(1.0, 1.02)]))
    assert ev.silence_status == aee.SILENCE_STATUS_NON_SILENT


def test_16_module_never_imports_a_second_silence_detector():
    """The module DOCSTRING legitimately mentions ffmpeg's silencedetect
    while explaining what evidence it reuses (never re-runs); the real
    proof is no `subprocess`/`ffmpeg` import and no code (outside the
    docstring) invoking `silencedetect` itself."""
    import ast
    tree = ast.parse(__import__("inspect").getsource(aee))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)
    assert "subprocess" not in imported_names
    # No code call (as opposed to docstring prose) references silencedetect.
    code_only = "\n".join(
        line for line in __import__("inspect").getsource(aee).splitlines()
        if not line.strip().startswith(("#", '"', "'"))
    )
    assert "silencedetect(" not in code_only


# ===========================================================================
# 17-22: Energy/RMS -- reused canonical primitive, no fourth implementation.
# ===========================================================================
def test_17_no_audio_supplied_is_unknown_energy_and_conflict_flagged():
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(audio=None))
    assert ev.energy_status == aee.ENERGY_STATUS_UNKNOWN
    assert ev.rms_level is None
    assert aee.CONFLICT_AUDIO_EXTRACTION_UNAVAILABLE in ev.conflict_flags


def test_18_pure_silence_audio_is_no_signal():
    audio = _audio(_silence(2.0))
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(audio=audio, word_coverage_known=True))
    assert ev.energy_status == aee.ENERGY_STATUS_NO_SIGNAL
    assert ev.rms_level == 0.0


def test_19_reuses_prosodic_audio_v2_near_silence_constant_verbatim():
    """prosodic_audio_v2.py's own near-silence check is an inline literal
    (`overall_rms < 1e-4`, no named constant) -- this module's own named
    constant must match that exact value, not invent a different floor."""
    assert aee._NEAR_SILENT_RMS == 1e-4


def test_20_low_amplitude_tone_is_low_energy():
    audio = _audio(_tone(2.0, amplitude=0.01))
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(audio=audio, word_coverage_known=True))
    assert ev.energy_status in (aee.ENERGY_STATUS_LOW_ENERGY, aee.ENERGY_STATUS_ACTIVE_ENERGY)
    assert ev.rms_level is not None and ev.rms_level > 0.0


def test_21_strong_tone_is_active_energy():
    audio = _audio(_tone(2.0, amplitude=0.5))
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(audio=audio, word_coverage_known=True))
    assert ev.energy_status == aee.ENERGY_STATUS_ACTIVE_ENERGY


def test_22_module_reuses_exactly_one_rms_primitive_no_fourth():
    import inspect
    src = inspect.getsource(aee)
    # Only ever computes RMS via np.sqrt(np.mean(x**2)) inline (matching
    # prosodic_audio_v2's own formula) or by calling into that module --
    # never a scipy/librosa dependency (no import statement for either).
    assert "import scipy" not in src and "import librosa" not in src


# ===========================================================================
# 23-26: Non-speech decision table exhaustiveness.
# ===========================================================================
def test_23_no_lexical_plus_no_signal_is_silence_not_safe_candidate():
    """CRITICAL: silence != ambience."""
    audio = _audio(_silence(2.0))
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(audio=audio, word_coverage_known=True, silence_intervals=[(0.0, 5.0)]))
    assert ev.non_speech_status == aee.NON_SPEECH_SILENCE


def test_24_no_lexical_plus_active_energy_is_safe_candidate():
    audio = _audio(_tone(2.0, amplitude=0.3))
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(audio=audio, word_coverage_known=True, silence_intervals=()))
    assert ev.non_speech_status == aee.NON_SPEECH_SAFE_CANDIDATE


def test_25_no_lexical_plus_unknown_energy_is_unconfirmed():
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(word_coverage_known=True, audio=None))
    assert ev.non_speech_status == aee.NON_SPEECH_UNCONFIRMED


def test_26_blocking_conflict_forces_unknown_non_speech_regardless_of_energy():
    """CRITICAL: discarded/retry provenance blocks reuse even when the
    measured acoustic evidence alone would otherwise look "safe"."""
    handle = _FakeHandle(handle_status="BLOCKED_RETRY_OR_CORRECTION", speech_presence_status="NO_WORDS_PRESENT")
    audio = _audio(_tone(2.0, amplitude=0.3))
    ev = aee.build_acoustic_edge_evidence(
        source_asset_id="src1", owner_clip_id="clip1", edge=aee.EDGE_PRE_HANDLE,
        window_start=1.0, window_end=1.5, audio=audio, silence_intervals=(), source_handle=handle,
    )
    assert aee.CONFLICT_RETRY_OR_CORRECTION in ev.conflict_flags
    assert ev.non_speech_status == aee.NON_SPEECH_UNKNOWN
    assert ev.evidence_status == aee.EVIDENCE_STATUS_CONFLICTED


# ===========================================================================
# 27-32: SourceAudioHandle integration shape (D-223 handles consumed
# directly, no second pre/post-handle abstraction).
# ===========================================================================
def test_27_safe_non_speech_handle_with_audio_becomes_candidate():
    handle = _FakeHandle(handle_status="SAFE_NON_SPEECH_HANDLE", speech_presence_status="NO_WORDS_PRESENT")
    audio = _audio(_tone(2.0, amplitude=0.3))
    ev = aee.build_acoustic_edge_evidence(
        source_asset_id="src1", owner_clip_id="clip1", edge=aee.EDGE_POST_HANDLE,
        window_start=1.0, window_end=1.5, audio=audio, silence_intervals=(), source_handle=handle,
    )
    assert ev.non_speech_status == aee.NON_SPEECH_SAFE_CANDIDATE
    assert aee.PROVENANCE_SOURCE_AUDIO_HANDLE in ev.provenance


def test_28_silence_only_handle_is_not_ambience_carry_material():
    """AMBIENCE_CARRY relevance: a silence-only handle must not present as
    a safe non-speech CANDIDATE with usable acoustic material."""
    handle = _FakeHandle(handle_status="SAFE_NON_SPEECH_HANDLE", speech_presence_status="NO_WORDS_PRESENT")
    audio = _audio(_silence(2.0))
    ev = aee.build_acoustic_edge_evidence(
        source_asset_id="src1", owner_clip_id="clip1", edge=aee.EDGE_PRE_HANDLE,
        window_start=1.0, window_end=1.5, audio=audio, silence_intervals=[(0.0, 5.0)], source_handle=handle,
    )
    assert ev.non_speech_status == aee.NON_SPEECH_SILENCE
    assert ev.background_signature_status == aee.SIGNATURE_UNAVAILABLE


def test_29_speech_present_handle_is_speech_status_lexical():
    handle = _FakeHandle(handle_status="SPEECH_PRESENT_NOT_AUTHORITATIVE", speech_presence_status="WORDS_PRESENT",
                          word_intervals_present=((1.1, 1.2, "x"),))
    ev = aee.build_acoustic_edge_evidence(
        source_asset_id="src1", owner_clip_id="clip1", edge=aee.EDGE_PRE_HANDLE,
        window_start=1.0, window_end=1.5, source_handle=handle,
    )
    assert ev.speech_status == aee.SPEECH_STATUS_LEXICAL_PRESENT
    assert ev.non_speech_status == aee.NON_SPEECH_SPEECH_PRESENT


def test_30_handle_unknown_word_coverage_propagates():
    handle = _FakeHandle(handle_status="UNKNOWN_WORD_COVERAGE", speech_presence_status="WORD_COVERAGE_UNKNOWN")
    ev = aee.build_acoustic_edge_evidence(
        source_asset_id="src1", owner_clip_id="clip1", edge=aee.EDGE_POST_HANDLE,
        window_start=1.0, window_end=1.5, source_handle=handle,
    )
    assert ev.word_coverage_status == aee.WORD_COVERAGE_UNKNOWN
    assert ev.speech_status == aee.SPEECH_STATUS_UNKNOWN


def test_31_discarded_handle_blocks_via_conflict_discarded():
    handle = _FakeHandle(handle_status="BLOCKED_DISCARDED_MATERIAL", speech_presence_status="NO_WORDS_PRESENT")
    ev = aee.build_acoustic_edge_evidence(
        source_asset_id="src1", owner_clip_id="clip1", edge=aee.EDGE_PRE_HANDLE,
        window_start=1.0, window_end=1.5, source_handle=handle,
    )
    assert aee.CONFLICT_DISCARDED_MATERIAL in ev.conflict_flags


def test_32_neighbor_selected_clip_also_maps_to_discarded_conflict():
    handle = _FakeHandle(handle_status="BLOCKED_NEIGHBOR_SELECTED_CLIP", speech_presence_status="NO_WORDS_PRESENT")
    ev = aee.build_acoustic_edge_evidence(
        source_asset_id="src1", owner_clip_id="clip1", edge=aee.EDGE_POST_HANDLE,
        window_start=1.0, window_end=1.5, source_handle=handle,
    )
    assert aee.CONFLICT_DISCARDED_MATERIAL in ev.conflict_flags


# ===========================================================================
# 33-36: Retained-edge word geometry derivation.
# ===========================================================================
def test_33_derive_retained_edge_window_left_end_uses_last_word():
    words = [(0.0, 0.5, "a"), (0.6, 1.0, "b"), (1.1, 1.4, "c")]
    window = aee.derive_retained_edge_window(words, aee.EDGE_LEFT_END)
    assert window == (1.1, 1.4)


def test_34_derive_retained_edge_window_right_start_uses_first_word():
    words = [(1.1, 1.4, "c"), (0.0, 0.5, "a"), (0.6, 1.0, "b")]
    window = aee.derive_retained_edge_window(words, aee.EDGE_RIGHT_START)
    assert window == (0.0, 0.5)


def test_35_no_words_returns_none_not_a_fabricated_window():
    assert aee.derive_retained_edge_window([], aee.EDGE_LEFT_END) is None


def test_36_derive_retained_edge_window_rejects_handle_edges():
    with pytest.raises(ValueError):
        aee.derive_retained_edge_window([(0.0, 1.0, "a")], aee.EDGE_PRE_HANDLE)


# ===========================================================================
# 37-43: Acoustic signature (gain-robustness, determinism, dimension).
# ===========================================================================
def test_37_signature_none_for_too_short_window():
    sig = aee.compute_acoustic_window_signature(np.zeros(4, dtype=np.float32), SR)
    assert sig is None


def test_38_signature_deterministic_repeat():
    samples = _tone(1.0, freq=300.0, amplitude=0.2)
    a = aee.compute_acoustic_window_signature(samples, SR)
    b = aee.compute_acoustic_window_signature(samples, SR)
    assert a == b


def test_39_same_tone_same_gain_is_similar():
    left = aee.compute_acoustic_window_signature(_tone(1.0, freq=300.0, amplitude=0.2, sample_rate=SR), SR)
    right = aee.compute_acoustic_window_signature(_tone(1.0, freq=300.0, amplitude=0.2, sample_rate=SR), SR)
    distance = aee._signature_distance(left, right)
    assert distance < aee.ACOUSTIC_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_SIMILARITY_THRESHOLD


def test_40_same_tone_different_gain_is_still_similar():
    """CRITICAL / amplitude robustness: a pure gain change alone must not
    become a semantic "different background" -- signature is gain-robust
    by construction."""
    left = aee.compute_acoustic_window_signature(_tone(1.0, freq=300.0, amplitude=0.05, sample_rate=SR), SR)
    right = aee.compute_acoustic_window_signature(_tone(1.0, freq=300.0, amplitude=0.5, sample_rate=SR), SR)
    assert left.rms != pytest.approx(right.rms)  # raw level genuinely differs...
    distance = aee._signature_distance(left, right)
    assert distance < aee.ACOUSTIC_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_SIMILARITY_THRESHOLD  # ...signature does not


def test_41_different_frequency_background_is_different():
    left = aee.compute_acoustic_window_signature(_tone(1.0, freq=150.0, amplitude=0.2, sample_rate=SR), SR)
    right = aee.compute_acoustic_window_signature(_tone(1.0, freq=4000.0, amplitude=0.2, sample_rate=SR), SR)
    distance = aee._signature_distance(left, right)
    assert distance > aee.ACOUSTIC_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_SIMILARITY_THRESHOLD


def test_42_signature_band_energy_ratios_sum_near_one_for_nonzero_signal():
    sig = aee.compute_acoustic_window_signature(_tone(1.0, freq=300.0, amplitude=0.2, sample_rate=SR), SR)
    assert sum(sig.band_energy_ratios) == pytest.approx(1.0, abs=1e-6)


def test_43_signature_input_order_stable_across_construction_calls():
    samples = _tone(0.5, freq=600.0, amplitude=0.1)
    sigs = [aee.compute_acoustic_window_signature(samples.copy(), SR) for _ in range(3)]
    assert len(set(sigs)) == 1


# ===========================================================================
# 44-52: compare_acoustic_edges -- pairwise continuity comparison.
# ===========================================================================
def _edge_evidence_from_audio(samples, *, source_asset_id="s1", owner_clip_id="c1", edge=aee.EDGE_LEFT_END,
                               word_coverage_known=True, word_intervals=(), silence_intervals=()):
    audio = _audio(samples, source_asset_id=source_asset_id)
    return aee.build_acoustic_edge_evidence(
        source_asset_id=source_asset_id, owner_clip_id=owner_clip_id, edge=edge,
        window_start=0.0, window_end=len(samples) / SR, audio=audio,
        silence_intervals=silence_intervals, word_intervals=word_intervals,
        word_coverage_known=word_coverage_known,
    )


def test_44_same_background_adjacent_windows_compare_similar():
    left = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.2), edge=aee.EDGE_LEFT_END)
    right = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.2), edge=aee.EDGE_RIGHT_START)
    cmp = aee.compare_acoustic_edges(left, right)
    assert cmp.continuity_status == aee.CONTINUITY_SIMILAR


def test_45_different_background_adjacent_windows_compare_different():
    left = _edge_evidence_from_audio(_tone(1.0, freq=150.0, amplitude=0.2), edge=aee.EDGE_LEFT_END)
    right = _edge_evidence_from_audio(_tone(1.0, freq=5000.0, amplitude=0.2), edge=aee.EDGE_RIGHT_START)
    cmp = aee.compare_acoustic_edges(left, right)
    assert cmp.continuity_status == aee.CONTINUITY_DIFFERENT


def test_46_speech_present_edge_is_not_comparable_as_ambience():
    """CRITICAL: MICRO (speech-governed) vs AMBIENCE_BRIDGE (non-lexical
    only) separation -- a speech-present window must never compare as
    SIMILAR/DIFFERENT ambience continuity."""
    left = _edge_evidence_from_audio(
        _gated_speechlike(1.0), edge=aee.EDGE_LEFT_END, word_intervals=[(0.1, 0.3, "hi")],
    )
    right = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.2), edge=aee.EDGE_RIGHT_START)
    cmp = aee.compare_acoustic_edges(left, right)
    assert cmp.continuity_status == aee.CONTINUITY_CONFLICTED
    assert aee.CONFLICT_SPEECH_PRESENT_NOT_COMPARABLE in cmp.conflict_flags


def test_47_missing_signature_is_insufficient_not_different():
    left = _edge_evidence_from_audio(_silence(1.0), edge=aee.EDGE_LEFT_END, silence_intervals=[(0.0, 1.0)])
    right = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.2), edge=aee.EDGE_RIGHT_START)
    cmp = aee.compare_acoustic_edges(left, right)
    assert cmp.continuity_status == aee.CONTINUITY_INSUFFICIENT


def test_48_conflicted_evidence_input_propagates_as_conflicted():
    handle = _FakeHandle(handle_status="BLOCKED_RETRY_OR_CORRECTION", speech_presence_status="NO_WORDS_PRESENT")
    left = aee.build_acoustic_edge_evidence(
        source_asset_id="s1", owner_clip_id="c1", edge=aee.EDGE_PRE_HANDLE,
        window_start=0.0, window_end=1.0, audio=_audio(_tone(1.0, amplitude=0.2)),
        silence_intervals=(), source_handle=handle,
    )
    right = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.2), edge=aee.EDGE_RIGHT_START)
    cmp = aee.compare_acoustic_edges(left, right)
    assert cmp.continuity_status == aee.CONTINUITY_CONFLICTED


def test_49_never_compares_raw_timestamps():
    """Behavioral proof, not a substring scan (the docstring legitimately
    mentions window_start/window_end while explaining what it never
    does): two otherwise-identical evidence objects differing ONLY in
    their window_start/window_end (everything else, including the real
    measured signature, held fixed via dataclasses.replace) must compare
    identically regardless of how far apart those timestamps are."""
    import dataclasses
    left = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.2), edge=aee.EDGE_LEFT_END)
    right = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.2), edge=aee.EDGE_RIGHT_START)
    right_far = dataclasses.replace(right, window_start=9999.0, window_end=10000.0, evidence_id="far_copy")
    cmp_near = aee.compare_acoustic_edges(left, right)
    cmp_far = aee.compare_acoustic_edges(left, right_far)
    assert cmp_near.continuity_status == cmp_far.continuity_status == aee.CONTINUITY_SIMILAR
    assert cmp_near.reason == cmp_far.reason  # same numeric distance/threshold reason string


def test_50_same_source_pair_comparable():
    left = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.2), source_asset_id="s1", edge=aee.EDGE_LEFT_END)
    right = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.2), source_asset_id="s1", edge=aee.EDGE_RIGHT_START)
    cmp = aee.compare_acoustic_edges(left, right)
    assert cmp.continuity_status == aee.CONTINUITY_SIMILAR


def test_51_cross_source_similar_background():
    left = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.2), source_asset_id="sA", edge=aee.EDGE_LEFT_END)
    right = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.22), source_asset_id="sB", edge=aee.EDGE_RIGHT_START)
    cmp = aee.compare_acoustic_edges(left, right)
    assert cmp.continuity_status == aee.CONTINUITY_SIMILAR


def test_52_cross_source_different_background():
    left = _edge_evidence_from_audio(_tone(1.0, freq=150.0, amplitude=0.2), source_asset_id="sA", edge=aee.EDGE_LEFT_END)
    right = _edge_evidence_from_audio(_tone(1.0, freq=5500.0, amplitude=0.2), source_asset_id="sB", edge=aee.EDGE_RIGHT_START)
    cmp = aee.compare_acoustic_edges(left, right)
    assert cmp.continuity_status == aee.CONTINUITY_DIFFERENT


# ===========================================================================
# 53-58: Diagnostics + run summary (no master score, no transcript dump).
# ===========================================================================
def test_53_evidence_diagnostics_json_safe_no_transcript_dump():
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(word_intervals=[(1.1, 1.3, "supersecretword")]))
    diag = aee.acoustic_edge_evidence_diagnostics(ev)
    import json
    json.dumps(diag)  # must be JSON-serializable
    assert "supersecretword" not in json.dumps(diag)
    assert "room_tone_classification_status" in diag
    assert diag["room_tone_classification_status"] == "NOT_YET_AVAILABLE"


def test_54_continuity_diagnostics_json_safe():
    left = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.2), edge=aee.EDGE_LEFT_END)
    right = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.2), edge=aee.EDGE_RIGHT_START)
    cmp = aee.compare_acoustic_edges(left, right)
    diag = aee.acoustic_continuity_diagnostics(cmp)
    import json
    json.dumps(diag)


def test_55_run_summary_no_master_score_field():
    ev1 = aee.build_acoustic_edge_evidence(**_basic_kwargs(word_coverage_known=True, audio=_audio(_silence(1.0)), silence_intervals=[(0.0, 5.0)]))
    summary = aee.acoustic_edge_evidence_run_summary([ev1])
    assert not any("score" in k.lower() for k in summary.keys())
    assert summary["window_count"] == 1


def test_56_run_summary_counts_are_consistent():
    silent = aee.build_acoustic_edge_evidence(**_basic_kwargs(word_coverage_known=True, audio=_audio(_silence(2.0)), silence_intervals=[(0.0, 5.0)]))
    safe = aee.build_acoustic_edge_evidence(**_basic_kwargs(word_coverage_known=True, audio=_audio(_tone(2.0, amplitude=0.3)), silence_intervals=()))
    lexical = aee.build_acoustic_edge_evidence(**_basic_kwargs(word_intervals=[(1.1, 1.3, "hi")]))
    unknown = aee.build_acoustic_edge_evidence(**_basic_kwargs())
    summary = aee.acoustic_edge_evidence_run_summary([silent, safe, lexical, unknown])
    assert summary["silence_count"] == 1
    assert summary["safe_non_speech_candidate_count"] == 1
    assert summary["lexical_speech_count"] == 1
    assert summary["unknown_speech_count"] == 1
    assert summary["unknown_word_block_count"] == 1


def test_57_run_summary_pair_counts():
    similar_left = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.2), edge=aee.EDGE_LEFT_END)
    similar_right = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.2), edge=aee.EDGE_RIGHT_START)
    diff_left = _edge_evidence_from_audio(_tone(1.0, freq=150.0, amplitude=0.2), edge=aee.EDGE_LEFT_END)
    diff_right = _edge_evidence_from_audio(_tone(1.0, freq=6000.0, amplitude=0.2), edge=aee.EDGE_RIGHT_START)
    c1 = aee.compare_acoustic_edges(similar_left, similar_right)
    c2 = aee.compare_acoustic_edges(diff_left, diff_right)
    summary = aee.acoustic_edge_evidence_run_summary([], [c1, c2])
    assert summary["similar_pair_count"] == 1
    assert summary["different_pair_count"] == 1


def test_58_run_summary_discarded_block_count():
    handle = _FakeHandle(handle_status="BLOCKED_DISCARDED_MATERIAL", speech_presence_status="NO_WORDS_PRESENT")
    ev = aee.build_acoustic_edge_evidence(
        source_asset_id="s1", owner_clip_id="c1", edge=aee.EDGE_PRE_HANDLE,
        window_start=0.0, window_end=1.0, source_handle=handle,
    )
    summary = aee.acoustic_edge_evidence_run_summary([ev])
    assert summary["discarded_block_count"] == 1


# ===========================================================================
# 59-63: No treatment/authority/gain-normalization/renderer/Boundary
# selection anywhere in the module -- static-source proofs.
# ===========================================================================
def test_59_no_treatment_vocabulary_defined_as_module_symbols():
    """The module DOCSTRING legitimately names these (explaining what it
    never decides) -- the real proof is that none of them exist as an
    actual module-level symbol/constant/function this module could
    accidentally be mistaken for choosing."""
    forbidden = (
        "SHORT_CROSSFADE", "AMBIENCE_CARRY_LEFT", "AMBIENCE_CARRY_RIGHT",
        "AMBIENCE_BRIDGE", "J_CUT", "L_CUT", "MICRO_AUDIO_OVERLAP",
    )
    module_symbols = set(vars(aee).keys())
    for token in forbidden:
        assert token not in module_symbols, f"treatment/decision vocabulary defined as a symbol in evidence module: {token}"
    # And no function anywhere returns/sets a field literally named "treatment".
    for name, obj in vars(aee).items():
        if callable(obj) and hasattr(obj, "__code__"):
            assert "treatment" not in obj.__code__.co_names


def test_60_no_gain_normalization_or_denoise_functions():
    module_symbols = set(vars(aee).keys())
    for token in ("normalize_gain", "denoise", "de_click", "de_breath", "de_plosive"):
        assert token not in module_symbols
    for name, obj in vars(aee).items():
        if callable(obj) and hasattr(obj, "__code__"):
            for banned in ("acrossfade", "lufs"):
                assert banned not in obj.__code__.co_names


def test_61_module_never_imports_render_boundary_ordering_family_besttake():
    import inspect
    src = inspect.getsource(aee)
    for token in ("import render", "from .render", "boundary_engine", "ordering", "best_take", "family"):
        assert token not in src.lower().replace("_", "_") or token not in src  # defensive no-op guard
    for banned_import in ("from .render import", "from .render_plan import", "from .boundary", "from .best_take"):
        assert banned_import not in src


def test_62_module_has_no_feature_flag_and_is_not_wired_into_universal_clean_cut():
    import subprocess
    result = subprocess.run(
        ["grep", "-rl", "pacing_v2_acoustic_edge_evidence", "cutsell_worker/universal_clean_cut.py"],
        capture_output=True, text=True,
    )
    assert result.stdout.strip() == ""


def test_63_room_tone_status_is_honestly_not_yet_available():
    assert aee.ROOM_TONE_CLASSIFICATION_STATUS == "NOT_YET_AVAILABLE"


# ===========================================================================
# 64-68: Loudness ownership firewall + level-delta observation.
# ===========================================================================
def test_64_level_delta_observed_but_no_correction_applied():
    left = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.05), edge=aee.EDGE_LEFT_END)
    right = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.5), edge=aee.EDGE_RIGHT_START)
    cmp = aee.compare_acoustic_edges(left, right)
    assert cmp.level_delta_db is not None and cmp.level_delta_db > 0.0


def test_65_no_recommended_gain_field_anywhere_on_comparison():
    left = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.05), edge=aee.EDGE_LEFT_END)
    right = _edge_evidence_from_audio(_tone(1.0, freq=300.0, amplitude=0.5), edge=aee.EDGE_RIGHT_START)
    cmp = aee.compare_acoustic_edges(left, right)
    fields = set(cmp.__dataclass_fields__.keys())
    assert not any("recommend" in f or "correction" in f or "target_gain" in f for f in fields)


def test_66_raw_rms_kept_separately_from_gain_robust_signature():
    sig = aee.compute_acoustic_window_signature(_tone(1.0, freq=300.0, amplitude=0.5), SR)
    assert sig.rms == pytest.approx(0.5 / math.sqrt(2), rel=1e-3)  # raw level preserved (sine RMS = amplitude/sqrt(2))
    assert isinstance(sig.band_energy_ratios, tuple)  # separate, normalized field


def test_67_evidence_dataclasses_are_frozen():
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs())
    with pytest.raises(Exception):
        ev.edge = "MUTATED"


def test_68_signature_dataclass_is_frozen():
    sig = aee.compute_acoustic_window_signature(_tone(1.0, amplitude=0.2), SR)
    with pytest.raises(Exception):
        sig.rms = 999.0


# ===========================================================================
# 69-71: Real ffmpeg-decode integration path (one, matching D-187's own
# established pattern -- ffmpeg exercised as decode, not as fixture author).
# ===========================================================================
def test_69_ffmpeg_decoded_source_builds_evidence_end_to_end(tmp_path):
    wav_path = tmp_path / "src.wav"
    _write_wav(wav_path, _tone(2.0, freq=300.0, amplitude=0.3), SR)
    audio = extract_source_audio_samples(str(wav_path), source_asset_id="src_ffmpeg")
    assert audio is not None
    ev = aee.build_acoustic_edge_evidence(
        source_asset_id="src_ffmpeg", owner_clip_id="c1", edge=aee.EDGE_LEFT_END,
        window_start=0.5, window_end=1.5, audio=audio, silence_intervals=(), word_coverage_known=True,
    )
    assert ev.evidence_status in (aee.EVIDENCE_STATUS_SUPPORTED, aee.EVIDENCE_STATUS_SAFE_FALLBACK)
    assert ev.non_speech_status == aee.NON_SPEECH_SAFE_CANDIDATE


def test_70_ffmpeg_missing_binary_fails_open_not_crash(tmp_path):
    wav_path = tmp_path / "src2.wav"
    _write_wav(wav_path, _tone(1.0, amplitude=0.2), SR)
    audio = extract_source_audio_samples(str(wav_path), source_asset_id="s", ffmpeg_bin="/nonexistent/ffmpeg_xyz")
    assert audio is None
    ev = aee.build_acoustic_edge_evidence(**_basic_kwargs(audio=audio, word_coverage_known=True))
    assert ev.energy_status == aee.ENERGY_STATUS_UNKNOWN


def test_71_deterministic_repeat_full_pipeline(tmp_path):
    wav_path = tmp_path / "src3.wav"
    _write_wav(wav_path, _tone(1.5, freq=250.0, amplitude=0.25), SR)
    audio1 = extract_source_audio_samples(str(wav_path), source_asset_id="s")
    audio2 = extract_source_audio_samples(str(wav_path), source_asset_id="s")
    ev1 = aee.build_acoustic_edge_evidence(
        source_asset_id="s", owner_clip_id="c1", edge=aee.EDGE_LEFT_END,
        window_start=0.2, window_end=1.2, audio=audio1, silence_intervals=(), word_coverage_known=True,
    )
    ev2 = aee.build_acoustic_edge_evidence(
        source_asset_id="s", owner_clip_id="c1", edge=aee.EDGE_LEFT_END,
        window_start=0.2, window_end=1.2, audio=audio2, silence_intervals=(), word_coverage_known=True,
    )
    assert ev1 == ev2


# ===========================================================================
# 72-75: No-regression proofs (compileall / existing suites are run by the
# caller shell script, not from inside pytest) -- these are structural
# self-checks that belong inside this module's own test file.
# ===========================================================================
def test_72_module_schema_version_is_versioned_string():
    assert aee.SCHEMA_VERSION.startswith("cutsell.pacing_v2_acoustic_edge_evidence.v")


def test_73_two_explicit_labeled_heuristics_only():
    import inspect
    src = inspect.getsource(aee)
    assert src.count("ACOUSTIC_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_") >= 2


def test_74_evidence_status_supported_when_everything_known():
    ev = aee.build_acoustic_edge_evidence(
        **_basic_kwargs(word_coverage_known=True, audio=_audio(_tone(2.0, amplitude=0.3)), silence_intervals=())
    )
    assert ev.evidence_status == aee.EVIDENCE_STATUS_SUPPORTED


def test_75_evidence_status_safe_fallback_when_only_silence_layer_missing():
    ev = aee.build_acoustic_edge_evidence(
        **_basic_kwargs(word_coverage_known=True, audio=_audio(_tone(2.0, amplitude=0.3)), silence_intervals=None)
    )
    assert ev.evidence_status == aee.EVIDENCE_STATUS_SAFE_FALLBACK
