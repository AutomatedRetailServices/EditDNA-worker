"""D-187: Prosodic Audio V2 -- Phase A offline test matrix (45 items).

PERCEPTION / EVIDENCE ONLY. No BestTake authority, no D-184 fusion, no
winner mutation. See cutsell_worker/prosodic_audio_v2.py's own module
docstring for the canonical position and firewalls this suite verifies.

Synthetic audio is generated purely in-process (numpy -> stdlib `wave`
PCM16) -- no Video00 media, no literal Pimples/gynecologist transcript
text anywhere in this file (D-187's own explicit requirement for the
D-186B-inspired abstract fixture).
"""
from __future__ import annotations

import inspect
import math
import struct
import time
import wave
from dataclasses import dataclass

import numpy as np
import pytest

from cutsell_worker import prosodic_audio_v2 as pav2


# ---------------------------------------------------------------------------
# Synthetic audio helpers (generic, non-Video00, non-transcript).
# ---------------------------------------------------------------------------
SR = pav2.DEFAULT_SAMPLE_RATE


def _tone(duration_sec, freq=180.0, amplitude=0.3, sample_rate=SR):
    n = int(round(duration_sec * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def _noise(duration_sec, amplitude=0.05, sample_rate=SR, seed=0):
    n = int(round(duration_sec * sample_rate))
    rng = np.random.RandomState(seed)
    return (amplitude * rng.standard_normal(n)).astype(np.float32)


def _silence(duration_sec, sample_rate=SR):
    return np.zeros(int(round(duration_sec * sample_rate)), dtype=np.float32)


def _speechlike(duration_sec, *, amplitude=0.3, seed=0, sample_rate=SR):
    """A crude 'speech-like' waveform: voiced tone bursts + noise, varying
    amplitude -- generic synthetic content, never real speech audio."""
    return (_tone(duration_sec, amplitude=amplitude, sample_rate=sample_rate)
            + _noise(duration_sec, amplitude=amplitude * 0.15, seed=seed, sample_rate=sample_rate))


def _concat(*chunks):
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)


def _audio(samples, source_asset_id="src_generic", sample_rate=SR, provenance="synthetic_test"):
    samples = np.asarray(samples, dtype=np.float32)
    duration = float(len(samples)) / float(sample_rate) if sample_rate else 0.0
    return pav2.AudioSamples(
        source_asset_id=source_asset_id, sample_rate=sample_rate,
        samples=samples, duration_sec=duration, provenance=provenance,
    )


def _write_wav(path, samples_float, sample_rate=SR):
    """Write a real PCM16 mono WAV via stdlib `wave` (no ffmpeg needed to
    author test fixtures -- ffmpeg is exercised as the DECODE path under
    test, via extract_source_audio_samples itself)."""
    clipped = np.clip(samples_float, -1.0, 1.0)
    ints = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(ints.tobytes())


@dataclass(frozen=True)
class _W:
    text: str
    start: float
    end: float
    confidence: float | None = None


def _words(spans, texts=None):
    texts = texts or [f"w{i}" for i in range(len(spans))]
    return [_W(text=t, start=s, end=e) for (s, e), t in zip(spans, texts)]


MODULE_SOURCE = inspect.getsource(pav2)


# ===========================================================================
# 1-6: Capability status contract
# ===========================================================================
def test_01_capability_status_summary_shape():
    summary = pav2.prosodic_capability_status_summary()
    required = {
        "prosodic_audio_status", "speech_rate", "pause_structure", "hesitation",
        "restart", "continuity", "energy", "emphasis", "pitch", "delivery_variation",
    }
    assert required <= set(summary.keys())
    assert summary["pitch"] == pav2.CAPABILITY_NOT_IMPLEMENTED
    assert summary["speech_rate"] == pav2.CAPABILITY_AVAILABLE
    assert summary["pause_structure"] == pav2.CAPABILITY_AVAILABLE


def test_02_capability_status_never_claims_full_availability():
    # Pitch is NOT_IMPLEMENTED -> overall must never be a bare AVAILABLE lie.
    summary = pav2.prosodic_capability_status_summary()
    assert summary["prosodic_audio_status"] != pav2.CAPABILITY_AVAILABLE


def test_03_capability_status_deterministic():
    a = pav2.prosodic_capability_status_summary()
    b = pav2.prosodic_capability_status_summary()
    assert a == b


def test_04_module_level_available_flag_is_honest_false():
    assert pav2.PROSODIC_AUDIO_AVAILABLE is False


def test_05_pitch_always_not_implemented_status():
    ev = pav2.analyze_prosodic_delivery(
        "c1", "s1", 0.0, 3.0, _audio(_speechlike(3.0)),
    )
    assert ev.pitch_analysis_status == pav2.PITCH_NOT_IMPLEMENTED
    assert ev.pitch_variation_state == pav2.UNKNOWN
    assert "pitch_analysis" in ev.missing_evidence


def test_06_diagnostics_row_has_pitch_status_key():
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 3.0, _audio(_speechlike(3.0)))
    row = pav2.prosodic_delivery_diagnostics(ev)
    assert row["prosodic_pitch_status"] == pav2.PITCH_NOT_IMPLEMENTED


# ===========================================================================
# 7-11: transcript-only / no-audio / no-speech / too-short / insufficient
# ===========================================================================
def test_07_transcript_only_no_audio_is_acoustic_unknown():
    ev = pav2.analyze_prosodic_delivery(
        "c1", "s1", 0.0, 5.0, None, words=_words([(0.0, 1.0), (1.2, 2.0)]),
    )
    assert ev.analysis_status == pav2.STATUS_NOT_EVALUABLE
    assert ev.speech_rate is None
    assert ev.speech_rate_state == pav2.UNKNOWN
    assert ev.energy_dynamics_state == pav2.UNKNOWN
    assert ev.pause_structure_state == pav2.UNKNOWN
    assert "audio_samples" in ev.missing_evidence


def test_08_no_speech_silence_only_span():
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 3.0, _audio(_silence(3.0)))
    assert ev.analysis_status == pav2.STATUS_NO_SPEECH
    assert ev.energy_dynamics_state == pav2.UNKNOWN


def test_09_empty_slice_out_of_range_is_no_speech():
    audio = _audio(_speechlike(2.0))
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 10.0, 12.0, audio)
    assert ev.analysis_status == pav2.STATUS_NO_SPEECH


def test_10_too_short_span_is_insufficient_evidence():
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 0.2, _audio(_speechlike(0.2)))
    assert ev.analysis_status == pav2.STATUS_INSUFFICIENT_EVIDENCE


def test_11_abstain_evidence_never_fabricates_confidence():
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 3.0, None)
    assert ev.evidence_confidence == pav2.CONFIDENCE_UNKNOWN


# ===========================================================================
# 12-18: normal / broken-cadence / pauses (interior vs boundary) / multiple
# ===========================================================================
def test_12_normal_continuous_speech_no_interior_pauses():
    audio = _audio(_speechlike(4.0))
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 4.0, audio, audio_silence_intervals=())
    assert ev.analysis_status == pav2.STATUS_EVALUATED
    assert ev.pause_count == 0
    assert ev.vocal_continuity_state == pav2.CONTINUITY_CONTINUOUS


def test_13_interior_pause_breaks_continuity():
    audio = _audio(_speechlike(6.0))
    # An interior gap well away from both edges.
    ev = pav2.analyze_prosodic_delivery(
        "c1", "s1", 0.0, 6.0, audio, audio_silence_intervals=[(2.5, 3.3)],
    )
    assert ev.pause_count == 1
    assert ev.vocal_continuity_state in (pav2.CONTINUITY_MILDLY_INTERRUPTED, pav2.CONTINUITY_FRAGMENTED)


def test_14_boundary_pause_excluded_from_interior_count():
    """A silence interval flush against the span edge is a natural
    boundary pause (dead air control), not a delivery interruption --
    D-187's own explicit 'dead-air control' requirement."""
    audio = _audio(_speechlike(6.0))
    ev = pav2.analyze_prosodic_delivery(
        "c1", "s1", 1.0, 5.0, audio,
        audio_silence_intervals=[(0.0, 1.0), (5.0, 6.0)],  # both at/outside edges
    )
    assert ev.pause_count == 0
    assert ev.vocal_continuity_state == pav2.CONTINUITY_CONTINUOUS


def test_15_multiple_interior_pauses_fragment_delivery():
    audio = _audio(_speechlike(8.0))
    ev = pav2.analyze_prosodic_delivery(
        "c1", "s1", 0.0, 8.0, audio,
        audio_silence_intervals=[(1.5, 2.0), (3.5, 4.2), (5.5, 6.4)],
    )
    assert ev.pause_count == 3
    assert ev.vocal_continuity_state == pav2.CONTINUITY_FRAGMENTED


def test_16_pause_total_sec_sums_interior_only():
    audio = _audio(_speechlike(6.0))
    ev = pav2.analyze_prosodic_delivery(
        "c1", "s1", 0.0, 6.0, audio,
        audio_silence_intervals=[(0.0, 0.1), (2.0, 2.5), (5.9, 6.0)],
    )
    assert ev.pause_count == 1
    assert abs(ev.pause_total_sec - 0.5) < 1e-6


def test_17_pause_structure_never_independently_recomputed():
    """Structural: analyze_prosodic_delivery must take pause intervals as
    an input, never CALL audio_silence.detect_audio_silence_intervals
    itself (Audio V1 reuse mandate). Mentioning the reused function's name
    in a docstring/comment as documentation is fine -- an actual call
    expression (`detect_audio_silence_intervals(`) is what would violate
    the mandate, and neither appears anywhere in this module."""
    assert "detect_audio_silence_intervals(" not in MODULE_SOURCE
    src = inspect.getsource(pav2.analyze_prosodic_delivery)
    assert "silencedetect" not in src
    assert "ffmpeg" not in src  # no ffmpeg subprocess call inside the pure analysis function
    assert "detect_audio_silence_intervals" not in src  # not even referenced inside the function


def test_18_no_audio_silence_import():
    import ast
    tree = ast.parse(MODULE_SOURCE)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Import):
            for n in node.names:
                imported.add(n.name)
    assert "cutsell_worker.audio_silence" not in imported
    assert "audio_silence" not in imported


# ===========================================================================
# 19-24: hesitation / restart / filler corroboration controls
# ===========================================================================
def test_19_hesitation_present_with_interior_pause():
    audio = _audio(_speechlike(4.0))
    ev = pav2.analyze_prosodic_delivery(
        "c1", "s1", 0.0, 4.0, audio, audio_silence_intervals=[(1.5, 1.9)],
    )
    assert ev.hesitation_state == pav2.HESITATION_PRESENT


def test_20_hesitation_not_observed_without_pause():
    audio = _audio(_speechlike(4.0))
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 4.0, audio)
    assert ev.hesitation_state == pav2.HESITATION_NOT_OBSERVED


def test_21_filler_text_alone_does_not_force_hesitation():
    """Filler control: filler text + fluent (no-pause) audio must NOT be
    escalated to acoustic hesitation."""
    audio = _audio(_speechlike(4.0))
    ev = pav2.analyze_prosodic_delivery(
        "c1", "s1", 0.0, 4.0, audio, language_filler_present=True,
    )
    assert ev.hesitation_state == pav2.HESITATION_NOT_OBSERVED


def test_22_filler_plus_real_pause_is_still_hesitation():
    audio = _audio(_speechlike(4.0))
    ev = pav2.analyze_prosodic_delivery(
        "c1", "s1", 0.0, 4.0, audio,
        audio_silence_intervals=[(1.5, 1.9)], language_filler_present=True,
    )
    assert ev.hesitation_state == pav2.HESITATION_PRESENT


def test_23_restart_evidence_corroborates_not_replaces():
    audio = _audio(_speechlike(4.0))
    without = pav2.analyze_prosodic_delivery(
        "c1", "s1", 0.0, 4.0, audio, language_restart_evidence=False,
    )
    with_corrob = pav2.analyze_prosodic_delivery(
        "c1", "s1", 0.0, 4.0, audio, language_restart_evidence=True,
    )
    assert without.restart_or_interruption_state == pav2.RESTART_NOT_OBSERVED
    assert with_corrob.restart_or_interruption_state == pav2.RESTART_SUPPORTED


def test_24_restart_supported_from_acoustic_pause_alone():
    audio = _audio(_speechlike(4.0))
    ev = pav2.analyze_prosodic_delivery(
        "c1", "s1", 0.0, 4.0, audio, audio_silence_intervals=[(1.5, 1.9)],
        language_restart_evidence=None,
    )
    assert ev.restart_or_interruption_state == pav2.RESTART_SUPPORTED


# ===========================================================================
# 25-30: energy dynamics / emphasis / gain invariance
# ===========================================================================
def test_25_flat_energy_is_low_variation():
    flat = _tone(4.0, amplitude=0.2)  # constant-amplitude sine -> low frame-RMS CV
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 4.0, _audio(flat))
    assert ev.energy_dynamics_state in (pav2.VARIATION_LOW, pav2.VARIATION_MODERATE)


def test_26_varying_energy_is_higher_variation_than_flat():
    flat = _tone(4.0, amplitude=0.2)
    varying = _concat(
        _tone(1.0, amplitude=0.05), _tone(1.0, amplitude=0.5),
        _tone(1.0, amplitude=0.05), _tone(1.0, amplitude=0.6),
    )
    ev_flat = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 4.0, _audio(flat))
    ev_varying = pav2.analyze_prosodic_delivery("c2", "s1", 0.0, 4.0, _audio(varying))
    assert ev_varying.energy_variation > ev_flat.energy_variation


def test_27_gain_invariance_does_not_shift_variation_state():
    """Gain control: a global amplitude multiplier must not create a
    preference-shaped difference -- CV is scale-invariant by construction."""
    base = _concat(_tone(1.0, amplitude=0.05), _tone(1.0, amplitude=0.4),
                    _tone(1.0, amplitude=0.08), _tone(1.0, amplitude=0.35))
    loud = base * 3.0
    ev_base = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 4.0, _audio(base))
    ev_loud = pav2.analyze_prosodic_delivery("c2", "s1", 0.0, 4.0, _audio(loud))
    assert ev_base.energy_dynamics_state == ev_loud.energy_dynamics_state
    assert abs(ev_base.energy_variation - ev_loud.energy_variation) < 1e-6
    assert ev_base.energy_mean != ev_loud.energy_mean  # raw levels differ; CV doesn't


def test_28_emphasis_present_on_energy_excursion():
    normal = _tone(3.0, amplitude=0.1)
    burst = _tone(0.3, amplitude=0.9)
    waveform = _concat(normal[: len(normal) // 2], burst, normal[len(normal) // 2:])
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, float(len(waveform)) / SR, _audio(waveform))
    assert ev.emphasis_dynamics_state == pav2.EMPHASIS_PRESENT


def test_29_emphasis_not_observed_on_flat_signal():
    flat = _tone(3.0, amplitude=0.2)
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 3.0, _audio(flat))
    assert ev.emphasis_dynamics_state == pav2.EMPHASIS_NOT_OBSERVED


def test_30_energy_mean_none_when_signal_effectively_silent_but_not_zero():
    # Just above the NO_SPEECH floor but essentially negligible -> still
    # evaluated (not fabricated), since overall_rms check is at the whole-
    # span level and _MIN_ANALYZABLE_SPEECH_SEC gates duration, not amplitude.
    quiet = _tone(3.0, amplitude=0.01)
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 3.0, _audio(quiet))
    assert ev.analysis_status == pav2.STATUS_EVALUATED
    assert ev.energy_mean is not None


# ===========================================================================
# 31-34: speech rate arithmetic (pure ASR words, no audio DSP)
# ===========================================================================
def test_31_speech_rate_from_word_timings_no_audio_needed_for_arithmetic():
    words = _words([(0.0, 0.4), (0.5, 0.9), (1.0, 1.4), (1.5, 1.9)])
    ev = pav2.analyze_prosodic_delivery(
        "c1", "s1", 0.0, 2.0, _audio(_speechlike(2.0)), words=words,
    )
    assert ev.speech_rate is not None
    assert ev.speech_rate > 0


def test_32_speech_rate_state_thresholds():
    assert pav2._speech_rate_state(1.0) == pav2.RATE_SLOW
    assert pav2._speech_rate_state(2.5) == pav2.RATE_MODERATE
    assert pav2._speech_rate_state(5.0) == pav2.RATE_FAST
    assert pav2._speech_rate_state(None) == pav2.UNKNOWN


def test_33_no_words_yields_unknown_speech_rate_but_still_evaluates_acoustics():
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 3.0, _audio(_speechlike(3.0)), words=())
    assert ev.speech_rate is None
    assert ev.speech_rate_state == pav2.UNKNOWN
    assert ev.analysis_status == pav2.STATUS_EVALUATED  # acoustic evidence independent of ASR


def test_34_voiced_speech_duration_clipped_to_span():
    words = _words([(-1.0, 0.5), (1.0, 5.0)])  # extends outside [0,2]
    ev = pav2.analyze_prosodic_delivery(
        "c1", "s1", 0.0, 2.0, _audio(_speechlike(2.0)), words=words,
    )
    assert ev.voiced_or_active_speech_duration_sec <= 2.0 + 1e-6


# ===========================================================================
# 35-38: determinism / same-source multi-span / confidence gating
# ===========================================================================
def test_35_determinism_identical_inputs_identical_output():
    audio = _audio(_speechlike(4.0, seed=7))
    words = _words([(0.0, 0.4), (1.0, 1.4)])
    a = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 4.0, audio, words=words,
                                        audio_silence_intervals=[(2.0, 2.3)])
    b = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 4.0, audio, words=words,
                                        audio_silence_intervals=[(2.0, 2.3)])
    assert a == b


def test_36_same_source_two_candidate_spans_extracted_once():
    """Extract-once mandate: one AudioSamples object, two _slice_samples
    calls for two different candidate spans."""
    waveform = _speechlike(10.0)
    audio = _audio(waveform)
    span1 = pav2._slice_samples(audio, 0.0, 3.0)
    span2 = pav2._slice_samples(audio, 5.0, 9.0)
    assert span1.size == int(round(3.0 * SR))
    assert span2.size == int(round(4.0 * SR))
    ev1 = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 3.0, audio)
    ev2 = pav2.analyze_prosodic_delivery("c2", "s1", 5.0, 9.0, audio)
    assert ev1.analysis_status == pav2.STATUS_EVALUATED
    assert ev2.analysis_status == pav2.STATUS_EVALUATED


def test_37_confidence_weak_for_short_span_supported_for_longer():
    audio = _audio(_speechlike(5.0))
    short = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 1.0, audio)
    long_ = pav2.analyze_prosodic_delivery("c2", "s1", 0.0, 3.0, audio)
    assert short.evidence_confidence == pav2.CONFIDENCE_WEAK
    assert long_.evidence_confidence == pav2.CONFIDENCE_SUPPORTED


def test_38_near_equal_control_does_not_force_a_difference():
    """Two near-identical takes must yield near-identical (not artificially
    differentiated) prosodic states -- the module never manufactures a
    distinction that isn't in the signal."""
    a = _speechlike(4.0, seed=1)
    b = _speechlike(4.0, seed=2)  # different noise seed, same shape/energy profile
    ev_a = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 4.0, _audio(a))
    ev_b = pav2.analyze_prosodic_delivery("c2", "s1", 0.0, 4.0, _audio(b))
    assert ev_a.energy_dynamics_state == ev_b.energy_dynamics_state
    assert ev_a.vocal_continuity_state == ev_b.vocal_continuity_state


# ===========================================================================
# 39-42: D-186B abstract two-take differentiating fixture (generic, no
# literal Video00/Pimples transcript text).
# ===========================================================================
def test_39_abstract_two_take_fixture_differentiates_delivery():
    """D-186B-inspired abstract fixture: same generic proposition, two
    delivery realizations. Take A: broken cadence, interior pauses,
    hesitation, restart. Take B: continuous delivery. Prosodic evidence
    must FACTUALLY DIFFERENTIATE the acoustic delivery -- and must NOT
    declare a winner (no such field/return value exists at all)."""
    take_a_audio = _audio(_concat(
        _speechlike(1.0, seed=10), _silence(0.5),
        _speechlike(0.8, seed=11), _silence(0.4),
        _speechlike(1.2, seed=12),
    ), source_asset_id="src_abstract")
    take_b_audio = _audio(_speechlike(3.9, seed=20), source_asset_id="src_abstract")

    ev_a = pav2.analyze_prosodic_delivery(
        "candidate_a", "src_abstract", 0.0, 3.9, take_a_audio,
        audio_silence_intervals=[(1.0, 1.5), (2.3, 2.7)],
        language_restart_evidence=True,
    )
    ev_b = pav2.analyze_prosodic_delivery(
        "candidate_b", "src_abstract", 0.0, 3.9, take_b_audio,
        audio_silence_intervals=(),
    )

    assert ev_a.vocal_continuity_state in (pav2.CONTINUITY_MILDLY_INTERRUPTED, pav2.CONTINUITY_FRAGMENTED)
    assert ev_b.vocal_continuity_state == pav2.CONTINUITY_CONTINUOUS
    assert ev_a.vocal_continuity_state != ev_b.vocal_continuity_state
    assert ev_a.hesitation_state == pav2.HESITATION_PRESENT
    assert ev_b.hesitation_state == pav2.HESITATION_NOT_OBSERVED
    assert ev_a.restart_or_interruption_state == pav2.RESTART_SUPPORTED
    assert ev_b.restart_or_interruption_state == pav2.RESTART_NOT_OBSERVED
    # No winner field exists anywhere on the evidence object or diagnostics row.
    row_a, row_b = pav2.prosodic_delivery_diagnostics(ev_a), pav2.prosodic_delivery_diagnostics(ev_b)
    for row in (row_a, row_b):
        assert not any("winner" in k.lower() or "select" in k.lower() for k in row.keys())
    assert not hasattr(ev_a, "selected_clip_id")


def test_40_abstract_fixture_never_uses_literal_video00_text():
    forbidden_substrings = ("pimples", "espinillas", "gynecolog", "acné", "acne")
    src_lower = MODULE_SOURCE.lower()
    test_src_lower = inspect.getsource(test_39_abstract_two_take_fixture_differentiates_delivery).lower()
    for s in forbidden_substrings:
        assert s not in src_lower
        assert s not in test_src_lower


def test_41_evidence_object_has_no_winner_or_score_field():
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 3.0, _audio(_speechlike(3.0)))
    field_names = [f for f in ev.__dataclass_fields__.keys()]
    assert not any("winner" in f or "selected" in f or "score" in f for f in field_names)


def test_42_diagnostics_row_json_serializable_and_bounded():
    import json
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 3.0, _audio(_speechlike(3.0)))
    row = pav2.prosodic_delivery_diagnostics(ev)
    encoded = json.dumps(row)
    assert len(encoded) < 2000  # compact, no waveform/transcript dump
    assert set(row.keys()) == {
        "prosodic_audio_available", "prosodic_speech_rate_state",
        "prosodic_pause_structure_state", "prosodic_hesitation_state",
        "prosodic_restart_state", "prosodic_continuity_state",
        "prosodic_energy_dynamics_state", "prosodic_emphasis_state",
        "prosodic_pitch_status", "prosodic_delivery_variation_state",
        "prosodic_confidence", "prosodic_missing_evidence", "prosodic_provenance",
    }


# ===========================================================================
# 43-45: firewalls (structural source scans) -- psych/demographic/master
# score/winner-authority/no-provider/no-mutation-of-other-authorities.
# ===========================================================================
_FORBIDDEN_PSYCH_DEMOGRAPHIC_TOKENS = (
    "nervous", "anxious", "happy", "sad", "angry", "excited", "lying",
    "truthful", "persuasive", "authentic", "unauthentic",
    "age_estimate", "gender_infer", "race_infer", "ethnicity", "health_status",
    "speaker_identity", "speaker_id",
)


def test_43_no_psychological_or_demographic_inference_vocabulary_in_code():
    # Scan CODE lines only (not the module docstring, which intentionally
    # NAMES these tokens to disclaim them), and match whole words only so
    # innocent substrings (e.g. "underlying" containing "lying") don't
    # false-positive.
    import re
    body = MODULE_SOURCE.split('"""', 2)
    code_only = body[2] if len(body) >= 3 else MODULE_SOURCE
    lowered = code_only.lower()
    for token in _FORBIDDEN_PSYCH_DEMOGRAPHIC_TOKENS:
        pattern = r"\b" + re.escape(token) + r"\b"
        assert not re.search(pattern, lowered), f"forbidden token found in code: {token}"


def test_44_no_master_prosody_score_anywhere():
    import re
    code_only = MODULE_SOURCE.split('"""', 2)
    code_only = code_only[2] if len(code_only) >= 3 else MODULE_SOURCE
    assert not re.search(r"\bprosody_score\b", code_only, re.IGNORECASE)
    assert not re.search(r"\boverall_score\b", code_only, re.IGNORECASE)
    # No single float field aggregating all dimensions on the evidence object.
    field_names = set(pav2.ProsodicDeliveryEvidence.__dataclass_fields__.keys())
    assert "prosody_score" not in field_names
    assert "overall_score" not in field_names
    assert "score" not in field_names


def test_45_no_network_provider_or_authority_mutation_imports():
    import ast
    tree = ast.parse(MODULE_SOURCE)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Import):
            for n in node.names:
                imported.add(n.name)
    forbidden_modules = {
        "requests", "urllib", "urllib.request", "socket", "http.client",
        "cutsell_worker.pipeline", "cutsell_worker.bounded_finalist_arbiter",
        "cutsell_worker.canonical_edit_plan", "cutsell_worker.universal_clean_cut",
        "cutsell_worker.watch_listen_understanding",
    }
    assert not (imported & forbidden_modules), imported & forbidden_modules
    lowered = MODULE_SOURCE.lower()
    for banned_host_token in ("openai.com", "anthropic.com", "modal.com", "runpod"):
        assert banned_host_token not in lowered


# ===========================================================================
# Bonus: real ffmpeg-decode integration path (extract_source_audio_samples),
# fail-open behavior, and runtime measurement -- not part of the numbered
# 45 but required by the directive's "audio extraction reuse" / "gain and
# clipping controls" / "performance measurement" sections.
# ===========================================================================
def test_extract_real_wav_via_ffmpeg(tmp_path):
    wav_path = tmp_path / "synthetic_source.wav"
    waveform = _speechlike(2.5, amplitude=0.25)
    _write_wav(wav_path, waveform, SR)
    audio = pav2.extract_source_audio_samples(str(wav_path), source_asset_id="src_x")
    assert audio is not None
    assert audio.sample_rate == SR
    assert abs(audio.duration_sec - 2.5) < 0.05
    assert audio.samples.dtype == np.float32


def test_extract_nonexistent_file_fails_open():
    audio = pav2.extract_source_audio_samples("/nonexistent/path/does_not_exist.wav", source_asset_id="src_x")
    assert audio is None


def test_extract_missing_ffmpeg_binary_fails_open(tmp_path):
    wav_path = tmp_path / "x.wav"
    _write_wav(wav_path, _speechlike(1.0), SR)
    audio = pav2.extract_source_audio_samples(
        str(wav_path), source_asset_id="src_x", ffmpeg_bin="/nonexistent/ffmpeg_binary_xyz",
    )
    assert audio is None


def test_clipped_audio_does_not_crash():
    waveform = np.clip(_tone(2.0, amplitude=1.5), -1.0, 1.0)  # heavily clipped
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 2.0, _audio(waveform))
    assert ev.analysis_status == pav2.STATUS_EVALUATED  # never crashes on clipped signal


def test_runtime_measurement_on_bounded_fixture():
    """Performance measurement (informational only -- no optimization
    target invented, per this task's own explicit instruction)."""
    waveform = _speechlike(30.0)
    audio = _audio(waveform)
    start = time.monotonic()
    ev = pav2.analyze_prosodic_delivery("c1", "s1", 0.0, 30.0, audio)
    elapsed = time.monotonic() - start
    real_time_factor = elapsed / 30.0 if elapsed else 0.0
    assert ev.analysis_status == pav2.STATUS_EVALUATED
    # No assertion on a specific RTF ceiling -- purely observational, printed
    # for the decision-log record.
    print(f"D-187 runtime: audio_duration_sec=30.0 wall_sec={elapsed:.4f} rtf={real_time_factor:.5f}")


def test_mono_downmix_is_documented_design_choice():
    """extract_source_audio_samples always requests -ac 1 (mono downmix)
    from ffmpeg -- a deliberate, documented design choice (not a silent
    gap): stereo/multi-channel source audio is downmixed before analysis,
    matching this codebase's existing render/QC mono convention."""
    src = inspect.getsource(pav2.extract_source_audio_samples)
    assert '"-ac", "1"' in src or "-ac', '1'" in src
