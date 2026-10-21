"""D-187: Prosodic Audio V2 -- Phase A, PERCEPTION / EVIDENCE ONLY.

Answers ONE question for a source-real spoken span: **HOW was the
message vocally delivered?** Never WHAT was said (Language Spine's own
authority -- `language_spine.py`/`language_utterance_attempt.py`),
never WHAT was visually performed (`local_performance.py`/Visual
Spine), never WHICH take wins (`pipeline.py`'s BestTake ladder / D-183
`TerminalBestTakeConfidence` / D-184 `bounded_finalist_arbiter.py`).

## Canonical position (per this task's own directive)

    RAW AUDIO
    -> AUDIO V1 (silence/dead-air/pauses/timing -- `audio_silence.py`,
       `silence_analysis.py`, UNCHANGED, reused verbatim here)
    -> PROSODIC AUDIO V2 (this module: observable vocal-delivery
       evidence)

    Language Spine + Visual/Performance + Audio V1 + Prosodic Audio V2
    + Media/Timing -> Watch+Listen Understanding LATER (not touched by
    this module at all).

This module is a freestanding evidence library. It is NOT wired into
`pipeline.py`'s live per-family loop, NOT wired into
`watch_listen_understanding.py`, and NOT consumed by
`bounded_finalist_arbiter.py` -- D-188 (a future, separately-authorized
task) owns any Prosodic-to-arbiter fusion. Phase A produces evidence
objects only; nothing here is called by any production code path.

## Audio dependency forensic (this task's own required audit)

`scipy`/`librosa`/`parselmouth`/`pyworld`/`soundfile`/`pydub` are NOT
installed in this environment (verified directly, not assumed) and are
NOT added by this module -- no new dependency burden. `numpy` (already
a hard requirement, see `requirements.txt`) and `ffmpeg` (already a
hard dependency of this codebase's own render/QC/`audio_silence.py`
pipeline) are the ONLY tools this module uses. Consequently:

- Speech rate: pure arithmetic over existing ASR `Word` timings -- no
  audio signal needed at all.
- Pause structure: REUSES `audio_silence.py`'s own
  `AUDIO_SILENCE_EVENT_KIND` `TemporalEvent`s (ffmpeg `silencedetect`,
  D-095.2) verbatim -- this module never runs a second silence
  detector, per this task's own explicit "audit first" instruction.
- Energy dynamics / emphasis: computed from ffmpeg-decoded raw PCM
  samples (`extract_source_audio_samples`) using plain `numpy` RMS-
  over-frames -- no DSP library required.
- Pitch/F0: honestly `PITCH_ANALYSIS_NOT_IMPLEMENTED` -- a raw
  autocorrelation pitch tracker written from scratch without any
  vetted DSP library would be exactly the "faking it" this task
  explicitly forbids; the task's own instruction explicitly permits
  this status rather than blocking Phase A.

## Firewalls (binding, structurally enforced and directly tested)

NO psychological inference (confidence/nervousness/emotion/lying/
persuasiveness/authenticity) from voice. NO demographic/biometric/
identity inference (age/gender/race/health/speaker identity). NO
master prosody score -- every dimension (rate, pause, hesitation,
continuity, energy, emphasis, pitch availability, variation) stays
separate; a future arbiter (D-188) reasons from evidence, never from
one magic number. NO winner authority -- nothing in this module can
set `selected_clip_id`, choose a BestTake, reject a winner, modify a
family, or touch a render plan (there is no code path here that even
imports anything from `pipeline.py`, `bounded_finalist_arbiter.py`,
`canonical_edit_plan.py`, or any render module).

## Source-real timing

Every `ProsodicDeliveryEvidence` carries `source_asset_id`,
`source_start`, `source_end` verbatim from its caller -- no synthetic
clock, no re-derived timeline.

## Language-Spine corroboration only

`language_restart_evidence` (an optional, caller-supplied bool from
`attempt_reconstruction._restart_evidence`/`LanguageAttempt.restart_
evidence`) and `language_filler_present` are consumed ONLY as
corroborating context for the acoustic restart/hesitation states --
this module never recreates word/meaning/attempt/proposition/relation
identity, and a transcript-only caller (no real audio) still yields
honest `UNKNOWN`/`NOT_AVAILABLE` acoustic fields (see
`test_02_transcript_only_is_acoustic_unknown`).
"""
from __future__ import annotations

from dataclasses import dataclass
import subprocess
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = "cutsell.prosodic_audio_v2.v1"

PROSODIC_AUDIO_AVAILABLE = False  # module-level fact for AT-REST honesty; a
# real per-candidate result's own `evidence_confidence`/`missing_evidence`
# fields are the actual per-call truth -- this constant mirrors D-184's own
# `PROSODIC_AUDIO_AVAILABLE` naming so a future consumer can grep one name.

_SUBPROCESS_TIMEOUT_SEC = 600
DEFAULT_SAMPLE_RATE = 16000
_FRAME_SEC = 0.020  # 20ms RMS analysis frame -- plain numpy, no DSP library.
_MIN_ANALYZABLE_SPEECH_SEC = 0.75  # below this, dynamics are not reliable.
_INTERIOR_PAUSE_MARGIN_SEC = 0.15  # a pause this close to either edge is a
# boundary pause, not an interior delivery interruption.

# ---------------------------------------------------------------------------
# Categorical vocabulary (compact, factual, never a value judgment).
# ---------------------------------------------------------------------------
UNKNOWN = "UNKNOWN"

RATE_SLOW = "SLOW"
RATE_MODERATE = "MODERATE"
RATE_FAST = "FAST"

CONTINUITY_CONTINUOUS = "CONTINUOUS"
CONTINUITY_MILDLY_INTERRUPTED = "MILDLY_INTERRUPTED"
CONTINUITY_FRAGMENTED = "FRAGMENTED"

VARIATION_LOW = "LOW_VARIATION"
VARIATION_MODERATE = "MODERATE_VARIATION"
VARIATION_HIGH = "HIGH_VARIATION"

HESITATION_PRESENT = "HESITATION_PATTERN_PRESENT"
HESITATION_NOT_OBSERVED = "HESITATION_NOT_OBSERVED"

RESTART_SUPPORTED = "RESTART_OR_INTERRUPTION_SUPPORTED"
RESTART_NOT_OBSERVED = "RESTART_NOT_OBSERVED"

EMPHASIS_PRESENT = "EMPHASIS_PATTERN_PRESENT"
EMPHASIS_NOT_OBSERVED = "EMPHASIS_NOT_OBSERVED"

PITCH_NOT_IMPLEMENTED = "PITCH_ANALYSIS_NOT_IMPLEMENTED"

CONFIDENCE_SUPPORTED = "SUPPORTED"
CONFIDENCE_WEAK = "WEAK"
CONFIDENCE_MIXED = "MIXED"
CONFIDENCE_UNKNOWN = "UNKNOWN"

STATUS_NO_SPEECH = "NO_SPEECH"
STATUS_NOT_EVALUABLE = "NOT_EVALUABLE"
STATUS_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
STATUS_EVALUATED = "EVALUATED"

# ---------------------------------------------------------------------------
# Capability status contract (this task's own explicit requirement) -- a
# static, code-derived source of truth for what THIS Phase A build can
# actually measure. Never silently claim more than this.
# ---------------------------------------------------------------------------
CAPABILITY_AVAILABLE = "AVAILABLE"
CAPABILITY_PARTIAL = "PARTIAL"
CAPABILITY_NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
CAPABILITY_NOT_EVALUABLE = "NOT_EVALUABLE"

PROSODIC_CAPABILITY_STATUS: Mapping[str, str] = {
    # Pure ASR-word arithmetic -- no audio signal required at all.
    "speech_rate": CAPABILITY_AVAILABLE,
    # Reuses audio_silence.py's own ffmpeg silencedetect evidence verbatim.
    "pause_structure": CAPABILITY_AVAILABLE,
    # Categorical inference from pause placement + optional Language-Spine
    # corroboration -- real, but coarser than a dedicated hesitation model.
    "hesitation": CAPABILITY_PARTIAL,
    "restart": CAPABILITY_PARTIAL,
    "continuity": CAPABILITY_AVAILABLE,
    # Real RMS-over-frames from decoded PCM via ffmpeg + numpy.
    "energy": CAPABILITY_AVAILABLE,
    # Energy-excursion-only (no pitch excursion, no duration-elongation
    # alignment yet) -- a real but partial emphasis signal.
    "emphasis": CAPABILITY_PARTIAL,
    # No parselmouth/pyworld/librosa in this environment -- honestly absent.
    "pitch": CAPABILITY_NOT_IMPLEMENTED,
    "delivery_variation": CAPABILITY_PARTIAL,
}


def prosodic_capability_status_summary() -> dict:
    """Compact `prosodic_audio_status` + per-capability status, so
    downstream code can know which evidence is actually real without
    guessing from field values."""
    values = set(PROSODIC_CAPABILITY_STATUS.values())
    if values == {CAPABILITY_NOT_IMPLEMENTED}:
        overall = CAPABILITY_NOT_IMPLEMENTED
    elif CAPABILITY_AVAILABLE in values and CAPABILITY_NOT_IMPLEMENTED not in values:
        overall = CAPABILITY_AVAILABLE
    else:
        overall = CAPABILITY_PARTIAL
    return {"prosodic_audio_status": overall, **PROSODIC_CAPABILITY_STATUS}


# ---------------------------------------------------------------------------
# Audio extraction (extract once per source; slice per candidate span).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class AudioSamples:
    """One source's fully-decoded mono waveform, extracted ONCE via ffmpeg
    -- callers slice `samples` per candidate span rather than re-invoking
    ffmpeg per candidate (this task's own "extract once" requirement).
    `eq=False`: a numpy array has no meaningful dataclass equality; identity
    comparison is what every caller here actually needs."""
    source_asset_id: str
    sample_rate: int
    samples: "np.ndarray"  # float32, mono, range approx [-1.0, 1.0]
    duration_sec: float
    provenance: str


def extract_source_audio_samples(
    local_path: str,
    *,
    source_asset_id: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    ffmpeg_bin: str = "ffmpeg",
) -> AudioSamples | None:
    """Decode the WHOLE source once to mono PCM via ffmpeg. Never raises --
    an unreadable file, a missing ffmpeg, or a timeout yields `None` (fail-
    open, same posture as `audio_silence.detect_audio_silence_intervals`).
    No new dependency: ffmpeg is already a hard dependency of this
    codebase's own render/QC path."""
    command = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostats",
        "-i", str(local_path), "-vn",
        "-ac", "1", "-ar", str(int(sample_rate)),
        "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ]
    try:
        completed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=_SUBPROCESS_TIMEOUT_SEC,
        )
    except Exception:  # noqa: BLE001 -- evidence source unavailable, never fatal
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None
    try:
        raw = np.frombuffer(completed.stdout, dtype="<i2")
        samples = raw.astype(np.float32) / 32768.0
    except Exception:  # noqa: BLE001
        return None
    duration_sec = float(len(samples)) / float(sample_rate) if sample_rate else 0.0
    return AudioSamples(
        source_asset_id=str(source_asset_id),
        sample_rate=int(sample_rate),
        samples=samples,
        duration_sec=duration_sec,
        provenance="ffmpeg_pcm_s16le_mono",
    )


def _slice_samples(audio: AudioSamples, start: float, end: float) -> "np.ndarray":
    lo = max(0, int(round(max(0.0, start) * audio.sample_rate)))
    hi = min(len(audio.samples), int(round(max(0.0, end) * audio.sample_rate)))
    if hi <= lo:
        return np.zeros(0, dtype=np.float32)
    return audio.samples[lo:hi]


def _rms_frames(samples: "np.ndarray", sample_rate: int, frame_sec: float = _FRAME_SEC) -> "np.ndarray":
    frame_len = max(1, int(round(frame_sec * sample_rate)))
    n_frames = len(samples) // frame_len
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float64)
    trimmed = samples[: n_frames * frame_len].astype(np.float64)
    framed = trimmed.reshape(n_frames, frame_len)
    return np.sqrt(np.mean(framed * framed, axis=1))


# ---------------------------------------------------------------------------
# ProsodicDeliveryEvidence
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProsodicDeliveryEvidence:
    """One candidate span's observable vocal-delivery evidence. Additive,
    diagnostic-only -- never a decision, never a score. Every numeric field
    may be `None` and every categorical field may be `UNKNOWN` when the
    underlying signal is genuinely unavailable; this module never invents a
    value to fill a gap (see module docstring's firewalls)."""
    candidate_id: str
    source_asset_id: str
    source_start: float
    source_end: float

    analysis_status: str  # STATUS_* -- EVALUATED/NO_SPEECH/NOT_EVALUABLE/INSUFFICIENT_EVIDENCE

    speech_duration_sec: float | None
    voiced_or_active_speech_duration_sec: float | None
    speech_rate: float | None  # words per second of active speech, or None
    speech_rate_state: str

    pause_count: int | None
    pause_total_sec: float | None
    pause_structure_state: str

    hesitation_state: str
    restart_or_interruption_state: str
    vocal_continuity_state: str

    energy_mean: float | None
    energy_variation: float | None  # coefficient of variation (scale-invariant)
    energy_dynamics_state: str

    emphasis_dynamics_state: str

    pitch_analysis_status: str
    pitch_variation_state: str

    delivery_variation_state: str

    evidence_confidence: str  # SUPPORTED/WEAK/MIXED/UNKNOWN
    missing_evidence: Tuple[str, ...]

    provenance: str


def _abstain_evidence(
    candidate_id: str, source_asset_id: str, source_start: float, source_end: float,
    *, status: str, missing_evidence: Tuple[str, ...], provenance: str,
) -> ProsodicDeliveryEvidence:
    return ProsodicDeliveryEvidence(
        candidate_id=candidate_id, source_asset_id=source_asset_id,
        source_start=source_start, source_end=source_end,
        analysis_status=status,
        speech_duration_sec=None, voiced_or_active_speech_duration_sec=None,
        speech_rate=None, speech_rate_state=UNKNOWN,
        pause_count=None, pause_total_sec=None, pause_structure_state=UNKNOWN,
        hesitation_state=UNKNOWN, restart_or_interruption_state=UNKNOWN,
        vocal_continuity_state=UNKNOWN,
        energy_mean=None, energy_variation=None, energy_dynamics_state=UNKNOWN,
        emphasis_dynamics_state=UNKNOWN,
        pitch_analysis_status=PITCH_NOT_IMPLEMENTED, pitch_variation_state=UNKNOWN,
        delivery_variation_state=UNKNOWN,
        evidence_confidence=CONFIDENCE_UNKNOWN,
        missing_evidence=tuple(missing_evidence),
        provenance=provenance,
    )


def _speech_rate_state(rate: float | None) -> str:
    if rate is None:
        return UNKNOWN
    if rate < 2.0:
        return RATE_SLOW
    if rate <= 3.3:
        return RATE_MODERATE
    return RATE_FAST


def _continuity_state(pause_count: int, span_duration: float, pause_total_sec: float) -> str:
    if span_duration <= 0:
        return UNKNOWN
    if pause_count == 0:
        return CONTINUITY_CONTINUOUS
    if pause_count <= 2 and (pause_total_sec / span_duration) < 0.25:
        return CONTINUITY_MILDLY_INTERRUPTED
    return CONTINUITY_FRAGMENTED


def _variation_state(cv: float | None) -> str:
    if cv is None:
        return UNKNOWN
    if cv < 0.25:
        return VARIATION_LOW
    if cv <= 0.55:
        return VARIATION_MODERATE
    return VARIATION_HIGH


def analyze_prosodic_delivery(
    candidate_id: str,
    source_asset_id: str,
    source_start: float,
    source_end: float,
    audio: AudioSamples | None,
    *,
    words: Sequence = (),
    audio_silence_intervals: Iterable[Tuple[float, float]] = (),
    language_restart_evidence: bool | None = None,
    language_filler_present: bool | None = None,
) -> ProsodicDeliveryEvidence:
    """Pure function: the SAME `audio`/`words`/`audio_silence_intervals`
    always yield the SAME evidence (determinism -- no randomness anywhere
    in this module). Never mutates `audio`; never calls a network provider;
    never infers emotion/psychology/demographics/identity (see module
    docstring's firewalls, and the structural tests in
    tests/test_cutsell_d187_prosodic_audio_v2.py)."""
    source_start = float(source_start)
    source_end = float(source_end)
    span_duration = max(0.0, source_end - source_start)
    provenance = "prosodic_audio_v2_phase_a"

    if audio is None:
        return _abstain_evidence(
            candidate_id, source_asset_id, source_start, source_end,
            status=STATUS_NOT_EVALUABLE,
            missing_evidence=("audio_samples", "pitch_analysis"),
            provenance=provenance,
        )

    # --- Speech rate: pure ASR-word arithmetic, no audio signal needed. ---
    span_words = [
        w for w in words
        if float(getattr(w, "start", 0.0)) < source_end and float(getattr(w, "end", 0.0)) > source_start
    ]
    voiced_speech_sec = sum(
        max(0.0, min(source_end, float(w.end)) - max(source_start, float(w.start))) for w in span_words
    )
    speech_rate = (len(span_words) / voiced_speech_sec) if (span_words and voiced_speech_sec > 0) else None

    # --- Pause structure: REUSE Audio V1 evidence, never recompute. ---
    interior_intervals = [
        (max(source_start, s), min(source_end, e))
        for s, e in audio_silence_intervals
        if e > source_start and s < source_end
    ]
    interior_only = [
        (s, e) for s, e in interior_intervals
        if s >= source_start + _INTERIOR_PAUSE_MARGIN_SEC and e <= source_end - _INTERIOR_PAUSE_MARGIN_SEC
    ]
    pause_count = len(interior_only)
    pause_total_sec = round(sum(e - s for s, e in interior_only), 4)

    # --- Slice the candidate span out of the once-extracted source. ---
    span_samples = _slice_samples(audio, source_start, source_end)
    if span_samples.size == 0:
        return _abstain_evidence(
            candidate_id, source_asset_id, source_start, source_end,
            status=STATUS_NO_SPEECH,
            missing_evidence=("audio_samples_in_span", "pitch_analysis"),
            provenance=provenance,
        )

    frames = _rms_frames(span_samples, audio.sample_rate)
    overall_rms = float(np.sqrt(np.mean(span_samples.astype(np.float64) ** 2))) if span_samples.size else 0.0
    # A near-silent span (no active signal above noise floor) is NO_SPEECH,
    # never a fabricated "flat delivery" verdict.
    if overall_rms < 1e-4:
        return _abstain_evidence(
            candidate_id, source_asset_id, source_start, source_end,
            status=STATUS_NO_SPEECH,
            missing_evidence=("active_speech_signal", "pitch_analysis"),
            provenance=provenance,
        )

    if span_duration < _MIN_ANALYZABLE_SPEECH_SEC or frames.size < 3:
        return _abstain_evidence(
            candidate_id, source_asset_id, source_start, source_end,
            status=STATUS_INSUFFICIENT_EVIDENCE,
            missing_evidence=("sufficient_span_duration", "pitch_analysis"),
            provenance=provenance,
        )

    energy_mean = float(np.mean(frames))
    energy_std = float(np.std(frames))
    # Coefficient of variation: SCALE-INVARIANT by construction -- a global
    # gain multiplier on the waveform cancels out of std/mean identically,
    # so this state never shifts merely because a recording is louder (this
    # task's own "gain control" requirement).
    energy_cv = (energy_std / energy_mean) if energy_mean > 1e-9 else None
    energy_dynamics_state = _variation_state(energy_cv)

    # --- Emphasis: energy-excursion-only (no pitch excursion available). ---
    if energy_mean > 1e-9:
        excursions = int(np.sum(frames > (energy_mean + 1.5 * energy_std)))
        emphasis_state = EMPHASIS_PRESENT if excursions >= 1 else EMPHASIS_NOT_OBSERVED
    else:
        emphasis_state = UNKNOWN

    # --- Hesitation / restart / continuity: categorical, from pause
    # placement, optionally corroborated (never overridden) by Language
    # Spine's own restart/filler evidence. ---
    continuity_state = _continuity_state(pause_count, span_duration, pause_total_sec)
    acoustic_hesitation = pause_count >= 1
    if acoustic_hesitation:
        hesitation_state = HESITATION_PRESENT
    elif language_filler_present:
        # Filler text alone, with continuous acoustic delivery, is NOT
        # escalated to acoustic hesitation -- Language evidence corroborates,
        # audio determines the acoustic shape (this task's own filler
        # control requirement).
        hesitation_state = HESITATION_NOT_OBSERVED
    else:
        hesitation_state = HESITATION_NOT_OBSERVED

    restart_state = (
        RESTART_SUPPORTED if (pause_count >= 1 or bool(language_restart_evidence))
        else RESTART_NOT_OBSERVED
    )

    delivery_variation_state = energy_dynamics_state  # pitch absent -> energy is the only variation axis

    missing = ["pitch_analysis"]
    confidence = CONFIDENCE_SUPPORTED if span_duration >= 1.5 else CONFIDENCE_WEAK

    return ProsodicDeliveryEvidence(
        candidate_id=candidate_id, source_asset_id=source_asset_id,
        source_start=source_start, source_end=source_end,
        analysis_status=STATUS_EVALUATED,
        speech_duration_sec=round(span_duration, 4),
        voiced_or_active_speech_duration_sec=round(voiced_speech_sec, 4) if voiced_speech_sec else None,
        speech_rate=round(speech_rate, 4) if speech_rate is not None else None,
        speech_rate_state=_speech_rate_state(speech_rate),
        pause_count=pause_count,
        pause_total_sec=pause_total_sec,
        pause_structure_state=continuity_state,
        hesitation_state=hesitation_state,
        restart_or_interruption_state=restart_state,
        vocal_continuity_state=continuity_state,
        energy_mean=round(energy_mean, 6),
        energy_variation=round(energy_cv, 6) if energy_cv is not None else None,
        energy_dynamics_state=energy_dynamics_state,
        emphasis_dynamics_state=emphasis_state,
        pitch_analysis_status=PITCH_NOT_IMPLEMENTED,
        pitch_variation_state=UNKNOWN,
        delivery_variation_state=delivery_variation_state,
        evidence_confidence=confidence,
        missing_evidence=tuple(missing),
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Compact per-candidate diagnostics row (this task's own required shape).
# ---------------------------------------------------------------------------
def prosodic_delivery_diagnostics(evidence: ProsodicDeliveryEvidence) -> dict:
    """JSON-safe, bounded per-candidate diagnostics row -- no waveform, no
    transcript dump. Field names match this task's own directive."""
    return {
        "prosodic_audio_available": evidence.analysis_status == STATUS_EVALUATED,
        "prosodic_speech_rate_state": evidence.speech_rate_state,
        "prosodic_pause_structure_state": evidence.pause_structure_state,
        "prosodic_hesitation_state": evidence.hesitation_state,
        "prosodic_restart_state": evidence.restart_or_interruption_state,
        "prosodic_continuity_state": evidence.vocal_continuity_state,
        "prosodic_energy_dynamics_state": evidence.energy_dynamics_state,
        "prosodic_emphasis_state": evidence.emphasis_dynamics_state,
        "prosodic_pitch_status": evidence.pitch_analysis_status,
        "prosodic_delivery_variation_state": evidence.delivery_variation_state,
        "prosodic_confidence": evidence.evidence_confidence,
        "prosodic_missing_evidence": evidence.missing_evidence,
        "prosodic_provenance": evidence.provenance,
    }
