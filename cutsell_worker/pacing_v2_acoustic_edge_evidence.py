"""D-230 -- Pacing V2 Acoustic Edge Evidence Foundation. OFFLINE ONLY.

Closes the ONE gap D-229's own forensic named (docs/CUTSELL_DECISIONS.md
D-229, verdict B): every existing speech/non-speech signal in this
codebase (`pacing_v2_source_audio_handle.py`'s own `_classify_speech_
presence`) is WORD-TIMING ONLY -- "no ASR word here" is treated as a
structural proof of safety only when it ALSO comes from a Boundary
authority's own already-applied, word-boundary-clamped trim; nothing
today measures the ACTUAL AUDIO at a join edge. This module adds that
missing acoustic layer -- genuinely measured (ffmpeg-decoded PCM, numpy
RMS/FFT, ffmpeg `silencedetect`), never word-timing-only, never a
learned model, never a new dependency.

## What this module explicitly does NOT do

Does not choose `SHORT_CROSSFADE`/`AMBIENCE_CARRY_LEFT`/`AMBIENCE_
CARRY_RIGHT`/`AMBIENCE_BRIDGE`/`J_CUT`/`L_CUT`/`MICRO_AUDIO_OVERLAP` --
those remain downstream Layer-2/Layer-3 decisions (D-098 Section 16).
Does not implement a "room tone" classifier -- `ROOM_TONE_
CLASSIFICATION_NOT_YET_AVAILABLE` remains the honest status (D-222/
D-225/D-228/D-229 precedent), even though this module now measures
enough acoustic descriptors that a FUTURE gate could build one. Does
not normalize gain, denoise, de-click, de-breath, de-plosive, EQ, or
compress -- loudness CORRECTION remains `finishing_contract.py`'s own
(D-024) separate, dormant territory; this module only OBSERVES level
(raw RMS) and a level-ROBUST signature (gain-normalized ratios/
centroid), never corrects it. Does not touch `cutsell_worker/
pacing_v2_source_audio_handle.py`, `render.py`, `render_plan.py`,
Boundary, Ordering, Family, or BestTake -- all read-only inputs, zero
mutation. Does not wire into `universal_clean_cut.py` or any live call
site, has no feature flag, and is used by offline tests only.

## Primitive reuse (D-230's own explicit "no fourth RMS implementation"
instruction)

D-229's own forensic found THREE independent RMS implementations
already in this repo (`prosodic_audio_v2.py`, `perceptual_watch_
listen.py`, `human_gold_decision_map.py`), none shared. This module
selects `prosodic_audio_v2.py`'s own `extract_source_audio_samples`/
`_slice_samples`/`_rms_frames` as the ONE canonical primitive to build
on: it is the only one of the three already living in the
"Perception/Evidence Only" architecture layer Join Understanding also
belongs to (D-098 Section 16.3), already proven dependency-free
(ffmpeg + numpy only, no scipy/librosa/parselmouth), and already
designed for the "extract once per source, slice per span" discipline
this module's own edge windows need. Silence evidence reuses `audio_
silence.detect_audio_silence_intervals` verbatim -- a second silence
detector is explicitly forbidden by this task's own instruction and
was never written.

## Acoustic signature (new, numpy-only, no model, no embedding)

`AcousticWindowSignature` adds ONE new descriptor this codebase did not
have before D-230: a small, deterministic, gain-ROBUST frequency-band
profile via `numpy.fft.rfft` (already a hard dependency; no scipy
needed for a real FFT). `band_energy_ratios` are magnitude-per-band
NORMALIZED by total spectral energy, and `spectral_centroid_hz` is a
magnitude-weighted mean frequency -- both mathematically invariant to
a global gain multiplier by construction (scaling every sample by a
constant scales every FFT magnitude by the same constant, which
cancels out of a ratio or a weighted average). This directly satisfies
this task's own "amplitude robustness" requirement: two windows of the
same background at different gain compare as SIMILAR on signature,
while the SEPARATE, un-normalized `rms` field still carries the real
absolute level for future loudness work (never used by this module to
decide, only to observe -- the "loudness ownership firewall" below).

## Firewalls (restated, not weakened)

- **Word-timing limit** (D-229's own restated finding): absence of a
  known word is never, by itself, "safe non-speech" -- see `non_
  speech_status`'s own decision table; `WORD_COVERAGE_UNKNOWN` always
  keeps a window at `NON_SPEECH_UNCONFIRMED`/`SPEECH_STATUS_UNKNOWN`,
  never `SAFE_NON_SPEECH_CANDIDATE`.
- **Discarded/retry/correction/meaning firewall**: when built FROM a
  `SourceAudioHandle` (D-223), this module reuses that handle's own
  already-computed `handle_status`/`discarded_overlap_status`/
  `meaning_safety_status` DIRECTLY -- it never re-derives them. A
  correction to D-229's own item 14: `pacing_v2_source_audio_handle.
  py`'s `_decide_handle_status` already checks `retry_evidence_kinds`
  BEFORE ever returning `HANDLE_STATUS_SAFE_NON_SPEECH` -- a handle
  already at `SAFE_NON_SPEECH_HANDLE` can never be retry/BTS-derived by
  construction, so no ADDITIONAL provenance re-check is needed beyond
  reading `handle_status` itself. (D-229 overstated this as a still-
  open gap; recorded here as an honest correction, not a new finding.)
- **Loudness ownership firewall**: this module OBSERVES `rms`/a
  `level_delta_db` between two edges; it never normalizes, corrects, or
  recommends a gain change. That remains `finishing_contract.py`'s own
  (D-024) or a future local gain-match step's territory (D-229 item
  28-29), never this module's.
- **Firewall-block vs. evidence-quality gap**: a firewall block
  (`CONFLICT_DISCARDED_MATERIAL`/`CONFLICT_RETRY_OR_CORRECTION`/
  `CONFLICT_MEANING_CRITICAL`, tracked in `_FIREWALL_CONFLICTS`) always
  forces `NON_SPEECH_UNKNOWN`/`EVIDENCE_STATUS_CONFLICTED` regardless of
  anything else -- discovered/fixed during this module's own test
  authoring: a first implementation draft conflated this with mere
  evidence-QUALITY gaps (`CONFLICT_AUDIO_EXTRACTION_UNAVAILABLE`/
  `CONFLICT_WINDOW_TOO_SHORT`), which wrongly downgraded an already-
  KNOWN `LEXICAL_SPEECH_PRESENT` window (from word timing, independent
  of acoustic availability) to `NON_SPEECH_UNKNOWN` merely because no
  audio track happened to be supplied. Evidence-quality gaps now fall
  through to the ordinary `INSUFFICIENT_EVIDENCE`/`NON_SPEECH_
  UNCONFIRMED` checks instead, never `CONFLICTED`/firewall-`UNKNOWN`.

## Room tone / background signature honesty

`ROOM_TONE_CLASSIFICATION_STATUS = "NOT_YET_AVAILABLE"` (module-level
constant, unchanged from D-222/D-225/D-228/D-229's own established
status). Nothing here claims to KNOW two windows share "the same room"
-- `compare_acoustic_edges` reports a bounded, explicitly-labeled
`ACOUSTIC_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_SIMILARITY_THRESHOLD`
categorical SIMILAR/DIFFERENT comparison over measured descriptors,
never a semantic "this is the same room" claim, and never a numeric
master score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .prosodic_audio_v2 import AudioSamples, _rms_frames, _slice_samples

SCHEMA_VERSION = "cutsell.pacing_v2_acoustic_edge_evidence.v1"

# ---------------------------------------------------------------------------
# Edge kind vocabulary (this task's own named four).
# ---------------------------------------------------------------------------
EDGE_LEFT_END = "LEFT_END"
EDGE_RIGHT_START = "RIGHT_START"
EDGE_PRE_HANDLE = "PRE_HANDLE"
EDGE_POST_HANDLE = "POST_HANDLE"
_KNOWN_EDGES = frozenset({EDGE_LEFT_END, EDGE_RIGHT_START, EDGE_PRE_HANDLE, EDGE_POST_HANDLE})

# ---------------------------------------------------------------------------
# Word coverage vocabulary.
# ---------------------------------------------------------------------------
WORD_COVERAGE_KNOWN_EMPTY = "KNOWN_EMPTY"
WORD_COVERAGE_KNOWN_PRESENT = "KNOWN_PRESENT"
WORD_COVERAGE_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Silence vocabulary (reuses audio_silence.py's own already-computed
# intervals; these buckets are a plain coverage-fraction categorization,
# never a second detector).
# ---------------------------------------------------------------------------
SILENCE_STATUS_SILENT = "SILENT"
SILENCE_STATUS_MOSTLY_SILENT = "MOSTLY_SILENT"
SILENCE_STATUS_NON_SILENT = "NON_SILENT"
SILENCE_STATUS_UNKNOWN = "UNKNOWN"
_SILENT_FRACTION_FLOOR = 0.95
_MOSTLY_SILENT_FRACTION_FLOOR = 0.10

# ---------------------------------------------------------------------------
# Energy vocabulary (reuses prosodic_audio_v2.py's own near-silence
# constant verbatim -- never a second, different silence threshold).
# ---------------------------------------------------------------------------
ENERGY_STATUS_NO_SIGNAL = "NO_SIGNAL"
ENERGY_STATUS_LOW_ENERGY = "LOW_ENERGY"
ENERGY_STATUS_ACTIVE_ENERGY = "ACTIVE_ENERGY"
ENERGY_STATUS_UNKNOWN = "UNKNOWN"
_NEAR_SILENT_RMS = 1e-4  # identical constant to prosodic_audio_v2.py's own "overall_rms < 1e-4" check.
ACOUSTIC_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_LOW_ENERGY_CEILING = 0.015
# ^ ONE explicit, isolated, offline heuristic (never real-media tuned),
# distinguishing LOW_ENERGY from ACTIVE_ENERGY once a window is already
# known to be above the near-silence floor. Not used to gate any
# firewall -- only descriptive.

# ---------------------------------------------------------------------------
# Speech status vocabulary (this task's own conservative names).
# ---------------------------------------------------------------------------
SPEECH_STATUS_LEXICAL_PRESENT = "LEXICAL_SPEECH_PRESENT"
SPEECH_STATUS_NO_LEXICAL_OBSERVED = "NO_LEXICAL_SPEECH_OBSERVED"
SPEECH_STATUS_UNKNOWN = "SPEECH_STATUS_UNKNOWN"

# ---------------------------------------------------------------------------
# Non-speech status vocabulary (this task's own conservative names).
# ---------------------------------------------------------------------------
NON_SPEECH_SAFE_CANDIDATE = "SAFE_NON_SPEECH_CANDIDATE"
NON_SPEECH_UNCONFIRMED = "NON_SPEECH_UNCONFIRMED"
NON_SPEECH_SPEECH_PRESENT = "SPEECH_PRESENT"
NON_SPEECH_SILENCE = "SILENCE"
NON_SPEECH_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Room-tone honesty (unchanged status, restated here for this module).
# ---------------------------------------------------------------------------
ROOM_TONE_CLASSIFICATION_STATUS = "NOT_YET_AVAILABLE"

# ---------------------------------------------------------------------------
# Signature availability vocabulary.
# ---------------------------------------------------------------------------
SIGNATURE_AVAILABLE = "SIGNATURE_AVAILABLE"
SIGNATURE_UNAVAILABLE = "SIGNATURE_UNAVAILABLE"
_SIGNATURE_MIN_SAMPLES = 64  # a structural FFT floor, not a tuned threshold.
_SIGNATURE_BAND_COUNT = 4

# ---------------------------------------------------------------------------
# Evidence status vocabulary (mirrors pacing_v2_timing_policy.py's own
# TIMING_STATUS_* shape -- a reused PATTERN, never a reused constant).
# ---------------------------------------------------------------------------
EVIDENCE_STATUS_SUPPORTED = "SUPPORTED"
EVIDENCE_STATUS_SAFE_FALLBACK = "SAFE_FALLBACK"
EVIDENCE_STATUS_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
EVIDENCE_STATUS_CONFLICTED = "CONFLICTED"
EVIDENCE_STATUS_UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Continuity comparison vocabulary.
# ---------------------------------------------------------------------------
CONTINUITY_SIMILAR = "SIMILAR"
CONTINUITY_DIFFERENT = "DIFFERENT"
CONTINUITY_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
CONTINUITY_CONFLICTED = "CONFLICTED"
ACOUSTIC_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_SIMILARITY_THRESHOLD = 0.20
# ^ THE ONE bounded, isolated, explicitly-labeled comparison threshold
# this task's own "no magic similarity score first" section allows when
# truly required. Tested directly (see test file); never silently
# reused as a production Layer-3 authority constant.

# ---------------------------------------------------------------------------
# Conflict-flag vocabulary (reuses this track's own "reason string" idiom).
# ---------------------------------------------------------------------------
CONFLICT_DISCARDED_MATERIAL = "blocked_discarded_material"
CONFLICT_RETRY_OR_CORRECTION = "blocked_retry_or_correction"
CONFLICT_MEANING_CRITICAL = "blocked_meaning_critical"
CONFLICT_WORD_COVERAGE_UNKNOWN = "word_coverage_unknown"
CONFLICT_NO_EDGE_GEOMETRY = "no_word_geometry_available_for_edge_window"
CONFLICT_AUDIO_EXTRACTION_UNAVAILABLE = "audio_extraction_unavailable"
CONFLICT_WINDOW_TOO_SHORT = "window_too_short_for_analysis"
CONFLICT_SIGNATURE_DIMENSION_MISMATCH = "signature_dimension_mismatch"
CONFLICT_SPEECH_PRESENT_NOT_COMPARABLE = "speech_present_not_comparable_as_ambience"

# Only these three represent an actual FIREWALL block (discarded/retry/
# correction/meaning-critical material this module must never treat as
# reusable, per the "Discarded/retry firewall" section of the module
# docstring). `CONFLICT_AUDIO_EXTRACTION_UNAVAILABLE`/`CONFLICT_WINDOW_
# TOO_SHORT`/`CONFLICT_NO_EDGE_GEOMETRY` are evidence-QUALITY gaps, not
# firewall blocks -- they must not override a speech_status already
# known from word timing (independent of acoustic evidence), and they
# fall through to `INSUFFICIENT_EVIDENCE`/`UNCONFIRMED` on their own via
# the ordinary energy/word-coverage checks below, never `CONFLICTED`.
_FIREWALL_CONFLICTS = frozenset({
    CONFLICT_DISCARDED_MATERIAL, CONFLICT_RETRY_OR_CORRECTION, CONFLICT_MEANING_CRITICAL,
})


def _has_firewall_conflict(conflicts: Sequence[str]) -> bool:
    return any(c in _FIREWALL_CONFLICTS for c in conflicts)


PROVENANCE_SOURCE_AUDIO_HANDLE = "source_audio_handle_reused"
PROVENANCE_RETAINED_EDGE_WORD_GEOMETRY = "retained_edge_word_geometry"
PROVENANCE_AUDIO_SILENCE_INTERVALS = "audio_silence_detect_audio_silence_intervals"
PROVENANCE_PROSODIC_AUDIO_V2_RMS = "prosodic_audio_v2_rms_frames"
PROVENANCE_FFT_SIGNATURE = "numpy_fft_band_energy_signature"


@dataclass(frozen=True)
class AcousticWindowSignature:
    """A small, deterministic, gain-ROBUST acoustic descriptor for one
    bounded window -- never a learned embedding, never a model. `rms` is
    the ONE deliberately un-normalized (absolute-level) field; `spectral_
    centroid_hz`/`band_energy_ratios` are both gain-invariant by
    construction (see module docstring)."""

    schema_version: str
    rms: float
    spectral_centroid_hz: Optional[float]
    band_energy_ratios: Tuple[float, ...]


@dataclass(frozen=True)
class AcousticEdgeEvidence:
    """One bounded acoustic window near one clip edge or `SourceAudioHandle`
    (D-223). Evidence only -- see module docstring for what this never
    decides."""

    schema_version: str
    evidence_id: str
    source_asset_id: str
    owner_clip_id: str
    edge: str
    window_start: float
    window_end: float
    word_coverage_status: str
    words_present_count: int
    silence_status: str
    silence_fraction: Optional[float]
    rms_level: Optional[float]
    energy_status: str
    speech_status: str
    non_speech_status: str
    background_signature_status: str
    signature: Optional[AcousticWindowSignature]
    evidence_status: str
    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]


@dataclass(frozen=True)
class AcousticContinuityComparison:
    """Pairwise, deterministic comparison of two `AcousticEdgeEvidence`
    windows. Never compares raw timestamps (same-source or cross-source
    alike) -- only measured descriptors. Never a treatment decision."""

    schema_version: str
    left_evidence_id: str
    right_evidence_id: str
    continuity_status: str
    signature_comparison_status: str
    level_delta_db: Optional[float]
    reason: str
    conflict_flags: Tuple[str, ...]
    provenance: Tuple[str, ...]


def _evidence_id(source_asset_id: str, owner_clip_id: str, edge: str, start: float, end: float) -> str:
    """Stable, deterministic -- derived purely from identity/geometry
    inputs, never a random UUID (same convention as `pacing_v2_source_
    audio_handle._handle_id`)."""
    return f"acoustic:{source_asset_id}:{owner_clip_id}:{edge}:{round(float(start), 3)}:{round(float(end), 3)}"


def derive_retained_edge_window(
    words: Sequence[Tuple[float, float, str]], edge: str,
) -> Optional[Tuple[float, float]]:
    """Derives a retained-clip-edge analysis window from ALREADY-EXISTING
    per-word geometry (the clip's own last/first word) -- never a fixed,
    invented duration (this task's own explicit instruction). Returns
    `None` when the clip carries no word timing at all (a non-verbal
    edge), which the caller must then honestly report as `INSUFFICIENT_
    EVIDENCE` rather than fabricate a window."""
    if not words:
        return None
    ordered = sorted(words, key=lambda w: (float(w[0]), float(w[1])))
    if edge == EDGE_LEFT_END:
        start, end, _ = ordered[-1]
    elif edge == EDGE_RIGHT_START:
        start, end, _ = ordered[0]
    else:
        raise ValueError(f"derive_retained_edge_window only supports {EDGE_LEFT_END}/{EDGE_RIGHT_START}, got {edge!r}")
    return float(start), float(end)


def _silence_fraction(window_start: float, window_end: float, silence_intervals: Sequence[Tuple[float, float]]) -> float:
    duration = max(0.0, window_end - window_start)
    if duration <= 0.0:
        return 0.0
    covered = 0.0
    for s, e in silence_intervals or ():
        lo = max(window_start, float(s))
        hi = min(window_end, float(e))
        if hi > lo:
            covered += hi - lo
    return min(1.0, covered / duration)


def _silence_status_from_fraction(fraction: Optional[float]) -> str:
    if fraction is None:
        return SILENCE_STATUS_UNKNOWN
    if fraction >= _SILENT_FRACTION_FLOOR:
        return SILENCE_STATUS_SILENT
    if fraction >= _MOSTLY_SILENT_FRACTION_FLOOR:
        return SILENCE_STATUS_MOSTLY_SILENT
    return SILENCE_STATUS_NON_SILENT


def _energy_status_from_rms(rms: Optional[float]) -> str:
    if rms is None:
        return ENERGY_STATUS_UNKNOWN
    if rms < _NEAR_SILENT_RMS:
        return ENERGY_STATUS_NO_SIGNAL
    if rms < ACOUSTIC_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_LOW_ENERGY_CEILING:
        return ENERGY_STATUS_LOW_ENERGY
    return ENERGY_STATUS_ACTIVE_ENERGY


def compute_acoustic_window_signature(samples: "np.ndarray", sample_rate: int) -> Optional[AcousticWindowSignature]:
    """Deterministic, numpy-only (no scipy/librosa) descriptor: raw RMS
    (absolute level, NOT gain-robust, kept separately for future loudness
    work) plus a magnitude-weighted spectral centroid and a small
    normalized band-energy-ratio vector (BOTH gain-invariant by
    construction -- see module docstring's own proof). Returns `None`
    when the window is too short for a meaningful FFT (a structural
    floor, `_SIGNATURE_MIN_SAMPLES`, never a tuned threshold)."""
    if samples is None or samples.size < _SIGNATURE_MIN_SAMPLES or sample_rate <= 0:
        return None
    windowed = samples.astype(np.float64)
    rms = float(np.sqrt(np.mean(windowed ** 2)))
    spectrum = np.fft.rfft(windowed)
    magnitude = np.abs(spectrum)
    freqs = np.fft.rfftfreq(windowed.size, d=1.0 / sample_rate)
    total_energy = float(np.sum(magnitude))
    if total_energy <= 1e-12:
        return AcousticWindowSignature(
            schema_version=SCHEMA_VERSION, rms=rms, spectral_centroid_hz=None,
            band_energy_ratios=tuple(0.0 for _ in range(_SIGNATURE_BAND_COUNT)),
        )
    centroid = float(np.sum(freqs * magnitude) / total_energy)
    nyquist = sample_rate / 2.0
    edges = [nyquist * i / _SIGNATURE_BAND_COUNT for i in range(_SIGNATURE_BAND_COUNT + 1)]
    ratios = []
    for i in range(_SIGNATURE_BAND_COUNT):
        lo, hi = edges[i], edges[i + 1]
        if i < _SIGNATURE_BAND_COUNT - 1:
            mask = (freqs >= lo) & (freqs < hi)
        else:
            mask = (freqs >= lo) & (freqs <= hi)
        ratios.append(float(np.sum(magnitude[mask])) / total_energy)
    return AcousticWindowSignature(
        schema_version=SCHEMA_VERSION, rms=rms, spectral_centroid_hz=centroid, band_energy_ratios=tuple(ratios),
    )


def _decide_non_speech_status(
    *, speech_status: str, blocking_conflicts: Sequence[str], energy_status: str,
) -> str:
    """The one, exhaustive, priority-ordered decision table (mirrors
    `pacing_v2_source_audio_handle._decide_handle_status`'s own shape).
    Word-absence alone NEVER reaches `SAFE_NON_SPEECH_CANDIDATE` without
    KNOWN word coverage (`speech_status` already encodes that: it is
    only `NO_LEXICAL_SPEECH_OBSERVED`, never merely absent, when coverage
    is actually known) and without measurable acoustic energy above the
    near-silence floor. A firewall block (discarded/retry/meaning-
    critical) always forces UNKNOWN regardless of speech_status; a mere
    evidence-quality gap (no audio, window too short) does NOT -- known
    lexical speech from word timing remains authoritative either way."""
    if _has_firewall_conflict(blocking_conflicts):
        return NON_SPEECH_UNKNOWN
    if speech_status == SPEECH_STATUS_LEXICAL_PRESENT:
        return NON_SPEECH_SPEECH_PRESENT
    if speech_status == SPEECH_STATUS_UNKNOWN:
        return NON_SPEECH_UNKNOWN
    # speech_status == SPEECH_STATUS_NO_LEXICAL_OBSERVED from here on.
    if energy_status == ENERGY_STATUS_UNKNOWN:
        return NON_SPEECH_UNCONFIRMED
    if energy_status == ENERGY_STATUS_NO_SIGNAL:
        return NON_SPEECH_SILENCE
    return NON_SPEECH_SAFE_CANDIDATE


def _decide_evidence_status(
    *, word_coverage_status: str, blocking_conflicts: Sequence[str],
    silence_status: str, energy_status: str, window_valid: bool,
) -> str:
    if not window_valid:
        return EVIDENCE_STATUS_UNKNOWN
    if _has_firewall_conflict(blocking_conflicts):
        return EVIDENCE_STATUS_CONFLICTED
    if word_coverage_status == WORD_COVERAGE_UNKNOWN:
        return EVIDENCE_STATUS_INSUFFICIENT
    if energy_status == ENERGY_STATUS_UNKNOWN:
        return EVIDENCE_STATUS_INSUFFICIENT
    if silence_status == SILENCE_STATUS_UNKNOWN:
        # Word coverage and energy are both known; only the independent
        # silence-interval layer is missing -- usable, degraded evidence.
        return EVIDENCE_STATUS_SAFE_FALLBACK
    return EVIDENCE_STATUS_SUPPORTED


def build_acoustic_edge_evidence(
    *,
    source_asset_id: str,
    owner_clip_id: str,
    edge: str,
    window_start: float,
    window_end: float,
    audio: Optional[AudioSamples] = None,
    silence_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    word_intervals: Sequence[Tuple[float, float, str]] = (),
    word_coverage_known: bool = False,
    source_handle: Optional[object] = None,
) -> AcousticEdgeEvidence:
    """Builds one `AcousticEdgeEvidence` for a bounded window.

    Two calling shapes, matching this task's own "D-223 handle objects
    should be consumable directly" instruction:

    1. `edge in (EDGE_PRE_HANDLE, EDGE_POST_HANDLE)` with `source_handle`
       set to a real `pacing_v2_source_audio_handle.SourceAudioHandle` --
       word coverage / discarded / retry / meaning-safety are read
       DIRECTLY from that handle's own already-computed fields, never
       re-derived (word_intervals/word_coverage_known are ignored in
       this shape).
    2. `edge in (EDGE_LEFT_END, EDGE_RIGHT_START)` with `word_intervals`/
       `word_coverage_known` supplied by the caller from the clip's own
       already-known word list (see `derive_retained_edge_window`) --
       no discarded/retry/meaning conflict applies structurally (a
       retained edge is inside the already-selected, already-approved
       span).

    `silence_intervals=None` means "never checked" (-> `SILENCE_STATUS_
    UNKNOWN`); pass `()` explicitly to mean "checked, found none" (->
    `SILENCE_STATUS_NON_SILENT`, fraction 0.0). Same convention for
    `audio=None` vs. an `AudioSamples` whose slice happens to be empty.
    """
    if edge not in _KNOWN_EDGES:
        raise ValueError(f"unknown edge kind: {edge!r}")

    window_valid = window_end > window_start
    conflict_flags: list[str] = []
    provenance: list[str] = [SCHEMA_VERSION]

    if source_handle is not None:
        provenance.append(PROVENANCE_SOURCE_AUDIO_HANDLE)
        handle_status = getattr(source_handle, "handle_status", None)
        speech_presence = getattr(source_handle, "speech_presence_status", None)
        word_intervals = tuple(getattr(source_handle, "word_intervals_present", ()) or ())
        if speech_presence == "WORDS_PRESENT":
            word_coverage_status = WORD_COVERAGE_KNOWN_PRESENT
            speech_status = SPEECH_STATUS_LEXICAL_PRESENT
        elif speech_presence == "NO_WORDS_PRESENT":
            word_coverage_status = WORD_COVERAGE_KNOWN_EMPTY
            speech_status = SPEECH_STATUS_NO_LEXICAL_OBSERVED
        else:
            word_coverage_status = WORD_COVERAGE_UNKNOWN
            speech_status = SPEECH_STATUS_UNKNOWN
        if handle_status == "BLOCKED_DISCARDED_MATERIAL" or handle_status == "BLOCKED_NEIGHBOR_SELECTED_CLIP":
            conflict_flags.append(CONFLICT_DISCARDED_MATERIAL)
        elif handle_status == "BLOCKED_RETRY_OR_CORRECTION":
            conflict_flags.append(CONFLICT_RETRY_OR_CORRECTION)
        elif handle_status == "BLOCKED_MEANING_CRITICAL":
            conflict_flags.append(CONFLICT_MEANING_CRITICAL)
    else:
        provenance.append(PROVENANCE_RETAINED_EDGE_WORD_GEOMETRY)
        if word_intervals:
            word_coverage_status = WORD_COVERAGE_KNOWN_PRESENT
            speech_status = SPEECH_STATUS_LEXICAL_PRESENT
        elif word_coverage_known:
            word_coverage_status = WORD_COVERAGE_KNOWN_EMPTY
            speech_status = SPEECH_STATUS_NO_LEXICAL_OBSERVED
        else:
            word_coverage_status = WORD_COVERAGE_UNKNOWN
            speech_status = SPEECH_STATUS_UNKNOWN
            conflict_flags.append(CONFLICT_WORD_COVERAGE_UNKNOWN)
        if not window_valid:
            conflict_flags.append(CONFLICT_NO_EDGE_GEOMETRY)

    words_present_count = len(word_intervals)

    if silence_intervals is None:
        silence_status = SILENCE_STATUS_UNKNOWN
        silence_fraction: Optional[float] = None
    else:
        provenance.append(PROVENANCE_AUDIO_SILENCE_INTERVALS)
        silence_fraction = _silence_fraction(window_start, window_end, silence_intervals) if window_valid else None
        silence_status = _silence_status_from_fraction(silence_fraction)

    rms_level: Optional[float] = None
    signature: Optional[AcousticWindowSignature] = None
    if audio is not None and window_valid:
        provenance.append(PROVENANCE_PROSODIC_AUDIO_V2_RMS)
        span = _slice_samples(audio, window_start, window_end)
        if span.size >= 1:
            rms_level = float(np.sqrt(np.mean(span.astype(np.float64) ** 2)))
        else:
            conflict_flags.append(CONFLICT_WINDOW_TOO_SHORT)
    elif audio is None:
        conflict_flags.append(CONFLICT_AUDIO_EXTRACTION_UNAVAILABLE)

    energy_status = _energy_status_from_rms(rms_level)

    non_speech_status = _decide_non_speech_status(
        speech_status=speech_status, blocking_conflicts=conflict_flags, energy_status=energy_status,
    )

    background_signature_status = SIGNATURE_UNAVAILABLE
    if (
        non_speech_status in (NON_SPEECH_SAFE_CANDIDATE, NON_SPEECH_UNCONFIRMED)
        and audio is not None and window_valid
    ):
        span = _slice_samples(audio, window_start, window_end)
        signature = compute_acoustic_window_signature(span, audio.sample_rate)
        if signature is not None:
            background_signature_status = SIGNATURE_AVAILABLE
            provenance.append(PROVENANCE_FFT_SIGNATURE)

    evidence_status = _decide_evidence_status(
        word_coverage_status=word_coverage_status, blocking_conflicts=conflict_flags,
        silence_status=silence_status, energy_status=energy_status, window_valid=window_valid,
    )

    return AcousticEdgeEvidence(
        schema_version=SCHEMA_VERSION,
        evidence_id=_evidence_id(source_asset_id, owner_clip_id, edge, window_start, window_end),
        source_asset_id=str(source_asset_id),
        owner_clip_id=str(owner_clip_id),
        edge=edge,
        window_start=float(window_start),
        window_end=float(window_end),
        word_coverage_status=word_coverage_status,
        words_present_count=words_present_count,
        silence_status=silence_status,
        silence_fraction=silence_fraction,
        rms_level=rms_level,
        energy_status=energy_status,
        speech_status=speech_status,
        non_speech_status=non_speech_status,
        background_signature_status=background_signature_status,
        signature=signature,
        evidence_status=evidence_status,
        conflict_flags=tuple(conflict_flags),
        provenance=tuple(provenance),
    )


def _signature_distance(a: AcousticWindowSignature, b: AcousticWindowSignature) -> float:
    ratio_diff = math.sqrt(sum((x - y) ** 2 for x, y in zip(a.band_energy_ratios, b.band_energy_ratios)))
    if a.spectral_centroid_hz is not None and b.spectral_centroid_hz is not None:
        denom = max(a.spectral_centroid_hz, b.spectral_centroid_hz, 1.0)
        centroid_diff = abs(a.spectral_centroid_hz - b.spectral_centroid_hz) / denom
    else:
        centroid_diff = 0.0
    return ratio_diff + centroid_diff


def compare_acoustic_edges(left: AcousticEdgeEvidence, right: AcousticEdgeEvidence) -> AcousticContinuityComparison:
    """Deterministic pairwise comparison. Never compares `window_start`/
    `window_end` (same-source or cross-source alike -- this task's own
    "never raw timestamps" instruction); only measured descriptors.
    Never selects a treatment."""
    conflict_flags: list[str] = []
    provenance = (SCHEMA_VERSION, "compare_acoustic_edges")

    level_delta_db: Optional[float] = None
    if left.rms_level and right.rms_level and left.rms_level > 0 and right.rms_level > 0:
        level_delta_db = 20.0 * math.log10(right.rms_level / left.rms_level)

    if left.evidence_status == EVIDENCE_STATUS_CONFLICTED or right.evidence_status == EVIDENCE_STATUS_CONFLICTED:
        return AcousticContinuityComparison(
            schema_version=SCHEMA_VERSION, left_evidence_id=left.evidence_id, right_evidence_id=right.evidence_id,
            continuity_status=CONTINUITY_CONFLICTED, signature_comparison_status=CONTINUITY_CONFLICTED,
            level_delta_db=level_delta_db, reason="one_or_both_edges_evidence_conflicted",
            conflict_flags=tuple(conflict_flags), provenance=provenance,
        )
    if left.non_speech_status == NON_SPEECH_SPEECH_PRESENT or right.non_speech_status == NON_SPEECH_SPEECH_PRESENT:
        conflict_flags.append(CONFLICT_SPEECH_PRESENT_NOT_COMPARABLE)
        return AcousticContinuityComparison(
            schema_version=SCHEMA_VERSION, left_evidence_id=left.evidence_id, right_evidence_id=right.evidence_id,
            continuity_status=CONTINUITY_CONFLICTED, signature_comparison_status=CONTINUITY_CONFLICTED,
            level_delta_db=level_delta_db, reason="lexical_speech_present_not_comparable_as_ambience",
            conflict_flags=tuple(conflict_flags), provenance=provenance,
        )
    if left.signature is None or right.signature is None:
        return AcousticContinuityComparison(
            schema_version=SCHEMA_VERSION, left_evidence_id=left.evidence_id, right_evidence_id=right.evidence_id,
            continuity_status=CONTINUITY_INSUFFICIENT, signature_comparison_status=CONTINUITY_INSUFFICIENT,
            level_delta_db=level_delta_db, reason="one_or_both_signatures_unavailable",
            conflict_flags=tuple(conflict_flags), provenance=provenance,
        )
    if len(left.signature.band_energy_ratios) != len(right.signature.band_energy_ratios):
        conflict_flags.append(CONFLICT_SIGNATURE_DIMENSION_MISMATCH)
        return AcousticContinuityComparison(
            schema_version=SCHEMA_VERSION, left_evidence_id=left.evidence_id, right_evidence_id=right.evidence_id,
            continuity_status=CONTINUITY_INSUFFICIENT, signature_comparison_status=CONTINUITY_INSUFFICIENT,
            level_delta_db=level_delta_db, reason="signature_band_count_mismatch",
            conflict_flags=tuple(conflict_flags), provenance=provenance,
        )
    distance = _signature_distance(left.signature, right.signature)
    similar = distance <= ACOUSTIC_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_SIMILARITY_THRESHOLD
    status = CONTINUITY_SIMILAR if similar else CONTINUITY_DIFFERENT
    reason = (
        f"signature_distance={distance:.4f} vs threshold="
        f"{ACOUSTIC_HEURISTIC_OFFLINE_NOT_REAL_MEDIA_TUNED_SIMILARITY_THRESHOLD:.4f}"
    )
    return AcousticContinuityComparison(
        schema_version=SCHEMA_VERSION, left_evidence_id=left.evidence_id, right_evidence_id=right.evidence_id,
        continuity_status=status, signature_comparison_status=status,
        level_delta_db=level_delta_db, reason=reason,
        conflict_flags=tuple(conflict_flags), provenance=provenance,
    )


def acoustic_edge_evidence_diagnostics(evidence: AcousticEdgeEvidence) -> dict:
    """Compact, JSON-safe diagnostics row -- no transcript dump (this
    task's own explicit instruction)."""
    return {
        "schema_version": evidence.schema_version,
        "evidence_id": evidence.evidence_id,
        "source_asset_id": evidence.source_asset_id,
        "owner_clip_id": evidence.owner_clip_id,
        "edge": evidence.edge,
        "window_start": evidence.window_start,
        "window_end": evidence.window_end,
        "word_coverage_status": evidence.word_coverage_status,
        "words_present_count": evidence.words_present_count,
        "silence_status": evidence.silence_status,
        "silence_fraction": evidence.silence_fraction,
        "rms_level": evidence.rms_level,
        "energy_status": evidence.energy_status,
        "speech_status": evidence.speech_status,
        "non_speech_status": evidence.non_speech_status,
        "room_tone_classification_status": ROOM_TONE_CLASSIFICATION_STATUS,
        "background_signature_status": evidence.background_signature_status,
        "signature_available": evidence.signature is not None,
        "evidence_status": evidence.evidence_status,
        "conflict_flags": list(evidence.conflict_flags),
        "provenance": list(evidence.provenance),
    }


def acoustic_continuity_diagnostics(comparison: AcousticContinuityComparison) -> dict:
    return {
        "schema_version": comparison.schema_version,
        "left_evidence_id": comparison.left_evidence_id,
        "right_evidence_id": comparison.right_evidence_id,
        "continuity_status": comparison.continuity_status,
        "signature_comparison_status": comparison.signature_comparison_status,
        "level_delta_db": comparison.level_delta_db,
        "reason": comparison.reason,
        "conflict_flags": list(comparison.conflict_flags),
        "provenance": list(comparison.provenance),
    }


def acoustic_edge_evidence_run_summary(
    evidences: Sequence[AcousticEdgeEvidence], comparisons: Sequence[AcousticContinuityComparison] = (),
) -> dict:
    """No master score (this task's own explicit instruction) -- plain
    counts only."""
    silence_count = sum(1 for e in evidences if e.non_speech_status == NON_SPEECH_SILENCE)
    non_silent_count = sum(1 for e in evidences if e.silence_status == SILENCE_STATUS_NON_SILENT)
    lexical_speech_count = sum(1 for e in evidences if e.non_speech_status == NON_SPEECH_SPEECH_PRESENT)
    safe_non_speech_candidate_count = sum(1 for e in evidences if e.non_speech_status == NON_SPEECH_SAFE_CANDIDATE)
    unknown_speech_count = sum(1 for e in evidences if e.non_speech_status == NON_SPEECH_UNKNOWN)
    signature_available_count = sum(1 for e in evidences if e.background_signature_status == SIGNATURE_AVAILABLE)
    signature_missing_count = sum(1 for e in evidences if e.background_signature_status == SIGNATURE_UNAVAILABLE)
    discarded_block_count = sum(1 for e in evidences if CONFLICT_DISCARDED_MATERIAL in e.conflict_flags)
    unknown_word_block_count = sum(1 for e in evidences if CONFLICT_WORD_COVERAGE_UNKNOWN in e.conflict_flags)
    similar_pair_count = sum(1 for c in comparisons if c.continuity_status == CONTINUITY_SIMILAR)
    different_pair_count = sum(1 for c in comparisons if c.continuity_status == CONTINUITY_DIFFERENT)
    insufficient_pair_count = sum(
        1 for c in comparisons if c.continuity_status in (CONTINUITY_INSUFFICIENT, CONTINUITY_CONFLICTED)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "window_count": len(evidences),
        "silence_count": silence_count,
        "non_silent_count": non_silent_count,
        "lexical_speech_count": lexical_speech_count,
        "safe_non_speech_candidate_count": safe_non_speech_candidate_count,
        "unknown_speech_count": unknown_speech_count,
        "signature_available_count": signature_available_count,
        "signature_missing_count": signature_missing_count,
        "similar_pair_count": similar_pair_count,
        "different_pair_count": different_pair_count,
        "insufficient_pair_count": insufficient_pair_count,
        "discarded_block_count": discarded_block_count,
        "unknown_word_block_count": unknown_word_block_count,
        "room_tone_classification_status": ROOM_TONE_CLASSIFICATION_STATUS,
    }
