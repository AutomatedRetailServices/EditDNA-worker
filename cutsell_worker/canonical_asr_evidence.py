"""D-052 Part A: provider-neutral canonical ASR evidence + deterministic
segment normalization.

See ``docs/CUTSELL_DECISIONS.md`` D-051 (the audit that found ASR
``segment_count`` itself already diverges run-to-run on an *identical*
source video, before any candidate/attempt/clean_cut logic runs at all)
and D-052 (this fix).

PROBLEM
=======
``take_segmentation.segment_takes`` consumes ``TranscriptSegment`` objects
exactly as Whisper happened to group them for this run. Whisper's own
segment boundaries are not a semantic decision -- they are an artifact of
beam-search/VAD internals -- yet every downstream stage (AttemptReconstructor
onward) has historically inherited whatever shape Whisper produced. D-051
proved this shape is not stable: the identical source video produced
segment_count 48 / 54 / ~5x across three otherwise-identical Modal canaries.

FIX SHAPE (this module)
========================
1. ``CanonicalASREvidence`` -- a provider-neutral contract representing
   WHAT WAS SAID (the flat, source-ordered word stream + language + model
   fingerprint), deliberately never anchored to Whisper's segment
   boundaries. ``evidence_hash`` is a pure function of the normalized word
   TEXT sequence only -- never of timestamps -- so it answers "did the
   transcript content change" independently of "did Whisper happen to
   group it differently."
2. ``normalize_transcript_segments`` -- flattens every ``TranscriptSegment``
   belonging to one source into a single word-level timeline (ignoring
   Whisper's original per-segment grouping entirely) and re-derives segment
   boundaries using ONLY deterministic, content-based rules: a word-gap
   threshold (the same ``0.75s`` constant ``take_segmentation`` already
   uses for its own internal ``_speech_units`` gap-splitting -- reused, not
   widened) plus sentence-ending punctuation. Whisper is free to have drawn
   its own segment boundaries anywhere; the same underlying WORDS always
   re-derive the same normalized segmentation, by construction.

This module is additive-only and provider-neutral: it does not require
faster-whisper to be installed (there is no heavy import here), does not
change ``TranscriptSegment``/``Word``'s shape, and is not wired into the
live pipeline by default -- see ``take_segmentation.py``'s
``CUTSELL_ASR_CANONICAL_NORMALIZATION`` flag (default OFF, preserving
today's exact behavior) for the opt-in integration point.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, Mapping, Tuple

from .contracts import TranscriptSegment, Word

_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]*\s*$")

# Same constant take_segmentation.py's own _speech_units already uses for its
# internal per-segment gap-splitting. Reused verbatim -- this module does not
# introduce a new threshold, it only applies the existing one across the
# FULL flattened word timeline instead of within Whisper's own per-segment
# grouping. Per the D-052 directive: "Do NOT globally widen thresholds."
DEFAULT_SPLIT_GAP_SEC = 0.75


def _normalize_word_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip()


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(str(text or "").strip()))


@dataclass(frozen=True)
class ASRConfigFingerprint:
    """Every behaviorally-relevant ASR setting, made explicit (D-052 Part A
    Section 1/5; extended by D-053 Section 2/10 with the full effective
    ``model.transcribe()`` decode config, ground-truthed against the exact
    pinned ``faster-whisper==1.0.0`` wheel's own signature -- see
    ``docs/CUTSELL_DECISIONS.md`` D-053 for how each default below was
    verified, not assumed). Two runs with an identical fingerprint used an
    identical effective decode configuration -- a difference here is a
    legitimate, named suspect for run-to-run variance; a match rules
    configuration out and points back at the audio/hardware/library-internal
    path instead.

    ``sampling_fallback_enabled`` (D-053 Section 3) is a derived, explicit
    answer to "can this configuration ever leave temperature=0.0 deterministic
    beam search for random sampling" -- true whenever ``temperature_ladder``
    has more than one entry (the library's own fallback-on-quality-gate
    behavior; see ``asr.py``'s module docstring for the exact trigger
    conditions).

    Deliberately a plain, hashable, provider-neutral value object -- this is
    metadata attached to evidence, never a knob any editorial stage reads.
    """
    model_name: str
    device: str
    compute_type: str
    task: str
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    temperature_ladder: Tuple[float, ...]
    compression_ratio_threshold: float | None
    log_prob_threshold: float | None
    no_speech_threshold: float | None
    condition_on_previous_text: bool
    prompt_reset_on_temperature: float
    word_timestamps: bool
    vad_filter: bool
    vad_parameters: Tuple[float, ...]
    language_hint: str | None
    initial_prompt: str | None

    @property
    def sampling_fallback_enabled(self) -> bool:
        return len(self.temperature_ladder) > 1

    def fingerprint(self) -> str:
        raw = "|".join([
            self.model_name,
            self.device,
            self.compute_type,
            self.task,
            str(self.beam_size),
            str(self.best_of),
            f"{self.patience:.2f}",
            f"{self.length_penalty:.2f}",
            f"{self.repetition_penalty:.2f}",
            str(self.no_repeat_ngram_size),
            ",".join(f"{value:.2f}" for value in self.temperature_ladder),
            f"{self.compression_ratio_threshold:.2f}" if self.compression_ratio_threshold is not None else "",
            f"{self.log_prob_threshold:.2f}" if self.log_prob_threshold is not None else "",
            f"{self.no_speech_threshold:.2f}" if self.no_speech_threshold is not None else "",
            str(self.condition_on_previous_text),
            f"{self.prompt_reset_on_temperature:.2f}",
            str(self.word_timestamps),
            str(self.vad_filter),
            ",".join(f"{value:.4f}" for value in self.vad_parameters),
            self.language_hint or "",
            self.initial_prompt or "",
        ]).encode("utf-8")
        return "asrcfg_" + hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class CanonicalASREvidence:
    """Provider-neutral canonical evidence for one source's ASR output.

    ``normalized_words`` is the single source of truth for "what was said";
    segment boundaries are deliberately NOT part of this contract's identity
    (see ``evidence_hash``). Any downstream re-segmentation
    (``normalize_transcript_segments`` in this module, or a future
    replacement) is expected to derive its own segment shape purely from
    this evidence, never from Whisper's original grouping.
    """
    source_asset_id: str
    normalized_words: Tuple[Word, ...]
    language: str | None
    asr_model: str
    asr_config_fingerprint: str
    evidence_hash: str


def compute_evidence_hash(source_asset_id: str, words: Iterable[Word], language: str | None) -> str:
    """Content-only hash: normalized word TEXT sequence + language +
    source lineage. Deliberately excludes every timestamp -- per the D-052
    directive, "Segment boundaries must NOT define semantic identity," and
    the same principle extends to word-level timing jitter that does not
    change what was actually said. Two ASR runs that transcribed the exact
    same words (even if Whisper grouped/timed them differently) always
    produce the same ``evidence_hash``.
    """
    word_text = "|".join(_normalize_word_text(word.text) for word in words if _normalize_word_text(word.text))
    raw = f"{source_asset_id}|{language or ''}|{word_text}".encode("utf-8")
    return "asrev_" + hashlib.sha256(raw).hexdigest()[:24]


def build_canonical_asr_evidence(
    segments: Iterable[TranscriptSegment],
    *,
    source_asset_id: str,
    language: str | None,
    asr_model: str,
    asr_config_fingerprint: str,
) -> CanonicalASREvidence:
    """Flatten every segment belonging to ``source_asset_id`` into one
    source-ordered word timeline. Segments are read only for their
    ``words`` -- their own boundary shape (start/end/text as Whisper drew
    it) is discarded here on purpose; ``normalize_transcript_segments``
    below is the one place a new segmentation is re-derived, from the
    words alone.
    """
    words: list[Word] = []
    for segment in segments:
        if segment.source_asset_id != source_asset_id:
            continue
        words.extend(segment.words)
    ordered_words = tuple(sorted(words, key=lambda word: (word.start, word.end)))
    return CanonicalASREvidence(
        source_asset_id=source_asset_id,
        normalized_words=ordered_words,
        language=language,
        asr_model=asr_model,
        asr_config_fingerprint=asr_config_fingerprint,
        evidence_hash=compute_evidence_hash(source_asset_id, ordered_words, language),
    )


def normalize_transcript_segments(
    segments: Iterable[TranscriptSegment],
    *,
    split_gap_sec: float = DEFAULT_SPLIT_GAP_SEC,
) -> Tuple[TranscriptSegment, ...]:
    """Deterministically re-segment ASR output from word-level evidence
    alone, independent of however Whisper originally grouped it (D-052
    Part A Section 4).

    Groups the input by ``source_asset_id`` (multiple sources are handled
    independently and the output preserves source order via input order of
    first appearance), flattens each source's words into one timeline
    ignoring the input's own segment boundaries, then re-splits purely on:

    - a word-timestamp gap >= ``split_gap_sec`` (speech gap), or
    - the previous word ending a sentence (period/question/exclamation,
      optionally followed by a closing quote/paren).

    Both rules are content/timing-only and were already present in
    ``take_segmentation``'s own boundary-fragment repair logic in spirit;
    this function is the segment-boundary-INDEPENDENT version of the same
    idea, applied before AttemptReconstructor ever sees a segment shape.

    Two inputs whose words carry the same text and (approximately) the same
    timestamps produce the same output regardless of how many original
    ``TranscriptSegment`` objects those words arrived in -- see
    ``tests/test_cutsell_d052_canonical_asr_evidence.py`` for the exact
    equivalence classes this is required to satisfy.
    """
    by_source: dict[str, list[TranscriptSegment]] = {}
    order: list[str] = []
    for segment in segments:
        if segment.source_asset_id not in by_source:
            by_source[segment.source_asset_id] = []
            order.append(segment.source_asset_id)
        by_source[segment.source_asset_id].append(segment)

    output: list[TranscriptSegment] = []
    for source_asset_id in order:
        source_segments = by_source[source_asset_id]
        words = tuple(sorted(
            (word for segment in source_segments for word in segment.words),
            key=lambda word: (word.start, word.end),
        ))
        if not words:
            continue

        chunks: list[list[Word]] = [[]]
        for index, word in enumerate(words):
            if index:
                previous = words[index - 1]
                gap = word.start - previous.end
                previous_text = _normalize_word_text(previous.text)
                if gap >= split_gap_sec or _ends_sentence(previous_text):
                    chunks.append([])
            chunks[-1].append(word)

        for chunk in chunks:
            if not chunk:
                continue
            text = " ".join(_normalize_word_text(word.text) for word in chunk if _normalize_word_text(word.text))
            if not text:
                continue
            output.append(TranscriptSegment(
                source_asset_id=source_asset_id,
                start=float(chunk[0].start),
                end=float(chunk[-1].end),
                text=text,
                words=tuple(chunk),
            ))
    return tuple(output)


def build_asr_config_fingerprint(
    *,
    model_name: str,
    device: str,
    compute_type: str,
    beam_size: int,
    best_of: int,
    temperature_ladder: Tuple[float, ...],
    condition_on_previous_text: bool,
    word_timestamps: bool,
    vad_filter: bool,
    language_hint: str | None,
    initial_prompt: str | None = None,
    task: str = "transcribe",
    patience: float = 1.0,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    compression_ratio_threshold: float | None = 2.4,
    log_prob_threshold: float | None = -1.0,
    no_speech_threshold: float | None = 0.6,
    prompt_reset_on_temperature: float = 0.5,
    vad_parameters: Tuple[float, ...] = (),
) -> ASRConfigFingerprint:
    return ASRConfigFingerprint(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        task=task,
        beam_size=beam_size,
        best_of=best_of,
        patience=patience,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        temperature_ladder=tuple(temperature_ladder),
        compression_ratio_threshold=compression_ratio_threshold,
        log_prob_threshold=log_prob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        prompt_reset_on_temperature=prompt_reset_on_temperature,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
        vad_parameters=tuple(vad_parameters),
        language_hint=language_hint,
        initial_prompt=initial_prompt,
    )
