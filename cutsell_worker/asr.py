"""ASR provider boundary for CutSell Flow B.

D-052 Part A Section 1/5: every behaviorally-relevant Faster-Whisper decode
setting is an explicit ``FasterWhisperASR`` field instead of an implicit
library default.

D-053 Section 1/2: every default value below is no longer merely "assumed to
match the library default" -- it is ground-truthed against the ACTUAL
installed wheel this repo pins (``faster-whisper==1.0.0``,
``requirements.cutsell.worker.txt``), by extracting and reading
``faster_whisper/transcribe.py``'s real ``WhisperModel.transcribe()``
signature and ``faster_whisper/vad.py``'s real ``VadOptions`` defaults
directly from that exact wheel. Nothing here is copied from documentation
for a different version. See ``docs/CUTSELL_DECISIONS.md`` D-053 for the
verification method and the live Modal audit that cross-checked it against
the actual GPU image.

D-053 Section 3: this ground-truthed signature confirms
``temperature: Union[float, List[float], Tuple[float, ...]] = [0.0, 0.2,
0.4, 0.6, 0.8, 1.0]`` -- i.e. LEGACY behavior is not a scalar deterministic
temperature, it is a fallback LADDER. The primary decode attempt is
temperature 0.0 (deterministic beam search); a segment only escalates to a
later, sampling (non-deterministic) temperature in the ladder when its
temperature-0.0 attempt fails the library's own quality gates
(``compression_ratio_threshold=2.4``, ``log_prob_threshold=-1.0``,
``no_speech_threshold=0.6`` -- exceeding any of these on one temperature
triggers a retry at the next). ``condition_on_previous_text=True`` (also a
library default) compounds any such escalation forward through the rest of
the transcript via its context window. This is the exact, confirmed
mechanism the D-051/D-052 audits named as the leading suspect for
word-sequence variance -- D-053 turns "leading suspect" into "confirmed
capable of causing it," and ``CUTSELL_ASR_DETERMINISTIC_CONFIG`` (below)
closes it by using a scalar ``temperature=(0.0,)`` -- no fallback rung to
ever reach.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Iterable, Mapping, Protocol, Tuple

from .canonical_asr_evidence import ASRConfigFingerprint, build_asr_config_fingerprint
from .contracts import TranscriptSegment, Word

# Ground-truthed from faster_whisper==1.0.0's own transcribe.py signature
# (see module docstring). This is the LEGACY (flag-off) ladder -- capable of
# reaching non-deterministic sampling temperatures; see
# DETERMINISTIC_TEMPERATURE below for the D-053 fix.
DEFAULT_TEMPERATURE_LADDER: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# D-053 Section 7: the one deliberate behavioral difference between LEGACY
# and the deterministic candidate -- a scalar temperature, so the
# decode-with-fallback loop has no second rung to ever escalate to,
# regardless of how a segment scores against the quality gates below.
DETERMINISTIC_TEMPERATURE: Tuple[float, ...] = (0.0,)

DEFAULT_BEAM_SIZE = 5
DEFAULT_BEST_OF = 5
DEFAULT_PATIENCE = 1.0
DEFAULT_LENGTH_PENALTY = 1.0
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_NO_REPEAT_NGRAM_SIZE = 0
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.4
DEFAULT_LOG_PROB_THRESHOLD = -1.0
DEFAULT_NO_SPEECH_THRESHOLD = 0.6
DEFAULT_CONDITION_ON_PREVIOUS_TEXT = True
DEFAULT_PROMPT_RESET_ON_TEMPERATURE = 0.5
DEFAULT_SUPPRESS_TOKENS: Tuple[int, ...] = (-1,)
DEFAULT_MAX_INITIAL_TIMESTAMP = 1.0
DEFAULT_TASK = "transcribe"
# Ground-truthed from faster_whisper==1.0.0's own vad.py VadOptions defaults
# (see module docstring) -- passing these explicitly changes nothing versus
# vad_parameters=None (which constructs the identical VadOptions() itself
# internally); this only makes them observable/fingerprinted.
DEFAULT_VAD_PARAMETERS: Tuple[float, ...] = (
    0.5,     # threshold
    250.0,   # min_speech_duration_ms
    float("inf"),  # max_speech_duration_s
    2000.0,  # min_silence_duration_ms
    1024.0,  # window_size_samples
    400.0,   # speech_pad_ms
)

_ASR_DETERMINISTIC_CONFIG_ENV = "CUTSELL_ASR_DETERMINISTIC_CONFIG"


class ASRProvider(Protocol):
    def transcribe(self, path: str, *, source_asset_id: str, language_hint: str | None = None) -> Tuple[TranscriptSegment, ...]: ...


def _env_str(values: Mapping[str, str], key: str, default: str) -> str:
    raw = values.get(key)
    return str(raw).strip() if raw else default


def _env_bool(values: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = values.get(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _vad_parameters_dict(values: Tuple[float, ...]) -> dict:
    threshold, min_speech_ms, max_speech_s, min_silence_ms, window_samples, speech_pad_ms = values
    return {
        "threshold": threshold,
        "min_speech_duration_ms": int(min_speech_ms),
        "max_speech_duration_s": max_speech_s,
        "min_silence_duration_ms": int(min_silence_ms),
        "window_size_samples": int(window_samples),
        "speech_pad_ms": int(speech_pad_ms),
    }


@dataclass
class FasterWhisperASR:
    model_name: str = "medium"
    device: str = "auto"
    # D-052/D-053: "auto" is faster-whisper's own default and is left
    # unchanged here -- pinning it to a fixed precision (e.g. float16/int8)
    # is deliberately kept a SEPARATE, separately-tested configuration
    # change (D-053 Section 4/10: "Do NOT change it yet"), not bundled into
    # this task's deterministic-temperature fix.
    compute_type: str = "auto"
    task: str = DEFAULT_TASK
    beam_size: int = DEFAULT_BEAM_SIZE
    best_of: int = DEFAULT_BEST_OF
    patience: float = DEFAULT_PATIENCE
    length_penalty: float = DEFAULT_LENGTH_PENALTY
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE
    temperature_ladder: Tuple[float, ...] = field(default_factory=lambda: DEFAULT_TEMPERATURE_LADDER)
    compression_ratio_threshold: float | None = DEFAULT_COMPRESSION_RATIO_THRESHOLD
    log_prob_threshold: float | None = DEFAULT_LOG_PROB_THRESHOLD
    no_speech_threshold: float | None = DEFAULT_NO_SPEECH_THRESHOLD
    condition_on_previous_text: bool = DEFAULT_CONDITION_ON_PREVIOUS_TEXT
    prompt_reset_on_temperature: float = DEFAULT_PROMPT_RESET_ON_TEMPERATURE
    initial_prompt: str | None = None
    prefix: str | None = None
    suppress_blank: bool = True
    suppress_tokens: Tuple[int, ...] = field(default_factory=lambda: DEFAULT_SUPPRESS_TOKENS)
    without_timestamps: bool = False
    max_initial_timestamp: float = DEFAULT_MAX_INITIAL_TIMESTAMP
    word_timestamps: bool = True
    vad_filter: bool = True
    # D-053 Section 2: explicit VAD parameters (ground-truthed library
    # defaults -- see module docstring). Threaded into the live call for
    # BOTH legacy and deterministic providers (identical values either way,
    # so this is observability-only, never a behavior change).
    vad_parameters: Tuple[float, ...] = field(default_factory=lambda: DEFAULT_VAD_PARAMETERS)
    max_new_tokens: int | None = None
    chunk_length: int | None = None
    clip_timestamps: str = "0"
    hallucination_silence_threshold: float | None = None

    @property
    def sampling_fallback_enabled(self) -> bool:
        """D-053 Section 3: true whenever the decode-with-fallback loop has
        more than one temperature rung to ever escalate to."""
        return len(self.temperature_ladder) > 1

    def config_fingerprint(self, *, language_hint: str | None = None) -> ASRConfigFingerprint:
        """D-052/D-053: the full effective decode config as one hashable,
        reportable value -- see canonical_asr_evidence.ASRConfigFingerprint.
        Two runs whose fingerprints match ruled out configuration as the
        source of any observed transcript/segmentation difference between
        them; a mismatch names configuration as a live suspect."""
        return build_asr_config_fingerprint(
            model_name=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            task=self.task,
            beam_size=self.beam_size,
            best_of=self.best_of,
            patience=self.patience,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            temperature_ladder=self.temperature_ladder,
            compression_ratio_threshold=self.compression_ratio_threshold,
            log_prob_threshold=self.log_prob_threshold,
            no_speech_threshold=self.no_speech_threshold,
            condition_on_previous_text=self.condition_on_previous_text,
            prompt_reset_on_temperature=self.prompt_reset_on_temperature,
            word_timestamps=self.word_timestamps,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
            language_hint=language_hint,
            initial_prompt=self.initial_prompt,
        )

    def transcribe(self, path: str, *, source_asset_id: str, language_hint: str | None = None) -> Tuple[TranscriptSegment, ...]:
        # Heavy dependency is loaded only when a real transcription runs.
        from faster_whisper import WhisperModel

        # D-053: every one of these is now ground-truthed against the exact
        # pinned faster_whisper==1.0.0 wheel's own transcribe() signature
        # (see module docstring) -- for the LEGACY (default-constructed)
        # provider, every value below is IDENTICAL to that signature's own
        # default, so this is a make-explicit-and-fingerprintable change
        # with zero behavior difference from before this call was threaded
        # through. The temperature ladder (scalar for the deterministic
        # provider, the multi-rung fallback ladder for legacy) is the one
        # value that is allowed to differ in effect.
        model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        temperature = self.temperature_ladder[0] if len(self.temperature_ladder) == 1 else list(self.temperature_ladder)
        segments, _info = model.transcribe(
            path,
            language=language_hint,
            task=self.task,
            beam_size=self.beam_size,
            best_of=self.best_of,
            patience=self.patience,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            temperature=temperature,
            compression_ratio_threshold=self.compression_ratio_threshold,
            log_prob_threshold=self.log_prob_threshold,
            no_speech_threshold=self.no_speech_threshold,
            condition_on_previous_text=self.condition_on_previous_text,
            prompt_reset_on_temperature=self.prompt_reset_on_temperature,
            initial_prompt=self.initial_prompt,
            prefix=self.prefix,
            suppress_blank=self.suppress_blank,
            suppress_tokens=list(self.suppress_tokens),
            without_timestamps=self.without_timestamps,
            max_initial_timestamp=self.max_initial_timestamp,
            word_timestamps=self.word_timestamps,
            vad_filter=self.vad_filter,
            vad_parameters=_vad_parameters_dict(self.vad_parameters),
            max_new_tokens=self.max_new_tokens,
            chunk_length=self.chunk_length,
            clip_timestamps=self.clip_timestamps,
            hallucination_silence_threshold=self.hallucination_silence_threshold,
        )
        output = []
        for segment in segments:
            words = tuple(
                Word(
                    text=str(getattr(word, "word", "")).strip(),
                    start=float(getattr(word, "start", 0.0) or 0.0),
                    end=float(getattr(word, "end", 0.0) or 0.0),
                    confidence=(float(getattr(word, "probability", 0.0)) if getattr(word, "probability", None) is not None else None),
                )
                for word in (getattr(segment, "words", None) or ())
                if str(getattr(word, "word", "")).strip()
            )
            text = str(getattr(segment, "text", "")).strip()
            if not text:
                continue
            output.append(TranscriptSegment(
                source_asset_id=source_asset_id,
                start=float(getattr(segment, "start", 0.0) or 0.0),
                end=float(getattr(segment, "end", 0.0) or 0.0),
                text=text,
                words=words,
            ))
        return tuple(output)


def build_deterministic_asr_provider(model_name: str, *, device: str = "auto", compute_type: str = "auto") -> FasterWhisperASR:
    """D-053 Section 7: the ONE conservative deterministic candidate
    configuration -- every decode parameter explicit at its ground-truthed
    library-default value EXCEPT ``temperature_ladder``, which is a scalar
    ``(0.0,)`` instead of the library's own multi-rung fallback ladder.
    ``compute_type`` is deliberately left at the caller's own value (still
    "auto" by default) -- Section 4/10 keeps that a separate, not-yet-made
    decision pending its own live tradeoff data.
    """
    return FasterWhisperASR(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        temperature_ladder=DETERMINISTIC_TEMPERATURE,
    )


def load_asr_provider_from_env(env: Mapping[str, str] | None = None, *, model_name: str) -> FasterWhisperASR:
    """D-052 Part A Section 5 / D-053 Section 10: construct a
    ``FasterWhisperASR`` with every behaviorally-relevant setting explicit
    and env-overridable, instead of leaving them as silent library defaults
    baked into a bare ``FasterWhisperASR(model_name=...)`` call. Setting no
    env vars reproduces today's exact (legacy, fallback-ladder) behavior.

    ``CUTSELL_ASR_DETERMINISTIC_CONFIG`` (default OFF): switches to the
    scalar-temperature deterministic candidate from
    ``build_deterministic_asr_provider`` -- see D-053's live ASR-only
    stability battery for the evidence this is gated on.

    ``CUTSELL_ASR_COMPUTE_TYPE`` is the one override named separately (by
    D-052): it lets a future, separately-tested change pin precision (e.g.
    ``float16``) without touching this module's code, while the default
    stays ``"auto"`` (unchanged behavior) until that follow-up is
    explicitly authorized and validated. It applies to either provider.
    """
    values = env if env is not None else os.environ
    device = _env_str(values, "CUTSELL_ASR_DEVICE", "auto")
    compute_type = _env_str(values, "CUTSELL_ASR_COMPUTE_TYPE", "auto")
    if _env_bool(values, _ASR_DETERMINISTIC_CONFIG_ENV):
        return build_deterministic_asr_provider(model_name, device=device, compute_type=compute_type)
    return FasterWhisperASR(model_name=model_name, device=device, compute_type=compute_type)
