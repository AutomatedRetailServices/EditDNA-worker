"""ASR provider boundary for CutSell Flow B.

D-052 Part A Section 1/5: every behaviorally-relevant Faster-Whisper decode
setting is now an explicit ``FasterWhisperASR`` field instead of an implicit
library default. Every default value below is chosen to be IDENTICAL to
faster-whisper's own documented library default -- this is a
make-it-explicit-and-reportable change only, not a behavior change. See
``docs/CUTSELL_DECISIONS.md`` D-051/D-052 for the full audit: faster-whisper
exposes no random-seed control at all for its beam-search/temperature-
fallback decoding, so "pinning" here means naming and fixing every knob the
library *does* expose, not achieving bit-for-bit reproducibility (that
remains an open, reported limitation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Iterable, Mapping, Protocol, Tuple

from .canonical_asr_evidence import ASRConfigFingerprint, build_asr_config_fingerprint
from .contracts import TranscriptSegment, Word

# faster-whisper's own documented default temperature fallback ladder: the
# primary decode attempt uses temperature 0.0 (deterministic beam search);
# only a segment that fails the library's compression-ratio/log-prob/
# no-speech quality gates escalates to the next (sampling, non-deterministic)
# temperature in this ladder. Most segments never leave 0.0 -- but D-052's
# audit note is that GPU floating-point non-determinism under
# ``compute_type="auto"`` can shift which segments are borderline enough to
# escalate, and ``condition_on_previous_text=True`` (also a library default,
# also now explicit below) compounds any resulting drift forward through the
# rest of the transcript. This is the leading, though not proven-exclusive,
# suspect the D-052 audit names for the D-051-observed segment_count
# variance -- see canonical_asr_evidence.py's module docstring.
DEFAULT_TEMPERATURE_LADDER: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_BEAM_SIZE = 5
DEFAULT_BEST_OF = 5
DEFAULT_CONDITION_ON_PREVIOUS_TEXT = True


class ASRProvider(Protocol):
    def transcribe(self, path: str, *, source_asset_id: str, language_hint: str | None = None) -> Tuple[TranscriptSegment, ...]: ...


def _env_str(values: Mapping[str, str], key: str, default: str) -> str:
    raw = values.get(key)
    return str(raw).strip() if raw else default


@dataclass
class FasterWhisperASR:
    model_name: str = "medium"
    device: str = "auto"
    # D-052: "auto" is faster-whisper's own default and is left unchanged
    # here -- pinning it to a fixed precision (e.g. float16/int8) is a
    # separately-tested configuration change per the D-052 directive
    # ("If pinning compute type could alter transcription quality
    # materially, keep it as a separately tested configuration change"),
    # not bundled into this make-explicit pass. See
    # ``load_asr_provider_from_env`` for the opt-in override.
    compute_type: str = "auto"
    beam_size: int = DEFAULT_BEAM_SIZE
    best_of: int = DEFAULT_BEST_OF
    temperature_ladder: Tuple[float, ...] = field(default_factory=lambda: DEFAULT_TEMPERATURE_LADDER)
    condition_on_previous_text: bool = DEFAULT_CONDITION_ON_PREVIOUS_TEXT
    word_timestamps: bool = True
    vad_filter: bool = True
    initial_prompt: str | None = None

    def config_fingerprint(self, *, language_hint: str | None = None) -> ASRConfigFingerprint:
        """D-052: the full effective decode config as one hashable,
        reportable value -- see canonical_asr_evidence.ASRConfigFingerprint.
        Two runs whose fingerprints match ruled out configuration as the
        source of any observed transcript/segmentation difference between
        them; a mismatch names configuration as a live suspect."""
        return build_asr_config_fingerprint(
            model_name=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature_ladder=self.temperature_ladder,
            condition_on_previous_text=self.condition_on_previous_text,
            word_timestamps=self.word_timestamps,
            vad_filter=self.vad_filter,
            language_hint=language_hint,
            initial_prompt=self.initial_prompt,
        )

    def transcribe(self, path: str, *, source_asset_id: str, language_hint: str | None = None) -> Tuple[TranscriptSegment, ...]:
        # Heavy dependency is loaded only when a real transcription runs.
        from faster_whisper import WhisperModel

        # D-052 NOTE: beam_size/best_of/temperature_ladder/
        # condition_on_previous_text/initial_prompt are now explicit
        # dataclass fields (audited, fingerprinted via
        # config_fingerprint()) but are DELIBERATELY NOT YET threaded into
        # this call. This sandbox has no faster-whisper install to verify
        # the exact defaults of the version actually running on the GPU
        # image against, and the D-052 directive is explicit: "Report
        # before changing any precision mode" / "keep as a separately
        # tested configuration change." Passing an assumed-but-unverified
        # default here risks a silent behavior change in the one direction
        # this whole audit exists to prevent. Wiring these fields into the
        # live call is the recommended, low-risk follow-up once verified
        # against the pinned faster-whisper version on the GPU image (see
        # docs/CUTSELL_DECISIONS.md D-052).
        model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        segments, _info = model.transcribe(
            path,
            language=language_hint,
            word_timestamps=self.word_timestamps,
            vad_filter=self.vad_filter,
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


def load_asr_provider_from_env(env: Mapping[str, str] | None = None, *, model_name: str) -> FasterWhisperASR:
    """D-052 Part A Section 5: construct an ``FasterWhisperASR`` with every
    behaviorally-relevant setting explicit and env-overridable, instead of
    leaving them as silent library defaults baked into a bare
    ``FasterWhisperASR(model_name=...)`` call. Every default below matches
    the dataclass's own defaults (which match faster-whisper's library
    defaults) -- setting no env vars reproduces today's exact behavior.

    ``CUTSELL_ASR_COMPUTE_TYPE`` is the one override this directive singled
    out by name: it lets a future, separately-tested change pin precision
    (e.g. ``float16``) without touching this module's code, while the
    default stays ``"auto"`` (unchanged behavior) until that follow-up is
    explicitly authorized and validated.
    """
    values = env if env is not None else os.environ
    return FasterWhisperASR(
        model_name=model_name,
        device=_env_str(values, "CUTSELL_ASR_DEVICE", "auto"),
        compute_type=_env_str(values, "CUTSELL_ASR_COMPUTE_TYPE", "auto"),
    )
