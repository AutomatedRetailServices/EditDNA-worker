"""ASR provider boundary for CutSell Flow B."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Tuple

from .contracts import TranscriptSegment, Word


class ASRProvider(Protocol):
    def transcribe(self, path: str, *, source_asset_id: str, language_hint: str | None = None) -> Tuple[TranscriptSegment, ...]: ...


@dataclass
class FasterWhisperASR:
    model_name: str = "medium"
    device: str = "auto"
    compute_type: str = "auto"

    def transcribe(self, path: str, *, source_asset_id: str, language_hint: str | None = None) -> Tuple[TranscriptSegment, ...]:
        # Heavy dependency is loaded only when a real transcription runs.
        from faster_whisper import WhisperModel

        model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        segments, _info = model.transcribe(
            path,
            language=language_hint,
            word_timestamps=True,
            vad_filter=True,
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
