"""Validated, deterministic inputs and outputs for Take Judge V2."""

from dataclasses import dataclass
import base64
import os
import re
import tempfile
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from worker.text_normalization import unicode_word_tokens


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a score")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


@dataclass(frozen=True)
class DeliveryFeatures:
    word_count: int
    words_per_second: float
    filler_count: int
    repeated_word_count: int
    incomplete_phrase: bool
    transcript_clarity_score: float
    silence_ratio: Optional[float]
    excessive_pause: bool
    very_short: bool
    abnormally_long: bool
    semantic_score: Optional[float]
    visual_score: Optional[float]

    def __post_init__(self) -> None:
        if min(self.word_count, self.filler_count, self.repeated_word_count) < 0:
            raise ValueError("delivery counts cannot be negative")
        if self.words_per_second < 0:
            raise ValueError("speech rate cannot be negative")
        for name in ("transcript_clarity_score", "silence_ratio", "semantic_score", "visual_score"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _unit(value, name))


@dataclass(frozen=True)
class TakeJudgeCandidate:
    candidate_id: str
    slot: str
    transcript: str
    duration_sec: float
    delivery: DeliveryFeatures
    frame_timestamps: Tuple[float, ...]
    image_count: int

    def __post_init__(self) -> None:
        if not self.candidate_id or self.duration_sec < 0 or self.image_count < 0:
            raise ValueError("invalid Take Judge candidate")
        if self.image_count != len(self.frame_timestamps):
            raise ValueError("image count does not match timestamps")


@dataclass(frozen=True)
class TakeJudgeCandidateScore:
    candidate_id: str
    delivery_score: float
    visual_performance_score: float
    clarity_score: float
    sales_effectiveness_score: float
    overall_score: float
    reason: str

    def __post_init__(self) -> None:
        if not self.candidate_id or not isinstance(self.reason, str):
            raise ValueError("invalid candidate score")
        for name in ("delivery_score", "visual_performance_score", "clarity_score",
                     "sales_effectiveness_score", "overall_score"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        object.__setattr__(self, "reason", self.reason[:240])


@dataclass(frozen=True)
class TakeJudgeV2Result:
    winner_id: Optional[str]
    candidate_scores: Tuple[TakeJudgeCandidateScore, ...]
    confidence: float
    abstain: bool
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", _unit(self.confidence, "confidence"))
        if type(self.abstain) is not bool or not isinstance(self.reason, str):
            raise ValueError("invalid Take Judge result")
        object.__setattr__(self, "reason", self.reason[:240])


@dataclass(frozen=True)
class TemporalFrameSample:
    candidate_id: str
    requested_frame_count: int
    successful_frame_timestamps: Tuple[float, ...]
    image_content: Tuple[Mapping[str, Any], ...]
    attempted_count: int
    failed_count: int


def temporal_timestamps(start: float, end: float, count: int) -> Tuple[float, ...]:
    """Return bin-centred timestamps, inset from both potentially unstable cuts."""
    if count <= 0 or end <= start:
        return ()
    duration = end - start
    margin = min(0.1, duration * 0.1)
    low, high = start + margin, end - margin
    if high < low:
        low = high = start + duration / 2.0
    span = high - low
    values = [low + span * ((index + 0.5) / count) for index in range(count)]
    unique: Dict[int, float] = {}
    for value in values:
        safe = min(end, max(start, value))
        unique.setdefault(round(safe * 1000), safe)
    return tuple(unique.values())


def sample_candidate_frames(
    clip: Mapping[str, Any], requested_count: int,
    fallback_source: str, extractor: Callable[[str, float, str], bool],
) -> TemporalFrameSample:
    candidate_id = str(clip.get("id") or "")
    timestamps = temporal_timestamps(float(clip.get("start", 0.0)), float(clip.get("end", 0.0)), requested_count)
    source = str(clip.get("source_local") or fallback_source)
    successes, images = [], []
    for timestamp in timestamps:
        path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
                path = handle.name
            if extractor(source, timestamp, path):
                with open(path, "rb") as image:
                    encoded = base64.b64encode(image.read()).decode("ascii")
                if encoded:
                    successes.append(timestamp)
                    images.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}})
        except (OSError, ValueError):
            pass
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
    return TemporalFrameSample(candidate_id, requested_count, tuple(successes), tuple(images),
                               len(timestamps), len(timestamps) - len(successes))


_FILLERS = frozenset(("um", "uh", "erm", "like", "basically", "actually"))


def delivery_features(clip: Mapping[str, Any]) -> DeliveryFeatures:
    text = str(clip.get("text") or "").strip()
    tokens = unicode_word_tokens(text)
    duration = max(0.0, float(clip.get("end", 0.0)) - float(clip.get("start", 0.0)))
    repeats = sum(left == right for left, right in zip(tokens, tokens[1:]))
    words = clip.get("words") if isinstance(clip.get("words"), Sequence) else ()
    pauses = []
    for left, right in zip(words, words[1:]):
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            pauses.append(max(0.0, float(right.get("start", 0.0)) - float(left.get("end", 0.0))))
    clarity = clip.get("transcript_clarity_score", clip.get("clarity_score", 1.0 if text else 0.0))
    incomplete = bool(tokens) and (len(tokens) <= 2 or text[-1:] not in ".?!")
    return DeliveryFeatures(
        word_count=len(tokens), words_per_second=(len(tokens) / duration if duration else 0.0),
        filler_count=sum(token in _FILLERS for token in tokens), repeated_word_count=repeats,
        incomplete_phrase=incomplete, transcript_clarity_score=float(clarity),
        silence_ratio=clip.get("silence_ratio"), excessive_pause=any(pause >= 1.0 for pause in pauses),
        very_short=duration < 0.75, abnormally_long=duration > 15.0,
        semantic_score=clip.get("semantic_score"), visual_score=clip.get("visual_score"),
    )

