"""Dependency-free provider protocols for model input/output boundaries."""

from typing import Any, Mapping, Optional, Protocol, Sequence


class ASRProvider(Protocol):
    def transcribe(self, audio_path: str) -> Sequence[Mapping[str, Any]]: ...


class SemanticClassifierProvider(Protocol):
    def classify(self, clips: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]: ...


class VisionProvider(Protocol):
    def score(self, image_path: str, text: str) -> float: ...


class TakeJudgeProvider(Protocol):
    def choose(self, takes: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]: ...
