"""Runtime model configuration for the active worker pipeline.

This module deliberately contains no client or heavyweight ML imports. Configuration is
loaded afresh on every call so callers and tests can safely change the environment.
"""

from dataclasses import dataclass
import os
from typing import Mapping, Optional, Tuple


class ModelConfigurationError(ValueError):
    """An environment variable has an invalid, non-secret configuration value."""

    def __init__(self, variable: str, expected: str) -> None:
        self.variable = variable
        self.expected = expected
        super().__init__(f"Invalid model configuration for {variable}: expected {expected}")


@dataclass(frozen=True)
class OpenAIClientConfig:
    timeout_seconds: float
    max_retries: int
    environment_sources: Tuple[str, ...]


@dataclass(frozen=True)
class ASRConfig:
    provider: str
    model_name: str
    enabled: bool
    timeout_seconds: Optional[float]
    retry_count: int
    modality: str
    device: str
    environment_sources: Tuple[str, ...]


@dataclass(frozen=True)
class SemanticLLMConfig:
    provider: str
    model_name: str
    enabled: bool
    timeout_seconds: Optional[float]
    retry_count: int
    modality: str
    environment_sources: Tuple[str, ...]


@dataclass(frozen=True)
class VisionConfig:
    provider: str
    model_name: str
    enabled: bool
    timeout_seconds: Optional[float]
    retry_count: int
    modality: str
    interval_seconds: float
    max_samples: int
    weight: float
    environment_sources: Tuple[str, ...]


@dataclass(frozen=True)
class OpenAIVisualBadTakeConfig:
    provider: str
    model_name: str
    enabled: bool
    timeout_seconds: Optional[float]
    retry_count: int
    modality: str
    environment_sources: Tuple[str, ...]


@dataclass(frozen=True)
class BoundaryRefinerConfig:
    provider: str
    model_name: str
    enabled: bool
    timeout_seconds: Optional[float]
    retry_count: int
    modality: str
    min_duration_seconds: float
    head_step_seconds: float
    tail_step_seconds: float
    frame_count: int
    environment_sources: Tuple[str, ...]


@dataclass(frozen=True)
class TakeJudgeConfig:
    provider: str
    model_name: str
    enabled: bool
    timeout_seconds: Optional[float]
    retry_count: int
    modality: str
    max_groups: int
    max_takes: int
    frame_count: int
    environment_sources: Tuple[str, ...]


@dataclass(frozen=True)
class GlobalComposerConfig:
    provider: str
    model_name: Optional[str]
    enabled: bool
    timeout_seconds: Optional[float]
    retry_count: int
    modality: str
    min_semantic_score: float
    max_per_slot: int
    environment_sources: Tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    openai: OpenAIClientConfig
    asr: ASRConfig
    semantic_llm: SemanticLLMConfig
    vision: VisionConfig
    visual_bad_take: OpenAIVisualBadTakeConfig
    boundary_refiner: BoundaryRefinerConfig
    take_judge: TakeJudgeConfig
    global_composer: GlobalComposerConfig


_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_VALUES = frozenset(("0", "false", "no", "off"))


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    if name not in env:
        return default
    normalized = env[name].strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ModelConfigurationError(name, "a boolean (1/0, true/false, yes/no, on/off)")


def _integer(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ModelConfigurationError(name, "an integer") from None


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ModelConfigurationError(name, "a number") from None


def load_model_config() -> ModelConfig:
    """Read and validate the canonical model configuration from the environment."""
    env = os.environ
    whisper_model = env.get("WHISPER_MODEL_NAME") or env.get("WHISPER_MODEL") or "medium"
    whisper_device = env.get("WHISPER_DEVICE", env.get("ASR_DEVICE", "auto"))

    timeout = _float(env, "OPENAI_TIMEOUT_SECONDS", 60.0)
    retries = _integer(env, "OPENAI_MAX_RETRIES", 0)
    if timeout <= 0:
        raise ModelConfigurationError("OPENAI_TIMEOUT_SECONDS", "a number greater than zero")
    if retries < 0:
        raise ModelConfigurationError("OPENAI_MAX_RETRIES", "a non-negative integer")

    return ModelConfig(
        openai=OpenAIClientConfig(
            timeout_seconds=timeout, max_retries=retries,
            environment_sources=("OPENAI_TIMEOUT_SECONDS", "OPENAI_MAX_RETRIES"),
        ),
        asr=ASRConfig(
            provider="faster-whisper", model_name=whisper_model,
            enabled=_boolean(env, "ASR_ENABLED", True), timeout_seconds=None,
            retry_count=0, modality="audio-to-text", device=whisper_device,
            environment_sources=("ASR_ENABLED", "WHISPER_MODEL_NAME", "WHISPER_MODEL", "WHISPER_DEVICE", "ASR_DEVICE"),
        ),
        semantic_llm=SemanticLLMConfig(
            provider="openai", model_name=env.get("EDITDNA_LLM_MODEL", "gpt-5.1"),
            enabled=_boolean(env, "EDITDNA_USE_LLM", False), timeout_seconds=None,
            retry_count=0, modality="text-to-structured-data",
            environment_sources=("EDITDNA_USE_LLM", "EDITDNA_LLM_MODEL"),
        ),
        vision=VisionConfig(
            provider="openai-clip", model_name="ViT-B/32",
            enabled=_boolean(env, "VISION_ENABLED", False), timeout_seconds=None,
            retry_count=0, modality="image-text-embedding",
            interval_seconds=_float(env, "VISION_INTERVAL_SEC", 2.0),
            max_samples=_integer(env, "VISION_MAX_SAMPLES", 50),
            weight=_float(env, "W_VISION", 0.7),
            environment_sources=("VISION_ENABLED", "VISION_INTERVAL_SEC", "VISION_MAX_SAMPLES", "W_VISION"),
        ),
        visual_bad_take=OpenAIVisualBadTakeConfig(
            provider="openai", model_name="gpt-4o",
            enabled=_boolean(env, "BAD_TAKES_ENABLED", False), timeout_seconds=None,
            retry_count=0, modality="image-and-text-to-label",
            environment_sources=("BAD_TAKES_ENABLED",),
        ),
        boundary_refiner=BoundaryRefinerConfig(
            provider="openai", model_name="gpt-5.1",
            enabled=_boolean(env, "BOUNDARY_REFINER_ENABLED", False), timeout_seconds=None,
            retry_count=0, modality="image-and-text-to-boundaries",
            min_duration_seconds=_float(env, "BOUNDARY_REFINER_MIN_DURATION_SEC", 3.0),
            head_step_seconds=_float(env, "BOUNDARY_REFINER_HEAD_STEP_SEC", 1.5),
            tail_step_seconds=_float(env, "BOUNDARY_REFINER_TAIL_STEP_SEC", 1.5), frame_count=3,
            environment_sources=("BOUNDARY_REFINER_ENABLED", "BOUNDARY_REFINER_MIN_DURATION_SEC", "BOUNDARY_REFINER_HEAD_STEP_SEC", "BOUNDARY_REFINER_TAIL_STEP_SEC"),
        ),
        take_judge=TakeJudgeConfig(
            provider="openai", model_name=env.get("TAKE_JUDGE_MODEL", "gpt-4o-mini"),
            enabled=_boolean(env, "TAKE_JUDGE_ENABLED", False), timeout_seconds=None,
            retry_count=0, modality="image-and-text-to-ranking",
            max_groups=_integer(env, "TAKE_JUDGE_MAX_GROUPS", 6),
            max_takes=_integer(env, "TAKE_JUDGE_MAX_TAKES", 3),
            frame_count=_integer(env, "TAKE_JUDGE_FRAMES", 1),
            environment_sources=("TAKE_JUDGE_ENABLED", "TAKE_JUDGE_MODEL", "TAKE_JUDGE_MAX_GROUPS", "TAKE_JUDGE_MAX_TAKES", "TAKE_JUDGE_FRAMES"),
        ),
        global_composer=GlobalComposerConfig(
            provider="local", model_name=None, enabled=False, timeout_seconds=None,
            retry_count=0, modality="structured-data-to-timeline",
            min_semantic_score=_float(env, "COMPOSER_MIN_SEMANTIC", 0.75),
            max_per_slot=_integer(env, "COMPOSER_MAX_PER_SLOT", 7),
            environment_sources=("COMPOSER_MIN_SEMANTIC", "COMPOSER_MAX_PER_SLOT"),
        ),
    )
