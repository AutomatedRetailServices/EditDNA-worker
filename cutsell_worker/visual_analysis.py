"""Visual-analysis provider boundary for CutSell Watch + Listen."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Protocol, Tuple

from .contracts import CandidateTake, MediaSignals
from .frame_sampling import FrameSample
from .providers import ProviderStatus


@dataclass(frozen=True)
class VisualObservation:
    clip_id: str
    face_visibility: float = 0.5
    eye_contact: float = 0.5
    framing_quality: float = 0.5
    product_visibility: float = 0.0
    motion_stability: float = 0.5
    continuity: float = 0.5
    visual_fumble: float = 0.0


@dataclass(frozen=True)
class VisualProviderResult:
    observations: Tuple[VisualObservation, ...]
    status: ProviderStatus


class VisualProvider(Protocol):
    def analyze(
        self,
        takes: Tuple[CandidateTake, ...],
        samples: Tuple[FrameSample, ...],
    ) -> VisualProviderResult: ...


class NoopVisualProvider:
    def analyze(self, takes, samples):
        return VisualProviderResult(
            observations=(),
            status=ProviderStatus("none", False, False, "not_requested"),
        )


def safe_visual_analyze(provider: VisualProvider, takes, samples) -> VisualProviderResult:
    try:
        return provider.analyze(takes, samples)
    except Exception as exc:
        return VisualProviderResult(
            observations=(),
            status=ProviderStatus(
                provider=provider.__class__.__name__,
                requested=True,
                available=False,
                status="provider_error",
                reason=exc.__class__.__name__,
            ),
        )


def apply_visual_observations(
    takes: Tuple[CandidateTake, ...],
    observations: Tuple[VisualObservation, ...],
) -> Tuple[CandidateTake, ...]:
    by_clip: Mapping[str, VisualObservation] = {item.clip_id: item for item in observations}
    output = []
    for take in takes:
        observation = by_clip.get(take.clip_id)
        if observation is None:
            output.append(take)
            continue
        base = take.signals or MediaSignals(take.source_asset_id, take.start, take.end)
        signals = replace(
            base,
            face_visibility=observation.face_visibility,
            eye_contact=observation.eye_contact,
            framing_quality=observation.framing_quality,
            product_visibility=observation.product_visibility,
            motion_stability=observation.motion_stability,
            continuity=observation.continuity,
            visual_fumble=observation.visual_fumble,
        )
        output.append(replace(take, signals=signals))
    return tuple(output)
