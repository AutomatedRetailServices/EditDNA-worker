"""Finishing -- durable interface contract only (D-024).

Not implemented or activated in Clean Cut Core V1. Defines the shape a
future delivery-finishing pass would have: dialogue loudness normalization,
true-peak control, a color/contrast/saturation/sharpening profile, H.264/
AAC delivery encode, fast-start (moov atom placement for progressive
playback), metadata hygiene, and decode verification.

Finishing operates strictly after Selection Freeze and Boundary/Render; it
must NEVER mutate semantics (spoken content, membership, or ordering) --
it only prepares the already-final rendered file for delivery. Nothing in
this module performs finishing; it exists so a future implementation has an
agreed input/output shape without another architecture reset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class FinishingSpec:
    target_loudness_lufs: float | None = None
    true_peak_ceiling_dbtp: float | None = None
    color_profile: str | None = None
    delivery_codec: str = "h264_aac"
    fast_start: bool = True
    metadata: dict | None = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


@dataclass(frozen=True)
class FinishingResult:
    status: str  # "PASS" | "FAIL"
    output_path: str | None
    decode_verified: bool
    detail: dict


class FinishingProvider(Protocol):
    """Contract a future concrete finishing pass must satisfy. Not
    implemented or invoked anywhere in Clean Cut Core V1."""

    def finish(self, rendered_video_path: str, spec: FinishingSpec) -> FinishingResult:
        ...
