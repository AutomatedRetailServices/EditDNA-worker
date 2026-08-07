"""OpenAI image-input adapter for CutSell Watch + Listen visual signals."""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

from .contracts import CandidateTake
from .frame_sampling import FrameSample
from .providers import ProviderStatus
from .visual_analysis import VisualObservation, VisualProviderResult


@dataclass
class OpenAIVisualProvider:
    model: str = "gpt-5.4-nano"
    client_factory: Callable[[], object] | None = None

    def _client(self):
        if self.client_factory is not None:
            return self.client_factory()
        from openai import OpenAI
        return OpenAI()

    @staticmethod
    def _image_url(path: str) -> str:
        raw = Path(path).read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")

    @staticmethod
    def _score(value) -> float:
        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError("visual score outside 0..1")
        return score

    def analyze(
        self,
        takes: Tuple[CandidateTake, ...],
        samples: Tuple[FrameSample, ...],
    ) -> VisualProviderResult:
        if not takes:
            return VisualProviderResult((), ProviderStatus("openai", True, True, "empty_input"))

        frames_by_clip: dict[str, list[FrameSample]] = {}
        for sample in samples:
            frames_by_clip.setdefault(sample.clip_id, []).append(sample)

        content = [{
            "type": "input_text",
            "text": (
                "Evaluate each video take using only its transcript and temporal frames. Return JSON only as "
                "{\"clips\":[{\"id\":...,\"face_visibility\":0..1,\"eye_contact\":0..1,"
                "\"framing_quality\":0..1,\"product_visibility\":0..1,\"motion_stability\":0..1,"
                "\"continuity\":0..1,\"visual_fumble\":0..1}]}. Include every take exactly once. "
                "visual_fumble means visible restart/mistake/distraction; higher is worse. Do not decide deletion."
            ),
        }]
        for take in takes:
            content.append({
                "type": "input_text",
                "text": json.dumps({
                    "take_id": take.clip_id,
                    "transcript": take.text,
                    "source_asset_id": take.source_asset_id,
                }, ensure_ascii=False),
            })
            for sample in sorted(frames_by_clip.get(take.clip_id, ()), key=lambda item: item.timestamp):
                content.append({
                    "type": "input_image",
                    "image_url": self._image_url(sample.path),
                    "detail": "low",
                })

        response = self._client().responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
        )
        data = json.loads(str(response.output_text).strip())
        items = data.get("clips")
        if not isinstance(items, list):
            raise ValueError("visual provider returned invalid payload")

        known = {take.clip_id for take in takes}
        seen = set()
        observations = []
        for item in items:
            clip_id = str(item.get("id") or "")
            if clip_id not in known or clip_id in seen:
                raise ValueError("visual provider returned invalid clip id")
            observations.append(VisualObservation(
                clip_id=clip_id,
                face_visibility=self._score(item.get("face_visibility")),
                eye_contact=self._score(item.get("eye_contact")),
                framing_quality=self._score(item.get("framing_quality")),
                product_visibility=self._score(item.get("product_visibility")),
                motion_stability=self._score(item.get("motion_stability")),
                continuity=self._score(item.get("continuity")),
                visual_fumble=self._score(item.get("visual_fumble")),
            ))
            seen.add(clip_id)
        if seen != known:
            raise ValueError("visual provider omitted takes")
        return VisualProviderResult(
            observations=tuple(observations),
            status=ProviderStatus("openai", True, True, "applied"),
        )
