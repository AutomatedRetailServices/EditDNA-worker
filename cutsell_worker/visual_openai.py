"""OpenAI image-input adapter for CutSell Watch + Listen visual signals."""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

from .contracts import CandidateTake
from .frame_sampling import FrameSample
from .openai_json import parse_json_object
from .providers import ProviderStatus
from .visual_analysis import VisualObservation, VisualProviderResult


_INSTRUCTION = (
    "You are CutSell Watch + Listen, evaluating sales/UGC/conversational recording takes like a careful human editor. "
    "Use the transcript plus the ordered temporal frames across each take. Judge the delivery over time, not a single pose. "
    "Return JSON only as {\"clips\":[{\"id\":...,\"face_visibility\":0..1,\"eye_contact\":0..1,"
    "\"framing_quality\":0..1,\"product_visibility\":0..1,\"motion_stability\":0..1,"
    "\"continuity\":0..1,\"visual_fumble\":0..1,\"expression_naturalness\":0..1,"
    "\"gesture_naturalness\":0..1,\"delivery_energy\":0..1,\"distraction_risk\":0..1}]}. "
    "Include every take exactly once. visual_fumble and distraction_risk are higher when worse. "
    "expression_naturalness should penalize visibly awkward/frozen/incongruent facial delivery but not normal individual style. "
    "gesture_naturalness should reward intentional, coherent hand/body movement and penalize accidental recording fumbles. "
    "delivery_energy means visually engaged/present delivery appropriate to the content, not exaggerated movement. "
    "Do not infer sensitive traits and do not decide deletion; only produce editing-quality signals."
)


@dataclass
class OpenAIVisualProvider:
    model: str = "gpt-4o-mini"
    client_factory: Callable[[], object] | None = None
    # Full raw recordings can contain dozens of segmented takes. Sending every
    # frame for every take in one Responses request caused oversized multimodal
    # requests in real validation. Keep requests bounded while preserving all takes.
    batch_size: int = 6

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

    def _analyze_batch(
        self,
        takes: Tuple[CandidateTake, ...],
        frames_by_clip: dict[str, list[FrameSample]],
    ) -> Tuple[VisualObservation, ...]:
        content = [{"type": "input_text", "text": _INSTRUCTION}]
        for take in takes:
            ordered = sorted(frames_by_clip.get(take.clip_id, ()), key=lambda item: item.timestamp)
            content.append({
                "type": "input_text",
                "text": json.dumps({
                    "take_id": take.clip_id,
                    "transcript": take.text,
                    "source_asset_id": take.source_asset_id,
                    "take_start_sec": take.start,
                    "take_end_sec": take.end,
                    "frame_count": len(ordered),
                }, ensure_ascii=False),
            })
            for sample in ordered:
                content.append({
                    "type": "input_text",
                    "text": json.dumps({
                        "timestamp_sec": round(sample.timestamp, 3),
                        "relative_position": round(sample.relative_position, 4),
                    }),
                })
                content.append({
                    "type": "input_image",
                    "image_url": self._image_url(sample.path),
                    "detail": "high",
                })

        response = self._client().responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
        )
        data = parse_json_object(response.output_text)
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
                expression_naturalness=self._score(item.get("expression_naturalness")),
                gesture_naturalness=self._score(item.get("gesture_naturalness")),
                delivery_energy=self._score(item.get("delivery_energy")),
                distraction_risk=self._score(item.get("distraction_risk")),
            ))
            seen.add(clip_id)
        if seen != known:
            raise ValueError("visual provider omitted takes")
        return tuple(observations)

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

        size = max(1, min(12, int(self.batch_size)))
        observations = []
        for start in range(0, len(takes), size):
            observations.extend(self._analyze_batch(takes[start:start + size], frames_by_clip))

        return VisualProviderResult(
            observations=tuple(observations),
            status=ProviderStatus("openai", True, True, "applied"),
        )
