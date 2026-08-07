"""OpenAI-backed Best Take ranking using Watch + Listen signals."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Tuple

from .contracts import CandidateTake, RankedTake
from .providers import ProviderStatus
from .take_judge_provider import TakeJudgeProviderResult


@dataclass
class OpenAITakeJudgeProvider:
    model: str = "gpt-5.4-mini"
    client_factory: Callable[[], object] | None = None

    def _client(self):
        if self.client_factory is not None:
            return self.client_factory()
        from openai import OpenAI
        return OpenAI()

    def rank(self, takes: Tuple[CandidateTake, ...]) -> TakeJudgeProviderResult:
        evidence = []
        for take in takes:
            signals = take.signals
            evidence.append({
                "id": take.clip_id,
                "transcript": take.text,
                "duration_sec": take.duration_sec,
                "complete_idea": take.complete_idea,
                "signals": ({
                    "silence_ratio": signals.silence_ratio,
                    "audio_quality": signals.audio_quality,
                    "face_visibility": signals.face_visibility,
                    "eye_contact": signals.eye_contact,
                    "framing_quality": signals.framing_quality,
                    "product_visibility": signals.product_visibility,
                    "motion_stability": signals.motion_stability,
                    "continuity": signals.continuity,
                    "visual_fumble": signals.visual_fumble,
                } if signals is not None else {}),
            })
        instruction = (
            "Rank these alternate recordings of the same sales idea from strongest to weakest. "
            "Prefer natural delivery, completeness, clarity, confident pacing, strong visual presentation, "
            "low fumble/distraction, and sales effectiveness. Do not delete any candidate. "
            "Return JSON only as {\"ranked\":[{\"id\":...,\"score\":0..1,\"reason\":...}]}. "
            "Include every candidate exactly once."
        )
        response = self._client().responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps({"takes": evidence}, ensure_ascii=False)},
            ],
        )
        data = json.loads(str(response.output_text).strip())
        items = data.get("ranked")
        if not isinstance(items, list):
            raise ValueError("take judge returned invalid payload")
        expected = {take.clip_id for take in takes}
        seen = set()
        ranked = []
        for item in items:
            clip_id = str(item.get("id") or "")
            if clip_id not in expected or clip_id in seen:
                raise ValueError("take judge returned invalid clip id")
            score = float(item.get("score"))
            if not 0.0 <= score <= 1.0:
                raise ValueError("take judge score outside 0..1")
            ranked.append(RankedTake(clip_id, score, str(item.get("reason") or "")[:240]))
            seen.add(clip_id)
        if seen != expected:
            raise ValueError("take judge omitted candidates")
        ranked.sort(key=lambda item: (-item.score, item.clip_id))
        return TakeJudgeProviderResult(
            tuple(ranked),
            ProviderStatus("openai", True, True, "applied"),
        )
