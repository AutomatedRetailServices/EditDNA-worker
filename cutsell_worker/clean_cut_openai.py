"""OpenAI-backed recording-mistake judge for conservative Clean Cut."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .clean_cut_provider import CleanCutJudgement, CleanCutProviderResult
from .contracts import CandidateTake
from .openai_json import parse_json_object
from .providers import ProviderStatus


@dataclass
class OpenAICleanCutProvider:
    model: str = "gpt-4o-mini"
    client_factory: Callable[[], object] | None = None

    def _client(self):
        if self.client_factory is not None:
            return self.client_factory()
        from openai import OpenAI
        return OpenAI()

    def judge(self, takes: tuple[CandidateTake, ...]) -> CleanCutProviderResult:
        evidence = []
        for index, take in enumerate(takes):
            signals = take.signals
            evidence.append({
                "index": index,
                "id": take.clip_id,
                "transcript": take.text,
                "duration_sec": round(take.duration_sec, 3),
                "previous_transcript": takes[index - 1].text if index > 0 else None,
                "next_transcript": takes[index + 1].text if index + 1 < len(takes) else None,
                "signals": ({
                    "audio_quality": signals.audio_quality,
                    "silence_ratio": signals.silence_ratio,
                    "motion_stability": signals.motion_stability,
                    "continuity": signals.continuity,
                    "visual_fumble": signals.visual_fumble,
                } if signals is not None else {}),
            })

        instruction = (
            "You are CutSell Clean Cut Judge. Your ONLY job is to distinguish valid creator speech "
            "from obvious recording mistakes. You have no authority to judge sales quality, hook quality, "
            "commercial role, usefulness, style, profanity, or whether a sentence is persuasive. "
            "For every candidate return exactly one action: KEEP, DELETE, or MIXED. "
            "DELETE only when the WHOLE candidate is clearly unusable production speech such as an explicit "
            "restart/stop direction, self-correction about filming, abandoned gibberish/restart, repeated false "
            "start, or reaction to a recording error. Profanity or emotion alone is NEVER a reason to delete. "
            "KEEP valid content even if casual, imperfect, profane, emotional, short, or commercially weak. "
            "MIXED when one candidate contains both an obvious blooper/restart portion and valid creator speech; "
            "MIXED must not be treated as a whole-candidate delete. When uncertain choose KEEP. "
            "Use neighboring transcript only to understand recording context. "
            "Return JSON only: {\"judgements\":[{\"id\":...,\"action\":\"keep|delete|mixed\","
            "\"confidence\":0..1,\"reason\":...}]}. Include every candidate exactly once."
        )
        response = self._client().responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps({"takes": evidence}, ensure_ascii=False)},
            ],
        )
        data = parse_json_object(response.output_text)
        items = data.get("judgements")
        if not isinstance(items, list):
            raise ValueError("clean cut judge returned invalid payload")
        judgements = []
        for item in items:
            judgements.append(CleanCutJudgement(
                clip_id=str(item.get("id") or ""),
                action=str(item.get("action") or "").lower(),
                confidence=float(item.get("confidence")),
                reason=str(item.get("reason") or ""),
            ))
        return CleanCutProviderResult(
            tuple(judgements),
            ProviderStatus("openai", True, True, "applied"),
        )
