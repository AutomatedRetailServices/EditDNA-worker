"""OpenAI-backed semantic retry grouping for CutSell."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Tuple

from .contracts import CandidateTake
from .openai_json import parse_json_object
from .providers import ProviderStatus
from .take_grouping_provider import TakeGroupingProviderResult


@dataclass
class OpenAITakeGroupingProvider:
    model: str = "gpt-4o-mini"
    client_factory: Callable[[], object] | None = None

    def _client(self):
        if self.client_factory is not None:
            return self.client_factory()
        from openai import OpenAI
        return OpenAI()

    def group(self, takes: Tuple[CandidateTake, ...]) -> TakeGroupingProviderResult:
        payload = []
        for take in takes:
            signals = take.signals
            payload.append({
                "id": take.clip_id,
                "source_asset_id": take.source_asset_id,
                "source_order": take.source_order,
                "start": take.start,
                "end": take.end,
                "text": take.text,
                "complete_idea": take.complete_idea,
                "signals": ({
                    "product_visibility": signals.product_visibility,
                    "continuity": signals.continuity,
                    "visual_fumble": signals.visual_fumble,
                } if signals is not None else {}),
            })

        instruction = (
            "Group these valid TikTok Shop/UGC recording takes by the creator's underlying spoken idea. "
            "Different wording can still be alternate takes when the creator is clearly retrying the same claim, hook, proof, story beat, CTA, or demonstration idea. "
            "Do not group clips merely because they share a broad commercial role or product topic. Preserve distinct claims/details as separate groups. "
            "Use transcript meaning, temporal proximity/source context, and available visual context. Every clip must appear exactly once. "
            "Return JSON only as {\"groups\":[[\"clip_id\",...],...],\"reason\":\"...\"}."
        )
        response = self._client().responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps({"takes": payload}, ensure_ascii=False)},
            ],
        )
        data = parse_json_object(response.output_text)
        groups = data.get("groups")
        if not isinstance(groups, list):
            raise ValueError("take grouping returned invalid groups")
        normalized = []
        for group in groups:
            if not isinstance(group, list):
                raise ValueError("take grouping returned invalid group")
            normalized.append(tuple(str(item) for item in group))
        return TakeGroupingProviderResult(
            tuple(normalized),
            ProviderStatus("openai", True, True, "applied"),
            str(data.get("reason") or "")[:500],
        )
