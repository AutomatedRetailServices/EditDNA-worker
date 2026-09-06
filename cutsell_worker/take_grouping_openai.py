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

    def _parse_or_repair_json(self, client: object, output_text: str) -> tuple[dict, bool]:
        """Parse provider JSON; on malformed JSON, ask once for format-only repair.

        The repair request is not allowed to reinterpret grouping semantics. It only
        converts the already-produced answer into valid JSON with the same clip ids
        and grouping intent. If repair also fails, the caller falls back safely.
        """
        try:
            return parse_json_object(output_text), False
        except Exception:
            repair = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Repair the following malformed take-grouping response into valid JSON only. "
                            "Do not add, remove, rename, regroup, or reinterpret clip ids. Preserve the original "
                            "grouping intent exactly. Return only {\"groups\":[[\"clip_id\",...],...],\"reason\":\"...\"}."
                        ),
                    },
                    {"role": "user", "content": str(output_text)[:30000]},
                ],
            )
            return parse_json_object(repair.output_text), True

    def group(
        self,
        takes: Tuple[CandidateTake, ...],
        context_text: str = "",
    ) -> TakeGroupingProviderResult:
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
                    "expression_naturalness": signals.expression_naturalness,
                    "gesture_naturalness": signals.gesture_naturalness,
                    "delivery_energy": signals.delivery_energy,
                } if signals is not None else {}),
            })

        instruction = (
            "Group these valid short-form recording takes by the creator's SAME underlying communication attempt. The footage may be sales, natural, or mixed; "
            "use whole-video context to understand which. Recognize retries even when wording changes: two takes belong together when the creator is clearly "
            "trying again to express the same specific idea/story beat/hook/demo/claim/reaction/CTA. In natural storytime or talking-head footage, repeated attempts "
            "at the same sentence or story point are also retries. Do NOT group clips merely because they share a broad topic, product, commercial role, or story. "
            "Preserve distinct facts, claims, details and sequential story beats as separate groups. Use transcript meaning, temporal proximity, retry/body-reset context "
            "and visual continuity. Every clip must appear exactly once. Return JSON only as "
            "{\"groups\":[[\"clip_id\",...],...],\"reason\":\"...\"}."
        )
        client = self._client()
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps({
                    "whole_video_context": context_text[:20000],
                    "takes": payload,
                }, ensure_ascii=False)},
            ],
        )
        data, repaired_json = self._parse_or_repair_json(client, response.output_text)
        groups = data.get("groups")
        if not isinstance(groups, list):
            raise ValueError("take grouping returned invalid groups")
        normalized = []
        for group in groups:
            if not isinstance(group, list):
                raise ValueError("take grouping returned invalid group")
            normalized.append(tuple(str(item) for item in group))
        reason = str(data.get("reason") or "")[:500]
        if repaired_json:
            reason = (reason + "; " if reason else "") + "json_format_repaired"
        return TakeGroupingProviderResult(
            tuple(normalized),
            ProviderStatus("openai", True, True, "applied"),
            reason,
        )
