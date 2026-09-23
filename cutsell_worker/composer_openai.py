"""OpenAI-backed flexible composer for CutSell postable drafts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Tuple

from .composer_provider import ComposerProviderResult
from .contracts import CandidateTake, EditStrategy, SemanticLabel
from .openai_json import parse_json_object
from .providers import ProviderStatus


@dataclass
class OpenAIComposerProvider:
    model: str = "gpt-4o-mini"
    client_factory: Callable[[], object] | None = None

    def _client(self):
        if self.client_factory is not None:
            return self.client_factory()
        from openai import OpenAI
        return OpenAI()

    def order(
        self,
        takes: Tuple[CandidateTake, ...],
        labels: Tuple[SemanticLabel, ...],
        strategy: EditStrategy,
        context_text: str = "",
    ) -> ComposerProviderResult:
        label_map = {item.clip_id: item for item in labels}
        clips = []
        for index, take in enumerate(takes):
            label = label_map.get(take.clip_id)
            signals = take.signals
            clips.append({
                "id": take.clip_id,
                "natural_index": index,
                "source_order": take.source_order,
                "start": take.start,
                "end": take.end,
                "text": take.text,
                "role": label.role.value if label is not None else "OTHER",
                "role_confidence": label.confidence if label is not None else 0.0,
                "visual": ({
                    "eye_contact": signals.eye_contact,
                    "product_visibility": signals.product_visibility,
                    "continuity": signals.continuity,
                    "expression_naturalness": signals.expression_naturalness,
                    "gesture_naturalness": signals.gesture_naturalness,
                    "delivery_energy": signals.delivery_energy,
                    "distraction_risk": signals.distraction_risk,
                } if signals is not None else {}),
            })

        instruction = (
            "You are CutSell's global story composer. The whole-video context was produced BEFORE editing and tells you whether the footage is "
            "sales, natural, or mixed. Respect that routing. For SALES footage, build the strongest coherent product sales story available in the real "
            "footage: visual/verbal/combined hook, product context, demo, problem, feature, benefit, proof, reaction, objection, result, CTA or other beats "
            "only when they actually exist. Do NOT force HOOK->PROBLEM->BENEFIT->PROOF->CTA. For NATURAL footage (storytime, yapping, talking-head, "
            "routine/lifestyle, commentary, education, vlog), do not invent a sales objective; preserve personality while creating a clear topic/story "
            "with setup, development and payoff/conclusion when present. Engaging yapping/story-building detail may stay; redundant repetition or tangents "
            "that kill momentum should not be favored. For MIXED footage, preserve the natural story and integrate genuine sales moments without making it "
            "feel artificially commercial. Prefer natural source order unless a reorder clearly improves comprehension, hook strength, story logic, demo flow "
            "or payoff. A silent visual hook/reaction can matter if represented by a valid clip. Never invent speech or claims, merge sentences, duplicate clips, "
            "or change clip contents. This provider only orders the already-selected clips, so include every input id exactly once. Use continuity, camera "
            "engagement, facial/body naturalness and delivery evidence to avoid awkward jumps. Return JSON only as "
            "{\"ordered_clip_ids\":[...],\"reason\":\"...\"}."
        )
        response = self._client().responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps({
                    "strategy": strategy.value,
                    "whole_video_context": context_text[:20000],
                    "clips": clips,
                }, ensure_ascii=False)},
            ],
        )
        data = parse_json_object(response.output_text)
        ordered = data.get("ordered_clip_ids")
        if not isinstance(ordered, list):
            raise ValueError("composer returned invalid ordered_clip_ids")
        return ComposerProviderResult(
            tuple(str(item) for item in ordered),
            ProviderStatus("openai", True, True, "applied"),
            str(data.get("reason") or "")[:500],
        )
