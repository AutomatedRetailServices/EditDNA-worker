"""OpenAI-backed flexible composer for CutSell sales/UGC drafts."""
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
            "You are CutSell's sales-aware flexible video composer. Order the provided already-valid clips into the most coherent, "
            "natural TikTok Shop/UGC sales edit for the detected strategy. Preserve the creator's intent and prefer natural source order "
            "unless a reorder clearly improves comprehension, opening strength, demonstration flow, payoff, or sales coherence. "
            "Do NOT force HOOK->PROBLEM->BENEFIT->PROOF->CTA. Roles are optional/repeatable; storytelling may dominate; CTA need not be last. "
            "Never invent speech, never merge sentences, never remove a clip, never duplicate a clip, and never change clip contents. "
            "Use continuity and delivery signals to avoid awkward jumps. Return JSON only as "
            "{\"ordered_clip_ids\":[...],\"reason\":\"...\"}. Include every input id exactly once."
        )
        response = self._client().responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps({"strategy": strategy.value, "clips": clips}, ensure_ascii=False)},
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
