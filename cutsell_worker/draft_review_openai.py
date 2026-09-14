"""OpenAI-backed global review of an assembled CutSell draft."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Tuple

from .contracts import CandidateTake, EditStrategy, SemanticLabel
from .draft_review_provider import DraftReviewResult
from .openai_json import parse_json_object
from .providers import ProviderStatus


@dataclass
class OpenAIDraftReviewProvider:
    model: str = "gpt-4o-mini"
    client_factory: Callable[[], object] | None = None

    def _client(self):
        if self.client_factory is not None:
            return self.client_factory()
        from openai import OpenAI
        return OpenAI()

    def review(
        self,
        takes: Tuple[CandidateTake, ...],
        labels: Tuple[SemanticLabel, ...],
        strategy: EditStrategy,
        context_text: str = "",
    ) -> DraftReviewResult:
        label_map = {label.clip_id: label for label in labels}
        clips = []
        for index, take in enumerate(takes):
            signals = take.signals
            label = label_map.get(take.clip_id)
            clips.append({
                "id": take.clip_id,
                "draft_index": index,
                "source_asset_id": take.source_asset_id,
                "start": take.start,
                "end": take.end,
                "text": take.text,
                "role": label.role.value if label is not None else "OTHER",
                "visual": ({
                    "eye_contact": signals.eye_contact,
                    "continuity": signals.continuity,
                    "visual_fumble": signals.visual_fumble,
                    "expression_naturalness": signals.expression_naturalness,
                    "gesture_naturalness": signals.gesture_naturalness,
                    "delivery_energy": signals.delivery_energy,
                    "distraction_risk": signals.distraction_risk,
                    "product_visibility": signals.product_visibility,
                } if signals is not None else {}),
            })

        instruction = (
            "You are CutSell's FINAL GLOBAL STORY REVIEWER. Review the complete proposed edit after cleanup, retry grouping, Best Take and composition. "
            "Use the whole-video context and the ordered selected clips as one continuous story. Decide whether a competent short-form editor could post "
            "this draft without repairing its LOGIC. For sales footage, judge product/sales-story coherence, hook relevance, progression, redundancy, demo/proof/"
            "reaction/CTA placement when those beats actually exist, and whether the edit feels natural rather than forced into a funnel. For natural footage, "
            "judge topic/story coherence, setup/development/payoff when present, engaging personality, momentum and unnecessary repetition/tangents without "
            "forcing a sales objective. In all modes, watch for repeated ideas, incoherent jumps, a clearly weaker duplicate take that survived, or a clip whose "
            "performance evidence makes it unsuitable. Do not punish intentional pauses/personality/humor that belong to the story. "
            "You MAY conservatively remove redundant or incoherent clips and MAY reorder existing selected clips when that clearly fixes the story. "
            "You may NOT add an unknown clip, duplicate a clip, rewrite speech, invent a claim, or invent missing footage. If the existing material cannot form a "
            "fully postable story, preserve the strongest coherent version and set postable=false with concise issues. "
            "Return JSON only as {\"ordered_clip_ids\":[...],\"postable\":true|false,\"issues\":[...],\"reason\":\"...\"}."
        )
        response = self._client().responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps({
                    "strategy": strategy.value,
                    "whole_video_context": context_text[:20000],
                    "proposed_edit": clips,
                }, ensure_ascii=False)},
            ],
        )
        data = parse_json_object(response.output_text)
        ordered = data.get("ordered_clip_ids")
        issues = data.get("issues") or []
        if not isinstance(ordered, list) or not isinstance(issues, list):
            raise ValueError("draft reviewer returned invalid payload")
        return DraftReviewResult(
            ordered_clip_ids=tuple(str(item) for item in ordered),
            postable=bool(data.get("postable", False)),
            issues=tuple(str(item) for item in issues),
            reason=str(data.get("reason") or ""),
            status=ProviderStatus("openai", True, True, "applied"),
        )
