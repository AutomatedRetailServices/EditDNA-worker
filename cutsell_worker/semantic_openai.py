"""OpenAI-backed commercial meaning provider for the clean CutSell worker."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Tuple

from .contracts import CandidateTake, SemanticLabel, SemanticRole
from .openai_json import parse_json_object
from .providers import ProviderStatus, SemanticProviderResult


@dataclass
class OpenAISemanticProvider:
    model: str = "gpt-4o-mini"
    client_factory: Callable[[], object] | None = None

    def _client(self):
        if self.client_factory is not None:
            return self.client_factory()
        from openai import OpenAI
        return OpenAI()

    def classify(self, takes: Tuple[CandidateTake, ...]) -> SemanticProviderResult:
        if not takes:
            return SemanticProviderResult(
                labels=(),
                status=ProviderStatus("openai", True, True, "empty_input"),
            )

        payload = {
            "clips": [
                {
                    "id": take.clip_id,
                    "text": take.text,
                    "source_asset_id": take.source_asset_id,
                }
                for take in takes
            ]
        }
        instruction = (
            "Classify each clip by its own primary commercial function. Allowed roles: "
            "HOOK, PROBLEM, FEATURES, BENEFITS, PROOF, STORY, CTA, OTHER. "
            "Roles are descriptive only and must never imply deletion. "
            "Return JSON only as {\"clips\":[{\"id\":...,\"role\":...,\"confidence\":0..1,\"reason\":...}]}. "
            "Include every input clip exactly once."
        )
        response = self._client().responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = parse_json_object(response.output_text)
        items = data.get("clips")
        if not isinstance(items, list):
            raise ValueError("semantic provider returned invalid payload")

        known = {take.clip_id for take in takes}
        labels = []
        seen = set()
        for item in items:
            clip_id = str(item.get("id") or "")
            if clip_id not in known or clip_id in seen:
                raise ValueError("semantic provider returned invalid clip id")
            role = SemanticRole(str(item.get("role")))
            confidence = float(item.get("confidence"))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("semantic confidence outside 0..1")
            reason = str(item.get("reason") or "")[:240]
            labels.append(SemanticLabel(clip_id, role, confidence, reason))
            seen.add(clip_id)
        if seen != known:
            raise ValueError("semantic provider omitted clips")
        return SemanticProviderResult(
            labels=tuple(labels),
            status=ProviderStatus("openai", True, True, "applied"),
        )
