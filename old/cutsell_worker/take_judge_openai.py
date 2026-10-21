"""OpenAI-backed Best Take ranking using Watch + Listen signals."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Tuple

from .contracts import CandidateTake, RankedTake
from .openai_json import parse_json_object
from .providers import ProviderStatus
from .take_judge import rank_takes
from .take_judge_provider import TakeJudgeProviderResult


def _safe_score(value: object, *, baseline_score: float) -> tuple[float, bool]:
    """Return a bounded model score, or the deterministic baseline when malformed."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return baseline_score, False
    if not 0.0 <= score <= 1.0:
        return baseline_score, False
    return score, True


def _identity_error(data: dict, expected: set[str]) -> str | None:
    items = data.get("ranked")
    if not isinstance(items, list):
        return "take judge returned invalid payload"
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            return "take judge returned non-object ranking item"
        clip_id = str(item.get("id") or "")
        if clip_id not in expected or clip_id in seen:
            return "take judge returned invalid clip id"
        seen.add(clip_id)
    if seen != expected:
        return "take judge omitted candidates"
    return None


@dataclass
class OpenAITakeJudgeProvider:
    model: str = "gpt-4o-mini"
    client_factory: Callable[[], object] | None = None

    def _client(self):
        if self.client_factory is not None:
            return self.client_factory()
        from openai import OpenAI
        return OpenAI()

    def _parse_or_repair_json(self, client: object, output_text: str) -> tuple[dict, bool]:
        """Parse Best Take JSON; on syntax failure, request one format-only repair."""
        try:
            return parse_json_object(output_text), False
        except Exception:
            repair = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Repair the following malformed Best Take ranking into valid JSON only. "
                            "Do not add, remove, rename, rerank, rescore, or reinterpret candidates. "
                            "Preserve the original ranking intent, clip ids, scores and reasons exactly. "
                            "Return only {\"ranked\":[{\"id\":...,\"score\":0..1,\"reason\":...}]}."
                        ),
                    },
                    {"role": "user", "content": str(output_text)[:30000]},
                ],
            )
            return parse_json_object(repair.output_text), True

    def _repair_candidate_identities(
        self,
        client: object,
        *,
        output_text: str,
        data: dict,
        expected: set[str],
    ) -> dict:
        """Request one constrained identity repair without changing ranking judgement.

        The repair model may only map malformed/foreign ids onto the exact candidate ids
        that were supplied to the original judge. It must preserve ordering, scores and
        reasons. The result is validated strictly afterwards; no local fuzzy remapping is
        allowed.
        """
        repair = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Repair ONLY candidate identity fields in this Best Take JSON. "
                        "Allowed candidate ids are supplied separately. Return every allowed id exactly once. "
                        "Do not rerank, rescore, rewrite reasons, add candidates, remove candidates, or infer a new judgement. "
                        "Preserve item order, scores and reasons from the original ranking as closely as possible; only correct id values "
                        "that are invalid, duplicated, or missing. If you cannot map identities unambiguously, preserve the uncertainty in "
                        "the JSON rather than inventing a new ranking. Return JSON only as "
                        "{\"ranked\":[{\"id\":...,\"score\":0..1,\"reason\":...}]} ."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "allowed_candidate_ids": sorted(expected),
                            "parsed_ranking": data,
                            "original_output": str(output_text)[:30000],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        return parse_json_object(repair.output_text)

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
                    "expression_naturalness": signals.expression_naturalness,
                    "gesture_naturalness": signals.gesture_naturalness,
                    "delivery_energy": signals.delivery_energy,
                    "distraction_risk": signals.distraction_risk,
                } if signals is not None else {}),
            })
        instruction = (
            "Rank alternate recordings of the same underlying idea from strongest to weakest like a careful human short-form editor. "
            "The content may be sales, storytime, yapping, talking-head, routine/lifestyle, commentary or education. Use combined Watch + Listen evidence. "
            "Prefer complete and clear speech, confidence, authentic pacing, camera engagement, useful eye contact, natural facial expression, body/gesture "
            "congruency with the words, clean audio, appropriate energy, stable framing and low visible fumble/distraction. When two takes express the same idea, "
            "lexical clarity and reaching a clear semantic endpoint matter more than simply being longer or more energetic: prefer the take that preserves the "
            "specific noun, object, action, diagnosis, product name, claim or conclusion instead of trailing off, repeating a broken word, or ending in vague/garbled "
            "wording. Do not infer facts that are not spoken; compare only the recordings provided. A grammatically complete line is NOT automatically a good "
            "take: frustration, breaking character, accidental laughter, searching for words, a body reset or a visible 'I got it wrong' reaction can make it a "
            "failed take. For genuine product footage, also consider product presentation and sales effectiveness, but never penalize non-sales content for lacking "
            "a product. Do not reward exaggerated performance merely for energy or duration. Preserve authentic personality and intentional humor/reactions that "
            "belong to the content rather than the recording process. Do not delete any candidate. Return JSON only as "
            "{\"ranked\":[{\"id\":...,\"score\":0..1,\"reason\":...}]}. Include every candidate exactly once."
        )
        client = self._client()
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps({"takes": evidence}, ensure_ascii=False)},
            ],
        )
        data, repaired_json = self._parse_or_repair_json(client, response.output_text)
        expected = {take.clip_id for take in takes}
        identity_repaired = False
        identity_error = _identity_error(data, expected)
        if identity_error is not None:
            data = self._repair_candidate_identities(
                client,
                output_text=response.output_text,
                data=data,
                expected=expected,
            )
            identity_repaired = True
            identity_error = _identity_error(data, expected)
            if identity_error is not None:
                raise ValueError(identity_error)

        items = data.get("ranked")
        if not isinstance(items, list):
            raise ValueError("take judge returned invalid payload")
        baseline_by_id = {item.clip_id: item for item in rank_takes(takes)}
        seen = set()
        ranked = []
        score_fallback_count = 0
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("take judge returned non-object ranking item")
            clip_id = str(item.get("id") or "")
            if clip_id not in expected or clip_id in seen:
                raise ValueError("take judge returned invalid clip id")
            baseline = baseline_by_id[clip_id]
            score, valid_score = _safe_score(item.get("score"), baseline_score=baseline.score)
            reason = str(item.get("reason") or "")[:240]
            if not valid_score:
                score_fallback_count += 1
                reason = f"{reason} [malformed score: deterministic baseline used]".strip()[:240]
            elif (repaired_json or identity_repaired) and not reason:
                reason = "provider_output_repaired"
            ranked.append(RankedTake(clip_id, score, reason))
            seen.add(clip_id)
        if seen != expected:
            raise ValueError("take judge omitted candidates")
        ranked.sort(key=lambda item: (-item.score, item.clip_id))
        status_bits = []
        if repaired_json:
            status_bits.append("json_format_repaired")
        if identity_repaired:
            status_bits.append("candidate_ids_repaired")
        if score_fallback_count:
            status_bits.append(f"score_fallback:{score_fallback_count}")
        return TakeJudgeProviderResult(
            tuple(ranked),
            ProviderStatus("openai", True, True, "applied", ",".join(status_bits)),
        )
