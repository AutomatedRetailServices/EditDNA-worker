"""OpenAI-backed recording-mistake judge for conservative Clean Cut."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .clean_cut_provider import CleanCutJudgement, CleanCutProviderResult
from .contracts import CandidateTake
from .openai_json import parse_json_object
from .providers import ProviderStatus
from .take_grouping import retry_similarity, semantic_key


def _word_count(take: CandidateTake) -> int:
    if take.words:
        return sum(1 for word in take.words if str(word.text or "").strip())
    return len(str(take.text or "").split())


def _is_ambiguous_microtake(
    take: CandidateTake,
    *,
    max_words: int = 5,
    max_duration_sec: float = 3.0,
) -> bool:
    return 0 < _word_count(take) <= max_words and 0.0 < take.duration_sec <= max_duration_sec


def _prefix_relation(fragment: CandidateTake, reference: CandidateTake | None) -> bool:
    if reference is None:
        return False
    fragment_tokens = semantic_key(fragment.text).split()
    reference_tokens = semantic_key(reference.text).split()
    return bool(fragment_tokens and reference_tokens[: len(fragment_tokens)] == fragment_tokens)


def _gap(left: CandidateTake | None, right: CandidateTake | None) -> float | None:
    if left is None or right is None or left.source_asset_id != right.source_asset_id:
        return None
    return round(max(0.0, right.start - left.end), 3)


def _word_confidence_summary(take: CandidateTake) -> tuple[float | None, float | None]:
    values = [float(word.confidence) for word in take.words if word.confidence is not None]
    if not values:
        return None, None
    return round(sum(values) / len(values), 4), round(min(values), 4)


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
        target_indexes = [index for index, take in enumerate(takes) if _is_ambiguous_microtake(take)]
        target_ids = {takes[index].clip_id for index in target_indexes}

        # The provider has zero authority over non-target speech. Return explicit KEEP
        # judgements for it so the generic provider boundary still receives every
        # candidate exactly once, without sending long/normal takes to the model.
        if not target_indexes:
            return CleanCutProviderResult(
                tuple(
                    CleanCutJudgement(take.clip_id, "keep", 1.0, "not_ambiguous_microtake")
                    for take in takes
                ),
                ProviderStatus("openai", True, True, "applied", "no_ambiguous_microtakes"),
            )

        evidence = []
        for index in target_indexes:
            take = takes[index]
            previous = (
                takes[index - 1]
                if index > 0 and takes[index - 1].source_asset_id == take.source_asset_id
                else None
            )
            following = (
                takes[index + 1]
                if index + 1 < len(takes) and takes[index + 1].source_asset_id == take.source_asset_id
                else None
            )
            avg_confidence, min_confidence = _word_confidence_summary(take)
            signals = take.signals
            evidence.append({
                "id": take.clip_id,
                "transcript": take.text,
                "duration_sec": round(take.duration_sec, 3),
                "word_count": _word_count(take),
                "complete_idea": bool(take.complete_idea),
                "asr_avg_confidence": avg_confidence,
                "asr_min_confidence": min_confidence,
                "previous_transcript": previous.text if previous is not None else None,
                "next_transcript": following.text if following is not None else None,
                "gap_from_previous_sec": _gap(previous, take),
                "gap_to_next_sec": _gap(take, following),
                "similarity_to_previous": (
                    round(retry_similarity(take.text, previous.text), 4)
                    if previous is not None else None
                ),
                "similarity_to_next": (
                    round(retry_similarity(take.text, following.text), 4)
                    if following is not None else None
                ),
                "exact_prefix_of_previous": _prefix_relation(take, previous),
                "exact_prefix_of_next": _prefix_relation(take, following),
                "words": [
                    {
                        "index": word_index,
                        "text": word.text,
                        "start": round(float(word.start), 3),
                        "end": round(float(word.end), 3),
                        "confidence": (
                            round(float(word.confidence), 4)
                            if word.confidence is not None else None
                        ),
                    }
                    for word_index, word in enumerate(take.words)
                ],
                "signals": ({
                    "audio_quality": signals.audio_quality,
                    "silence_ratio": signals.silence_ratio,
                    "motion_stability": signals.motion_stability,
                    "continuity": signals.continuity,
                    "visual_fumble": signals.visual_fumble,
                } if signals is not None else {}),
            })

        instruction = (
            "You are CutSell Clean Cut Judge. The listed takes are ONLY short ambiguous microtakes that survived "
            "deterministic cleanup. Your ONLY job is to distinguish valid creator speech from obvious recording "
            "mistakes. previous_transcript and next_transcript are READ-ONLY context, not candidates. You have no "
            "authority to judge sales quality, hook quality, commercial role, usefulness, style, profanity, or "
            "whether a sentence is persuasive. For every listed candidate return exactly one action: KEEP, DELETE, "
            "or MIXED. DELETE only when the WHOLE candidate is clearly unusable production speech such as an "
            "explicit restart/stop direction, abandoned gibberish, repeated false start, or a fragment that is "
            "structurally shown to be an abandoned retry by the supplied neighbor timing/similarity/prefix evidence. "
            "Grammatical incompleteness by itself is NOT enough to DELETE. A lone pronoun, preposition, noun, or "
            "short phrase must be KEPT unless the supplied context makes the failed-retry/blooper relationship clear. "
            "A short intentional reaction such as Yeah, No, Bye, What just happened, laughter-related speech, "
            "profanity, or emotion is valid and must be KEPT when it makes sense in neighboring context. ASR "
            "confidence can support uncertainty but low ASR confidence alone is never a delete reason. MIXED when "
            "one candidate contains both an obvious blooper/restart portion and a single contiguous span of valid "
            "creator speech. For MIXED, if and only if the provided word list makes the valid span unambiguous, "
            "return inclusive keep_start_word_index and keep_end_word_index using ONLY supplied indexes. Never "
            "invent timestamps or words. If the valid speech is not one contiguous span, or the boundary is uncertain, "
            "leave both indexes null. MIXED must never imply deleting the whole take. CONFIDENCE means certainty that "
            "the requested action is safe for automatic editing. If DELETE is strongly supported by structural/context "
            "evidence, use confidence >=0.95. If DELETE is plausible but not safe enough to auto-apply, use <=0.93. "
            "When uncertain choose KEEP. Return JSON only: "
            "{\"judgements\":[{\"id\":...,\"action\":\"keep|delete|mixed\",\"confidence\":0..1,"
            "\"reason\":...,\"keep_start_word_index\":null|int,\"keep_end_word_index\":null|int}]}. "
            "Include every LISTED candidate exactly once."
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

        target_judgements: dict[str, CleanCutJudgement] = {}
        for item in items:
            clip_id = str(item.get("id") or "")
            if clip_id not in target_ids or clip_id in target_judgements:
                raise ValueError("clean cut judge returned invalid target id")
            start_index = item.get("keep_start_word_index")
            end_index = item.get("keep_end_word_index")
            target_judgements[clip_id] = CleanCutJudgement(
                clip_id=clip_id,
                action=str(item.get("action") or "").lower(),
                confidence=float(item.get("confidence")),
                reason=str(item.get("reason") or ""),
                keep_start_word_index=(int(start_index) if start_index is not None else None),
                keep_end_word_index=(int(end_index) if end_index is not None else None),
            )
        if set(target_judgements) != target_ids:
            raise ValueError("clean cut judge omitted ambiguous microtake")

        judgements = []
        for take in takes:
            if take.clip_id in target_judgements:
                judgements.append(target_judgements[take.clip_id])
            else:
                judgements.append(CleanCutJudgement(
                    take.clip_id,
                    "keep",
                    1.0,
                    "not_ambiguous_microtake",
                ))
        return CleanCutProviderResult(
            tuple(judgements),
            ProviderStatus("openai", True, True, "applied", "selective_microtake_review"),
        )
