"""Validated Chat Completions adapters for the repository's OpenAI operations.

Chat Completions remains the default API family because external SDK documentation was
not available during this refactor and changing API families could alter model support.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import json
import logging
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .openai_client import (
    OpenAIProviderError, OpenAIResponseValidationError, OpenAITimeoutError,
    create_openai_client,
)
from worker.take_judge_v2 import (
    TakeJudgeCandidate, TakeJudgeCandidateScore, TakeJudgeV2Result, TemporalFrameSample,
)
from worker.semantic_slot_v2 import CanonicalSlot, EvidenceTag, SemanticClauseInput, SlotClassificationResult

logger = logging.getLogger("editdna.openai_provider")


class Slot(str, Enum):
    HOOK = "HOOK"; STORY = "STORY"; PROBLEM = "PROBLEM"; BENEFITS = "BENEFITS"
    FEATURES = "FEATURES"; PROOF = "PROOF"; CTA = "CTA"


class Verdict(str, Enum):
    GOOD = "GOOD"; BAD = "BAD"


@dataclass(frozen=True)
class SemanticResult:
    clip_id: str; slot: Slot; keep: bool; semantic_score: float; reason: str


@dataclass(frozen=True)
class TakeJudgeResult:
    winner_id: str; scores: Mapping[str, float]


@dataclass(frozen=True)
class BoundaryResult:
    HEAD: Verdict; MID: Verdict; TAIL: Verdict


def _json(raw: str) -> Any:
    text = (raw or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
            if text.startswith("json"):
                text = text[4:].lstrip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise OpenAIResponseValidationError("OpenAI response failed validation")
    try:
        return json.loads(text[start:end + 1])
    except (TypeError, ValueError) as exc:
        raise OpenAIResponseValidationError("OpenAI response failed validation") from exc


def _score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenAIResponseValidationError("OpenAI response failed validation")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise OpenAIResponseValidationError("OpenAI response failed validation")
    return result


def _content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError
        return content
    except Exception as exc:
        raise OpenAIResponseValidationError("OpenAI response failed validation") from exc


def _chat(operation: str, model: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
    started = time.monotonic()
    try:
        response = create_openai_client().chat.completions.create(model=model, messages=messages, **kwargs)
        content = _content(response)
    except OpenAIResponseValidationError:
        raise
    except Exception as exc:
        # Avoid interpolating SDK exceptions: they can contain request/provider content.
        if exc.__class__.__name__ in ("APITimeoutError", "Timeout"):
            error: OpenAIProviderError = OpenAITimeoutError("OpenAI operation timed out")
        else:
            error = OpenAIProviderError("OpenAI operation failed")
        logger.warning("OpenAI operation failed", extra={"operation": operation, "model": model,
                       "elapsed_ms": int((time.monotonic() - started) * 1000),
                       "error_category": error.__class__.__name__})
        raise error from exc
    logger.info("OpenAI operation completed", extra={"operation": operation, "model": model,
                "elapsed_ms": int((time.monotonic() - started) * 1000), "validation": "pending"})
    return content


def classify_semantic(model: str, messages: Sequence[Mapping[str, Any]], clip_ids: Iterable[str], **kwargs: Any) -> Dict[str, SemanticResult]:
    known = set(clip_ids)
    data = _json(_chat("semantic_classification", model, messages, **kwargs))
    results: Dict[str, SemanticResult] = {}
    for item in data.get("clips", []):
        if not isinstance(item, dict) or item.get("id") not in known:
            continue
        try:
            cid = item["id"]
            slot = Slot(item["slot"])
            if type(item.get("keep")) is not bool:
                raise OpenAIResponseValidationError("OpenAI response failed validation")
            reason = item.get("reason", "")
            if not isinstance(reason, str):
                raise OpenAIResponseValidationError("OpenAI response failed validation")
            results[cid] = SemanticResult(cid, slot, item["keep"], _score(item.get("semantic_score")), reason[:240])
        except (KeyError, ValueError, OpenAIResponseValidationError) as exc:
            raise OpenAIResponseValidationError("OpenAI response failed validation") from exc
    if not results:
        raise OpenAIResponseValidationError("OpenAI response failed validation")
    return results


def classify_semantic_v2(model: str, clauses: Sequence[SemanticClauseInput], **kwargs: Any) -> Dict[str, SlotClassificationResult]:
    """Classify target clauses using only privacy-reviewed structured context."""
    ids = [clause.clause_id for clause in clauses]
    if not ids or len(ids) != len(set(ids)):
        raise OpenAIResponseValidationError("OpenAI response failed validation")
    definitions = {
        "HOOK": "earns opening attention through a question, surprising claim, curiosity gap, result, or audience callout",
        "PROBLEM": "pain, frustration, risk, objection, failed alternative, or undesirable current state",
        "FEATURES": "what the product is, contains, includes, does mechanically, or how it works",
        "BENEFIT": "user outcome or practical value created by the product",
        "PROOF": "demonstration, before/after, measurable result, testimonial, social proof, or observed outcome",
        "STORY": "personal narrative, experience sequence, background, discovery, or transformation context",
        "CTA": "direct request to buy, click, add to cart, try, order, or check a link",
        "OTHER": "greeting, filler, production talk, unrelated content, or an insufficient fragment",
    }
    allowed_evidence_tags = [tag.value for tag in EvidenceTag]
    instructions = (
        "Classify each target clause's own sales role. Adjacent text is context only; never classify it as part "
        "of the target. Be conservative and abstain for incomplete, poor, tied, insufficient, or non-sales text. "
        "Enthusiasm alone is not a hook. Return JSON only: {results:[{id,primary_slot,secondary_slot|null,"
        "confidence,secondary_confidence|null,completeness,sales_relevance,standalone_quality,abstain,reason,evidence_tags}]}. "
        "Reason must be safe and at most 160 characters. evidence_tags must be a JSON list containing only "
        "values from this exact allowed list, or an empty list when no tag applies: "
        + json.dumps(allowed_evidence_tags)
        + ". Input signal field names are context, not evidence tags; never return arbitrary or invented tags. "
        "Definitions: " + json.dumps(definitions)
    )
    raw = _chat("semantic_classification_v2", model, [
        {"role": "system", "content": [{"type": "text", "text": instructions}]},
        {"role": "user", "content": [{"type": "text", "text": json.dumps({
            "operation": "semantic_classification_v2", "clauses": [item.provider_dict() for item in clauses],
        }, ensure_ascii=False)}]},
    ], **kwargs)
    data = _json(raw)
    try:
        items = data["results"]
        if not isinstance(items, list):
            raise ValueError
        output: Dict[str, SlotClassificationResult] = {}
        for item in items:
            cid = item["id"]
            if cid not in ids or cid in output:
                raise ValueError
            primary = CanonicalSlot(item["primary_slot"])
            secondary = CanonicalSlot(item["secondary_slot"]) if item.get("secondary_slot") is not None else None
            confidence = _score(item["confidence"])
            secondary_confidence = _score(item["secondary_confidence"]) if item.get("secondary_confidence") is not None else None
            if primary == secondary or (secondary is None) != (secondary_confidence is None):
                raise ValueError
            if secondary_confidence is not None and secondary_confidence > confidence:
                raise ValueError
            abstain = item["abstain"]
            reason = item["reason"]
            if (type(abstain) is not bool or not isinstance(reason, str) or len(reason) > 160
                    or reason != reason.strip() or not reason.isprintable()):
                raise ValueError
            tags = tuple(EvidenceTag(tag) for tag in item["evidence_tags"])
            if len(tags) != len(set(tags)):
                raise ValueError
            output[cid] = SlotClassificationResult(
                primary, secondary, confidence, secondary_confidence, _score(item["completeness"]),
                _score(item["sales_relevance"]), _score(item["standalone_quality"]), abstain, reason, tags,
            )
        if set(output) != set(ids):
            raise ValueError
        return output
    except (KeyError, TypeError, ValueError) as exc:
        raise OpenAIResponseValidationError("OpenAI response failed validation") from exc


def judge_takes(model: str, messages: Sequence[Mapping[str, Any]], candidate_ids: Iterable[str], **kwargs: Any) -> TakeJudgeResult:
    known = set(candidate_ids); data = _json(_chat("take_judge", model, messages, **kwargs))
    winner = data.get("winner_id")
    if winner not in known:
        raise OpenAIResponseValidationError("OpenAI response failed validation")
    scores: Dict[str, float] = {}
    for item in data.get("scores", []):
        if not isinstance(item, dict) or item.get("id") not in known:
            raise OpenAIResponseValidationError("OpenAI response failed validation")
        scores[item["id"]] = _score(item.get("score"))
    return TakeJudgeResult(winner, scores)


def judge_takes_v2(
    model: str, slot: str, candidates: Sequence[TakeJudgeCandidate],
    frames: Sequence[TemporalFrameSample], **kwargs: Any,
) -> TakeJudgeV2Result:
    """Compare one sibling group using sanitized structured evidence."""
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise OpenAIResponseValidationError("OpenAI response failed validation")
    frame_map = {sample.candidate_id: sample for sample in frames}
    if set(frame_map) - set(ids):
        raise OpenAIResponseValidationError("OpenAI response failed validation")
    evidence = []
    for candidate in candidates:
        evidence.append({
            "candidate_id": candidate.candidate_id, "transcript": candidate.transcript,
            "duration_sec": candidate.duration_sec, "delivery": asdict(candidate.delivery),
            "frame_timestamps": list(candidate.frame_timestamps), "image_count": candidate.image_count,
        })
    instructions = (
        "Compare only the supplied sibling candidates for the assigned slot. Evaluate spoken clarity, "
        "natural delivery, confidence, eye contact and visible engagement, distracting facial/body "
        "behavior, pacing, completeness, slot suitability, and sales effectiveness without rewarding "
        "exaggerated or artificial performance. Abstain when evidence is weak or effectively tied. "
        "Return JSON only with winner_id (null when abstaining), confidence, abstain, reason, and "
        "candidate_scores containing candidate_id, delivery_score, visual_performance_score, "
        "clarity_score, sales_effectiveness_score, overall_score, and reason. All scores are 0 to 1."
    )
    content: list[Mapping[str, Any]] = [{"type": "text", "text": json.dumps({
        "operation": "take_judge_v2", "slot": slot, "candidate_ids": ids, "candidates": evidence,
    }, ensure_ascii=False)}]
    for candidate_id in ids:
        sample = frame_map.get(candidate_id)
        if sample and sample.image_content:
            content.append({"type": "text", "text": "Temporal frames for candidate " + candidate_id})
            content.extend(sample.image_content)
    raw = _chat("take_judge_v2", model, [
        {"role": "system", "content": [{"type": "text", "text": instructions}]},
        {"role": "user", "content": content},
    ], **kwargs)
    data = _json(raw)
    try:
        abstain = data["abstain"]
        if type(abstain) is not bool:
            raise ValueError
        winner = data.get("winner_id")
        if (not abstain and winner not in ids) or (winner is not None and winner not in ids):
            raise ValueError
        scores = []
        seen = set()
        raw_scores = data["candidate_scores"]
        if not isinstance(raw_scores, list):
            raise ValueError
        for item in raw_scores:
            cid = item["candidate_id"]
            if cid not in ids or cid in seen:
                raise ValueError
            seen.add(cid)
            scores.append(TakeJudgeCandidateScore(
                cid, item["delivery_score"], item["visual_performance_score"], item["clarity_score"],
                item["sales_effectiveness_score"], item["overall_score"], item.get("reason", ""),
            ))
        if seen != set(ids):
            raise ValueError
        if not abstain:
            overall_scores = {score.candidate_id: score.overall_score for score in scores}
            highest = max(overall_scores.values())
            top_ids = {candidate_id for candidate_id, score in overall_scores.items() if score == highest}
            if winner not in top_ids:
                raise ValueError
            if len(top_ids) != 1:
                return TakeJudgeV2Result(None, tuple(scores), data["confidence"], True,
                                         "Top candidate scores are tied")
        return TakeJudgeV2Result(winner, tuple(scores), data["confidence"], abstain, data.get("reason", ""))
    except (KeyError, TypeError, ValueError) as exc:
        raise OpenAIResponseValidationError("OpenAI response failed validation") from exc


def refine_boundaries(model: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> BoundaryResult:
    data = _json(_chat("boundary_refinement", model, messages, **kwargs)); values: Dict[str, Verdict] = {}
    for item in data.get("frames", []):
        try:
            label = item["label"].upper()
            if label not in ("HEAD", "MID", "TAIL") or label in values:
                raise ValueError
            values[label] = Verdict(item["verdict"].upper())
        except (KeyError, AttributeError, ValueError, TypeError) as exc:
            raise OpenAIResponseValidationError("OpenAI response failed validation") from exc
    if set(values) != {"HEAD", "MID", "TAIL"}:
        raise OpenAIResponseValidationError("OpenAI response failed validation")
    return BoundaryResult(values["HEAD"], values["MID"], values["TAIL"])


def detect_bad_take(model: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Verdict:
    raw = _chat("visual_bad_take", model, messages, **kwargs).strip().upper()
    try:
        return Verdict(raw)
    except ValueError as exc:
        raise OpenAIResponseValidationError("OpenAI response failed validation") from exc


def score_multimodal_clause(model: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> Tuple[float, str]:
    data = _json(_chat("multimodal_clause_scoring", model, messages, **kwargs))
    reason = data.get("reason", "no_reason")
    if not isinstance(reason, str):
        raise OpenAIResponseValidationError("OpenAI response failed validation")
    return _score(data.get("score")), reason[:240]
