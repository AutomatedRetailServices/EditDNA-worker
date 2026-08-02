"""Validated Chat Completions adapters for the repository's OpenAI operations.

Chat Completions remains the default API family because external SDK documentation was
not available during this refactor and changing API families could alter model support.
"""

from dataclasses import dataclass
from enum import Enum
import json
import logging
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .openai_client import (
    OpenAIProviderError, OpenAIResponseValidationError, OpenAITimeoutError,
    create_openai_client,
)

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
