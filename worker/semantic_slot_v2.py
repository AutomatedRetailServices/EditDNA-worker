"""Conservative, context-aware semantic slot classification primitives."""

from dataclasses import asdict, dataclass
from enum import Enum
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class CanonicalSlot(str, Enum):
    HOOK = "HOOK"
    PROBLEM = "PROBLEM"
    BENEFIT = "BENEFIT"
    FEATURES = "FEATURES"
    PROOF = "PROOF"
    STORY = "STORY"
    CTA = "CTA"
    OTHER = "OTHER"


class EvidenceTag(str, Enum):
    QUESTION_HOOK = "question_hook"
    BOLD_CLAIM = "bold_claim"
    PAIN_POINT = "pain_point"
    DESIRED_OUTCOME = "desired_outcome"
    PRODUCT_ATTRIBUTE = "product_attribute"
    PRODUCT_MECHANISM = "product_mechanism"
    PRACTICAL_BENEFIT = "practical_benefit"
    PERSONAL_EXPERIENCE = "personal_experience"
    SOCIAL_PROOF = "social_proof"
    DEMONSTRATION = "demonstration"
    MEASURABLE_RESULT = "measurable_result"
    OBJECTION_HANDLING = "objection_handling"
    URGENCY = "urgency"
    DIRECT_ACTION = "direct_action"
    INCOMPLETE_CONTEXT = "incomplete_context"
    NON_SALES_CONTENT = "non_sales_content"


@dataclass(frozen=True)
class SlotClassificationResult:
    primary_slot: CanonicalSlot
    secondary_slot: Optional[CanonicalSlot]
    confidence: float
    secondary_confidence: Optional[float]
    completeness: float
    sales_relevance: float
    standalone_quality: float
    abstain: bool
    reason: str
    evidence_tags: Tuple[EvidenceTag, ...]


@dataclass(frozen=True)
class ClauseSignals:
    word_count: int
    question_mark: bool
    imperative_verb: bool
    first_person_narrative: bool
    measurable_number: bool
    product_attribute: bool
    direct_action_phrase: bool
    incomplete_phrase: bool
    production_talk: bool
    heuristic_slot: str


@dataclass(frozen=True)
class SemanticClauseInput:
    clause_id: str
    transcript: str
    preceding_transcript: Optional[str]
    following_transcript: Optional[str]
    duration: float
    word_count: int
    sentence_completeness: float
    heuristic_slot: str
    semantic_score: Optional[float]
    signals: ClauseSignals

    def provider_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        return value


_IMPERATIVES = frozenset(("buy", "click", "tap", "try", "order", "shop", "get", "grab", "check", "add"))
_ACTIONS = ("click the link", "tap the link", "shop now", "get yours", "add to cart", "check the link", "order now")
_ATTRIBUTES = ("contains", "includes", "made with", "comes with", "ingredient", "ceramic", "plate", "formula", "feature")
_PRODUCTION = ("cut that", "take two", "start over", "say that again", "camera", "mic check", "rolling")
_DEPENDENT = ("and", "but", "because", "so", "which", "that", "also", "then")


def derive_signals(text: str, heuristic_slot: str) -> ClauseSignals:
    normalized = " ".join((text or "").strip().lower().split())
    words = re.findall(r"[a-z0-9']+", normalized)
    first = words[0] if words else ""
    incomplete = not words or (len(words) <= 4 and first in _DEPENDENT) or normalized.endswith((" and", " but", " because", " so"))
    return ClauseSignals(
        word_count=len(words),
        question_mark="?" in text,
        imperative_verb=first in _IMPERATIVES,
        first_person_narrative=bool(re.search(r"\b(i|i'm|i've|my|we|our)\b", normalized)),
        measurable_number=bool(re.search(r"\b\d+(?:\.\d+)?(?:%|\s*(?:days?|weeks?|inches?|lbs?|pounds?))?\b", normalized)),
        product_attribute=any(phrase in normalized for phrase in _ATTRIBUTES),
        direct_action_phrase=any(phrase in normalized for phrase in _ACTIONS),
        incomplete_phrase=incomplete,
        production_talk=any(phrase in normalized for phrase in _PRODUCTION),
        heuristic_slot=heuristic_slot,
    )


def sentence_completeness(text: str, signals: ClauseSignals) -> float:
    if not text.strip():
        return 0.0
    if signals.incomplete_phrase:
        return 0.25
    if signals.word_count < 3:
        return 0.45
    return 1.0 if text.rstrip().endswith((".", "!", "?")) else 0.8


def source_index(clip: Mapping[str, Any]) -> Any:
    return clip.get("source_index", clip.get("meta", {}).get("source_index", 0))


def build_clause_inputs(clips: Sequence[Mapping[str, Any]]) -> Tuple[SemanticClauseInput, ...]:
    result = []
    for index, clip in enumerate(clips):
        text = str(clip.get("text") or "")
        heuristic = str(clip.get("slot") or "OTHER")
        signals = derive_signals(text, heuristic)
        same_previous = index > 0 and source_index(clips[index - 1]) == source_index(clip)
        same_following = index + 1 < len(clips) and source_index(clips[index + 1]) == source_index(clip)
        result.append(SemanticClauseInput(
            clause_id=str(clip.get("id")), transcript=text,
            preceding_transcript=str(clips[index - 1].get("text") or "") if same_previous else None,
            following_transcript=str(clips[index + 1].get("text") or "") if same_following else None,
            duration=max(0.0, float(clip.get("end", 0.0)) - float(clip.get("start", 0.0))),
            word_count=signals.word_count, sentence_completeness=sentence_completeness(text, signals),
            heuristic_slot=heuristic,
            semantic_score=float(clip["semantic_score"]) if clip.get("semantic_score") is not None else None,
            signals=signals,
        ))
    return tuple(result)
