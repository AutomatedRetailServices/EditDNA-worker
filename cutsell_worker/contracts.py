"""Versioned contracts shared by the CutSell clean worker stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = "cutsell.v1"


class JobState(str, Enum):
    PREPARING = "preparing"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    COMPOSING = "composing"
    DRAFT_READY = "draft_ready"
    RENDERING = "rendering"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELED = "canceled"


class SemanticRole(str, Enum):
    HOOK = "HOOK"
    PROBLEM = "PROBLEM"
    FEATURES = "FEATURES"
    BENEFITS = "BENEFITS"
    PROOF = "PROOF"
    STORY = "STORY"
    CTA = "CTA"
    OTHER = "OTHER"


class EditStrategy(str, Enum):
    DIRECT_SALES = "direct_sales"
    STORYTELLING = "storytelling"
    TESTIMONIAL = "testimonial"
    DEMO = "demo_product_led"
    EDUCATIONAL = "educational"
    FACELESS = "faceless_voiceover"
    MIXED = "mixed"


@dataclass(frozen=True)
class SourceAsset:
    source_asset_id: str
    project_id: str
    user_id: str
    original_name: str
    source_order: int
    duration_sec: float
    uri: str
    has_audio: bool = True
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    confidence: Optional[float] = None


@dataclass(frozen=True)
class TranscriptSegment:
    source_asset_id: str
    start: float
    end: float
    text: str
    words: Tuple[Word, ...] = ()


@dataclass(frozen=True)
class MediaSignals:
    source_asset_id: str
    start: float
    end: float
    silence_ratio: float = 0.0
    audio_quality: float = 0.5
    face_visibility: float = 0.5
    eye_contact: float = 0.5
    framing_quality: float = 0.5
    product_visibility: float = 0.0
    motion_stability: float = 0.5
    continuity: float = 0.5
    visual_fumble: float = 0.0


@dataclass(frozen=True)
class CandidateTake:
    clip_id: str
    source_asset_id: str
    source_order: int
    start: float
    end: float
    text: str
    words: Tuple[Word, ...] = ()
    signals: Optional[MediaSignals] = None
    complete_idea: bool = True

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class CleanCutDecision:
    clip_id: str
    keep: bool
    reason: str
    confidence: float


@dataclass(frozen=True)
class SemanticLabel:
    clip_id: str
    role: SemanticRole
    confidence: float
    reason: str = ""


@dataclass(frozen=True)
class RankedTake:
    clip_id: str
    score: float
    reason: str


@dataclass(frozen=True)
class TakeGroup:
    group_id: str
    semantic_key: str
    candidate_ids: Tuple[str, ...]
    ranked: Tuple[RankedTake, ...]
    selected_clip_id: str


@dataclass(frozen=True)
class DraftClip:
    clip_id: str
    source_asset_id: str
    source_order: int
    start: float
    end: float
    text: str
    caption_text: str
    semantic_role: SemanticRole = SemanticRole.OTHER
    take_group_id: Optional[str] = None
    selected: bool = True


@dataclass(frozen=True)
class DraftTimeline:
    schema_version: str
    project_id: str
    strategy: EditStrategy
    selected: Tuple[DraftClip, ...]
    alternates: Tuple[DraftClip, ...]
    discarded: Tuple[DraftClip, ...]
    diagnostics: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessingRequest:
    project_id: str
    user_id: str
    sources: Tuple[SourceAsset, ...]
    preferred_source_order: Tuple[str, ...] = ()
    audio_overlap: bool = False
    language_hint: Optional[str] = None


@dataclass(frozen=True)
class ProcessingResult:
    schema_version: str
    project_id: str
    state: JobState
    draft: DraftTimeline
    stage_status: Dict[str, object]
