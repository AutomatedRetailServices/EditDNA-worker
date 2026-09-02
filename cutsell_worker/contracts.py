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
    expression_naturalness: float = 0.5
    gesture_naturalness: float = 0.5
    delivery_energy: float = 0.5
    distraction_risk: float = 0.0


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
    # D-050A: canonical identity/provenance metadata (see
    # canonical_identity.py's module docstring for the full design note
    # and ID ownership table). Additive/shadow-only -- optional and
    # defaulted to None so every existing construction site stays valid
    # unchanged, and nothing in the active pipeline reads these fields to
    # make an editorial decision yet.
    #   source_span_id -- physical observation identity, minted once per
    #                     raw span in take_segmentation.py. Timestamp-
    #                     sensitive by design (see canonical_identity.py).
    #   attempt_id     -- canonical semantic identity for the delivery
    #                     attempt this candidate represents, minted once
    #                     in attempt_reconstruction.py's `_merge_attempt`
    #                     (covers both fused and singleton-passthrough
    #                     attempts). Content/membership-anchored, never
    #                     timestamp-anchored.
    #   realization_id -- canonical semantic identity for "one specific
    #                     recorded delivery of an idea", minted once in
    #                     pipeline.py immediately before take-grouping.
    #                     Carried forward unchanged by every later
    #                     `dataclasses.replace()` (physical trims/splits)
    #                     -- never independently recomputed downstream.
    source_span_id: Optional[str] = None
    attempt_id: Optional[str] = None
    realization_id: Optional[str] = None

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
    words: Tuple[Word, ...] = ()
    semantic_role: SemanticRole = SemanticRole.OTHER
    take_group_id: Optional[str] = None
    selected: bool = True
    audio_muted: bool = False
    audio_volume: float = 1.0
    # Local face/pose/motion evidence carried through from the CandidateTake
    # this clip was built from (see local_performance.py). Optional and
    # defaulted to None so every existing construction site (serde.py's
    # external-payload deserialization included) stays valid unchanged.
    # Selection-time consumers (e.g. unified_selection_google.py) must treat
    # a missing signals as "no evidence available", never as a zero score.
    signals: Optional[MediaSignals] = None
    # D-036: physical-fragment provenance (Boundary-only). `clip_id` remains
    # the SEMANTIC identity CanonicalEditPlan/FinalEditReviewer/Selection
    # Freeze reason about and must never be mutated to satisfy a downstream
    # physical check. When a Boundary pass (e.g. human_boundary_polish_v5's
    # micro-gap split) divides one already-frozen semantic clip into two or
    # more physical render pieces, EVERY resulting piece must set these so a
    # unique physical identity survives independently of `clip_id`, which
    # may legitimately repeat across siblings:
    #   render_fragment_id       -- unique per physical piece (never reused).
    #   parent_semantic_clip_id  -- the semantic clip_id (pre-split) all
    #                               siblings reconstruct together.
    #   fragment_index/fragment_count -- this piece's position among its
    #                               siblings, in rendered order.
    #   boundary_reason          -- which Boundary operation produced it, for
    #                               observability (e.g.
    #                               "remove_micro_visual_reset_word_gap").
    # All default None/absent: a clip nobody has ever split carries no
    # fragment provenance at all -- `effective_render_fragment_id`/
    # `effective_parent_semantic_clip_id` below fall back to `clip_id`.
    render_fragment_id: Optional[str] = None
    parent_semantic_clip_id: Optional[str] = None
    fragment_index: Optional[int] = None
    fragment_count: Optional[int] = None
    boundary_reason: Optional[str] = None
    # D-050A: canonical identity/provenance metadata (see
    # canonical_identity.py's module docstring for the full design note
    # and ID ownership table). Additive/shadow-only, same convention as
    # D-036 above -- all optional, all defaulted to None, nothing reads
    # these to make an editorial decision yet.
    #   realization_id       -- carried unchanged from the CandidateTake
    #                           this clip was built from (pipeline.py's
    #                           `_draft_clip`); never recomputed here or
    #                           by any later physical split.
    #   semantic_idea_id /
    #   retry_family_id      -- minted from this clip's final (post-
    #                           semantic-equivalence) `take_group_id` --
    #                           D-050A intentionally mints both fields
    #                           identically; see canonical_identity.py.
    #   parent_realization_id -- mirrors `parent_semantic_clip_id`'s own
    #                           pattern exactly: absent on a clip nobody
    #                           has split, set to the pre-split clip's
    #                           `realization_id` by the one physical-split
    #                           site that produced this fragment. The
    #                           fragment's own `realization_id` field is
    #                           NOT changed by a split -- it stays equal to
    #                           the parent's, which is the actual
    #                           "physical split preserves realization
    #                           identity" invariant; this field is the
    #                           explicit, observable marker that a split
    #                           happened at all.
    realization_id: Optional[str] = None
    semantic_idea_id: Optional[str] = None
    retry_family_id: Optional[str] = None
    parent_realization_id: Optional[str] = None


def effective_render_fragment_id(clip) -> str:
    """The clip's physical render identity -- its own explicit
    `render_fragment_id` when a Boundary split minted one, else its
    `clip_id` (an unsplit clip's semantic and physical identity coincide).
    Duck-typed on any object carrying `clip_id`/`render_fragment_id` (a
    `DraftClip` or a `render_plan.RenderSegment`) so this has no import-time
    dependency on either module."""
    explicit = getattr(clip, "render_fragment_id", None)
    return str(explicit) if explicit else str(clip.clip_id)


def effective_parent_semantic_clip_id(clip) -> Optional[str]:
    """The semantic clip this piece is a physical fragment of, or None when
    it carries no fragment provenance at all (never split, or split by code
    that predates D-036 and has not been updated to set this). Returning
    None -- rather than falling back to `clip_id` -- is deliberate: legitimacy
    requires POSITIVE evidence of a real split, not merely two segments that
    happen to share a `clip_id`; see `post_render_watch_listen_qc.
    check_no_duplicate_render_segments`."""
    explicit = getattr(clip, "parent_semantic_clip_id", None)
    return str(explicit) if explicit else None


@dataclass(frozen=True)
class TextOverlay:
    overlay_id: str
    text: str
    start: float
    end: float
    x: float = 0.5
    y: float = 0.2
    scale: float = 1.0


@dataclass(frozen=True)
class MediaOverlay:
    overlay_id: str
    kind: str
    uri: str
    start: float
    end: float
    x: float = 0.5
    y: float = 0.5
    width: float = 0.4
    source_start: float = 0.0
    source_end: Optional[float] = None
    mute_audio: bool = True


@dataclass(frozen=True)
class DraftTimeline:
    schema_version: str
    project_id: str
    strategy: EditStrategy
    selected: Tuple[DraftClip, ...]
    alternates: Tuple[DraftClip, ...]
    discarded: Tuple[DraftClip, ...]
    diagnostics: Dict[str, object] = field(default_factory=dict)
    captions_enabled: bool = True
    caption_preset: str = "classic"
    text_overlays: Tuple[TextOverlay, ...] = ()
    media_overlays: Tuple[MediaOverlay, ...] = ()


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
