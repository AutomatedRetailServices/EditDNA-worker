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
    #                     pipeline.py on the complete candidate pool,
    #                     before ANY editorial stage (clean_cut, provider
    #                     judgements, hybrid/composite resolution,
    #                     grouping) can keep, discard, or transform a
    #                     candidate -- D-050D1 relocated this from its
    #                     original post-composite-resolution point after
    #                     an audit found every candidate those upstream
    #                     stages removed never received an identity at
    #                     all. Carried forward unchanged by every later
    #                     `dataclasses.replace()` (physical trims/splits)
    #                     -- never independently recomputed downstream.
    source_span_id: Optional[str] = None
    attempt_id: Optional[str] = None
    realization_id: Optional[str] = None
    # D-235P: additive-only canonical word-membership provenance for this
    # reconstructed attempt -- the exact set of canonical, source-scoped
    # word ordinal positions (`language_spine.LanguageWord.word_index`'s
    # own numbering) this candidate's own `.words` occupy. Same D-050A
    # "shadow field, no consumer yet" precedent as the three ids above:
    # defaulted to `()` so every existing construction site stays valid
    # unchanged, and NOT populated by any live call site in
    # `take_segmentation.py`/`attempt_reconstruction.py`/`pipeline.py` --
    # only `shared_attempt_word_identity.py`'s own OFFLINE derivation
    # helper computes this (fresh, from `.words`, never read back from
    # this field), and only a caller that explicitly attaches it via
    # `dataclasses.replace()` would ever see it non-empty. See
    # `shared_attempt_word_identity.py`'s module docstring for the full
    # design note and docs/CUTSELL_DECISIONS.md D-235P.
    word_indices: Tuple[int, ...] = ()

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
    # D-076: carried unchanged from the CandidateTake this clip was built
    # from (pipeline.py's `_draft_clip`, same passthrough convention as
    # `realization_id` above) -- `source_span_id`/`attempt_id` exist on
    # CandidateTake but were never threaded through to DraftClip before
    # this, so the Semantic Ledger's own `RealizationRecord.source_span_
    # ids`/`.attempt_id` (which already read these via getattr, unchanged)
    # were always empty/None in practice. Additive/shadow-only, same
    # convention as every other D-050A identity field: nothing reads these
    # to make an editorial decision except realization_resolver.py's own
    # PRE_GROUP_SEMANTIC_PRESERVATION candidate-discovery relation check
    # (D-076), which only ever uses them as a strong-relation PROOF
    # REQUIREMENT, never as a ranking signal.
    source_span_id: Optional[str] = None
    attempt_id: Optional[str] = None
    # D-050C1.6 (F5, composite completeness safety): carried unchanged
    # from the CandidateTake this clip was built from (pipeline.py's
    # `_draft_clip`, same passthrough pattern as `realization_id` above).
    # Optional/defaulted so every existing construction site stays valid
    # unchanged; nothing authoritative reads this field today -- it exists
    # so `realization_resolver.py`'s SHADOW composite model can refuse to
    # assemble a composite out of fragments that were never a complete,
    # independently-usable delivery in the first place.
    complete_idea: Optional[bool] = None
    # D-235W: carried unchanged from the CandidateTake this clip was built
    # from (pipeline.py's `_draft_clip`, same passthrough pattern as
    # `source_span_id`/`attempt_id` above) -- see `CandidateTake.word_
    # indices`'s own docstring (D-235P) for the canonical, source-scoped
    # word-ordinal contract. `DraftClip` already carries `.words` (the raw
    # `Word` objects) and `.attempt_id`/`.source_span_id`/`.clip_id`/
    # `.source_asset_id` -- exactly the shape `shared_attempt_word_
    # identity.build_reconstructed_attempt_word_membership()` needs -- so
    # a future caller can compute exact word-membership identity directly
    # from a `DraftClip` without any new field; this one is additive
    # observability/parity with `CandidateTake`, not a hard dependency of
    # that computation. Defaulted to `()` so every existing construction
    # site (serde.py's external-payload deserialization included) stays
    # valid unchanged.
    word_indices: Tuple[int, ...] = ()


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
    # D-235X Part B: the ONE computed-once D-235Q `CompleteLostSemanticAtom
    # Materiality` result per lost atom, keyed by its own `lost_atom_
    # provenance_id` (D-235S) -- never `clip_id` (a clip may carry several
    # lost-atom rows/ordinals; the provenance id is the exact same-atom
    # identity D-235T's own suppression check keys on). Populated by
    # `final_story_coherence_validation.py` ONLY behind the SAME existing
    # `CUTSELL_LOST_ATOM_MATERIALITY_FREEZE_AUTHORITY_ENABLED` flag D-235R/
    # W already gate on -- `{}` (not merely unused) when the flag is off,
    # so flag-off stays byte-identical. Consumed downstream by
    # `repair_loop.run_repair_loop`'s own optional parameter of the same
    # name so D-235R and D-235T read the SAME computed result, never each
    # independently recomputing it. Values are real
    # `complete_lost_semantic_atom_materiality.CompleteLostSemanticAtomMateriality`
    # instances (never JSON-projected here, unlike `diagnostics` above --
    # this field is a typed-object carrier, not a diagnostics/logging
    # channel; see this task's own decision-log entry).
    lost_atom_materiality_by_provenance_id: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessingRequest:
    project_id: str
    user_id: str
    sources: Tuple[SourceAsset, ...]
    preferred_source_order: Tuple[str, ...] = ()
    audio_overlap: bool = False
    language_hint: Optional[str] = None
    # D-134: `audio_overlap` above is preserved byte-for-byte as the raw
    # legacy wire value (still unread by any engine consumer, exactly as
    # before). `dialogue_overlap_enabled` is the NEW canonical field the
    # future Dialogue/Pacing Transition authority will eventually consume
    # (D-129's canonical naming) -- normalized once, at the single dict ->
    # ProcessingRequest boundary in serde.request_from_dict, from either
    # this field's own wire value (if explicitly sent) or a fallback to
    # `audio_overlap` (see serde._normalize_dialogue_overlap). No consumer
    # reads it yet; it changes zero current editorial behavior.
    dialogue_overlap_enabled: bool = False
    # D-134: compact, request-level (not per-family) normalized diagnostics
    # for the Overlap field -- see serde._normalize_dialogue_overlap for the
    # precedence rule that produces these four values.
    overlap_diagnostics: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessingResult:
    schema_version: str
    project_id: str
    state: JobState
    draft: DraftTimeline
    stage_status: Dict[str, object]
    # D-235X Part A: the ONE production data-source seam GAP A (docs/
    # CUTSELL_DECISIONS.md D-235W) named -- an optional, additive per-
    # source live lost-atom exact-identity context, built ONCE inside
    # `pipeline.py::build_flow_b_draft` (the smallest owner with BOTH live
    # `CandidateTake.word_indices`/`.words` AND the live Language Spine's
    # `LanguageAttempt`/`PropositionCandidate` evidence) behind the SAME
    # `CUTSELL_LOST_ATOM_MATERIALITY_FREEZE_AUTHORITY_ENABLED` flag.
    # `None` (not just empty) whenever that flag is off, or whenever no
    # live Language Spine evidence was built for any source (fail-closed
    # default; see this task's own decision-log entry for the honest
    # triple-flag dependency). When present, a dict with exactly three
    # keys -- `exact_match_by_clip_id` (clip_id -> real
    # `shared_attempt_word_identity.AttemptLanguageIdentityMatch`),
    # `proposition_candidate_ids_by_attempt_id`, and
    # `proposition_slot_evidence_by_id` -- consumed only by
    # `universal_clean_cut.py`'s own D-235W/X call sites into
    # `apply_final_story_coherence_validation`/`apply_post_authority_
    # story_validation`. Never JSON-projected (typed-object carrier, same
    # as `DraftTimeline.lost_atom_materiality_by_provenance_id` above).
    lost_atom_exact_identity_context: Optional[Dict[str, object]] = None
