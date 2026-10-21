"""Faceless / Product / Hands / Demo Visual-Mode SAFETY FOUNDATION -- D-280.

D-276 established the canonical A-roll redefinition: a face is NOT
required for primary editorial footage. Canonical primary visual modes
include `TALKING_HEAD_A_ROLL`, `FACELESS_PRODUCT_A_ROLL`,
`PRODUCT_HANDS_A_ROLL`, and `DEMO_ACTION_A_ROLL` -- product/hands/demo
footage may itself BE the primary A-roll, never automatically B-roll.

D-258/D-260's own visual stack has real, proven evidence for face/gaze/
headroom/face-center/face-scale, and NONE for product/hands/demo. Left
unaddressed, that asymmetry risks:

    NO_FACE -> BAD_CLIP / INVALID_A_ROLL / VISUAL_FINISHING_FAILURE

for canonical faceless/product/demo primary footage. This module is a
SAFETY / ROUTING FOUNDATION, not advanced product understanding -- its
one job is letting the engine answer two narrow questions:

    "Should face-based evidence apply here?"
    "Can this clip remain valid primary footage even with no face?"

## Scope discipline (binding, D-280's own scope banner)

NO PRODUCT RECOGNITION. NO SKU RECOGNITION. NO DEMO-SEMANTIC
UNDERSTANDING. NO SPOKEN-TEXT-TO-DEMO MATCHING. NO CLIP-BASED B-ROLL
RANKING. NO AI B-ROLL SUGGESTION/AUTO-PLACEMENT. NO NEW VISUAL-FINISHING
NUMERIC THRESHOLD (the one rate this module reads,
`visual_finishing_policy.MIN_RELIABLE_FACE_DETECTION_RATE`, is REUSED
verbatim as the "strong face evidence" bar for classification -- never
redeclared or extended). This module:

- never calls ffmpeg, cv2, or mediapipe itself -- it consumes only
  already-computed facts (a `VisualClipMeasurement`'s own
  `face_detection_rate`, or a caller-supplied `VisualModeEvidence`);
- never fabricates a product/hands bbox or a positive product/hands/
  demo presence -- every field D-257 already established has no real
  detector for is represented as `UNAVAILABLE`, never `NOT_PRESENT`
  (Stage 7/8's own "unknown product location != known product absence"
  doctrine, restated from D-259/D-260, never weakened here);
- is intentionally NOT a full scene classifier: `classify_visual_mode`
  only ever returns `TALKING_HEAD` (real, strong face evidence) or
  `UNKNOWN` (everything else) -- it never infers `PRODUCT_HANDS`/
  `PRODUCT_ONLY`/`DEMO_ACTION` from the mere ABSENCE of a face (Stage 23's
  own explicit "no face evidence -> do NOT automatically infer
  PRODUCT_HANDS" instruction); the full seven-value `VisualMode`
  vocabulary exists so a FUTURE detector, or a manual/editorial
  assignment, can construct a `VisualModeClassification` for any of
  the other modes directly, without this module needing to change;
- never determines editorial/timeline role -- `visual_mode` and
  `EditorialContentRole` are proven independent (Stage 16/17).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .visual_finishing_policy import MIN_RELIABLE_FACE_DETECTION_RATE

VISUAL_MODE_CONTRACT_VERSION = 1


# ---------------------------------------------------------------------------
# STAGE 1: bounded, immutable visual-mode vocabulary.
# ---------------------------------------------------------------------------

class VisualMode(str, Enum):
    TALKING_HEAD = "TALKING_HEAD"
    TALKING_HEAD_WITH_PRODUCT = "TALKING_HEAD_WITH_PRODUCT"
    PRODUCT_HANDS = "PRODUCT_HANDS"
    PRODUCT_ONLY = "PRODUCT_ONLY"
    DEMO_ACTION = "DEMO_ACTION"
    SUPPORTING_VISUAL = "SUPPORTING_VISUAL"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# STAGE 2: evidence-status vocabulary for a CLASSIFICATION as a whole --
# never confused with per-field evidence availability (STAGE 27, below).
# ---------------------------------------------------------------------------

class VisualModeEvidenceState(str, Enum):
    SUPPORTED = "SUPPORTED"
    LIKELY = "LIKELY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# STAGE 7/27: per-field evidence availability. Four DISTINCT states,
# never collapsed -- this is the concrete mechanism behind Stage 8's
# "unknown product location != known product absence" doctrine.
# ---------------------------------------------------------------------------

class EvidenceAvailability(str, Enum):
    PRESENT = "PRESENT"
    NOT_PRESENT = "NOT_PRESENT"        # positively, measurably confirmed absent
    UNAVAILABLE = "UNAVAILABLE"        # no detector/capability exists at all today
    NOT_APPLICABLE = "NOT_APPLICABLE"  # this evidence family does not apply to this mode
    UNKNOWN = "UNKNOWN"                # a detector exists but this result is indeterminate


# ---------------------------------------------------------------------------
# STAGE 4: face-required helper.
# ---------------------------------------------------------------------------

_FACE_REQUIRED_MODES = frozenset({VisualMode.TALKING_HEAD, VisualMode.TALKING_HEAD_WITH_PRODUCT})
_FACE_INDEPENDENT_MODES = frozenset({
    VisualMode.PRODUCT_HANDS, VisualMode.PRODUCT_ONLY, VisualMode.DEMO_ACTION, VisualMode.SUPPORTING_VISUAL,
})


def requires_face_evidence(mode: VisualMode) -> bool | None:
    """Stage 4: `True` for TALKING_HEAD/TALKING_HEAD_WITH_PRODUCT,
    `False` for PRODUCT_HANDS/PRODUCT_ONLY/DEMO_ACTION/SUPPORTING_VISUAL,
    and `None` (never a guessed `True` or `False`) for `UNKNOWN` -- "fail
    conservative / insufficient evidence, not automatic rejection" is
    this module's own literal instruction for the unsettled case."""
    if mode in _FACE_REQUIRED_MODES:
        return True
    if mode in _FACE_INDEPENDENT_MODES:
        return False
    return None


# ---------------------------------------------------------------------------
# STAGE 6: pure evidence-family routing -- which evidence families a
# given mode cares about. No detector implementation implied.
# ---------------------------------------------------------------------------

EVIDENCE_FAMILY_FACE = "FACE"
EVIDENCE_FAMILY_GAZE = "GAZE"
EVIDENCE_FAMILY_HEADROOM = "HEADROOM"
EVIDENCE_FAMILY_FRAMING = "FRAMING"
EVIDENCE_FAMILY_SPEECH_FLUENCY = "SPEECH_FLUENCY"
EVIDENCE_FAMILY_PRODUCT_HANDS = "PRODUCT_HANDS_EVIDENCE"
EVIDENCE_FAMILY_VISUAL_CONTINUITY = "GENERIC_VISUAL_CONTINUITY"
EVIDENCE_FAMILY_MOTION_ACTION_CONTINUITY = "MOTION_ACTION_CONTINUITY"

_EVIDENCE_ROUTING: dict[VisualMode, tuple[str, ...]] = {
    VisualMode.TALKING_HEAD: (
        EVIDENCE_FAMILY_FACE, EVIDENCE_FAMILY_GAZE, EVIDENCE_FAMILY_HEADROOM,
        EVIDENCE_FAMILY_FRAMING, EVIDENCE_FAMILY_SPEECH_FLUENCY,
    ),
    VisualMode.TALKING_HEAD_WITH_PRODUCT: (
        EVIDENCE_FAMILY_FACE, EVIDENCE_FAMILY_GAZE, EVIDENCE_FAMILY_HEADROOM,
        EVIDENCE_FAMILY_FRAMING, EVIDENCE_FAMILY_SPEECH_FLUENCY, EVIDENCE_FAMILY_PRODUCT_HANDS,
    ),
    VisualMode.PRODUCT_HANDS: (
        EVIDENCE_FAMILY_SPEECH_FLUENCY, EVIDENCE_FAMILY_VISUAL_CONTINUITY, EVIDENCE_FAMILY_PRODUCT_HANDS,
    ),
    VisualMode.PRODUCT_ONLY: (
        EVIDENCE_FAMILY_SPEECH_FLUENCY, EVIDENCE_FAMILY_VISUAL_CONTINUITY, EVIDENCE_FAMILY_PRODUCT_HANDS,
    ),
    VisualMode.DEMO_ACTION: (
        EVIDENCE_FAMILY_SPEECH_FLUENCY, EVIDENCE_FAMILY_MOTION_ACTION_CONTINUITY,
    ),
    VisualMode.SUPPORTING_VISUAL: (
        EVIDENCE_FAMILY_SPEECH_FLUENCY, EVIDENCE_FAMILY_VISUAL_CONTINUITY,
    ),
    VisualMode.UNKNOWN: (),
}


def applicable_evidence_families(mode: VisualMode) -> tuple[str, ...]:
    """Stage 6: pure lookup, no fallback guessing -- `UNKNOWN` legitimately
    routes to no evidence family at all (Stage 21's own conservative
    default)."""
    return _EVIDENCE_ROUTING.get(mode, ())


# ---------------------------------------------------------------------------
# STAGE 8 (D-259/D-260 restated): unknown product location != known
# product absence.
# ---------------------------------------------------------------------------

def product_location_known(product_presence: EvidenceAvailability) -> bool:
    """A crop/reframe decision may treat product location as SAFE to
    reason about ONLY when this returns `True` -- i.e. `product_presence`
    is a positively resolved state (`PRESENT` or `NOT_PRESENT`), never
    `UNAVAILABLE`/`UNKNOWN`/`NOT_APPLICABLE`. Since D-257/D-258/D-260
    established that no product-bbox detector exists today, every real
    clip in this codebase currently resolves this `False` -- the
    honest, correct, expected V1 answer (mirrors `visual_finishing_
    policy.py`'s own `product_safety_established` fail-closed default,
    restated at the classification layer rather than duplicated)."""
    return product_presence in (EvidenceAvailability.PRESENT, EvidenceAvailability.NOT_PRESENT)


# ---------------------------------------------------------------------------
# STAGE 7/25: pure evidence + classification types.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VisualModeEvidence:
    """Stage 7: typed, optional evidence seams. Absence of a detector is
    always `UNAVAILABLE`/`UNKNOWN` -- never a fabricated `NOT_PRESENT`
    or an invented bbox. `uploaded_role_hint` (Stage 24) is carried for
    observability only; nothing in this module lets it change a
    classification."""

    face_detection_rate: float | None = None
    face_evidence_availability: EvidenceAvailability = EvidenceAvailability.UNKNOWN

    product_bbox: tuple[float, float, float, float] | None = None
    product_presence: EvidenceAvailability = EvidenceAvailability.UNAVAILABLE

    hands_bbox: tuple[float, float, float, float] | None = None
    hands_presence: EvidenceAvailability = EvidenceAvailability.UNAVAILABLE

    demo_action_presence: EvidenceAvailability = EvidenceAvailability.UNAVAILABLE

    uploaded_role_hint: str | None = None

    def __post_init__(self) -> None:
        # Stage 7: "do not fabricate coordinates" -- a bbox may only be
        # present alongside a positively PRESENT evidence state; never
        # paired with UNAVAILABLE/UNKNOWN/NOT_PRESENT/NOT_APPLICABLE.
        if self.product_bbox is not None and self.product_presence != EvidenceAvailability.PRESENT:
            raise ValueError("product_bbox may only be set when product_presence is PRESENT")
        if self.hands_bbox is not None and self.hands_presence != EvidenceAvailability.PRESENT:
            raise ValueError("hands_bbox may only be set when hands_presence is PRESENT")


@dataclass(frozen=True)
class VisualModeClassification:
    clip_id: str | None
    mode: VisualMode
    evidence_state: VisualModeEvidenceState
    requires_face: bool | None
    applicable_evidence_families: tuple[str, ...]
    evidence: VisualModeEvidence
    reasons: tuple[str, ...] = field(default_factory=tuple)


class EditorialContentRole(str, Enum):
    """Stage 16/17: the main-engine-side sibling of D-279's manual-
    timeline `TimelineAssetRole.PRIMARY_SOURCE`/`SUPPLEMENTAL_BROLL`
    (same naming spirit; deliberately NOT that same import -- D-279's
    registry is a separate, mobile-only manual-timeline feature, and
    conflating the two would make an unrelated feature's vocabulary
    changes silently ripple into the core engine's own role concept).
    Reused/extended, never redefined per visual mode: `PRODUCT_HANDS` +
    `PRIMARY_A_ROLL` and `PRODUCT_HANDS` + `SUPPLEMENTAL_BROLL` are both
    valid, independent combinations (Stage 16's own literal example)."""

    PRIMARY_A_ROLL = "PRIMARY_A_ROLL"
    SUPPLEMENTAL_BROLL = "SUPPLEMENTAL_BROLL"


VISUAL_MODE_ROLE_FORBIDDEN_EQUIVALENCES: frozenset[tuple[VisualMode, EditorialContentRole]] = frozenset()
"""Stage 16: literally empty -- no `(visual_mode, role)` pair is ever
forbidden by this module. Kept as a named, typed constant (rather than
simply omitting any such check) so the doctrine 'PRODUCT_HANDS ==
BROLL' is never true is provable by inspecting this set's own contents,
not merely by the absence of code that would have encoded it."""


# ---------------------------------------------------------------------------
# STAGE 22/23: the ONE conservative, rule-based V1 classifier. Its job
# is narrow (Stage 3): "should face-based evidence apply here", never
# "what exact product action is happening".
# ---------------------------------------------------------------------------

REASON_STRONG_FACE_EVIDENCE = "STRONG_FACE_EVIDENCE"
REASON_NO_FACE_EVIDENCE_NO_INFERENCE = "NO_FACE_EVIDENCE_NO_INFERENCE"
REASON_INSUFFICIENT_CLASSIFICATION_EVIDENCE = "INSUFFICIENT_CLASSIFICATION_EVIDENCE"


def classify_visual_mode(
    *,
    clip_id: str | None,
    face_detection_rate: float | None,
    evidence: VisualModeEvidence | None = None,
) -> VisualModeClassification:
    """Stage 22/23/24: `face_detection_rate` is the smallest evidence
    already available today (D-258's own `VisualClipMeasurement.face_
    detection_rate`). Strong, reliable face evidence (reusing D-260's
    own `MIN_RELIABLE_FACE_DETECTION_RATE` bar verbatim, never a new
    threshold) is a `TALKING_HEAD` candidate. The absence of face
    evidence is NEVER, by itself, turned into an automatic `PRODUCT_
    HANDS`/`PRODUCT_ONLY`/`DEMO_ACTION` inference (Stage 23's own
    explicit instruction) -- this V1 classifier has no real product/
    hands/demo detector to justify that inference, so it stays
    `UNKNOWN`, honestly marked `LIKELY` face-independent rather than
    falsely `SUPPORTED` as any specific face-independent mode. A future
    detector, or a manual/editorial assignment, constructs a genuine
    `PRODUCT_HANDS`/`PRODUCT_ONLY`/`DEMO_ACTION`/`SUPPORTING_VISUAL`
    `VisualModeClassification` directly -- this function is not the
    only way to produce one."""
    resolved_evidence = evidence if evidence is not None else VisualModeEvidence(
        face_detection_rate=face_detection_rate,
    )

    if face_detection_rate is not None and face_detection_rate >= MIN_RELIABLE_FACE_DETECTION_RATE:
        return VisualModeClassification(
            clip_id=clip_id, mode=VisualMode.TALKING_HEAD, evidence_state=VisualModeEvidenceState.SUPPORTED,
            requires_face=True, applicable_evidence_families=applicable_evidence_families(VisualMode.TALKING_HEAD),
            evidence=resolved_evidence, reasons=(REASON_STRONG_FACE_EVIDENCE,),
        )
    if face_detection_rate is None:
        return VisualModeClassification(
            clip_id=clip_id, mode=VisualMode.UNKNOWN, evidence_state=VisualModeEvidenceState.UNKNOWN,
            requires_face=None, applicable_evidence_families=(),
            evidence=resolved_evidence, reasons=(REASON_INSUFFICIENT_CLASSIFICATION_EVIDENCE,),
        )
    # Real, measured low/zero face-detection evidence: genuine evidence
    # AGAINST talking-head, but not (by itself) evidence FOR any
    # specific face-independent mode.
    return VisualModeClassification(
        clip_id=clip_id, mode=VisualMode.UNKNOWN, evidence_state=VisualModeEvidenceState.LIKELY,
        requires_face=None, applicable_evidence_families=(),
        evidence=resolved_evidence, reasons=(REASON_NO_FACE_EVIDENCE_NO_INFERENCE,),
    )
