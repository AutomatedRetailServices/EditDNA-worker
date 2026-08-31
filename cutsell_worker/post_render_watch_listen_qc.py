"""PostRenderWatchListenQC -- durable interface contract only (D-024).

Clean Cut Core V1 does not implement or activate this stage. This module
defines the SHAPE a future physical/output QA pass must have so it can be
added later without another architecture reset -- see the canonical
directive's "POST-RENDER WATCH + LISTEN QC -- INTERFACE NOW" section.

Scope, once implemented: physical/output QA AFTER render --
clipped words, unsafe word boundaries, awkward physical cuts, lingering
accidental silence, body/mic/camera reset debris, awkward post-line
expression, A/V synchronization, framing integrity, decode/export
integrity. It must NEVER change semantic membership. A finding whose root
cause is semantic (e.g. a clipped word reveals the wrong take was selected)
routes back upstream to the owning canonical authority; a finding that is
purely physical/timing (e.g. a cut lands 40ms into a word) routes to
BoundaryEngine. Nothing in this module performs that upstream/Boundary
routing itself -- it only defines the result shape a caller would act on.

No provider is implemented. `PostRenderWatchListenQCProvider` is a Protocol
so a future implementation can be type-checked against this contract
without this module depending on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .canonical_edit_plan import CanonicalEditPlan

CLIPPED_WORD = "CLIPPED_WORD"
UNSAFE_WORD_BOUNDARY = "UNSAFE_WORD_BOUNDARY"
AWKWARD_PHYSICAL_CUT = "AWKWARD_PHYSICAL_CUT"
LINGERING_ACCIDENTAL_SILENCE = "LINGERING_ACCIDENTAL_SILENCE"
RESET_DEBRIS = "RESET_DEBRIS"
AWKWARD_POST_LINE_EXPRESSION = "AWKWARD_POST_LINE_EXPRESSION"
AV_SYNC_DRIFT = "AV_SYNC_DRIFT"
FRAMING_INTEGRITY = "FRAMING_INTEGRITY"
DECODE_EXPORT_INTEGRITY = "DECODE_EXPORT_INTEGRITY"

_PHYSICAL_KINDS = frozenset({
    CLIPPED_WORD, UNSAFE_WORD_BOUNDARY, AWKWARD_PHYSICAL_CUT,
    LINGERING_ACCIDENTAL_SILENCE, RESET_DEBRIS, AWKWARD_POST_LINE_EXPRESSION,
    AV_SYNC_DRIFT, FRAMING_INTEGRITY, DECODE_EXPORT_INTEGRITY,
})


@dataclass(frozen=True)
class PostRenderFinding:
    kind: str
    start: float
    end: float
    detail: dict
    # "BoundaryEngine" for a purely physical/timing finding; the name of the
    # owning canonical authority (e.g. "StoryValidator") when the physical
    # symptom's real cause is upstream and semantic.
    routes_to: str


@dataclass(frozen=True)
class PostRenderQCResult:
    status: str  # "PASS" | "FAIL"
    findings: tuple[PostRenderFinding, ...]


class PostRenderWatchListenQCProvider(Protocol):
    """Contract a future concrete provider must satisfy. Not implemented in
    Clean Cut Core V1 -- no provider is constructed or invoked anywhere in
    the active pipeline."""

    def review(self, rendered_video_path: str, edit_plan: CanonicalEditPlan) -> PostRenderQCResult:
        ...
