"""Future Sales/TikTok Shop extension contract -- INTERFACE ONLY (D-024).

Sales Funnel and TikTok Shop styling are explicitly OUT OF SCOPE for Clean
Cut Core V1 and are NOT activated anywhere by this module. Nothing here is
imported by the active pipeline. This exists only so a future composition/
styling layer has a defined shape to build against, per the canonical
directive's "FUTURE SALES / TIKTOK SHOP EXTENSION CONTRACT" section.

Future architecture (not built here):

    Clean Cut Core -> Semantic Freeze -> OPTIONAL SalesComposition
    -> OPTIONAL StyleProfile -> Render -> Final QC

`CanonicalEditPlan.EditPlanClip.annotations` (canonical_edit_plan.py) is the
dormant per-clip extension point this future layer would populate --
Clean Cut Core V1 never writes to it. The reserved (not applied) annotation
keys are named below so a future implementation and this contract agree on
vocabulary from the start.

Do NOT hardcode or activate, here or anywhere in Clean Cut Core V1, any of
the specific pacing/style constants a future TikTokShop StyleProfile might
one day own (the directive names these explicitly as forbidden in Clean
Cut V1): a universal loudness floor, a universal pause-deletion threshold,
universal trim handles, a fixed visual-change cadence, or fixed zoom
percentages. Those are style decisions, not semantic Clean Cut ones, and
belong entirely inside a future StyleProfile implementation -- never in
this contract module and never in the semantic pipeline.
"""
from __future__ import annotations

from typing import Protocol

# Reserved per-clip annotation keys for CanonicalEditPlan.EditPlanClip.annotations.
# Never populated by Clean Cut Core V1; listed here purely as agreed vocabulary.
ANNOTATION_SEMANTIC_ROLE = "semantic_role"
ANNOTATION_PRODUCT_RELEVANCE = "product_relevance"
ANNOTATION_EMPHASIS_OPPORTUNITY = "emphasis_opportunity"
ANNOTATION_VISUAL_ACTION_OPPORTUNITY = "visual_action_opportunity"
ANNOTATION_PROTECTED_PAUSE = "protected_pause"
ANNOTATION_PROTECTED_WORD_BOUNDARY = "protected_word_boundary"

RESERVED_ANNOTATION_KEYS = frozenset({
    ANNOTATION_SEMANTIC_ROLE,
    ANNOTATION_PRODUCT_RELEVANCE,
    ANNOTATION_EMPHASIS_OPPORTUNITY,
    ANNOTATION_VISUAL_ACTION_OPPORTUNITY,
    ANNOTATION_PROTECTED_PAUSE,
    ANNOTATION_PROTECTED_WORD_BOUNDARY,
})

# Named future profiles (not implemented). A StyleProfile owns configurable,
# non-semantic pacing/presentation behavior (aggressive social pacing,
# silence-threshold signals, pause targets, contextual punch-ins/reframing,
# product-detail emphasis, visual-change cadence, captions/graphics rhythm)
# -- never semantic membership.
NATURAL_CLEAN = "NaturalClean"
TIGHT_SOCIAL = "TightSocial"
TIKTOK_SHOP = "TikTokShop"


class StyleProfile(Protocol):
    """Contract a future concrete style profile must satisfy. Not
    implemented or invoked anywhere in Clean Cut Core V1."""

    name: str

    def apply(self, edit_plan: object) -> object:
        ...
