"""D-056.3 CONTRADICTION-SAFE COMPOSITE CONTRACT -- the ONE provider-neutral,
deterministic contradiction primitive every caller in this codebase that
needs to know "do these two texts factually contradict each other" must
use.

Root defect this module fixes (see docs/CUTSELL_DECISIONS.md D-056.2/D-056.3
for the full live evidence): before this module existed,
``final_story_coherence_validation.py``'s StoryValidator computed a
negation/number contradiction check inline, TWICE, in two separate
functions (``_resolve_residual_family`` and ``_contradiction_findings``) --
already a duplicated-algorithm risk within one file. Meanwhile
``canonical_edit_plan.py``'s composite-acceptance gate (``is_accepted_
composite`` -- the actual field that decides whether a 2+-member winning
group is reported ``is_composite: true`` / ``coverage_status: complete``,
the object Selection Freeze/Boundary/Renderer actually consume) NEVER
checked for a factual contradiction between the composite's own members at
all. A contradictory pair (e.g. one member negating what the other
asserts, or the two disagreeing on a number) could therefore be accepted
as a resolved composite by ``canonical_edit_plan.py`` while StoryValidator's
own ``contradiction_findings``/FinalEditReviewer's independent CONTRADICTION
check still (correctly) flagged the exact same pair -- two safety layers
that DISAGREED instead of one shared, structurally-impossible-to-violate
contract. Live evidence: D-056.2 Run B (`tg_539b31f663aaf9e13f`) and Run C
(`tg_f4b9e7c1fe3e28a1af`), both a negation-conflicting pair CanonicalEditPlan
marked `is_composite: true, coverage_status: complete` while FinalEditReviewer
independently flagged the same pair CONTRADICTION.

This primitive is EXTRACTED, not reinvented: it reuses
``final_sibling_grouping._numbers``/``_negations`` verbatim -- the exact
same signals that module already requires to MATCH before it will merge two
takes, and the exact same signals StoryValidator has used since D-011 to
flag a factually-incompatible retry-family pair. Every caller that needs a
contradiction verdict between two pieces of text -- StoryValidator's own
checks, and CanonicalEditPlan's composite-acceptance gate -- now calls this
ONE function, so they can never independently disagree again.

Deliberately narrow, matching the existing primitive's own scope (see
``final_story_coherence_validation.py``'s module docstring "Not implemented
in V1" note): a positive-vs-negative (negation) mismatch, or an incompatible
number/percentage disagreement. General non-numeric/non-negation factual
contradiction remains an honest, documented gap -- not silently claimed to
be solved here.
"""
from __future__ import annotations

from dataclasses import dataclass

from .final_sibling_grouping import _negations, _numbers


@dataclass(frozen=True)
class TextContradiction:
    """The one contradiction verdict shape every caller consumes.

    ``number_conflict``: both texts carry at least one number token and
    their number sets differ (e.g. "5%" vs "10%") -- a restated IDENTICAL
    number (both sides say "5%") is never a conflict, only a genuine
    disagreement.

    ``negation_conflict``: exactly one of the two texts carries an explicit
    negation marker (a bare positive-vs-negative disagreement -- "soy la
    unica" vs "no soy la unica"). Both sides negating, or neither, is not a
    conflict by this signal.
    """

    number_conflict: bool
    negation_conflict: bool

    @property
    def has_conflict(self) -> bool:
        return self.number_conflict or self.negation_conflict


def detect_text_contradiction(left_text: str, right_text: str) -> TextContradiction:
    """The shared contradiction contract. Deterministic, provider-neutral,
    no external call -- same posture as the primitive it extracts (never a
    Gemini/arbiter judgment call; this is evidence-based, not a semantic
    guess). Every caller -- StoryValidator's ``_resolve_residual_family``/
    ``_contradiction_findings``, and CanonicalEditPlan's composite-
    acceptance gate -- MUST call this function rather than re-deriving the
    same signals inline, so a future change to the underlying number/
    negation extraction logic can never leave one caller's contradiction
    verdict out of sync with another's."""
    left_numbers, right_numbers = _numbers(left_text), _numbers(right_text)
    left_negations, right_negations = _negations(left_text), _negations(right_text)
    number_conflict = bool(left_numbers) and bool(right_numbers) and left_numbers != right_numbers
    negation_conflict = bool(left_negations) != bool(right_negations)
    return TextContradiction(number_conflict=number_conflict, negation_conflict=negation_conflict)


def any_pair_contradicts(texts: list[str]) -> bool:
    """True when ANY two texts in ``texts`` contradict each other per
    ``detect_text_contradiction``. Convenience for a composite's full
    member set (2 members today per the existing ``_MAX_COMPOSITE_SIZE``/
    bounded-pair conventions elsewhere in this codebase, but this helper is
    not itself bounded to 2 -- it checks every pair)."""
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if detect_text_contradiction(texts[i], texts[j]).has_conflict:
                return True
    return False
