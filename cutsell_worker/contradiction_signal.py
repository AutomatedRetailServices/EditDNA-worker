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

D-056.5 PROPOSITION-COMPLETENESS GATE (see docs/CUTSELL_DECISIONS.md D-056.4
for the live forensic that proved this): the D-056.3 primitive's
``negation_conflict`` signal was itself a false-positive risk -- whole-clip
negation-token PRESENCE/ABSENCE was sufficient to flag a conflict even when
one realization was an incomplete retry that trailed off before ever
reaching the shared proposition at all, or when a realization's negation
marker was attached to a DIFFERENT, adjacent clause (a rhetorical negation)
than the content actually shared with the other realization. D-056.4's live
example: realization A's "no creo... que los cánceres son hereditarios"
(rejecting a BROADER claim) sits in a different sentence from A's own
restatement of the shared "solo un 5-10%" figure; realization B never
reaches an equivalent clause because its recording was cut off -- yet
whole-clip presence/absence alone flagged these as contradicting. Fixed by
scoping the negation-token comparison to the sentence/clause that
corresponds to an ACTUAL clause of the other realization -- a shared NUMBER
is decisive on its own (the most specific anchor two realizations can
share); otherwise a clause only counts as corresponding when it clears both
a minimum shared-content-token count and a minimum coverage ratio (the same
two-part shape `final_sibling_grouping._same_retry_idea` already uses for
whole-clip retry matching, applied here at clause granularity so a single
incidental shared connecting word never counts as "the same proposition").
A negation elsewhere in the text, or a clause neither side ever reached, no
longer counts. An incomplete realization that DOES
complete the compared clause before trailing off elsewhere is unaffected --
its negation token still counts, because it still co-occurs with the shared
anchor in its own sentence.

This primitive is EXTRACTED, not reinvented: it reuses
``final_sibling_grouping._numbers``/``_content``/``_tokens`` verbatim -- the
exact same signals that module already requires to MATCH before it will
merge two takes, and the exact same signals StoryValidator has used since
D-011 to flag a factually-incompatible retry-family pair. The negation-
marker vocabulary itself is a local superset of ``final_sibling_grouping.
_NEGATIONS`` (adds "nadie"/"ni", the same class of general Spanish negation
marker as the existing set, never a Video00-specific phrase) -- kept local
to this module rather than edited into ``final_sibling_grouping.py`` because
that set is also consumed by retry-grouping and by ``final_story_coherence_
validation``'s unrelated lost-semantic-atoms coverage ledger; widening it
there would ripple into checks neither D-056.3 nor D-056.5 authorize
touching. Every caller that needs a contradiction verdict between two
pieces of text -- StoryValidator's own checks, and CanonicalEditPlan's
composite-acceptance gate -- calls this ONE function, so they can never
independently disagree, and a future improvement to the underlying
completeness/anchor logic benefits every caller identically.

Deliberately narrow, matching the existing primitive's own scope (see
``final_story_coherence_validation.py``'s module docstring "Not implemented
in V1" note): a positive-vs-negative (negation) mismatch, or an incompatible
number/percentage disagreement. General non-numeric/non-negation factual
contradiction (e.g. an implicit causal-direction reversal with no negation
marker at all) remains an honest, documented gap -- not silently claimed to
be solved here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .final_sibling_grouping import _content, _numbers, _tokens

# D-056.5: superset of final_sibling_grouping._NEGATIONS, scoped to this
# contradiction contract only -- see module docstring for why it is not
# edited into final_sibling_grouping.py itself. "nadie" (nobody) and "ni"
# (nor) are ordinary Spanish negation markers, the same general class as
# the existing five -- not a phrase tied to any specific video's content.
_NEGATION_MARKERS = frozenset({"no", "not", "never", "nunca", "sin", "without", "nadie", "ni"})

# Standard sentence-boundary heuristic (split after a run of terminal
# punctuation followed by whitespace) -- general-purpose, not tied to any
# specific language's idiom or any video's own phrasing. A text with no
# such boundary at all (a single clause, or a fragment cut off before ever
# reaching one) is treated as one whole "sentence" by `_sentences` below,
# so a short, complete-but-unpunctuated utterance is never penalized for
# lacking a period.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextContradiction:
    """The one contradiction verdict shape every caller consumes.

    ``number_conflict``: both texts carry at least one number token and
    their number sets differ (e.g. "5%" vs "10%") -- a restated IDENTICAL
    number (both sides say "5%") is never a conflict, only a genuine
    disagreement.

    ``negation_conflict``: exactly one of the two texts carries an explicit
    negation marker IN THE CLAUSE ADDRESSING THE PROPOSITION SHARED WITH
    THE OTHER TEXT (D-056.5's proposition-completeness gate -- see module
    docstring) -- a bare positive-vs-negative disagreement on the SAME
    claim ("soy la unica" vs "no soy la unica"), not a negation attached to
    an unrelated clause, and not a clause a truncated retry never reached.
    Both sides negating that clause, or neither, is not a conflict by this
    signal.
    """

    number_conflict: bool
    negation_conflict: bool

    @property
    def has_conflict(self) -> bool:
        return self.number_conflict or self.negation_conflict


def _negation_tokens(text: str) -> frozenset[str]:
    return frozenset(token for token in _tokens(text) if token in _NEGATION_MARKERS)


def _sentences(text: str) -> tuple[str, ...]:
    """Splits `text` on ordinary sentence boundaries. A text with no
    boundary at all (never reached one, or is inherently one clause) comes
    back as a single one-element tuple containing the whole text -- so
    "does this sentence contain the shared anchor" degrades gracefully to
    "does the whole text contain the shared anchor" for a short or
    unpunctuated utterance, rather than losing it entirely."""
    raw = str(text or "").strip()
    if not raw:
        return ()
    return tuple(part for part in _SENTENCE_SPLIT_RE.split(raw) if part.strip())


# D-056.5: same two-part shape (a minimum shared-token count AND a minimum
# coverage ratio) `final_sibling_grouping._same_retry_idea` already uses to
# decide whether two whole clips are "the same retry idea" -- reused here
# at clause level so a single incidental shared connecting word (e.g. two
# otherwise-unrelated sentences that both happen to mention "science")
# never counts as "the same proposition" on its own.
_MIN_SHARED_CLAUSE_TOKENS = 2
_MIN_SHARED_CLAUSE_COVERAGE = 0.5


def _clauses_address_same_proposition(clause: str, other_text: str) -> bool:
    """True when `clause` (one sentence from one realization) corresponds
    to at least one sentence of `other_text` closely enough to be "the same
    specific point" rather than a merely-adjacent or coincidentally-
    overlapping one. A shared NUMBER is decisive on its own -- an exact
    shared figure is essentially never a coincidence, and is precisely the
    anchor D-056.4's own live example shares ("5"/"10" from "5-10%" restated
    on both sides). Otherwise requires both a minimum shared-content-token
    count and a minimum coverage ratio (of the SMALLER side, so a short
    clause fully contained in a longer one still counts)."""
    clause_numbers = _numbers(clause)
    clause_content = _content(clause)
    other_sentences = _sentences(other_text) or (other_text,)
    for other_sentence in other_sentences:
        if clause_numbers and clause_numbers & _numbers(other_sentence):
            return True
        other_content = _content(other_sentence)
        if not clause_content or not other_content:
            continue
        shared = len(clause_content & other_content)
        smaller = min(len(clause_content), len(other_content))
        if shared >= _MIN_SHARED_CLAUSE_TOKENS and smaller and shared / smaller >= _MIN_SHARED_CLAUSE_COVERAGE:
            return True
    return False


def _negations_about_shared_proposition(text: str, other_text: str) -> frozenset[str]:
    """D-056.5 PROPOSITION-COMPLETENESS GATE: a negation token counts
    toward a contradiction verdict only when it appears in a clause of
    `text` that actually corresponds to some clause of `other_text` (see
    `_clauses_address_same_proposition`) -- the sentence that genuinely
    addresses the proposition in common, never an adjacent or unrelated
    clause (a rhetorical negation attached to a different point entirely),
    and never a clause the recording never reached at all (an incomplete
    retry that trails off before restating the shared point carries no
    negation there, by construction -- it never said anything in that
    clause to negate).

    Never widens what counts as a conflict relative to the pre-D-056.5
    primitive -- it can only DROP a negation token that whole-clip
    presence/absence would previously have counted, never add one.
    """
    found: set[str] = set()
    for sentence in _sentences(text) or (text,):
        if _clauses_address_same_proposition(sentence, other_text):
            found |= _negation_tokens(sentence)
    return frozenset(found)


def detect_text_contradiction(left_text: str, right_text: str) -> TextContradiction:
    """The shared contradiction contract. Deterministic, provider-neutral,
    no external call -- same posture as the primitives it extracts (never a
    Gemini/arbiter judgment call; this is evidence-based, not a semantic
    guess). Every caller -- StoryValidator's ``_resolve_residual_family``/
    ``_contradiction_findings``/its residual-family exemption check, and
    CanonicalEditPlan's composite-acceptance gate -- MUST call this
    function rather than re-deriving the same signals inline, so a future
    change to the underlying number/negation/completeness logic can never
    leave one caller's contradiction verdict out of sync with another's."""
    left_numbers, right_numbers = _numbers(left_text), _numbers(right_text)
    number_conflict = bool(left_numbers) and bool(right_numbers) and left_numbers != right_numbers

    left_negations = _negations_about_shared_proposition(left_text, right_text)
    right_negations = _negations_about_shared_proposition(right_text, left_text)
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
