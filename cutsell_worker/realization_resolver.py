"""D-050C1: the Unified Realization Resolver, in SHADOW AUTHORITY mode.

See docs/CUTSELL_DECISIONS.md D-050 (audit), D-050A (canonical identities),
D-050B (Semantic Ledger), and D-050C1 (this module) for full context.

SHADOW AUTHORITY CONTRACT (binding for every line in this file)
=================================================================
This module computes what ONE unified Realization Resolution authority
WOULD decide for every semantic idea in a `SemanticLedger`, using the
Ledger's own recorded evidence (realizations, canonical claims, delivery
evidence, discard/replacement history) as its ONLY inputs. It is not
consulted by, and never feeds back into, `DraftTimeline.selected`/
`discarded`, `CanonicalEditPlan`, `StoryValidator`, `FinalEditReviewer`, or
Selection Freeze -- today's engine (DeliveryScorer, `_semantic_best_take`,
`deterministic_best_take_authority`, `ClaimCoverageBestTake`,
`CompositeResolver`) remains the sole active authority. Its output is
wired into diagnostics ONLY (see `universal_clean_cut.py`'s call site,
under a dedicated `realization_resolver_shadow` key, never
`semantic_ledger`'s own key) for observability and for the parity report
this directive requires. See `resolve_realizations_shadow`'s own docstring
for the one-pass decision model, and the module-level "NO BEHAVIOR
CUTOVER" note at the bottom of this docstring for the explicit audit this
directive requires.

WHY ONE PASS, NOT A SEQUENTIAL OVERRIDE STACK
==============================================
Today's engine reaches a winner through up to four sequential overrides
(DeliveryScorer's local top score -> `_semantic_best_take`'s hybrid label
override -> `ClaimCoverageBestTake`'s claim-driven override -> a
`CompositeResolver` composite when neither single realization suffices),
each one only ever correcting the previous stage's mistake within its own
narrow concern. D-050's own architecture audit named this the duplicated-
authority problem: no single stage ever asks "considering safety,
completeness, consistency, delivery quality, and contextual value
together, which realization(s) actually deserve to win?" This module asks
exactly that question once per semantic idea, in the fixed precedence
order the D-050C1 directive specifies:

    semantic safety (no contradiction silently resolved)
    > critical claim completeness (every CRITICAL requirement covered)
    > factual/negation consistency (a contradiction blocks, never averages)
    > delivery quality (evidence, not authority -- a tie-breaker among
      already-safe, already-complete candidates)
    > contextual richness (a secondary tie-breaker only)

A candidate that fails a higher-precedence test is never reachable by a
lower-precedence one, no matter how good its delivery score -- this is
lexicographic/constraint-based resolution, not a weighted score.

HARD INVARIANTS (D-050C1 Section on Hard Invariants A-E)
==========================================================
A. Idea survival     -- every `semantic_idea_id` the Ledger knows about
                         gets exactly one `RealizationResolution` below;
                         none is silently dropped from the report.
B. Critical claim     -- a resolution's `decision_status` is never
   preservation          RESOLVED_WINNER/RESOLVED_COMPOSITE while
                         `missing_critical_claim_ids` is non-empty --
                         REVIEW_REQUIRED is the only allowed status then.
C. No duplicate       -- a composite (see `_find_minimal_composite`) never
   retry realization      includes two members whose own covered
                         requirement-group sets are equal (one would add
                         zero net coverage -- see composite criterion 5).
D. Discard requires   -- `discarded_realization_ids` only ever contains a
   safety                 realization that is either (a) fully redundant
                         with the chosen winner/composite's own coverage,
                         or (b) already carries a Ledger-verified
                         replacement. Every other non-winner realization
                         is reported under `retained_for_contextual_value`
                         instead of silently vanishing -- never a bare
                         "discarded" the resolver cannot justify.
E. Physical quality   -- see `resolve_orphan_realizations_shadow`: a
   cannot silently        realization the CURRENT engine discarded
   delete unique          BEFORE it ever reached grouping (no
   semantics               `semantic_idea_id` at all -- exactly D-049 Case
                         A's shape) can never be silently confirmed as a
                         safe discard by this module; without a Ledger-
                         verified replacement its shadow verdict is always
                         REVIEW_REQUIRED, never DISCARD_CONFIRMED.

CANONICAL CLAIM DEDUP (shadow-only -- see `build_requirement_groups`)
=======================================================================
Two canonical claims are folded into the same "requirement group" -- the
actual unit of critical-coverage bookkeeping below -- only when ALL of:
  1. same `claim_type` (a NEGATION-typed claim can never merge with a
     same-topic non-negated claim -- `classify_claim` already gives a
     negated proposition its own claim_type, so "has X" vs "does not have
     X" are structurally unable to collide here);
  2. compatible polarity: neither claim's own `content_tokens` carries a
     negation-content marker ("not"/"never"/"nunca"/"sin"/"without")
     while the other lacks one entirely;
  3. compatible quantitative meaning: `_claim_digit_values` extracts every
     digit run from the claim's own RAW clause text (`CanonicalClaimRecord.
     text`, a D-050C1 addition -- see that field's own docstring), not
     `content_tokens` alone: `semantic_claims._content`'s >=3-character
     token floor silently drops a bare short number ("5%" is 2 characters)
     while keeping a longer one ("10%", "5-10%") intact, so comparing
     `content_tokens` alone would let "5% vs 10%" falsely dedup the moment
     one side's digit happened to fall below that floor. Reading raw text
     instead means whenever EITHER claim has any digit evidence at all,
     the two claims' normalized numeric-value sets must be EXACTLY equal
     -- "5%" and "10%" (or "5-10%" and "8-12%") never dedup, while "5-10%"
     restated with different spacing/tokenization the second time (the
     literal D-049 Case B shape) still reduces to the same `{"5","10"}`
     both times and passes;
  4. high normalized equivalence on the remaining (non-digit) content
     tokens -- Jaccard overlap at or above `_CLAIM_DEDUP_THRESHOLD`.
This is metadata-only clustering over `CanonicalClaimRecord`s already in
the Ledger; it never touches `semantic_claims.py`'s own authoritative
`Claim`/`extract_claims` output.

COMPOSITE MODEL (shadow-only -- see `_find_minimal_composite`)
==================================================================
A candidate set of realizations is a valid shadow composite for one
semantic idea only when ALL SEVEN hold (D-050C1.6 F5 added criteria 6-7
after the D-050C1.5 full sweep caught a 3-fragment assembly production's
own bounded resolution would never attempt):
  1. every member belongs to the SAME `semantic_idea_id` (never
     cross-idea);
  2. the union of members' covered requirement groups is a superset of
     every CRITICAL requirement group for that idea;
  3. no two members are on opposite sides of a detected contradiction
     signal (`_detect_contradiction_signals`) -- a genuine factual
     conflict blocks composite formation entirely (goes to
     REVIEW_REQUIRED instead), it is never silently resolved by
     "including both retries";
  4. no member is itself a realization the engine's own Ledger discard
     history flags as physically unsafe without a verified replacement
     (mirrors invariant E for composite members specifically);
  5. every member contributes at least one requirement group no other
     chosen member already covers (no redundant member -- this is
     invariant C, applied inside the composite search itself);
  6. all members occupy non-overlapping time windows
     (`_composite_members_temporally_compatible` -- two physically-
     overlapping deliveries can never be genuine sequential pieces of one
     realization). NOTE: `complete_idea` (a member's own "is this a
     complete delivery on its own" signal) is deliberately NOT a
     composite-eligibility gate -- an earlier version of this criterion
     excluded `complete_idea is False` realizations and was empirically
     wrong: `test_complementary_critical_claims_require_a_composite`
     composites two realizations BOTH marked incomplete BY DESIGN, which
     is the entire point of a composite (production's own
     `claim_coverage_best_take.py` composite formation never checks
     completeness either). This is a deliberately narrower, honest proxy
     for full narrative/causal-order validity, which would need the
     bounded `CausalOrderArbiter` this codebase
     already defines elsewhere, not wired into this shadow resolver yet);
  7. bounded size (`_MAX_COMPOSITE_SIZE = 2`, matching production's own
     ClaimCoverageBestTake composite bound) -- among all sets satisfying
     1-6, the SMALLEST member count wins, and ties break on the
     lexicographically smallest sorted tuple of realization_ids -- fully
     deterministic, never an arbitrary pick. When no 2-piece set
     satisfies 1-6, the resolver returns REVIEW_REQUIRED rather than
     search wider -- "prefer conservative fail-safe over invented
     composition" (D-050C1.6 Phase 5).

NO BEHAVIOR CUTOVER (non-negotiable)
=====================================
Grep-auditable: nothing in `pipeline.py`, `deterministic_best_take_
authority.py`, `claim_coverage_best_take.py`, `composite_resolver.py`,
`canonical_edit_plan.py`, `final_story_coherence_validation.py`,
`final_edit_reviewer.py`, `selection_boundary_contract.py`, or any
Boundary/Render/QC module imports this module or reads a
`RealizationResolution`/`ResolverReport` value. Its only production call
site is `universal_clean_cut.py`, writing into
`diagnostics["realization_resolver_shadow"]`, strictly after Freeze's own
gate has already been decided from the CURRENT engine's output -- see
that call site's own comment.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
import re
from typing import Mapping, Sequence

from .claim_coverage_best_take import (
    _content_overlap_coefficient,
    _HINDSIGHT_ALIGNABLE_CLAIM_TYPES,
    _HINDSIGHT_ALIGNMENT_AMBIGUOUS_FLOOR,
    _HINDSIGHT_PROTECTED_CLAIM_TYPES,
    _is_low_information_incidental,
)
from .contracts import DraftTimeline
from .final_sibling_grouping import _content, _negations
from .semantic_atom_importance import blocks_freeze, classify_number_atom
from .semantic_claims import (
    ACTION_EVENT,
    CAUSE_EFFECT,
    CONTRASTIVE_HINDSIGHT_NEGATION,
    ClaimEquivalenceArbiter,
    DIAGNOSIS_IDENTIFICATION,
    ENTITY_RELATION,
    MEASUREMENT_QUANTITY,
    NEGATION,
    STATE_RESULT,
    TEMPORAL_RELATION,
    UNIQUE_CONCLUSION,
    _CAUSE_EFFECT_MARKERS,
    _TEMPORAL_MARKERS,
    _negation_role_hard_exclusion,
    classify_claim,
)
from .semantic_ledger import (
    CanonicalClaimRecord,
    DELIVERY_SCORE_WINNER,
    DiscardRecord,
    ENGINE_BLOCKED_UNRESOLVED,
    ENGINE_RESOLVED_COMPOSITE,
    ENGINE_RESOLVED_WINNER,
    ENGINE_REVIEW_REQUIRED,
    RealizationRecord,
    SEMANTIC_WINNER_OVERRIDE,
    SemanticLedger,
)

# --- Requirement-group dedup -------------------------------------------------

_NEGATION_CONTENT_MARKERS = frozenset({"not", "never", "nunca", "sin", "without"})
_CLAIM_DEDUP_THRESHOLD = 0.7
# A lower bar than dedup: two claims are "topically related" (candidates for
# a *contradiction*, not a merge) once they share a substantial fraction of
# their non-digit content -- see `_detect_contradiction_signals`.
_CONTRADICTION_RELATEDNESS_THRESHOLD = 0.5

_CRITICAL = "CRITICAL"


_DIGIT_RUN_RE = re.compile(r"\d+")


def _digit_tokens(tokens: frozenset) -> frozenset:
    return frozenset(token for token in tokens if any(ch.isdigit() for ch in token))


def _claim_digit_values(claim: CanonicalClaimRecord) -> frozenset[str]:
    """Same normalization as `_numeric_values`, but read from the claim's
    own raw `text` (D-050C1 addition to `CanonicalClaimRecord`) when
    available rather than its already-filtered `content_tokens`.
    `semantic_claims._content`'s >=3-character floor silently drops a bare
    short digit run ("5", "10") while keeping a longer one ("10%",
    "5-10%") intact -- a real, asymmetric information loss a synthetic
    fixture won't show but real ASR text does (a single-digit percentage
    like "5%" is exactly 2 characters and vanishes from `content_tokens`
    entirely). Reading the raw clause text instead means "5%" vs "10%"
    reliably surfaces as a genuine value conflict rather than silently
    falling through to "no number on this side, skip the check" and
    deduping two claims that actually disagree. Falls back to
    `content_tokens` for a record built without `text` (every hand-built
    fixture in this codebase's own tests, and any Ledger populated before
    this field existed) -- never a hard requirement, always additive."""
    source = claim.text if claim.text else " ".join(claim.content_tokens)
    return frozenset(_DIGIT_RUN_RE.findall(source))


def _negation_tokens(tokens: frozenset) -> frozenset:
    return frozenset(token for token in tokens if token in _NEGATION_CONTENT_MARKERS)


def _claim_has_negation(claim: CanonicalClaimRecord) -> bool:
    """D-050C1.6 F2: same "read the raw text, not the filtered tokens"
    fix already applied to quantitative meaning (`_claim_digit_values`),
    now applied to polarity. `semantic_claims._content`'s >=3-character
    floor drops bare "no" (2 characters) entirely while keeping "not"/
    "never"/"nunca"/"sin"/"without" intact -- an asymmetry that silently
    blinded the polarity check to every Spanish negation phrased with a
    bare "no" (the D-050C1.5 multilingual-fixture finding, F2). Reuses
    `final_sibling_grouping._negations` -- the SAME general, unfiltered
    negation-marker check `semantic_claims.classify_claim` itself already
    calls to classify a claim NEGATION in the first place -- rather than
    reimplementing it, per this directive's "reuse... existing production
    semantics" instruction. Falls back to the content-token check for a
    record built without `text` (every hand-built fixture in this
    codebase's own tests)."""
    if claim.text:
        return bool(_negations(claim.text))
    return bool(_negation_tokens(claim.content_tokens))


def _jaccard(left: frozenset, right: frozenset) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _overlap_coefficient(left: frozenset, right: frozenset) -> float:
    """Containment-style similarity (Szymkiewicz-Simpson overlap
    coefficient): the shared fraction of the SMALLER side, not the union.
    Chosen over plain Jaccard for the rest-token equivalence check because
    a real sibling retry restatement routinely adds filler words around an
    identical core claim ("Solo un 5-10% son de carácter hereditario" vs
    "Así que estoy convencida que solo un 5-10% ... son de carácter
    hereditario") -- the shorter claim's own content is fully contained in
    the longer one, but plain Jaccard over the union would be dragged down
    by the longer claim's extra filler and could wrongly fail to dedup the
    literal D-049 Case B restatement. `0.0` for a degenerate empty side
    (never mistaken for "fully equivalent")."""
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


# D-050C1.6 Phase 6 (F4): below this floor, two claims' remaining content
# overlap is too thin for a paraphrase judgment to plausibly apply at all
# -- confidently NOT equivalent, the arbiter is never consulted (mirrors
# `semantic_claims.AMBIGUOUS_COVERAGE_FLOOR`'s own "too little overlap"
# floor). Between this floor and `_CLAIM_DEDUP_THRESHOLD` is the genuinely
# ambiguous band deterministic overlap alone cannot safely decide.
_DEDUP_AMBIGUOUS_FLOOR = 0.4


def _claims_dedup_equivalent(
    left: CanonicalClaimRecord, right: CanonicalClaimRecord, *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
    arbiter_log: list[dict] | None = None,
) -> bool:
    """See module docstring's CANONICAL CLAIM DEDUP section -- the hard
    gates (type, polarity, quantitative meaning) implemented verbatim,
    each a hard gate never a weighted score, followed by the deterministic
    overlap-coefficient check. D-050C1.6 F4: only when a pair clears every
    hard gate AND its overlap falls in the genuinely ambiguous band
    (`_DEDUP_AMBIGUOUS_FLOOR` to `_CLAIM_DEDUP_THRESHOLD`) is the existing,
    provider-neutral `semantic_claims.ClaimEquivalenceArbiter` contract
    (the SAME one `claim_coverage_best_take.py` already uses -- no new
    model/provider) consulted, bounded to exactly this one pair. Fails
    open to FALSE (distinct claims, never silently collapsed) whenever no
    arbiter is supplied, the arbiter raises, or it does not explicitly
    return `True` -- deterministic safe equivalence is always tried first
    and is sufficient on its own for the overwhelming majority of pairs;
    the arbiter only ever narrows a band deterministic overlap alone
    cannot safely decide, exactly like `resolve_ambiguous_coverage`'s own
    posture for coverage. Every consultation (queried, whether the
    arbiter was actually available, its verdict) is appended to
    `arbiter_log` when supplied, for `resolve_realizations_shadow`'s own
    diagnostics -- "record arbiter usage in resolver diagnostics"."""
    if left.claim_type != right.claim_type:
        return False
    if _claim_has_negation(left) != _claim_has_negation(right):
        return False
    # Quantitative meaning: compared from the claim's own raw TEXT (see
    # `_claim_digit_values`), not `content_tokens` alone -- either side
    # having ANY digit evidence in its raw text is enough to require exact
    # agreement, because text-based extraction survives the token-length
    # floor that would otherwise hide a real short-number conflict ("5%"
    # vs "10%").
    # D-050C1.6 F1: require agreement only when BOTH sides show digit
    # evidence in their raw text. `_claim_digit_values` reads raw text (not
    # the token-filtered set), so an EMPTY result now genuinely means "this
    # claim's text never mentions a number" -- not a tokenization artifact
    # -- and omitting a number is not the same as asserting a conflicting
    # one (a real conflict always has a number on both sides: that's what
    # makes it a conflict). Gating on "either side" here (the original
    # D-050C1 rule) wrongly blocked dedup between a claim and its own
    # paraphrase that merely adds an incidental year -- found via the
    # D-050C1.5 full sweep's `test_incidental_year_safely_omitted_...` /
    # `test_redundant_date_repeated_...` fixtures.
    left_values, right_values = _claim_digit_values(left), _claim_digit_values(right)
    if left_values and right_values:
        if left_values != right_values:
            return False
    left_digits, right_digits = _digit_tokens(left.content_tokens), _digit_tokens(right.content_tokens)
    left_rest = left.content_tokens - left_digits
    right_rest = right.content_tokens - right_digits
    overlap = _overlap_coefficient(left_rest, right_rest)
    if overlap >= _CLAIM_DEDUP_THRESHOLD:
        return True
    if overlap < _DEDUP_AMBIGUOUS_FLOOR or claim_equivalence_arbiter is None or not left.text or not right.text:
        return False
    try:
        covered, confidence, reason = claim_equivalence_arbiter.claim_covered(left.text, right.text)
        verdict = bool(covered) is True
    except Exception:
        verdict, confidence, reason = False, 0.0, "arbiter_exception"
    if arbiter_log is not None:
        arbiter_log.append({
            "left_claim_id": left.canonical_claim_id, "right_claim_id": right.canonical_claim_id,
            "overlap": overlap, "verdict": verdict, "confidence": confidence, "reason": reason,
        })
    return verdict


@dataclass(frozen=True)
class RequirementGroup:
    """One deduplicated audience-facing requirement for a semantic idea --
    the actual unit of critical-coverage bookkeeping below. `group_id` is
    the lexicographically smallest member `canonical_claim_id` (stable and
    deterministic, never an incrementing counter)."""

    group_id: str
    claim_type: str
    importance: str
    member_claim_ids: tuple[str, ...]


_IMPORTANCE_RANK = {"CRITICAL": 3, "SUPPORTING": 2, "CONTEXTUAL": 1, "REDUNDANT": 0}


def _effective_importance(raw_importance: str, group_members: Sequence[CanonicalClaimRecord]) -> str:
    """D-050C1.6 F1: downgrades a requirement group's raw CRITICAL
    importance to SUPPORTING when it is both (a) low-information/incidental
    on its OWN terms -- reusing `claim_coverage_best_take._is_low_
    information_incidental` directly (ONE shared utility, per this
    directive's instruction, rather than reimplementing D-047/D-048 FIX
    2's own careful marker logic) -- and (b) source-exclusive: no OTHER
    realization independently produced this same requirement, mirroring
    production's own `_override_blocked_by_incidental_self_source_claims`
    compound rule ("independently corroborated elsewhere" always keeps a
    claim critical, exclusivity alone is never disqualifying). A bare
    year/date context, an incidental temporal aside, or a low-information
    self-referential remark ("no se por que me pasaba eso") stops forcing
    a composite or disqualifying an otherwise-complete winner UNLESS a
    second, independent realization also raised it (then it's evidently
    not a fluke) or it carries its own substantive marker (diagnosis,
    correction, a genuinely unit-qualified number, cause-effect,
    unique-conclusion, or result-state language -- `_is_low_information_
    incidental` already protects every one of those). This is general
    linguistic-marker logic, not a Video00-specific heuristic -- exactly
    the same claim shapes `claim_coverage_best_take.py` itself protects
    against, applied here at the shared-requirement-group level."""
    if raw_importance != _CRITICAL:
        return raw_importance
    if not all(_is_low_information_incidental(c) for c in group_members):
        return raw_importance
    distinct_realizations = {rid for c in group_members for rid in c.source_realization_ids}
    if len(distinct_realizations) > 1:
        return raw_importance  # corroborated by more than one realization -- not a fluke
    return "SUPPORTING"


def build_requirement_groups(
    claims: Sequence[CanonicalClaimRecord], *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
    arbiter_log: list[dict] | None = None,
) -> tuple[RequirementGroup, ...]:
    """Greedy, deterministic clustering (same style as `semantic_claims.
    dedupe_claims`): claims are visited in a stable order (sorted by
    `canonical_claim_id`) and each either joins the first existing group
    whose representative it is dedup-equivalent to, or starts a new one.
    D-050C1.6 F4: `claim_equivalence_arbiter`/`arbiter_log` pass straight
    through to `_claims_dedup_equivalent` -- see that function's own
    docstring for exactly when the arbiter is (and is never) consulted."""
    ordered = sorted(claims, key=lambda c: c.canonical_claim_id)
    groups: list[list[CanonicalClaimRecord]] = []
    for claim in ordered:
        joined = False
        for group in groups:
            if _claims_dedup_equivalent(
                claim, group[0], claim_equivalence_arbiter=claim_equivalence_arbiter, arbiter_log=arbiter_log,
            ):
                group.append(claim)
                joined = True
                break
        if not joined:
            groups.append([claim])
    result = []
    for group in groups:
        member_ids = tuple(sorted({c.canonical_claim_id for c in group}))
        importance = max((c.importance for c in group), key=lambda imp: _IMPORTANCE_RANK.get(imp, 0))
        importance = _effective_importance(importance, group)
        result.append(RequirementGroup(
            group_id=member_ids[0], claim_type=group[0].claim_type,
            importance=importance, member_claim_ids=member_ids,
        ))
    return tuple(sorted(result, key=lambda g: g.group_id))


def _covered_group_ids(realization: RealizationRecord, groups: Sequence[RequirementGroup]) -> frozenset[str]:
    claim_ids = frozenset(realization.claim_ids)
    return frozenset(g.group_id for g in groups if claim_ids & frozenset(g.member_claim_ids))


# --- Contradiction detection --------------------------------------------------

@dataclass(frozen=True)
class ContradictionSignal:
    realization_a: str
    realization_b: str
    claim_type: str
    reason: str


def _detect_contradiction_signals(
    claims_by_realization: Mapping[str, tuple[CanonicalClaimRecord, ...]],
) -> tuple[ContradictionSignal, ...]:
    """Two claims from DIFFERENT realizations of the same idea are a
    contradiction signal -- not a dedup merge, not independent coverage --
    when they are the same `claim_type`, topically related (substantial
    overlap on their non-digit content), and NOT dedup-equivalent (see
    `_claims_dedup_equivalent`): same topic, incompatible specifics
    (different numbers, or one negated and the other not). D-020's
    "contradictory retries block Selection Freeze for human review" is the
    live engine's own posture for exactly this shape; this module mirrors
    it as a hard gate on composite formation (see module docstring,
    COMPOSITE MODEL criterion 3) and on straight single-winner selection."""
    signals: list[ContradictionSignal] = []
    realization_ids = sorted(claims_by_realization)
    for i, rid_a in enumerate(realization_ids):
        for rid_b in realization_ids[i + 1:]:
            for claim_a in claims_by_realization[rid_a]:
                for claim_b in claims_by_realization[rid_b]:
                    # D-050C1.6 F2/F3: a genuine polarity conflict is, BY
                    # CONSTRUCTION, almost always a claim_type MISMATCH --
                    # `classify_claim` gives a negated proposition its own
                    # NEGATION type, so "no soy la unica" (NEGATION) vs
                    # "soy la unica" (e.g. UNIQUE_CONCLUSION) can never
                    # share a claim_type. The original same-claim_type-only
                    # gate structurally could never see this shape at all.
                    # Comparable now when types match OR either side is
                    # NEGATION or CORRECTION (the two types whose whole
                    # purpose is to assert something ABOUT another claim).
                    same_type = claim_a.claim_type == claim_b.claim_type
                    is_correction_pair = "CORRECTION" in (claim_a.claim_type, claim_b.claim_type)
                    comparable_types = {claim_a.claim_type, claim_b.claim_type} & {"NEGATION", "CORRECTION"}
                    if not same_type and not comparable_types:
                        continue
                    if same_type and _claims_dedup_equivalent(claim_a, claim_b):
                        continue
                    digits_a, digits_b = _digit_tokens(claim_a.content_tokens), _digit_tokens(claim_b.content_tokens)
                    rest_a = claim_a.content_tokens - digits_a
                    rest_b = claim_b.content_tokens - digits_b
                    relatedness = _jaccard(rest_a, rest_b)
                    negation_mismatch = _claim_has_negation(claim_a) != _claim_has_negation(claim_b)
                    values_a, values_b = _claim_digit_values(claim_a), _claim_digit_values(claim_b)
                    quantity_conflict = bool(values_a) and bool(values_b) and values_a != values_b
                    # F3: a CORRECTION claim is very often anaphoric ("it
                    # was actually 10%") with little or no lexical overlap
                    # with the claim it corrects -- requiring topical
                    # relatedness here (as for an ordinary same-topic
                    # conflict) would let a genuinely ambiguous correction
                    # slip through ungated. A CORRECTION-involving pair is
                    # always examined for conflict; `_correction_
                    # explicitly_supersedes` below (which DOES require its
                    # own topical-relatedness check) is what decides
                    # whether it's a safe, explicit correction or a
                    # blocking, unresolved one.
                    conflict_detected = negation_mismatch or quantity_conflict
                    if not is_correction_pair and relatedness < _CONTRADICTION_RELATEDNESS_THRESHOLD:
                        conflict_detected = False
                    if conflict_detected:
                        reason = "negation_polarity_conflict" if negation_mismatch else "quantitative_value_conflict"
                        if _correction_explicitly_supersedes(claim_a, claim_b):
                            continue  # F3: an explicit, safe correction -- not a contradiction
                        if _correction_explicitly_supersedes(claim_b, claim_a):
                            continue
                        signals.append(ContradictionSignal(rid_a, rid_b, claim_a.claim_type, reason))
    return tuple(signals)


def _correction_explicitly_supersedes(
    correction_claim: CanonicalClaimRecord, prior_claim: CanonicalClaimRecord,
) -> bool:
    """D-050C1.6 F3: a CORRECTION-typed claim is only ever trusted to
    supersede a conflicting prior numeric/factual claim when its OWN raw
    text explicitly names and rejects the prior claim's specific value --
    e.g. "actually it was 10%, not 5%" (contains "5" within a short window
    of a negation marker) -- never merely because a generic correction
    marker ("actually", "I checked") is present somewhere in the clip.
    "Actually, I checked, and it was 2020" carries a correction marker but
    never actually restates or negates "2019" -- exactly the ambiguous
    shape this directive requires falling open to REVIEW_REQUIRED rather
    than guessed at. Also requires SOME non-digit topical overlap between
    the two claims (guards against "correction of a different entity":
    two claims can share a bare digit substring -- "5" a dose vs "5%" a
    rate -- purely by coincidence, with no shared topic at all). `False`
    whenever `correction_claim` isn't CORRECTION-typed, either claim lacks
    raw `text`, the prior value never appears (explicitly rejected) in the
    correction claim's own text, or the two claims share no topical
    content at all."""
    if correction_claim.claim_type != "CORRECTION":
        return False
    if not correction_claim.text or not prior_claim.text:
        return False
    prior_values = _claim_digit_values(prior_claim)
    if not prior_values:
        return False
    correction_digits = _digit_tokens(correction_claim.content_tokens)
    prior_digits = _digit_tokens(prior_claim.content_tokens)
    if _jaccard(correction_claim.content_tokens - correction_digits, prior_claim.content_tokens - prior_digits) < 0.2:
        return False  # no shared topic at all -- never treat as the same proposition
    correction_text = correction_claim.text.casefold()
    negation_markers = ("no ", "not ", "sin ", "nunca", "never")
    for value in prior_values:
        idx = correction_text.find(value)
        if idx == -1:
            continue
        window = correction_text[max(0, idx - 25):idx]
        if any(marker in window for marker in negation_markers):
            return True
    return False


# --- Delivery evidence (not authority) ---------------------------------------

def _delivery_scores_by_clip(ledger: SemanticLedger) -> dict[str, float]:
    """Raw DeliveryScorer evidence (`take_judge.rank_takes`'s own per-clip
    score, as already recorded on the Ledger's DELIVERY_SCORE_WINNER
    decisions) -- consumed here strictly as a tie-breaker among candidates
    already proven safe and complete, never to override a safety or
    coverage verdict. See module docstring's precedence order."""
    scores: dict[str, float] = {}
    for decision in ledger.decisions():
        if decision.decision_type != DELIVERY_SCORE_WINNER:
            continue
        for row in decision.evidence.get("ranked") or ():
            clip_id = row.get("clip_id")
            score = row.get("score")
            if clip_id and score is not None:
                try:
                    scores[clip_id] = max(scores.get(clip_id, float("-inf")), float(score))
                except (TypeError, ValueError):
                    continue
    return scores


def _realization_delivery_score(record: RealizationRecord, clip_scores: Mapping[str, float]) -> float | None:
    candidates = [clip_scores[c] for c in record.clip_ids if c in clip_scores]
    return max(candidates) if candidates else None


# D-058 Phase 2: same confidence floor `pipeline.py`'s own
# `_semantic_best_take` already applies before it will override a local
# DeliveryScorer pick -- reused verbatim so the Resolver's own notion of
# "high confidence" can never disagree with the stage that produced the
# evidence in the first place.
_HIGH_CONFIDENCE_SEMANTIC_WINNER_THRESHOLD = 0.85


def _semantic_winner_confidence_by_realization(
    ledger: SemanticLedger, idea_id: str, candidate_ids: tuple[str, ...],
) -> dict[str, float]:
    """The Ledger's own recorded `SEMANTIC_WINNER_OVERRIDE` evidence for this
    idea, keyed by realization_id -> highest recorded confidence. Absence
    from this mapping means no such decision was ever recorded for that
    realization -- never guessed at as high confidence."""
    candidate_set = frozenset(candidate_ids)
    confidence_by_realization: dict[str, float] = {}
    for decision in ledger.decisions():
        if decision.decision_type != SEMANTIC_WINNER_OVERRIDE:
            continue
        if decision.semantic_idea_id != idea_id:
            continue
        rid = decision.subject_realization_id
        if rid not in candidate_set:
            continue
        confidence = float(decision.evidence.get("confidence") or 0.0)
        if confidence > confidence_by_realization.get(rid, 0.0):
            confidence_by_realization[rid] = confidence
    return confidence_by_realization


# --- Composite search ---------------------------------------------------------

# D-050C1.6 F5: bounded at 2, matching production's own ClaimCoverageBestTake
# composite bound (claim_coverage_best_take.py never assembles more than a
# pair either) -- the D-050C1.5 full sweep's one real POTENTIAL_REGRESSION
# was exactly a 3-fragment assembly production's own bounded resolution
# would never attempt, correctly deferring to REVIEW_REQUIRED instead. See
# module docstring's COMPOSITE MODEL section for the full criteria list.
_MAX_COMPOSITE_SIZE = 2

# D-050C1.6 F5: a small allowance for adjacent (not overlapping) windows,
# same tolerance concept as `claim_coverage_best_take._time_compatible`.
_COMPOSITE_TIME_TOLERANCE_SEC = 0.05


def _composite_members_temporally_compatible(members: Sequence[RealizationRecord]) -> bool:
    """D-050C1.6 F5 criterion (sentence-fragment discontinuity / order
    safety): every pair of composite members must occupy non-overlapping
    time windows -- two physically-overlapping deliveries can never be
    genuine sequential pieces of one coherent realization, whatever their
    claim coverage says. This is a structural, general proxy for "pieces
    do not create sentence-fragment discontinuity" and "narrative/causal
    order is valid": true semantic causal-order validation would need the
    bounded `CausalOrderArbiter` this codebase already defines elsewhere
    (`causal_order_validator.py`) -- deliberately NOT wired into this
    shadow resolver yet (that arbiter-wiring work is F4's own later
    phase); this deterministic timing check is what the resolver can
    safely enforce without one, and it is honestly narrower than full
    narrative coherence, not a claim to have solved it."""
    windows = sorted((m.start, m.end) for m in members)
    for (_, end_a), (start_b, _) in zip(windows, windows[1:]):
        if start_b < end_a - _COMPOSITE_TIME_TOLERANCE_SEC:
            return False
    return True


def _find_minimal_composite(
    candidate_ids: Sequence[str],
    realizations: Mapping[str, RealizationRecord],
    groups: Sequence[RequirementGroup],
    critical_group_ids: frozenset[str],
    unsafe_ids: frozenset[str],
    contradiction_pairs: frozenset[frozenset[str]],
) -> tuple[str, ...] | None:
    """See module docstring's COMPOSITE MODEL. Tries member counts from 2
    upward (never proposes a 1-member "composite" -- that is just a single
    winner) so the first valid set found is already minimal; ties within
    one size break on sorted-tuple order for full determinism."""
    coverage_by_id = {rid: _covered_group_ids(realizations[rid], groups) for rid in candidate_ids}
    # NOTE on `complete_idea` and composite eligibility: an EARLIER version
    # of this function excluded any `complete_idea is False` realization
    # from composite membership, reasoning that "each member is
    # independently usable" (D-050C1.6 Phase 5) meant each piece had to be
    # complete on its own. Empirically wrong, caught by re-running the full
    # CleanCutBench sweep after adding it: `test_complementary_critical_
    # claims_require_a_composite` composites TWO realizations BOTH marked
    # `complete_idea=False` by design -- that is the whole point of a
    # composite (`claim_coverage_best_take.py`'s own real composite
    # formation never checks completeness either, only claim coverage and
    # time compatibility). Gating on it here would have silently regressed
    # a validated production behavior. "Independently usable" is enforced
    # instead by the two checks below (temporal non-overlap + bounded size
    # + no-redundant-member) -- completeness alone is not the right signal.
    safe_ids = [rid for rid in candidate_ids if rid not in unsafe_ids]
    for size in range(2, min(_MAX_COMPOSITE_SIZE, len(safe_ids)) + 1):
        for combo in sorted(combinations(sorted(safe_ids), size)):
            if any(frozenset(pair) in contradiction_pairs for pair in combinations(combo, 2)):
                continue
            if not _composite_members_temporally_compatible([realizations[rid] for rid in combo]):
                continue
            union_coverage: frozenset[str] = frozenset()
            for rid in combo:
                union_coverage |= coverage_by_id[rid]
            if not critical_group_ids.issubset(union_coverage):
                continue
            # Criterion 5: every member must contribute >=1 group no other
            # chosen member already covers (no redundant member).
            redundant = False
            for rid in combo:
                others_coverage: frozenset[str] = frozenset()
                for other in combo:
                    if other != rid:
                        others_coverage |= coverage_by_id[other]
                if coverage_by_id[rid].issubset(others_coverage):
                    redundant = True
                    break
            if redundant:
                continue
            return combo
    return None


# --- Resolution result --------------------------------------------------------

RESOLVED_WINNER = "RESOLVED_WINNER"
RESOLVED_COMPOSITE = "RESOLVED_COMPOSITE"
REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True)
class RealizationResolution:
    semantic_idea_id: str
    candidate_realization_ids: tuple[str, ...]
    winner_realization_id: str | None
    composite_realization_ids: tuple[str, ...]
    covered_canonical_claim_ids: tuple[str, ...]
    missing_critical_claim_ids: tuple[str, ...]
    discarded_realization_ids: tuple[str, ...]
    retained_for_contextual_value: tuple[str, ...]
    decision_status: str
    decision_reason: str
    confidence: float
    evidence: Mapping[str, object] = field(default_factory=dict)

    @property
    def review_required(self) -> bool:
        return self.decision_status == REVIEW_REQUIRED


PRE_GROUP_REJECTED = "PRE_GROUP_REJECTED"
REPLACEMENT_VERIFIED_SAFE = "REPLACEMENT_VERIFIED_SAFE"
# D-073: a SECOND, additive certification path alongside REPLACEMENT_
# VERIFIED_SAFE (PATH A -- see VERIFICATION_METHOD_LEXICAL below, mapping
# it to "existing Path A concept" per the D-073 directive). Both verdicts
# mean the exact same thing to every existing consumer of this field
# (`apply_authoritative_realization_resolution`'s own `if o.verdict ==
# REVIEW_REQUIRED` filter, its only reader below, never matches either
# VERIFIED verdict) -- PATH B never widens what "safe" means, it only adds
# a second, independently-verified way to reach the SAME safe conclusion.
REPLACEMENT_VERIFIED_SEMANTIC = "REPLACEMENT_VERIFIED_SEMANTIC"

VERIFICATION_METHOD_LEXICAL = "lexical"
VERIFICATION_METHOD_SEMANTIC = "semantic"


# D-076: SEMANTIC_PRESERVATION_PROOF -- see docs/CUTSELL_DECISIONS.md D-075
# for the accepted design and D-076 for this implementation. One reusable,
# directional evidence object answering the same question regardless of
# which existing authority produced it: "is discarded content D's required
# meaning already accounted for by selected realization R?"
# GROUPED_SAME_IDEA is minted by StoryValidator's own existing
# `_same_idea_paraphrase_credit` (final_story_coherence_validation.py,
# D-061 Phase 1) -- unchanged, not reframed into this dataclass here (it
# already IS a proof-shaped decision at its own point of use).
# LEXICAL_REPLACEMENT/SEMANTIC_REPLACEMENT are this module's own existing
# PATH A/PATH B verdicts (`REPLACEMENT_VERIFIED_SAFE`/`REPLACEMENT_
# VERIFIED_SEMANTIC`), reframed into this shape by `build_semantic_
# preservation_proofs` below purely for StoryValidator's consumption --
# PATH A/PATH B's OWN verdict logic is never touched.
# PRE_GROUP_SEMANTIC_PRESERVATION is the one genuinely NEW certification
# path this directive adds -- see `_attempt_pre_group_semantic_
# preservation` below.
PROOF_METHOD_GROUPED_SAME_IDEA = "GROUPED_SAME_IDEA"
PROOF_METHOD_LEXICAL_REPLACEMENT = "LEXICAL_REPLACEMENT"
PROOF_METHOD_SEMANTIC_REPLACEMENT = "SEMANTIC_REPLACEMENT"
PROOF_METHOD_PRE_GROUP_SEMANTIC_PRESERVATION = "PRE_GROUP_SEMANTIC_PRESERVATION"
# D-079: the one genuinely NEW certification path THIS directive adds --
# see `_attempt_intra_idea_semantic_preservation` below. Closes the D-078
# forensic's own gap: a discarded realization that DID reach grouping (has
# a `semantic_idea_id`) and lost, WITHIN its own idea, to that idea's own
# resolved winner/composite (`resolve_realizations_shadow`'s own per-idea
# resolution) was never evaluated by ANY claim-level certification chain
# at all -- PATH A/PATH B only ever run for TRUE orphans (no semantic_
# idea_id), D-076's pre-group only for discards that never reached
# `hybrid_editorial_chunks`'s own semantic judgment. This is the THIRD,
# remaining population.
PROOF_METHOD_INTRA_IDEA_SEMANTIC_PRESERVATION = "INTRA_IDEA_SEMANTIC_PRESERVATION"


@dataclass(frozen=True)
class SemanticPreservationProof:
    """D-076: an additive, directional evidence object -- never a second
    authority. Every field is populated by the SAME certification chain
    PATH B already uses (`_certify_directional_semantic_preservation`);
    this dataclass only packages the result for StoryValidator's
    consumption (`final_story_coherence_validation.py` reads `.verified`
    and, when true, the remaining fields -- it computes nothing of its own
    from this object). `verified=False` proofs are still returned (never
    dropped) so `rejection_reason` stays visible in diagnostics per
    Section 14 -- StoryValidator itself only ever acts on a verified one."""

    discarded_id: str
    preserving_id: str | None
    proof_method: str  # "" | one of the PROOF_METHOD_* constants above
    preserved_claim_ids: tuple[str, ...] = ()
    # D-076 Section 6: an atom (today, always a number) that IS absent from
    # the preserving realization's own text but that the EXISTING
    # `semantic_atom_importance.classify_number_atom`/`blocks_freeze`
    # classifiers already establish as non-required -- reused verbatim,
    # never a new importance tier. Always recorded, never silently dropped,
    # regardless of whether the overall proof ends up verified.
    nonrequired_omissions: tuple[Mapping[str, object], ...] = ()
    hard_gate_results: Mapping[str, object] = field(default_factory=dict)
    arbiter_invoked: bool = False
    verified: bool = False
    rejection_reason: str = ""
    # D-077 Section 12: every individual claim-equivalence arbiter
    # consultation attempted for this discard (whether or not it was
    # actually reachable/available, and regardless of the overall proof's
    # final verdict) -- deterministic preservation result, eligibility,
    # invocation, provider/model, verdict, confidence, and best-effort
    # cost/budget evidence per consultation (see `_arbiter_provenance`).
    # Empty whenever no claim pair of this discard ever reached the
    # genuinely ambiguous cross-type band (D-077 Section 2's own
    # "deterministic first" requirement -- most pairs never populate
    # this at all).
    arbiter_consultations: tuple[Mapping[str, object], ...] = ()
    # D-076 Section 4 observability: which strong-relation evidence (if
    # any) made a PRE_GROUP_SEMANTIC_PRESERVATION candidate eligible for
    # certification in the first place -- "" for the other three proof
    # methods, which have their own, separately-diagnosed discovery
    # mechanisms (D-072's pre-guard candidate, StoryValidator's take-judge
    # grouping).
    relationship_evidence: str = ""
    candidate_discovery_method: str = ""


@dataclass(frozen=True)
class OrphanRealizationReview:
    """See module docstring's Hard Invariant E: a realization the current
    engine discarded with no `semantic_idea_id` at all. D-050D1 splits
    this into two contracts that used to be conflated under one blanket
    "unresolved orphan" umbrella:

      PRE_GROUP_REJECTED  -- this candidate never reached hybrid
                              editorial's own semantic judgment at all
                              (an ordinary deterministic clean_cut/
                              provider-judgement rejection, or a
                              draft_review removal of content grouping
                              never touched). It legitimately never had
                              a semantic_idea_id to lose -- there is
                              nothing here for D-049 Case A to be
                              concerned about, because nothing here was
                              ever judged meaningful enough to route
                              through a semantic delete path in the
                              first place. Safe; does not block Freeze.

      REVIEW_REQUIRED     -- this candidate DID reach hybrid_editorial's
                              own semantic delete decision (`discarding_
                              stage == "hybrid_editorial_chunks"`) and
                              was explicitly judged meaningful/unique
                              enough to record a `delete_basis`, with no
                              verified replacement -- D-049 Case A's
                              exact shape, the "true orphan" this module
                              can never silently confirm as safe. Still
                              blocks Freeze, unconditionally, regardless
                              of how many PRE_GROUP_REJECTED realizations
                              exist alongside it.

    (REPLACEMENT_VERIFIED_SAFE is the third, pre-existing verdict: a
    discard -- of either origin -- whose replacement IS verified.)

    This distinction is drawn ONLY from `DiscardRecord.discarding_stage`,
    itself set once and unconditionally by the Ledger's own read-only
    reconstruction (semantic_ledger.py) -- it never reclassifies a
    hybrid_editorial delete as safe, and never demotes it, regardless of
    identity closure elsewhere. `apply_authoritative_realization_
    resolution` escalates `any_review_required` only for the
    `REVIEW_REQUIRED` verdict; `PRE_GROUP_REJECTED` realizations are
    still fully recorded here (Section 5's Ledger-registration
    requirement) but never force that escalation on their own.

    D-073: a would-be REVIEW_REQUIRED orphan (reached hybrid_editorial's
    own semantic delete decision, no LEXICAL replacement ever verified)
    gets exactly ONE more chance, attempted here and here only: PATH B
    semantic replacement certification (`_attempt_semantic_replacement_
    certification`), using ONLY evidence the Ledger already reconstructed
    (claims, negation role, realization state, the D-072-surfaced
    pre-guard candidate). Success upgrades the verdict to
    `REPLACEMENT_VERIFIED_SEMANTIC`; anything short of full, directional,
    claim-level preservation leaves it REVIEW_REQUIRED, unchanged. This
    never touches, weakens, or bypasses PATH A (`REPLACEMENT_VERIFIED_
    SAFE`) -- PATH A is tried first, unconditionally, exactly as before;
    PATH B is only ever consulted when PATH A already failed."""

    realization_id: str
    discard_reason: str
    replacement_realization_id: str | None
    replacement_verified: bool
    verdict: str  # "REPLACEMENT_VERIFIED_SAFE" | "REPLACEMENT_VERIFIED_SEMANTIC" | "REVIEW_REQUIRED" | "PRE_GROUP_REJECTED"
    decision_reason: str
    # D-073 additive fields -- "" / None / empty mapping whenever PATH B
    # was never attempted (PATH A already succeeded, or this is a
    # PRE_GROUP_REJECTED discard) or found nothing to certify.
    verification_method: str = ""  # "" | "lexical" | "semantic"
    semantic_replacement_evidence: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolverReport:
    idea_resolutions: Mapping[str, RealizationResolution]
    orphan_reviews: tuple[OrphanRealizationReview, ...]
    # D-050C1.6 F4: every claim-equivalence arbiter consultation this run
    # made (see `_claims_dedup_equivalent`'s own docstring) -- "record
    # arbiter usage in resolver diagnostics". Empty whenever no arbiter
    # was supplied to `resolve_realizations_shadow`, or none of this run's
    # claim pairs ever fell in the ambiguous band.
    arbiter_consultations: tuple[dict, ...] = ()

    @property
    def total_ideas(self) -> int:
        return len(self.idea_resolutions)

    @property
    def review_required_count(self) -> int:
        return sum(1 for r in self.idea_resolutions.values() if r.review_required) + sum(
            1 for o in self.orphan_reviews if o.verdict == REVIEW_REQUIRED
        )


# D-073 Section 8/12: CAUSE_EFFECT and TEMPORAL_RELATION claims can express
# OPPOSITE meanings ("A caused B" vs "B caused A", "X then Y" vs "Y then X")
# while sharing an IDENTICAL bag-of-words content-token set --
# `_claims_dedup_equivalent`'s deterministic overlap check (and every other
# overlap-coefficient check in this module) is structurally blind to word
# ORDER/argument direction, and this codebase has no existing, tested
# causal/temporal-direction arbiter question to safely delegate to either
# (the bounded `ClaimEquivalenceArbiter.claim_covered` protocol was
# designed and tested for topical/paraphrase coverage, never validated
# against direction-reversal specifically). D-073 Section 12 requires
# causal and temporal reversal to NEVER verify, unconditionally -- not
# "never verify without an arbiter" -- so PATH B never attempts to
# preserve a CAUSE_EFFECT/TEMPORAL_RELATION claim by any means, regardless
# of a configured arbiter's own verdict. A future, separately-authorized
# directive introducing a dedicated, tested causal/temporal-order check
# could safely revisit this; PATH B does not invent one now (mirrors this
# module's own existing stance on `CausalOrderArbiter` -- "deliberately
# NOT wired in yet").
_DIRECTION_SENSITIVE_CLAIM_TYPES = frozenset({CAUSE_EFFECT, TEMPORAL_RELATION})

# D-073 Section 8/12 (continued): `classify_claim` (semantic_claims.py) only
# ever assigns CAUSE_EFFECT to a clause split on one of `_CAUSE_EFFECT_
# MARKERS`'s own CONNECTOR phrases ("because", "due to", "therefore", ...);
# a bare causal VERB with no connector ("The medication caused the rash.")
# is classified plain ACTION_EVENT -- `_DIRECTION_SENSITIVE_CLAIM_TYPES`
# alone therefore never fires for this realistic, common shape, and its
# reversal ("The rash caused the need for medication.") would otherwise
# pass `_claim_content_subsumed` as a false "superset" (both sides share
# the same bag of words; only the causal argument order differs). This is
# PATH B's own additional, deterministic, TEXT-level safety net -- not a
# change to `classify_claim`/`extract_claims` (out of scope, wide blast
# radius) -- reusing the SAME connector vocabulary already established for
# causal/temporal detection elsewhere in this codebase
# (`_CAUSE_EFFECT_MARKERS`, `_TEMPORAL_MARKERS`), plus a small, generic,
# bilingual set of causal VERB markers this module owns itself (no
# Video00-specific fact, disease, or product name, matching every other
# marker list's own convention). Any claim whose raw text contains one of
# these markers is treated as direction-sensitive regardless of its
# assigned `claim_type` -- fails closed, exactly like
# `_DIRECTION_SENSITIVE_CLAIM_TYPES` itself, no arbiter escape hatch.
_CAUSAL_VERB_MARKERS = (
    "caused", "causes", "causing", "led to", "leads to", "leading to",
    "resulted in", "results in", "resulting in", "triggered", "triggers",
    "brought on", "gave rise to",
    "causo", "causó", "provoco", "provocó", "genero", "generó",
    "produjo", "llevo a", "llevó a", "resulto en", "resultó en",
)


def _claim_signals_direction_sensitive(claim: CanonicalClaimRecord) -> bool:
    """True when `claim` must never be certified preserved by PATH B under
    any circumstance -- either `classify_claim` already assigned it
    CAUSE_EFFECT/TEMPORAL_RELATION, or its own raw text carries a causal/
    temporal connector or causal verb marker that a bag-of-words content-
    token comparison cannot safely disambiguate for direction. See
    `_DIRECTION_SENSITIVE_CLAIM_TYPES` and `_CAUSAL_VERB_MARKERS` docstrings
    for the full reasoning."""
    if claim.claim_type in _DIRECTION_SENSITIVE_CLAIM_TYPES:
        return True
    text = (claim.text or "").lower()
    if not text:
        return False
    for marker in _CAUSAL_VERB_MARKERS + _CAUSE_EFFECT_MARKERS + _TEMPORAL_MARKERS:
        if marker in text:
            return True
    return False


# D-073.1 (same-idea proxy safety audit): a claim/candidate pairing pass
# every deterministic content/negation/digit gate below AND still be an
# unsafe "replacement" when R's matching claim is REPORTED/ATTRIBUTED
# speech ("Some customers said it did not work for them...") while D's own
# claim is a direct, unattributed assertion of the identical words ("It did
# not work for me."). Bag-of-words content-token comparison cannot tell
# "I assert X" from "someone else is quoted asserting X" -- and in
# practice R's own clause reporting the third-party claim is very often
# the SETUP half of a contrastive rebuttal ("...but it worked great for
# me"), i.e. R's own NET assertion is the OPPOSITE of what the reported
# clause's bare words say. Proven via a concrete adversarial fixture
# during the D-073.1 audit (before this fix: `_claim_content_subsumed`
# and `_claims_dedup_equivalent` both incorrectly certified this pairing).
# Fix is a small, generic, bilingual reporting-verb/attribution marker set
# this module owns itself -- same established pattern as
# `_CAUSAL_VERB_MARKERS` -- not a change to `classify_claim`/`extract_
# claims` and not a new claim type. Only blocks the ASYMMETRIC case (R's
# matching claim carries attribution language D's own claim does not) --
# a D claim that itself already carries the same attribution framing may
# still match a same-attribution R claim normally.
_REPORTED_ATTRIBUTION_MARKERS = (
    "said", "says", "claimed", "claims", "reported", "reports",
    "mentioned", "mentions", "according to",
    "dijo", "dijeron", "afirmo", "afirmó", "afirmaron",
    "segun", "según", "comento", "comentó", "mencionó", "mencionaron",
)


def _claim_has_reported_attribution(claim: CanonicalClaimRecord) -> bool:
    text = (claim.text or "").lower()
    if not text:
        return False
    return any(marker in text for marker in _REPORTED_ATTRIBUTION_MARKERS)


def _preservation_blocked_by_attribution_asymmetry(
    d_claim: CanonicalClaimRecord, r_claim: CanonicalClaimRecord,
) -> bool:
    """True when `r_claim` may never be used to certify `d_claim` preserved
    -- R's own text attributes the claim to a third party ("said",
    "according to", ...) while D's own text asserts it directly, with no
    such attribution. See the module-level comment above this function's
    own marker set for the concrete adversarial case this closes."""
    return _claim_has_reported_attribution(r_claim) and not _claim_has_reported_attribution(d_claim)


def _realization_digit_values(text: str) -> frozenset[str]:
    """D-073 Section 8 hard NUMBER gate, read from the REALIZATION's own
    raw text (`RealizationRecord.text` -- each realization's own clip
    text, set once at Ledger construction and never merged/collapsed
    across sibling realizations) rather than from canonical claims.

    `mint_canonical_claim_id` (canonical_identity.py) deliberately groups
    claims by `(claim_type, content_tokens)` only -- NOT exact text --
    by design, for an unrelated D-050C purpose (crediting a same-idea
    paraphrase across sibling realizations). A side effect: two claims
    whose only difference is a digit ("...measured 3 centimeters..." vs
    "...measured 5 centimeters...") can share IDENTICAL content_tokens
    (the digit itself is not a content token) and therefore mint to the
    SAME `canonical_claim_id` -- meaning D's own realization and R's own
    realization can end up pointing at the literal same
    `CanonicalClaimRecord` object in `ledger.claims()`, whose `.text`
    reflects whichever clip was registered last. Comparing claims alone
    would then make D's claim trivially "preserved" by R's -- against
    itself. This function is PATH B's own independent, additional
    safety net against exactly that collision: it never trusts the
    Ledger's claim identity for numeric safety, only each realization's
    own untouched raw text. Not a fix to `mint_canonical_claim_id` itself
    (out of D-073's scope -- shared infra with a wide, unrelated blast
    radius); scoped entirely to this module's own PATH B gate."""
    return frozenset(_DIGIT_RUN_RE.findall(text or ""))


def _claim_content_subsumed(d_claim: CanonicalClaimRecord, r_claim: CanonicalClaimRecord) -> bool:
    """D-073 Sections 3/12: a cross-TYPE, deterministic superset check --
    every one of D's own content tokens appears verbatim in R's, AND every
    digit value D's own text carries is also present in R's (R may carry
    ADDITIONAL digit values beyond D's own -- a subset check, not exact
    equality: Section 3's "R may contain additional safe information"
    applies to numbers exactly like any other fact), AND negation polarity
    agrees. Deliberately not gated on claim_type equality (unlike
    `_claims_dedup_equivalent`): a richer candidate claim that folds D's
    fact together with an ADDITIONAL one is exactly the safe "R may
    contain additional information" direction Section 3 requires PATH B
    to recognize. Structurally cannot pass in the unsafe direction -- D's
    own words (or D's own numbers) can never simply be absent from R for
    this to return True."""
    if not d_claim.content_tokens:
        return False
    if not d_claim.content_tokens.issubset(r_claim.content_tokens):
        return False
    d_values = _claim_digit_values(d_claim)
    if d_values and not d_values.issubset(_claim_digit_values(r_claim)):
        return False
    if _claim_has_negation(d_claim) != _claim_has_negation(r_claim):
        return False
    return True


# D-077: the ONE new claim-relationship this directive adds -- a genuinely
# cross-TYPE ambiguous paraphrase, reachable ONLY when the caller explicitly
# opts in (`_claim_preserved`'s own `allow_cross_type_ambiguous_bridge`
# parameter, default False everywhere -- PATH B's own call site never
# passes it, so PATH B stays byte-identical). Candidate (R-side) types
# allowed here are every "asserts something specific" type EXCEPT the four
# whose whole purpose is asserting something ABOUT another claim or whose
# bag-of-words representation cannot safely encode argument direction
# (NEGATION, CORRECTION, CAUSE_EFFECT, TEMPORAL_RELATION -- the last two
# also independently checked via `_claim_signals_direction_sensitive`
# below, belt and braces).
_CROSS_TYPE_BRIDGE_ALLOWED_CANDIDATE_TYPES = frozenset({
    ENTITY_RELATION, STATE_RESULT, DIAGNOSIS_IDENTIFICATION,
    ACTION_EVENT, MEASUREMENT_QUANTITY, UNIQUE_CONCLUSION,
})


def _cross_type_ambiguous_bridge_eligible(
    d_claim: CanonicalClaimRecord, r_claim: CanonicalClaimRecord,
) -> tuple[bool, str]:
    """D-077 Sections 2/3/4/9: is this (d_claim, r_claim) pair a genuine
    ambiguous-middle-band cross-type paraphrase eligible for the bounded
    claim-equivalence arbiter -- never a candidate-discovery mechanism
    (Section 4's strong-relation requirement is enforced entirely upstream,
    by `_find_pre_group_candidates`; this function only ever judges a
    claim PAIR already belonging to an already-verified strong-relation
    realization pair). Returns `(eligible, gate_reason)` -- `gate_reason`
    is always populated, verified or not, for Section 12's own
    observability requirement.

    ENTITY/DIAGNOSIS SAFETY (Section 3/9), drawn structurally rather than
    via a new entity extractor ("no new provider/model/protocol"): requires
    `d_claim.claim_type == ACTION_EVENT` -- the marker-less, generic-
    statement fallback `classify_claim` assigns only when NOTHING more
    specific fired (no reporting/identification/quantity/state-result
    marker). A claim already classified ENTITY_RELATION/STATE_RESULT/
    DIAGNOSIS_IDENTIFICATION/MEASUREMENT_QUANTITY/UNIQUE_CONCLUSION/
    CORRECTION/NEGATION already asserts something SPECIFIC of its own (a
    diagnosis, a measured value, a reported result, a correction, a
    negation) -- D-073's own protected `test_different_diagnosis_never_
    verifies_even_with_confirming_arbiter` fixture ("...it was gastritis"
    vs "...it was an ulcer") is EXACTLY this shape (D's own claim there is
    ENTITY_RELATION, not ACTION_EVENT) and this gate never reaches it: the
    pre-existing exact-type-match dedup gate is what protects it, tried
    first, unchanged. An ACTION_EVENT `d_claim`, by construction, never
    itself names a specific diagnosis/measured value/reported result, so a
    richer candidate that develops it into one is always an ADDITION on
    R's side, never a substitution of a fact D itself already named --
    the same "R may contain additional safe information" direction
    `_claim_content_subsumed` already establishes for the literal-subset
    case, extended here across a type boundary for a genuine paraphrase.
    Same-type pairs are explicitly excluded (`_claims_dedup_equivalent`
    already tries -- and, for the ambiguous band, already arbitrates --
    every same-type pair; this function only ever adds the ONE new,
    additional relationship a same-type check cannot express)."""
    if d_claim.claim_type != ACTION_EVENT:
        return False, "source_not_generic_action_event"
    if r_claim.claim_type == d_claim.claim_type:
        return False, "same_type_handled_by_existing_dedup"
    if r_claim.claim_type not in _CROSS_TYPE_BRIDGE_ALLOWED_CANDIDATE_TYPES:
        return False, "candidate_type_not_bridge_eligible"
    if _claim_signals_direction_sensitive(d_claim) or _claim_signals_direction_sensitive(r_claim):
        return False, "direction_sensitive_claim"
    if _claim_has_negation(d_claim) != _claim_has_negation(r_claim):
        return False, "negation_polarity_mismatch"
    d_values, r_values = _claim_digit_values(d_claim), _claim_digit_values(r_claim)
    if d_values and not d_values.issubset(r_values):
        return False, "number_mismatch"
    d_digits, r_digits = _digit_tokens(d_claim.content_tokens), _digit_tokens(r_claim.content_tokens)
    overlap = _overlap_coefficient(d_claim.content_tokens - d_digits, r_claim.content_tokens - r_digits)
    if overlap < _DEDUP_AMBIGUOUS_FLOOR or overlap >= _CLAIM_DEDUP_THRESHOLD:
        return False, "outside_ambiguous_band"
    return True, "ambiguous_cross_type_candidate"


def _arbiter_provenance(claim_equivalence_arbiter: ClaimEquivalenceArbiter | None) -> Mapping[str, object]:
    """D-077 Section 12 (provider/model/cost-budget observability): best-
    effort, read-only introspection of whatever the CONFIGURED arbiter
    implementation happens to expose -- never fabricated, never required.
    `GoogleClaimEquivalenceArbiter` (D-061 Phase 2) exposes `.model` and
    `.settings.provider`/`.ledger.reserved_usd`/`.ledger.remaining_usd`;
    a test double (e.g. an always-YES arbiter) exposes none of these and
    every field simply falls back to "" / None via `getattr`."""
    if claim_equivalence_arbiter is None:
        return {"provider": "", "model": "", "reserved_usd": None, "remaining_usd": None}
    settings = getattr(claim_equivalence_arbiter, "settings", None)
    ledger = getattr(claim_equivalence_arbiter, "ledger", None)
    return {
        "provider": getattr(settings, "provider", "") or type(claim_equivalence_arbiter).__name__,
        "model": getattr(claim_equivalence_arbiter, "model", ""),
        "reserved_usd": getattr(ledger, "reserved_usd", None),
        "remaining_usd": getattr(ledger, "remaining_usd", None),
    }


# D-079 Phase 1/4: a small, FIXED, general (non-Video00-specific,
# bilingual) set of idiomatic discourse-aside phrases that carry a bare
# negation TOKEN ("no") without asserting a factual negation of anything
# -- "goes without saying" / "needless to say" phrasing, not "did not
# have X". D-078's own forensic (docs/CUTSELL_DECISIONS.md) traced a live
# CRITICAL_CLAIM_LOST false-positive to exactly this shape: "no hay que
# preguntar" ("no need to ask") tacked onto an otherwise-plain statement
# tripped `classify_claim`'s own unconditional "any negation token -> the
# WHOLE clause is NEGATION" rule (correct and untouched -- see below),
# which then tripped `claim_coverage()`'s negation-flip guard even though
# nothing about the clause's actual asserted content was negated.
#
# EXACT-PHRASE match only (never a bare "no"/"not" token) -- this is what
# makes it safe: "did not have gastritis" / "no tuve gastritis" contains
# no such phrase and is completely unaffected. Used ONLY by
# `_rhetorical_aside_negation_bridge_eligible` below, itself reachable
# ONLY via the new, opt-in `allow_rhetorical_aside_negation_bridge`
# parameter (default False everywhere except D-079's own new intra-idea
# certification call site) -- `classify_claim`, `_negations`,
# `_claim_has_negation`, and every existing NEGATION hard gate in this
# module (D-073's direction-sensitivity check, D-073.1's attribution
# check, PATH B's own dedup/subsumed/hindsight strategies) are completely
# untouched. This is an ADDITIVE, narrow exception -- never a global
# weakening of negation detection.
_RHETORICAL_ASIDE_NEGATION_MARKERS = (
    "no hay que preguntar", "no hay que decir", "no hace falta decir",
    "sin duda", "no cabe duda", "ni que decir tiene",
    "no need to ask", "needless to say", "no doubt", "without a doubt",
    "goes without saying",
)


def _rhetorical_aside_negation_bridge_eligible(
    d_claim: CanonicalClaimRecord,
) -> tuple[bool, str, CanonicalClaimRecord | None]:
    """D-079 Phase 1/4: does `d_claim`'s own NEGATION classification stem
    ONLY from a fixed, general rhetorical-aside phrase, with no genuine
    negation of its own asserted content? Returns `(eligible, reason,
    residual_claim_or_None)`. `residual_claim` -- when eligible -- is a
    NEW `CanonicalClaimRecord` (never mutates `d_claim`) built from
    `d_claim`'s own text with ONLY the matched phrase removed, then
    RE-classified via the SAME, unmodified `classify_claim` -- used only
    as an internal probe for the matching strategies below, never
    surfaced as `d_claim`'s own type/importance to any caller
    (`_lost_critical_claims` keeps validating the ORIGINAL NEGATION/
    CRITICAL claim; only the certification attempt uses the residual).

    Fails closed (not eligible) whenever:
      - `d_claim` is not NEGATION-typed at all;
      - its negation_role is already CONTRASTIVE_HINDSIGHT_NEGATION (D-066
        owns that shape via its own claim-vs-claim alignment machinery --
        never doubly handled here);
      - no exact marker phrase is present in the raw text;
      - stripping the phrase leaves no residual content at all;
      - the RESIDUAL text still contains ANY negation marker anywhere
        (`_negations`) -- a genuine second, real negation elsewhere in
        the same clause must never be bridged away;
      - the residual, independently re-classified, is STILL NEGATION --
        proof the match didn't actually isolate the true negation."""
    if d_claim.claim_type != NEGATION:
        return False, "not_negation_claim", None
    if d_claim.negation_role == CONTRASTIVE_HINDSIGHT_NEGATION:
        return False, "hindsight_negation_handled_elsewhere", None
    text = d_claim.text or ""
    if not text:
        return False, "no_raw_text", None
    lowered = text.lower()
    marker = next((m for m in _RHETORICAL_ASIDE_NEGATION_MARKERS if m in lowered), None)
    if marker is None:
        return False, "no_rhetorical_aside_marker", None
    pattern = re.compile(re.escape(marker), re.IGNORECASE)
    residual_text = pattern.sub(" ", text)
    residual_text = re.sub(r"\s*,\s*,\s*", ", ", residual_text)  # collapse a doubled comma left by the strip
    residual_text = residual_text.strip(" ,.")
    if not residual_text:
        return False, "no_residual_content", None
    if _negations(residual_text):
        return False, "residual_still_negated", None
    residual_type, _residual_importance, _evidence = classify_claim(residual_text)
    if residual_type == NEGATION:
        return False, "residual_reclassifies_negation", None
    residual_tokens = frozenset(_content(residual_text))
    if not residual_tokens:
        return False, "no_residual_content_tokens", None
    residual_claim = replace(
        d_claim, claim_type=residual_type, content_tokens=residual_tokens, text=residual_text,
    )
    return True, "rhetorical_aside_negation_isolated", residual_claim


def _claim_preserved(
    d_claim: CanonicalClaimRecord,
    r_claims: Sequence[CanonicalClaimRecord],
    *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None,
    arbiter_log: list[dict] | None,
    allow_cross_type_ambiguous_bridge: bool = False,
    allow_rhetorical_aside_negation_bridge: bool = False,
) -> tuple[CanonicalClaimRecord | None, str]:
    """D-073 Sections 3/5/8: is `d_claim` (one claim of the discarded
    realization D) preserved by some claim of the candidate replacement R?
    Returns (preserving_claim_or_None, method): "dedup_equivalent" reuses
    `_claims_dedup_equivalent` verbatim -- the SAME hard claim-type,
    negation-polarity, and quantitative-value gates plus deterministic
    overlap plus bounded arbiter escalation this module already uses for
    requirement-group dedup, giving PATH B the NUMBER/FACTUAL_NEGATION/
    entity-substitution safety Section 8 requires for free, by
    construction, not by a separate check. "content_subsumed" (D-073
    Section 3/12: "R may contain additional safe information") is a
    cross-TYPE superset check (`_claim_content_subsumed`) for exactly the
    case a richer candidate claim absorbs D's own fact into a differently-
    classified claim (e.g. `classify_claim` promotes a combined clause to
    MEASUREMENT_QUANTITY the moment a number appears, even though D's own
    half of it was plain ACTION_EVENT) -- `_claims_dedup_equivalent`'s
    exact claim_type gate would otherwise reject a genuinely safe superset
    match. Still hard-gated on digit-value and negation-polarity agreement.
    "hindsight_semantic" is D-066's own CONTRASTIVE_HINDSIGHT_NEGATION safe
    contract (same protected-type exclusion, same digit-evidence gate,
    same ambiguous-overlap floor, same bounded arbiter question --
    imported constants/functions, not redefined), applied to the Ledger's
    own claim representation since `semantic_claims.Claim`'s own helper
    functions use a different attribute name (`claim_id` vs
    `canonical_claim_id`) than `CanonicalClaimRecord`. Direction-sensitive
    claims (CAUSE_EFFECT/TEMPORAL_RELATION `claim_type`, OR raw text
    carrying a causal/temporal connector or causal-verb marker --
    `_claim_signals_direction_sensitive`) can NEVER be certified preserved
    by this function, unconditionally -- see `_DIRECTION_SENSITIVE_CLAIM_
    TYPES`'s and `_CAUSAL_VERB_MARKERS`'s own docstrings for why even a
    configured arbiter is not trusted here. Stable order (sorted by
    canonical_claim_id) -- first match wins, never an arbitrary pick.
    D-073.1: `r_claims` is filtered ONCE up front to drop any candidate
    claim blocked by `_preservation_blocked_by_attribution_asymmetry` --
    R's own reported/attributed speech can never certify a D claim asserted
    directly, in ANY of the matching strategies below (the three always
    tried, plus D-077's opt-in `cross_type_ambiguous_bridge` -- see that
    function's own docstring for the proven adversarial case).
    Fails closed (returns `(None, "none")`) whenever nothing matches."""
    if _claim_signals_direction_sensitive(d_claim):
        return None, "none"
    r_claims = tuple(
        r_claim for r_claim in r_claims
        if not _preservation_blocked_by_attribution_asymmetry(d_claim, r_claim)
    )
    for r_claim in sorted(r_claims, key=lambda c: c.canonical_claim_id):
        if _claims_dedup_equivalent(
            d_claim, r_claim, claim_equivalence_arbiter=claim_equivalence_arbiter, arbiter_log=arbiter_log,
        ):
            return r_claim, "dedup_equivalent"
    for r_claim in sorted(r_claims, key=lambda c: c.canonical_claim_id):
        if _claim_content_subsumed(d_claim, r_claim):
            return r_claim, "content_subsumed"
    if d_claim.claim_type == NEGATION and d_claim.negation_role == CONTRASTIVE_HINDSIGHT_NEGATION:
        for r_claim in sorted(r_claims, key=lambda c: c.canonical_claim_id):
            if r_claim.claim_type in _HINDSIGHT_PROTECTED_CLAIM_TYPES:
                continue
            if r_claim.claim_type not in _HINDSIGHT_ALIGNABLE_CLAIM_TYPES:
                continue
            if _claim_digit_values(d_claim) or _claim_digit_values(r_claim):
                continue
            if _negation_role_hard_exclusion(r_claim.text):
                continue
            overlap = _content_overlap_coefficient(d_claim.content_tokens, r_claim.content_tokens)
            if overlap < _HINDSIGHT_ALIGNMENT_AMBIGUOUS_FLOOR:
                continue
            if claim_equivalence_arbiter is None or not d_claim.text or not r_claim.text:
                continue
            try:
                covered, confidence, reason = claim_equivalence_arbiter.claim_covered(d_claim.text, r_claim.text)
                verdict = bool(covered) is True
            except Exception:
                verdict, confidence, reason = False, 0.0, "arbiter_exception"
            if arbiter_log is not None:
                arbiter_log.append({
                    "left_claim_id": d_claim.canonical_claim_id, "right_claim_id": r_claim.canonical_claim_id,
                    "overlap": overlap, "verdict": verdict, "confidence": confidence, "reason": reason,
                    "method": "hindsight_semantic",
                })
            if verdict:
                return r_claim, "hindsight_semantic"
    # D-077 Section 1/2/4: the ONE new arbiter-reachable relationship --
    # opt-in only (`allow_cross_type_ambiguous_bridge`, default False),
    # tried LAST (deterministic-first, per Section 2), after dedup_
    # equivalent/content_subsumed/hindsight_semantic have all already
    # failed deterministically for every candidate. `_cross_type_ambiguous_
    # bridge_eligible` re-verifies every hard gate (number, negation,
    # causal/temporal direction, entity/diagnosis-safety-by-construction)
    # independently before this point is ever reached -- see that
    # function's own docstring. A diagnostic entry is recorded for every
    # ELIGIBLE candidate regardless of arbiter availability or verdict
    # (Section 12: "confirm no arbiter call on deterministic pass/fail
    # cases" -- an ineligible pair, i.e. every deterministic pass/fail
    # case, never reaches this loop body at all, let alone logs or calls).
    if allow_cross_type_ambiguous_bridge:
        for r_claim in sorted(r_claims, key=lambda c: c.canonical_claim_id):
            eligible, gate_reason = _cross_type_ambiguous_bridge_eligible(d_claim, r_claim)
            if not eligible:
                continue
            if claim_equivalence_arbiter is None or not d_claim.text or not r_claim.text:
                if arbiter_log is not None:
                    arbiter_log.append({
                        "left_claim_id": d_claim.canonical_claim_id, "right_claim_id": r_claim.canonical_claim_id,
                        "method": "cross_type_ambiguous_bridge", "deterministic_result": "ambiguous",
                        "arbiter_eligible": True, "arbiter_invoked": False,
                        "verdict": False, "confidence": None, "reason": "arbiter_unavailable",
                        **_arbiter_provenance(claim_equivalence_arbiter),
                    })
                continue
            try:
                covered, confidence, reason = claim_equivalence_arbiter.claim_covered(d_claim.text, r_claim.text)
                verdict = bool(covered) is True
            except Exception:
                verdict, confidence, reason = False, 0.0, "arbiter_exception"
            if arbiter_log is not None:
                arbiter_log.append({
                    "left_claim_id": d_claim.canonical_claim_id, "right_claim_id": r_claim.canonical_claim_id,
                    "method": "cross_type_ambiguous_bridge", "deterministic_result": "ambiguous",
                    "arbiter_eligible": True, "arbiter_invoked": True,
                    "verdict": verdict, "confidence": confidence, "reason": reason,
                    **_arbiter_provenance(claim_equivalence_arbiter),
                })
            if verdict:
                return r_claim, "cross_type_ambiguous_bridge"
    # D-079 Phase 1/4: the ONE new arbiter-adjacent relationship this
    # directive adds -- opt-in only (`allow_rhetorical_aside_negation_
    # bridge`, default False), tried LAST, after every other strategy
    # (including the cross-type bridge above) has already failed for
    # `d_claim` itself. Eligibility is computed ONCE (it depends only on
    # `d_claim`'s own text, not on any candidate) via
    # `_rhetorical_aside_negation_bridge_eligible`; when eligible, the
    # SAME matching strategies above (dedup_equivalent, content_subsumed,
    # and -- only if the caller ALSO enabled it -- the cross-type bridge)
    # are retried against the RESIDUAL claim in place of `d_claim` --
    # never a new matching primitive, never a new hard-gate exemption.
    # `d_claim` itself (its own canonical_claim_id, claim_type, importance)
    # is NEVER mutated or replaced in the caller's own bookkeeping --
    # only this one internal certification attempt uses the residual.
    if allow_rhetorical_aside_negation_bridge:
        eligible, gate_reason, residual_claim = _rhetorical_aside_negation_bridge_eligible(d_claim)
        if eligible and residual_claim is not None:
            for r_claim in sorted(r_claims, key=lambda c: c.canonical_claim_id):
                if _claims_dedup_equivalent(
                    residual_claim, r_claim,
                    claim_equivalence_arbiter=claim_equivalence_arbiter, arbiter_log=arbiter_log,
                ):
                    return r_claim, "rhetorical_aside_negation_bridge"
            for r_claim in sorted(r_claims, key=lambda c: c.canonical_claim_id):
                if _claim_content_subsumed(residual_claim, r_claim):
                    return r_claim, "rhetorical_aside_negation_bridge"
            if allow_cross_type_ambiguous_bridge:
                for r_claim in sorted(r_claims, key=lambda c: c.canonical_claim_id):
                    eligible2, _gate2 = _cross_type_ambiguous_bridge_eligible(residual_claim, r_claim)
                    if not eligible2:
                        continue
                    if claim_equivalence_arbiter is None or not residual_claim.text or not r_claim.text:
                        if arbiter_log is not None:
                            arbiter_log.append({
                                "left_claim_id": d_claim.canonical_claim_id, "right_claim_id": r_claim.canonical_claim_id,
                                "method": "rhetorical_aside_negation_bridge", "deterministic_result": "ambiguous",
                                "arbiter_eligible": True, "arbiter_invoked": False,
                                "verdict": False, "confidence": None, "reason": "arbiter_unavailable",
                                **_arbiter_provenance(claim_equivalence_arbiter),
                            })
                        continue
                    try:
                        covered, confidence, reason = claim_equivalence_arbiter.claim_covered(
                            residual_claim.text, r_claim.text,
                        )
                        verdict = bool(covered) is True
                    except Exception:
                        verdict, confidence, reason = False, 0.0, "arbiter_exception"
                    if arbiter_log is not None:
                        arbiter_log.append({
                            "left_claim_id": d_claim.canonical_claim_id, "right_claim_id": r_claim.canonical_claim_id,
                            "method": "rhetorical_aside_negation_bridge", "deterministic_result": "ambiguous",
                            "arbiter_eligible": True, "arbiter_invoked": True,
                            "verdict": verdict, "confidence": confidence, "reason": reason,
                            **_arbiter_provenance(claim_equivalence_arbiter),
                        })
                    if verdict:
                        return r_claim, "rhetorical_aside_negation_bridge"
    return None, "none"


def _attempt_semantic_replacement_certification(
    discard: DiscardRecord,
    record: RealizationRecord,
    ledger: SemanticLedger,
    *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None,
) -> tuple[str | None, str | None, str, dict]:
    """D-073: PATH B, the Unified Resolver's own second, additive
    replacement-certification path -- see module docstring's DIRECTIONAL
    SEMANTIC REPLACEMENT CONTRACT. Attempted ONLY by `resolve_orphan_
    realizations_shadow`, and only for a discarded realization that
    already reached `hybrid_editorial_chunks`'s own semantic delete
    decision with no PATH A (lexical) verified replacement -- exactly the
    population that would otherwise become REVIEW_REQUIRED. Returns
    `(new_verdict_or_None, replacement_realization_id_or_None,
    decision_reason, evidence)`; `new_verdict` is
    `REPLACEMENT_VERIFIED_SEMANTIC` only when every one of the following
    holds, in order (any failure returns `(None, None, reason, evidence)`
    immediately -- fail-closed, never a partial/best-effort certification):

      1. A pre-guard candidate exists at all (`discard.pre_guard_
         candidate_clip_id`, D-072's own already-computed evidence --
         never guessed, never re-derived).
      2. That candidate clip maps to an EXISTING realization in this same
         Ledger.
      3. That realization is `state == "selected"` -- R is actually
         selected/kept; this IS this module's only available proxy for
         "verified same semantic idea/retry relation" for a TRUE orphan
         (one with no `semantic_idea_id` at all, by definition -- it never
         reached grouping, so no formal idea/retry-family id exists to
         match against; see module docstring).
      4. That realization is not proven incomplete (`complete_idea is not
         False`) -- a truncated/failed/unfinished candidate can never
         certify a replacement (Section 7).
      5. D has at least one extractable claim (a genuinely empty claim set
         is treated as unresolved/uncertain content, never vacuously safe
         -- WHEN UNCERTAIN, KEEP).
      6. No contradiction signal between D's and R's claims
         (`_detect_contradiction_signals`, reused verbatim).
      7. EVERY one of D's claims has a verified preserving claim on R's
         side (`_claim_preserved`) -- not just CRITICAL ones (Section 6:
         "not merely critical-claim-covered"; checking ALL claims is a
         strictly stronger, safer bar than critical-only).

    Never widens PATH A, never touches D-063/D-066's own rules (reuses
    their outputs only), never creates a second decision-making authority
    (this IS the existing Unified Resolver's own orphan-review authority,
    called only from within it)."""
    arbiter_log: list[dict] = []
    evidence: dict = {
        "semantic_replacement_evaluated": True,
        "candidate_replacement_realization_id": None,
        "same_idea_verified": False,
        "critical_claims_preserved": False,
        "unique_required_content_preserved": False,
        "hard_gate_results": {},
        "arbiter_invoked": False,
        "semantic_replacement_verified": False,
        "semantic_replacement_reason": "",
        "preserved_claim_ids": [],
    }

    candidate_clip_id = discard.pre_guard_candidate_clip_id
    if not candidate_clip_id:
        evidence["semantic_replacement_reason"] = "no_pre_guard_candidate"
        return None, None, evidence["semantic_replacement_reason"], evidence

    realizations = ledger.realizations()
    candidate_realization_id = next(
        (rid for rid, rec in realizations.items() if candidate_clip_id in rec.clip_ids), None,
    )
    if candidate_realization_id is None:
        evidence["semantic_replacement_reason"] = "candidate_realization_not_found"
        return None, None, evidence["semantic_replacement_reason"], evidence
    candidate_record = realizations[candidate_realization_id]

    verified, reason = _certify_directional_semantic_preservation(
        record, candidate_record, candidate_realization_id, ledger,
        claim_equivalence_arbiter=claim_equivalence_arbiter, evidence=evidence,
    )
    if not verified:
        return None, None, reason, evidence
    return REPLACEMENT_VERIFIED_SEMANTIC, candidate_realization_id, reason, evidence


def _sanitize_claim_for_nonrequired_omissions(
    claim: CanonicalClaimRecord, omitted_digit_values: frozenset[str],
) -> CanonicalClaimRecord:
    """D-076 Section 6: strips ONLY the specific digit VALUES the caller
    has already classified as non-required (via the existing
    `semantic_atom_importance` classifiers -- see `_classify_pre_group_
    number_omissions`) from this one claim's own `content_tokens` and raw
    `text` -- never any other token, never a negation marker, never a
    non-digit word. Returns `claim` UNCHANGED (the same object) when
    `omitted_digit_values` is empty, which is PATH B's own call --
    byte-identical behavior guaranteed by construction, not convention.

    Exists because the realization-level NUMBER gate above operates on
    the WHOLE realization's text, but `_claim_preserved`'s own content-
    token subset/dedup checks operate per-claim and have no visibility
    into that gate's own classification: a required clause can share ONE
    clause with an already-cleared CONTEXTUAL digit (e.g. "I had gastritis
    in 2023."), and without this the per-claim subset check would still
    fail on that literal digit token even though the realization-level
    gate already proved it safe to omit."""
    if not omitted_digit_values:
        return claim
    sanitized_text = claim.text
    for value in omitted_digit_values:
        sanitized_text = re.sub(r"\b" + re.escape(value) + r"\b", " ", sanitized_text)
    return replace(
        claim, content_tokens=claim.content_tokens - omitted_digit_values, text=sanitized_text,
    )


def _certify_directional_semantic_preservation(
    record: RealizationRecord,
    candidate_record: RealizationRecord,
    candidate_realization_id: str,
    ledger: SemanticLedger,
    *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None,
    evidence: dict,
    nonrequired_digit_omissions: frozenset[str] = frozenset(),
    allow_cross_type_ambiguous_bridge: bool = False,
    allow_rhetorical_aside_negation_bridge: bool = False,
) -> tuple[bool, str]:
    """D-073/D-076/D-077/D-079 shared directional preservation
    certification chain -- D's required meaning subset of R's required
    meaning. Takes an ALREADY-DISCOVERED candidate; contains no candidate-
    discovery logic of its own (each caller's own discovery method is what
    legitimately differs -- D-072's pre-guard candidate for PATH B,
    D-076's strong-relation discovery for PRE_GROUP_SEMANTIC_PRESERVATION,
    D-079's idea-own-winner discovery for INTRA_IDEA_SEMANTIC_
    PRESERVATION). Reused verbatim by all four -- this is "reuse the exact
    existing D-073/D-073.1 semantic preservation chain," not a second
    implementation of it.

    `nonrequired_digit_omissions` (D-076 Section 6, default empty): digit
    VALUES from D's own realization text that the CALLER has already
    classified, via the existing `semantic_atom_importance` classifiers,
    as safe to omit (e.g. a bare incidental year) -- PATH B's own call
    passes nothing here, so its NUMBER gate stays byte-identical to before
    D-076 (any missing digit still fails closed).

    `allow_cross_type_ambiguous_bridge` (D-077, default False): threaded
    straight through to `_claim_preserved`'s own identically-named
    parameter -- PATH B's own call passes nothing here, so its per-claim
    matching stays byte-identical to before D-077; only the pre-group and
    (D-079) intra-idea paths opt in.

    `allow_rhetorical_aside_negation_bridge` (D-079, default False):
    threaded straight through to `_claim_preserved`'s own identically-
    named parameter -- PATH B's and D-076's own calls pass nothing here,
    so their per-claim matching stays byte-identical to before D-079;
    only the new intra-idea path opts in. Mutates and returns `evidence`
    in place; caller owns constructing/initializing it."""
    evidence["candidate_replacement_realization_id"] = candidate_realization_id

    if candidate_record.state != "selected":
        evidence["semantic_replacement_reason"] = "candidate_not_selected"
        return False, evidence["semantic_replacement_reason"]
    evidence["same_idea_verified"] = True

    if candidate_record.complete_idea is False:
        evidence["semantic_replacement_reason"] = "candidate_incomplete"
        return False, evidence["semantic_replacement_reason"]

    d_realization_values = _realization_digit_values(record.text)
    r_realization_values = _realization_digit_values(candidate_record.text)
    # Subset, not exact equality: every number D asserts must survive in R,
    # but R may legitimately carry ADDITIONAL numbers beyond D's own (the
    # same safe-superset direction Section 3 requires everywhere else --
    # see `_claim_content_subsumed`'s own digit check). A digit value the
    # caller has already pre-classified as non-required (D-076) is
    # excluded from the "missing" set here, never from the gate itself.
    missing_values = d_realization_values - r_realization_values - nonrequired_digit_omissions
    evidence["hard_gate_results"]["realization_number_match"] = not missing_values
    if missing_values:
        evidence["semantic_replacement_reason"] = "number_mismatch"
        return False, evidence["semantic_replacement_reason"]

    claims = ledger.claims()
    d_claims = tuple(
        _sanitize_claim_for_nonrequired_omissions(claims[cid], nonrequired_digit_omissions)
        for cid in record.claim_ids if cid in claims
    )
    r_claims = tuple(claims[cid] for cid in candidate_record.claim_ids if cid in claims)

    if not d_claims:
        evidence["semantic_replacement_reason"] = "no_extractable_claims_uncertain_content"
        return False, evidence["semantic_replacement_reason"]

    contradictions = _detect_contradiction_signals({
        record.realization_id: d_claims, candidate_realization_id: r_claims,
    })
    evidence["hard_gate_results"]["contradiction"] = bool(contradictions)
    if contradictions:
        evidence["semantic_replacement_reason"] = "contradiction_detected"
        evidence["hard_gate_results"]["contradiction_detail"] = [
            {"claim_type": c.claim_type, "reason": c.reason} for c in contradictions
        ]
        return False, evidence["semantic_replacement_reason"]

    arbiter_log: list[dict] = []
    preserved_ids: list[str] = []
    preserved_methods: dict[str, str] = {}
    for d_claim in sorted(d_claims, key=lambda c: c.canonical_claim_id):
        preserving, method = _claim_preserved(
            d_claim, r_claims, claim_equivalence_arbiter=claim_equivalence_arbiter, arbiter_log=arbiter_log,
            allow_cross_type_ambiguous_bridge=allow_cross_type_ambiguous_bridge,
            allow_rhetorical_aside_negation_bridge=allow_rhetorical_aside_negation_bridge,
        )
        if preserving is None:
            evidence["semantic_replacement_reason"] = "required_claim_not_preserved"
            evidence["hard_gate_results"]["unpreserved_claim_id"] = d_claim.canonical_claim_id
            evidence["arbiter_invoked"] = bool(arbiter_log)
            # D-077 Section 12: every consultation attempted for this
            # discard, even the failing one -- never silently dropped.
            evidence["arbiter_consultations"] = list(arbiter_log)
            return False, evidence["semantic_replacement_reason"]
        preserved_ids.append(d_claim.canonical_claim_id)
        preserved_methods[d_claim.canonical_claim_id] = method

    evidence["preserved_claim_ids"] = preserved_ids
    # D-079 Phase 1: per-claim matching method, keyed by the claim's OWN
    # canonical_claim_id -- observability only, so a consumer can tell
    # "cross_type_ambiguous_bridge"/"rhetorical_aside_negation_bridge"
    # certified this specific claim apart from a plain deterministic match.
    evidence["preserved_claim_methods"] = dict(preserved_methods)
    evidence["critical_claims_preserved"] = True
    evidence["unique_required_content_preserved"] = True
    evidence["arbiter_invoked"] = bool(arbiter_log)
    evidence["arbiter_consultations"] = list(arbiter_log)
    evidence["semantic_replacement_verified"] = True
    evidence["semantic_replacement_reason"] = "semantic_replacement_certified_claim_level_preservation_verified"
    return True, evidence["semantic_replacement_reason"]


# D-076 Section 3: the two discard populations this directive extends
# Resolver evaluation to -- both currently classified `PRE_GROUP_REJECTED`
# by `resolve_orphan_realizations_shadow` (unchanged by this directive; a
# SEPARATE, additive pass over these same two stages, never a modification
# of that function or its own verdict). `hybrid_editorial_chunks` is
# deliberately excluded here -- that population already has its own
# candidate-discovery mechanism (D-072's pre-guard candidate) and its own
# verdict machinery (PATH A/PATH B); D-076 reframes THOSE verdicts into
# proofs in `build_semantic_preservation_proofs` below rather than
# re-evaluating them a second time.
_PRE_GROUP_PROOF_ELIGIBLE_STAGES = frozenset({
    "clean_cut_or_composite_resolution", "draft_review_removed",
})


def _pre_group_relationship_evidence(record: RealizationRecord, candidate: RealizationRecord) -> str:
    """D-076 Section 4: STRONG relation required. Same `source_asset_id`
    AND (shared `attempt_id` OR overlapping `source_span_ids`) -- looks at
    provenance identity ONLY, never at either realization's own text.
    Temporal proximity alone, topical overlap alone, and shared nouns
    alone are NEVER sufficient and are not even consulted here: this
    function has no access to start/end timing or text at all, by
    construction, so it cannot accidentally fall back to either. Returns
    the relation kind, or "" when no qualifying relation exists."""
    if not record.source_asset_id or record.source_asset_id != candidate.source_asset_id:
        return ""
    if record.attempt_id and candidate.attempt_id and record.attempt_id == candidate.attempt_id:
        return "attempt_relation"
    if set(record.source_span_ids) & set(candidate.source_span_ids):
        return "source_span_relation"
    return ""


def _find_pre_group_candidates(
    record: RealizationRecord, ledger: SemanticLedger,
) -> tuple[tuple[str, str], ...]:
    """D-076 Section 4: discovers every SELECTED realization with a strong
    provenance relation to `record` -- never an arbitrary search by
    topical similarity. Stable order (sorted by realization_id); the
    caller tries each in turn against the full certification chain and
    stops at the first that verifies (mirrors `_claim_preserved`'s own
    "first match wins" convention) -- this function only discovers
    ELIGIBLE candidates, it never certifies one itself."""
    qualifying = []
    for rid, candidate in sorted(ledger.realizations().items(), key=lambda pair: pair[0]):
        if candidate.state != "selected":
            continue
        relation = _pre_group_relationship_evidence(record, candidate)
        if relation:
            qualifying.append((rid, relation))
    return tuple(qualifying)


def _classify_pre_group_number_omissions(
    d_text: str, r_text: str,
) -> tuple[bool, frozenset[str], tuple[Mapping[str, object], ...]]:
    """D-076 Section 6: classifies every digit VALUE present in D's own
    realization text but absent from R's, via the EXISTING `semantic_atom_
    importance.classify_number_atom`/`blocks_freeze` classifiers -- reused
    verbatim, no new importance tier invented here. A CRITICAL or
    UNCERTAIN missing digit still fails closed (WHEN UNCERTAIN, KEEP,
    exactly `blocks_freeze`'s own existing rule); only a confidently
    CONTEXTUAL missing digit (e.g. a bare incidental year) may be recorded
    as a nonrequired omission. Returns `(gate_passes, safe_to_omit_
    values, omission_records)` -- `omission_records` is always populated
    for every classified atom, verified or not, so nothing is silently
    dropped from diagnostics (Section 6's own requirement); `gate_passes`
    is False the moment any single missing digit fails to clear
    `blocks_freeze`, at which point iteration stops (fail-closed, no
    partial credit)."""
    missing = _realization_digit_values(d_text) - _realization_digit_values(r_text)
    if not missing:
        return True, frozenset(), ()
    safe_to_omit: set[str] = set()
    omissions: list[Mapping[str, object]] = []
    for atom in sorted(missing):
        classification = classify_number_atom(atom, d_text)
        record_row = {
            "atom": classification.atom, "atom_type": classification.atom_type,
            "importance": classification.importance, "evidence": classification.evidence,
            "resolved_by": classification.resolved_by,
        }
        if blocks_freeze(classification.importance):
            return False, frozenset(), (record_row,)
        safe_to_omit.add(atom)
        omissions.append(record_row)
    return True, frozenset(safe_to_omit), tuple(omissions)


def _attempt_pre_group_semantic_preservation(
    record: RealizationRecord, ledger: SemanticLedger, *, claim_equivalence_arbiter: ClaimEquivalenceArbiter | None,
) -> SemanticPreservationProof:
    """D-076: the one genuinely NEW certification path -- PRE_GROUP_
    SEMANTIC_PRESERVATION, for a content-bearing discard that never
    reached `hybrid_editorial_chunks`'s own semantic judgment (so PATH A/
    PATH B never evaluate it) and never entered a `take_judge_groups`
    contest (so StoryValidator's own `_same_idea_paraphrase_credit` never
    evaluates it either) -- see docs/CUTSELL_DECISIONS.md D-074/D-075 for
    the discovered gap this closes.

    Tries every strong-relation candidate `_find_pre_group_candidates`
    discovers, in order, running the SAME `_certify_directional_semantic_
    preservation` chain PATH B uses, with one additive relaxation (Section
    6): a missing digit already classified CONTEXTUAL by the existing
    `semantic_atom_importance` classifiers is recorded as a nonrequired
    omission rather than failing the NUMBER gate outright -- CRITICAL/
    UNCERTAIN missing digits still fail closed exactly as PATH B's own
    unmodified gate does. Stops at the first candidate that verifies;
    returns an unverified proof (never silently omitted -- Section 14)
    when no candidate exists or every candidate fails."""
    evidence: dict = {
        "semantic_replacement_evaluated": True,
        "candidate_replacement_realization_id": None,
        "same_idea_verified": False,
        "critical_claims_preserved": False,
        "unique_required_content_preserved": False,
        "hard_gate_results": {},
        "arbiter_invoked": False,
        "semantic_replacement_verified": False,
        "semantic_replacement_reason": "",
        "preserved_claim_ids": [],
    }

    candidates = _find_pre_group_candidates(record, ledger)
    if not candidates:
        return SemanticPreservationProof(
            discarded_id=record.realization_id, preserving_id=None,
            proof_method=PROOF_METHOD_PRE_GROUP_SEMANTIC_PRESERVATION,
            rejection_reason="no_strong_relation_candidate",
            candidate_discovery_method="pre_group_strong_relation",
        )

    realizations = ledger.realizations()
    last_reason = "no_strong_relation_candidate"
    last_candidate_evidence: dict = {}
    for candidate_realization_id, relation in candidates:
        candidate_record = realizations[candidate_realization_id]
        _passes, safe_to_omit, omissions = _classify_pre_group_number_omissions(
            record.text, candidate_record.text,
        )
        candidate_evidence = dict(evidence)
        candidate_evidence["hard_gate_results"] = {}
        if not _passes:
            last_reason = "number_mismatch"
            continue
        # D-077 Section 1/4: the pre-group path is the ONLY caller that
        # opts into the cross-type ambiguous claim-equivalence bridge --
        # PATH B's own call to this same shared function (`_attempt_
        # semantic_replacement_certification` above) never passes this,
        # so PATH B stays byte-identical.
        verified, reason = _certify_directional_semantic_preservation(
            record, candidate_record, candidate_realization_id, ledger,
            claim_equivalence_arbiter=claim_equivalence_arbiter, evidence=candidate_evidence,
            nonrequired_digit_omissions=safe_to_omit, allow_cross_type_ambiguous_bridge=True,
        )
        last_reason = reason
        last_candidate_evidence = candidate_evidence
        if verified:
            return SemanticPreservationProof(
                discarded_id=record.realization_id, preserving_id=candidate_realization_id,
                proof_method=PROOF_METHOD_PRE_GROUP_SEMANTIC_PRESERVATION,
                preserved_claim_ids=tuple(candidate_evidence.get("preserved_claim_ids") or ()),
                nonrequired_omissions=omissions,
                hard_gate_results=dict(candidate_evidence.get("hard_gate_results") or {}),
                arbiter_invoked=bool(candidate_evidence.get("arbiter_invoked")),
                arbiter_consultations=tuple(candidate_evidence.get("arbiter_consultations") or ()),
                verified=True,
                relationship_evidence=relation,
                candidate_discovery_method="pre_group_strong_relation",
            )

    return SemanticPreservationProof(
        discarded_id=record.realization_id, preserving_id=None,
        proof_method=PROOF_METHOD_PRE_GROUP_SEMANTIC_PRESERVATION,
        rejection_reason=last_reason,
        arbiter_consultations=tuple(last_candidate_evidence.get("arbiter_consultations") or ()),
        candidate_discovery_method="pre_group_strong_relation",
    )


def resolve_pre_group_semantic_preservation_shadow(
    ledger: SemanticLedger, *, claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
) -> tuple[SemanticPreservationProof, ...]:
    """D-076: additive, shadow-only pass over the two discard populations
    named in Section 3 (`_PRE_GROUP_PROOF_ELIGIBLE_STAGES`) -- never
    touches `resolve_orphan_realizations_shadow`'s own discard walk or
    verdict for these same records (that function's `PRE_GROUP_REJECTED`
    verdict for them is completely unchanged; this is a SEPARATE,
    parallel evaluation feeding StoryValidator's consumption only, per
    Section 15's own "validation behavior, not semantic selection
    mutation" framing). Mechanical garbage/false starts retain their
    existing safe handling for free: a discard with no extractable claims
    at all fails the shared certification chain's own existing "no
    extractable claims" gate (WHEN UNCERTAIN, KEEP), never a new floor
    invented here."""
    realizations = ledger.realizations()
    proofs = []
    for discard in ledger.discards():
        if discard.discarding_stage not in _PRE_GROUP_PROOF_ELIGIBLE_STAGES:
            continue
        record = realizations.get(discard.discarded_realization_id)
        if record is None:
            continue
        proofs.append(
            _attempt_pre_group_semantic_preservation(
                record, ledger, claim_equivalence_arbiter=claim_equivalence_arbiter,
            )
        )
    return tuple(proofs)


# --- D-079: INTRA_IDEA_SEMANTIC_PRESERVATION ---------------------------------

def _find_intra_idea_candidates(resolution: RealizationResolution, exclude_id: str) -> tuple[str, ...]:
    """D-079 Phase 1: candidate discovery for a losing member of an
    ALREADY-RESOLVED idea -- never a search, never a new relation: the
    idea's own resolved winner and/or composite members, exactly as
    `resolve_realizations_shadow`'s own per-idea resolution already
    established them. Stable order (winner first, then composite members
    sorted) -- the caller tries each in turn and stops at the first that
    verifies, mirroring every other candidate-discovery function in this
    module's own "first match wins" convention."""
    candidates: list[str] = []
    if resolution.winner_realization_id and resolution.winner_realization_id != exclude_id:
        candidates.append(resolution.winner_realization_id)
    for cid in sorted(resolution.composite_realization_ids):
        if cid != exclude_id and cid not in candidates:
            candidates.append(cid)
    return tuple(candidates)


def _attempt_intra_idea_semantic_preservation(
    record: RealizationRecord, resolution: RealizationResolution, ledger: SemanticLedger, *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None,
) -> SemanticPreservationProof:
    """D-079: the one genuinely NEW certification path this directive
    adds -- for a realization that DID reach grouping (has a
    `semantic_idea_id`, hence never evaluated by `resolve_orphan_
    realizations_shadow`/PATH A/PATH B/D-076's pre-group pass at all) and
    lost, WITHIN its own already-resolved idea, to that idea's own winner
    or composite. Candidate discovery is the idea's OWN resolution --
    stronger than D-076's own "strong relation" search, since it is
    literally "the already-computed winner of MY OWN idea." Runs the
    SAME shared `_certify_directional_semantic_preservation` chain PATH B
    and D-076 use, with the SAME Section 6 nonrequired-digit-omission
    relaxation D-076 already established, PLUS (D-079's own addition)
    `allow_cross_type_ambiguous_bridge` and `allow_rhetorical_aside_
    negation_bridge` -- both opt-in, both previously introduced (D-077,
    D-079) as narrow, hard-gated, additive strategies inside the SAME
    unmodified `_claim_preserved` chain. Stops at the first candidate
    that verifies; returns an unverified proof (never silently omitted)
    when no candidate exists or every candidate fails."""
    evidence: dict = {
        "semantic_replacement_evaluated": True,
        "candidate_replacement_realization_id": None,
        "same_idea_verified": False,
        "critical_claims_preserved": False,
        "unique_required_content_preserved": False,
        "hard_gate_results": {},
        "arbiter_invoked": False,
        "semantic_replacement_verified": False,
        "semantic_replacement_reason": "",
        "preserved_claim_ids": [],
    }

    candidate_ids = _find_intra_idea_candidates(resolution, record.realization_id)
    if not candidate_ids:
        return SemanticPreservationProof(
            discarded_id=record.realization_id, preserving_id=None,
            proof_method=PROOF_METHOD_INTRA_IDEA_SEMANTIC_PRESERVATION,
            rejection_reason="no_resolved_idea_winner",
            candidate_discovery_method="intra_idea_own_resolution",
        )

    realizations = ledger.realizations()
    last_reason = "no_resolved_idea_winner"
    last_candidate_evidence: dict = {}
    for candidate_realization_id in candidate_ids:
        candidate_record = realizations.get(candidate_realization_id)
        if candidate_record is None:
            continue
        _passes, safe_to_omit, omissions = _classify_pre_group_number_omissions(
            record.text, candidate_record.text,
        )
        candidate_evidence = dict(evidence)
        candidate_evidence["hard_gate_results"] = {}
        if not _passes:
            last_reason = "number_mismatch"
            continue
        verified, reason = _certify_directional_semantic_preservation(
            record, candidate_record, candidate_realization_id, ledger,
            claim_equivalence_arbiter=claim_equivalence_arbiter, evidence=candidate_evidence,
            nonrequired_digit_omissions=safe_to_omit,
            allow_cross_type_ambiguous_bridge=True,
            allow_rhetorical_aside_negation_bridge=True,
        )
        last_reason = reason
        last_candidate_evidence = candidate_evidence
        if verified:
            return SemanticPreservationProof(
                discarded_id=record.realization_id, preserving_id=candidate_realization_id,
                proof_method=PROOF_METHOD_INTRA_IDEA_SEMANTIC_PRESERVATION,
                preserved_claim_ids=tuple(candidate_evidence.get("preserved_claim_ids") or ()),
                nonrequired_omissions=omissions,
                hard_gate_results=dict(candidate_evidence.get("hard_gate_results") or {}),
                arbiter_invoked=bool(candidate_evidence.get("arbiter_invoked")),
                arbiter_consultations=tuple(candidate_evidence.get("arbiter_consultations") or ()),
                verified=True,
                relationship_evidence="intra_idea_resolved_winner",
                candidate_discovery_method="intra_idea_own_resolution",
            )

    return SemanticPreservationProof(
        discarded_id=record.realization_id, preserving_id=None,
        proof_method=PROOF_METHOD_INTRA_IDEA_SEMANTIC_PRESERVATION,
        rejection_reason=last_reason,
        arbiter_consultations=tuple(last_candidate_evidence.get("arbiter_consultations") or ()),
        candidate_discovery_method="intra_idea_own_resolution",
    )


def resolve_intra_idea_semantic_preservation_shadow(
    ledger: SemanticLedger, *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
    resolver_report: ResolverReport | None = None,
) -> tuple[SemanticPreservationProof, ...]:
    """D-079: additive, shadow-only pass over every idea `resolve_
    realizations_shadow` already resolved to RESOLVED_WINNER/RESOLVED_
    COMPOSITE -- never touches that function's own per-idea resolution,
    `discarded_realization_ids`, or `retained_for_contextual_value`
    bookkeeping (Hard Invariant D is untouched); this is a SEPARATE,
    parallel evaluation feeding `_lost_critical_claims`'s claim-scoped
    consumption only (D-079 Phase 2), exactly D-076's own "validation
    behavior, not semantic selection mutation" framing.

    `resolver_report`: pass an ALREADY-COMPUTED `resolve_realizations_
    shadow` result when a caller (e.g. `universal_clean_cut.py`, which
    also needs it for its own `realization_resolver_shadow` diagnostics)
    has already run it, to avoid a redundant second pass -- computed
    internally when omitted."""
    report = (
        resolver_report if resolver_report is not None
        else resolve_realizations_shadow(ledger, claim_equivalence_arbiter=claim_equivalence_arbiter)
    )
    realizations = ledger.realizations()
    proofs: list[SemanticPreservationProof] = []
    for resolution in report.idea_resolutions.values():
        if resolution.decision_status not in (RESOLVED_WINNER, RESOLVED_COMPOSITE):
            continue
        winner_and_composite = set(resolution.composite_realization_ids)
        if resolution.winner_realization_id:
            winner_and_composite.add(resolution.winner_realization_id)
        for realization_id in sorted(resolution.candidate_realization_ids):
            if realization_id in winner_and_composite:
                continue
            record = realizations.get(realization_id)
            if record is None:
                continue
            proofs.append(
                _attempt_intra_idea_semantic_preservation(
                    record, resolution, ledger, claim_equivalence_arbiter=claim_equivalence_arbiter,
                )
            )
    return tuple(proofs)


def build_semantic_preservation_proofs(
    ledger: SemanticLedger, *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
    pre_group_proofs: Sequence[SemanticPreservationProof] | None = None,
    intra_idea_proofs: Sequence[SemanticPreservationProof] | None = None,
) -> dict[str, SemanticPreservationProof]:
    """D-076 Section 1/7 (D-079: now three new-path sources): the single
    unified lookup StoryValidator consumes -- keyed by every `clip_id` a
    proved realization carries (a realization can merge multiple legacy
    clip_ids), so a lookup by any one of them finds the same proof.
    Combines existing verdict sources plus the new paths, minting NOTHING
    itself:

      - `hybrid_editorial_chunks` discards: reframes `resolve_orphan_
        realizations_shadow`'s own existing PATH A/PATH B verdicts
        (called here, never duplicated) into `LEXICAL_REPLACEMENT`/
        `SEMANTIC_REPLACEMENT` proofs. PATH A/PATH B's own verdict logic
        is untouched -- this only relabels an already-computed result.
      - The two D-076 Section 3 populations: `resolve_pre_group_semantic_
        preservation_shadow`'s own new `PRE_GROUP_SEMANTIC_PRESERVATION`
        proofs.
      - D-079's third, remaining population: `resolve_intra_idea_
        semantic_preservation_shadow`'s own new `INTRA_IDEA_SEMANTIC_
        PRESERVATION` proofs (a discard that DID reach grouping and lost,
        within its own idea, to that idea's own resolved winner/
        composite).

    `GROUPED_SAME_IDEA` is deliberately NOT produced here -- it is
    StoryValidator's own existing, already-working `_same_idea_paraphrase_
    credit` mechanism, consulted first and independently at its own point
    of use; this function only supplies the additional evidence that
    mechanism cannot see.

    `pre_group_proofs`/`intra_idea_proofs`: pass the ALREADY-COMPUTED
    result of the corresponding `resolve_*_shadow` function when a caller
    (e.g. `universal_clean_cut.py`, which also needs the full raw list for
    diagnostics) has already run it, to avoid a redundant second pass (and
    a redundant second arbiter consultation for the same content) --
    computed internally when omitted, so this function stays usable
    standalone."""
    realizations = ledger.realizations()
    proofs_by_clip: dict[str, SemanticPreservationProof] = {}

    for review in resolve_orphan_realizations_shadow(ledger, claim_equivalence_arbiter=claim_equivalence_arbiter):
        if review.verdict == REPLACEMENT_VERIFIED_SAFE:
            method = PROOF_METHOD_LEXICAL_REPLACEMENT
        elif review.verdict == REPLACEMENT_VERIFIED_SEMANTIC:
            method = PROOF_METHOD_SEMANTIC_REPLACEMENT
        else:
            continue
        record = realizations.get(review.realization_id)
        proof = SemanticPreservationProof(
            discarded_id=review.realization_id, preserving_id=review.replacement_realization_id,
            proof_method=method,
            preserved_claim_ids=tuple((review.semantic_replacement_evidence or {}).get("preserved_claim_ids") or ()),
            hard_gate_results=dict((review.semantic_replacement_evidence or {}).get("hard_gate_results") or {}),
            arbiter_invoked=bool((review.semantic_replacement_evidence or {}).get("arbiter_invoked")),
            arbiter_consultations=tuple((review.semantic_replacement_evidence or {}).get("arbiter_consultations") or ()),
            verified=True,
        )
        for clip_id in (record.clip_ids if record is not None else ()):
            proofs_by_clip[clip_id] = proof

    resolved_pre_group_proofs = (
        pre_group_proofs if pre_group_proofs is not None
        else resolve_pre_group_semantic_preservation_shadow(ledger, claim_equivalence_arbiter=claim_equivalence_arbiter)
    )
    for proof in resolved_pre_group_proofs:
        if not proof.verified:
            continue
        record = realizations.get(proof.discarded_id)
        for clip_id in (record.clip_ids if record is not None else ()):
            proofs_by_clip[clip_id] = proof

    resolved_intra_idea_proofs = (
        intra_idea_proofs if intra_idea_proofs is not None
        else resolve_intra_idea_semantic_preservation_shadow(ledger, claim_equivalence_arbiter=claim_equivalence_arbiter)
    )
    for proof in resolved_intra_idea_proofs:
        if not proof.verified:
            continue
        record = realizations.get(proof.discarded_id)
        for clip_id in (record.clip_ids if record is not None else ()):
            proofs_by_clip[clip_id] = proof

    return proofs_by_clip


def build_preserved_claim_id_index(
    *proof_collections: Sequence[SemanticPreservationProof] | Mapping[str, SemanticPreservationProof],
) -> dict[str, SemanticPreservationProof]:
    """D-079 Phase 1/2: the single, CLAIM-SCOPED lookup `_lost_critical_
    claims` consumes -- canonical_claim_id -> the one VERIFIED proof that
    lists it in `preserved_claim_ids`. Deliberately NOT clip-keyed and NOT
    idea-keyed (Phase 1's own "do not match by idea id... do not suppress
    on whole-clip equivalence" requirement) -- membership is decided
    purely by whether the EXACT canonical claim id appears in a verified
    proof's own `preserved_claim_ids` tuple, which is itself only ever
    populated by the shared per-claim `_certify_directional_semantic_
    preservation` loop (PATH B, D-076, D-079) -- never by `GROUPED_SAME_
    IDEA` (StoryValidator's own clip-level credit, which carries no
    `preserved_claim_ids` at all and is never passed to this function).

    Accepts any mix of proof sequences (raw `resolve_*_shadow` lists) or
    clip-keyed dicts (`build_semantic_preservation_proofs`'s own output,
    where multiple clip_ids can point at the identical proof object) --
    every source is de-duplicated by object identity before indexing, so
    a proof consulted by both codepaths is only counted once and never
    double-recorded. Unverified proofs are skipped entirely; a claim with
    no verified proof anywhere is simply absent from the returned dict,
    which is `_lost_critical_claims`'s own fail-closed signal (Phase 1:
    'if canonical identity cannot be established, current fail-closed
    behavior remains')."""
    index: dict[str, SemanticPreservationProof] = {}
    seen_proof_object_ids: set[int] = set()
    for collection in proof_collections:
        values = collection.values() if isinstance(collection, Mapping) else collection
        for proof in values:
            if proof is None or id(proof) in seen_proof_object_ids:
                continue
            seen_proof_object_ids.add(id(proof))
            if not bool(getattr(proof, "verified", False)):
                continue
            for canonical_claim_id in getattr(proof, "preserved_claim_ids", ()) or ():
                index.setdefault(canonical_claim_id, proof)
    return index


# ---------------------------------------------------------------------------
# D-089 Part A: CANONICAL EFFECTIVE-IMPORTANCE SINGLE TRUTH
#
# Live defect (D-088 forensic, run 33957582102): the SAME canonical claim id
# was raw CRITICAL in `semantic_claims.extract_claims` (a NEGATION-typed
# sentence carrying a rhetorical aside), effectively SUPPORTING in the
# Ledger's requirement-group view (`_effective_importance`: incidental +
# source-exclusive), treated as non-required by this resolver (RESOLVED_
# WINNER, missing_critical_claim_ids empty, the loser retained as a
# contextual alternate) and by ClaimCoverageBestTake (`suppressed_
# incidental_overrides`), and then re-extracted as raw CRITICAL by
# StoryValidator's `_lost_critical_claims`, which emitted CRITICAL_CLAIM_LOST
# and blocked Freeze. Two importance truths for one canonical proposition.
#
# This index is the ONE authoritative, deterministic per-canonical-claim
# importance lookup StoryValidator consumes. It invents no classifier: the
# effective importance is exactly the requirement-group importance the
# resolver itself computes (`build_requirement_groups` -> `_effective_
# importance`, the shared `_is_low_information_incidental` + source-
# exclusivity rule), computed per idea over the idea's own realizations'
# claims -- the same inputs `_resolve_one_idea` uses. Keyed strictly by
# canonical_claim_id; a consumer must match the EXACT id (never idea id,
# clip similarity or topic). A canonical id that appears in more than one
# idea with disagreeing effective importance fails closed to CRITICAL.
# ---------------------------------------------------------------------------

EFFECTIVE_IMPORTANCE_RAW_RETAINED = "raw_importance_retained"
EFFECTIVE_IMPORTANCE_INCIDENTAL_SOURCE_EXCLUSIVE = "incidental_source_exclusive_downgrade"
EFFECTIVE_IMPORTANCE_CROSS_IDEA_CONFLICT = "cross_idea_conflict_fail_closed"


@dataclass(frozen=True)
class EffectiveClaimImportance:
    canonical_claim_id: str
    claim_type: str
    text: str
    raw_importance: str
    effective_importance: str
    reason: str
    semantic_idea_id: str
    requirement_group_id: str
    source_realization_ids: tuple[str, ...]
    source_exclusive: bool


def build_effective_claim_importance_index(
    ledger: SemanticLedger, *, claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
) -> dict[str, EffectiveClaimImportance]:
    """D-089 Section 2: canonical_claim_id -> the authoritative effective
    importance of that exact proposition, derived from the Ledger's own
    per-idea requirement groups (the identical `build_requirement_groups`
    call `_resolve_one_idea` makes, with the same optional arbiter). Pure
    function of Ledger state; deterministic (ideas and claims are visited
    in sorted order)."""
    realizations = ledger.realizations()
    claims_by_id = ledger.claims()
    index: dict[str, EffectiveClaimImportance] = {}
    for idea_id in sorted(ledger.ideas()):
        idea = ledger.ideas()[idea_id]
        candidate_ids = tuple(sorted(idea.realization_ids))
        idea_claims = [
            claims_by_id[cid]
            for rid in candidate_ids if rid in realizations
            for cid in realizations[rid].claim_ids if cid in claims_by_id
        ]
        if not idea_claims:
            continue
        groups = build_requirement_groups(idea_claims, claim_equivalence_arbiter=claim_equivalence_arbiter)
        for group in groups:
            for canonical_claim_id in group.member_claim_ids:
                claim = claims_by_id.get(canonical_claim_id)
                if claim is None:
                    continue
                source_rids = tuple(sorted(set(claim.source_realization_ids) & set(candidate_ids))) or tuple(claim.source_realization_ids)
                source_exclusive = len(set(source_rids)) <= 1
                if claim.importance == _CRITICAL and group.importance != _CRITICAL:
                    reason = EFFECTIVE_IMPORTANCE_INCIDENTAL_SOURCE_EXCLUSIVE
                else:
                    reason = EFFECTIVE_IMPORTANCE_RAW_RETAINED
                entry = EffectiveClaimImportance(
                    canonical_claim_id=canonical_claim_id, claim_type=claim.claim_type, text=claim.text,
                    raw_importance=claim.importance, effective_importance=group.importance, reason=reason,
                    semantic_idea_id=idea_id, requirement_group_id=group.group_id,
                    source_realization_ids=source_rids, source_exclusive=source_exclusive,
                )
                existing = index.get(canonical_claim_id)
                if existing is None:
                    index[canonical_claim_id] = entry
                elif existing.effective_importance != entry.effective_importance:
                    # Section 6: one canonical id, two authoritative answers
                    # -> never the weaker one. Fail closed to CRITICAL.
                    index[canonical_claim_id] = replace(
                        existing, effective_importance=_CRITICAL, reason=EFFECTIVE_IMPORTANCE_CROSS_IDEA_CONFLICT,
                        semantic_idea_id=existing.semantic_idea_id,
                    )
    return index


def build_effective_claim_importance_diagnostics(index: Mapping[str, EffectiveClaimImportance]) -> dict:
    """JSON-safe view for `diagnostics["canonical_effective_importance"]`:
    every entry whose effective importance differs from its raw one (the
    only rows that can change a StoryValidator outcome) plus counts."""
    rows = [
        {
            "canonical_claim_id": e.canonical_claim_id, "claim_type": e.claim_type, "text": e.text,
            "raw_importance": e.raw_importance, "effective_importance": e.effective_importance,
            "reason": e.reason, "semantic_idea_id": e.semantic_idea_id,
            "requirement_group_id": e.requirement_group_id,
            "source_realization_ids": list(e.source_realization_ids), "source_exclusive": e.source_exclusive,
        }
        for e in sorted(index.values(), key=lambda e: e.canonical_claim_id)
        if e.effective_importance != e.raw_importance or e.reason == EFFECTIVE_IMPORTANCE_CROSS_IDEA_CONFLICT
    ]
    return {
        "schema_version": "cutsell.canonical_effective_importance.v1",
        "claim_count": len(index),
        "downgraded_count": sum(1 for r in rows if r["effective_importance"] != r["raw_importance"]),
        "cross_idea_conflict_count": sum(1 for r in rows if r["reason"] == EFFECTIVE_IMPORTANCE_CROSS_IDEA_CONFLICT),
        "entries": rows,
    }


def resolve_orphan_realizations_shadow(
    ledger: SemanticLedger, *, claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
) -> tuple[OrphanRealizationReview, ...]:
    """Hard Invariant E, standalone: a realization discarded before ever
    reaching grouping never enters the per-idea loop below (it has no
    `semantic_idea_id` to loop over), so it is walked here directly from
    the Ledger's own discard history. This is the module's proof of the
    D-049 Case A required shadow result -- see
    tests/test_cutsell_d050c1_realization_resolver.py's own Case A fixture.

    D-050D1: three-way verdict (see `OrphanRealizationReview`'s own
    docstring for the full contract) -- a verified replacement is always
    safe regardless of origin; absent that, a discard that never reached
    `hybrid_editorial_chunks`'s own semantic delete decision is
    `PRE_GROUP_REJECTED` (never had semantic understanding applied, so
    D-049 Case A does not apply); a discard that DID reach that stage is
    `REVIEW_REQUIRED` -- unchanged from before this directive, never
    downgraded.

    D-073: a would-be-REVIEW_REQUIRED discard gets exactly one additional,
    additive chance here: PATH B semantic replacement certification
    (`_attempt_semantic_replacement_certification`, optional
    `claim_equivalence_arbiter` -- defaults to None, i.e. deterministic-
    only). PATH A (`REPLACEMENT_VERIFIED_SAFE`) is tried first,
    unconditionally, exactly as before this directive; PATH B is
    consulted only when PATH A already found nothing AND this discard
    reached hybrid_editorial's own semantic judgment. `PRE_GROUP_REJECTED`
    discards never reach PATH B either -- they never needed a
    replacement-safety verdict in the first place."""
    reviews = []
    realizations = ledger.realizations()
    for discard in ledger.discards():
        record = realizations.get(discard.discarded_realization_id)
        if record is None or record.semantic_idea_id is not None:
            continue  # not an orphan -- either unknown, or reached grouping normally
        verification_method = ""
        semantic_evidence: Mapping[str, object] = {}
        if discard.replacement_verified and discard.replacement_realization_id:
            verdict = REPLACEMENT_VERIFIED_SAFE
            reason = "ledger_confirms_verified_replacement_realization"
            replacement_realization_id = discard.replacement_realization_id
            verification_method = VERIFICATION_METHOD_LEXICAL
        elif discard.discarding_stage != "hybrid_editorial_chunks":
            verdict = PRE_GROUP_REJECTED
            reason = "ordinary_pre_grouping_rejection_never_reached_semantic_understanding"
            replacement_realization_id = discard.replacement_realization_id
        else:
            semantic_verdict, semantic_replacement_id, semantic_reason, semantic_evidence = (
                _attempt_semantic_replacement_certification(
                    discard, record, ledger, claim_equivalence_arbiter=claim_equivalence_arbiter,
                )
            )
            if semantic_verdict == REPLACEMENT_VERIFIED_SEMANTIC:
                verdict = REPLACEMENT_VERIFIED_SEMANTIC
                reason = semantic_reason
                replacement_realization_id = semantic_replacement_id
                verification_method = VERIFICATION_METHOD_SEMANTIC
            else:
                verdict = REVIEW_REQUIRED
                reason = "hybrid_editorial_semantic_delete_with_no_verified_replacement_never_silently_confirmed"
                replacement_realization_id = discard.replacement_realization_id
        reviews.append(OrphanRealizationReview(
            realization_id=discard.discarded_realization_id, discard_reason=discard.reason,
            replacement_realization_id=replacement_realization_id,
            replacement_verified=verdict in (REPLACEMENT_VERIFIED_SAFE, REPLACEMENT_VERIFIED_SEMANTIC),
            verdict=verdict, decision_reason=reason,
            verification_method=verification_method, semantic_replacement_evidence=semantic_evidence,
        ))
    return tuple(reviews)


def _resolve_one_idea(
    idea_id: str,
    candidate_ids: tuple[str, ...],
    ledger: SemanticLedger,
    clip_scores: Mapping[str, float],
    *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
    arbiter_log: list[dict] | None = None,
) -> RealizationResolution:
    realizations = ledger.realizations()
    claims_by_id = ledger.claims()
    claims_by_realization = {
        rid: tuple(claims_by_id[cid] for cid in realizations[rid].claim_ids if cid in claims_by_id)
        for rid in candidate_ids
    }
    all_claims = [claim for claims in claims_by_realization.values() for claim in claims]
    groups = build_requirement_groups(
        all_claims, claim_equivalence_arbiter=claim_equivalence_arbiter, arbiter_log=arbiter_log,
    )
    critical_group_ids = frozenset(g.group_id for g in groups if g.importance == _CRITICAL)

    contradictions = _detect_contradiction_signals(claims_by_realization)
    contradiction_pairs = frozenset(frozenset((c.realization_a, c.realization_b)) for c in contradictions)

    # Semantic safety (top precedence): an unresolved contradiction between
    # still-live candidates of this idea blocks a confident verdict
    # entirely -- never averaged, never silently picked around.
    if contradictions:
        covered_by_id = {rid: _covered_group_ids(realizations[rid], groups) for rid in candidate_ids}
        return RealizationResolution(
            semantic_idea_id=idea_id, candidate_realization_ids=candidate_ids, winner_realization_id=None,
            composite_realization_ids=(), covered_canonical_claim_ids=(),
            missing_critical_claim_ids=tuple(sorted(critical_group_ids)),
            discarded_realization_ids=(), retained_for_contextual_value=candidate_ids,
            decision_status=REVIEW_REQUIRED, decision_reason="contradiction_signal_blocks_selection_freeze",
            confidence=0.0,
            evidence={
                "contradictions": [
                    {"realization_a": c.realization_a, "realization_b": c.realization_b,
                     "claim_type": c.claim_type, "reason": c.reason}
                    for c in contradictions
                ],
                "covered_group_ids": {rid: sorted(ids) for rid, ids in covered_by_id.items()},
            },
        )

    # NOTE: a candidate's engine-side `state` (selected/alternate/discarded)
    # is deliberately NOT treated as a safety signal here. Every id in
    # `candidate_ids` already has a `semantic_idea_id` (it reached
    # grouping), so an engine-side "discarded" realization at this point
    # only ever means the CURRENT engine's own BestTake/ClaimCoverage
    # competition picked a different winner -- exactly the D-049 Case B
    # shape this resolver must be free to re-examine independently, not a
    # physical-safety exclusion. A genuinely unsafe realization never
    # reaches this loop at all: it has no `semantic_idea_id` (D-049 Case
    # A's shape) and is handled entirely by
    # `resolve_orphan_realizations_shadow` instead.
    unsafe_ids: frozenset[str] = frozenset()
    coverage_by_id = {rid: _covered_group_ids(realizations[rid], groups) for rid in candidate_ids}

    # Critical claim completeness, single realization: prefer a single
    # candidate over a composite whenever one already covers every
    # CRITICAL requirement group on its own.
    complete_single = [
        rid for rid in candidate_ids
        if rid not in unsafe_ids and critical_group_ids.issubset(coverage_by_id[rid])
    ]

    # D-058 Phase 2: the evidence hierarchy among candidates that already
    # each satisfy full critical coverage on their own. See
    # docs/CUTSELL_DECISIONS.md D-057's gastritis forensic: the previous
    # `_pick_winner` ranked by raw DeliveryScorer score FIRST, so an
    # incomplete take (higher watch/listen score) could -- and did -- beat
    # a complete take the Ledger's own event history already recorded as
    # the high-confidence semantic winner (`SEMANTIC_WINNER_OVERRIDE`,
    # `pipeline.py`'s `_semantic_best_take`). Delivery score now breaks
    # ties only among candidates already equivalent on every stronger tier
    # below it -- it can no longer silently override recorded semantic
    # evidence.
    semantic_winner_confidence = _semantic_winner_confidence_by_realization(ledger, idea_id, candidate_ids)

    def _critical_claim_richness(rid: str) -> int:
        return sum(
            len(group.member_claim_ids) for group in groups
            if group.group_id in (coverage_by_id[rid] & critical_group_ids)
        )

    def _pick_winner(pool: Sequence[str]) -> str:
        # Tier 1 -- semantic validity/completeness: a realization proven
        # incomplete (`complete_idea is False`, D-050C1.6) is never
        # preferred over one that is complete or of unknown completeness;
        # unknown is never guessed at as incomplete (CLAUDE.md "WHEN
        # UNCERTAIN, KEEP").
        # Tier 2 -- high-confidence semantic winner evidence: the Ledger's
        # own recorded `SEMANTIC_WINNER_OVERRIDE` decision for this idea
        # (see `_semantic_winner_confidence_by_realization`), at or above
        # the same 0.85 confidence floor `pipeline.py`'s own
        # `_semantic_best_take` already uses to apply an override --
        # reused verbatim, not a new number.
        # Tier 3 -- critical claim coverage quality: how many individual
        # critical claims (not just requirement groups) this candidate
        # covers -- a finer signal than the boolean "covers all critical
        # groups" test every pool member already passed.
        # Tier 4 -- delivery quality (DeliveryScorer/watch-listen score):
        # breaks ties only among candidates already equivalent above.
        # Tier 5 -- contextual richness (count of covered non-critical
        # groups); the realization_id itself is the final, fully
        # deterministic tiebreaker.
        def sort_key(rid: str):
            proven_incomplete = realizations[rid].complete_idea is False
            has_high_confidence_semantic_winner = (
                semantic_winner_confidence.get(rid, 0.0) >= _HIGH_CONFIDENCE_SEMANTIC_WINNER_THRESHOLD
            )
            score = _realization_delivery_score(realizations[rid], clip_scores)
            richness = len(coverage_by_id[rid] - critical_group_ids)
            return (
                1 if proven_incomplete else 0,
                0 if has_high_confidence_semantic_winner else 1,
                -_critical_claim_richness(rid),
                -(score if score is not None else float("-inf")),
                -richness,
                rid,
            )
        return sorted(pool, key=sort_key)[0]

    if complete_single:
        # D-058 Phase 2: if more than one candidate already carries
        # high-confidence semantic winner evidence for DIFFERENT
        # realizations, the recorded evidence itself disagrees -- never
        # guess between two confidently-recorded semantic verdicts. Falls
        # through to REVIEW_REQUIRED, the same fail-closed posture
        # contradiction signals already use above.
        high_confidence_pool = [
            rid for rid in complete_single
            if semantic_winner_confidence.get(rid, 0.0) >= _HIGH_CONFIDENCE_SEMANTIC_WINNER_THRESHOLD
        ]
        if len(high_confidence_pool) >= 2:
            covered_by_id = {rid: coverage_by_id[rid] for rid in complete_single}
            return RealizationResolution(
                semantic_idea_id=idea_id, candidate_realization_ids=candidate_ids, winner_realization_id=None,
                composite_realization_ids=(), covered_canonical_claim_ids=(),
                missing_critical_claim_ids=tuple(sorted(critical_group_ids)),
                discarded_realization_ids=(), retained_for_contextual_value=candidate_ids,
                decision_status=REVIEW_REQUIRED,
                decision_reason="conflicting_high_confidence_semantic_winner_evidence",
                confidence=0.0,
                evidence={
                    "conflicting_realization_ids": sorted(high_confidence_pool),
                    "covered_group_ids": {rid: sorted(ids) for rid, ids in covered_by_id.items()},
                },
            )
        winner = _pick_winner(complete_single)
        covered = coverage_by_id[winner]
        chosen_ids = (winner,)
        composite_ids: tuple[str, ...] = ()
        status, reason = RESOLVED_WINNER, "single_realization_full_critical_coverage"
    else:
        composite = _find_minimal_composite(
            candidate_ids, realizations, groups, critical_group_ids, unsafe_ids, contradiction_pairs,
        )
        if composite is not None:
            covered = frozenset()
            for rid in composite:
                covered |= coverage_by_id[rid]
            chosen_ids = composite
            composite_ids = composite
            winner = None
            status, reason = RESOLVED_COMPOSITE, "minimal_composite_covers_all_critical_claims"
        else:
            missing = critical_group_ids - frozenset().union(*coverage_by_id.values()) if coverage_by_id else critical_group_ids
            return RealizationResolution(
                semantic_idea_id=idea_id, candidate_realization_ids=candidate_ids, winner_realization_id=None,
                composite_realization_ids=(), covered_canonical_claim_ids=(),
                missing_critical_claim_ids=tuple(sorted(critical_group_ids)),
                discarded_realization_ids=(), retained_for_contextual_value=candidate_ids,
                decision_status=REVIEW_REQUIRED,
                decision_reason="no_single_or_composite_realization_covers_all_critical_claims",
                confidence=0.0, evidence={"uncovered_critical_group_ids": sorted(missing)},
            )

    covered_claim_ids = tuple(sorted(
        cid for group in groups if group.group_id in covered for cid in group.member_claim_ids
    ))

    discarded_ids = []
    retained_ids = []
    for rid in candidate_ids:
        if rid in chosen_ids:
            continue
        record = realizations[rid]
        if coverage_by_id[rid].issubset(covered):
            discarded_ids.append(rid)
        elif record.replacement_realization_id in chosen_ids and record.discard_reason:
            discarded_ids.append(rid)
        else:
            retained_ids.append(rid)

    return RealizationResolution(
        semantic_idea_id=idea_id, candidate_realization_ids=candidate_ids,
        winner_realization_id=winner, composite_realization_ids=composite_ids,
        covered_canonical_claim_ids=covered_claim_ids, missing_critical_claim_ids=(),
        discarded_realization_ids=tuple(sorted(discarded_ids)),
        retained_for_contextual_value=tuple(sorted(retained_ids)),
        decision_status=status, decision_reason=reason,
        confidence=1.0 if status == RESOLVED_WINNER else 0.85,
        evidence={"covered_group_ids": sorted(covered)},
    )


def resolve_realizations_shadow(
    ledger: SemanticLedger, *, claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
) -> ResolverReport:
    """The one-pass decision model (module docstring). Produces exactly one
    `RealizationResolution` per `semantic_idea_id` the Ledger knows about
    (Hard Invariant A) plus one `OrphanRealizationReview` per pre-grouping
    discard (Hard Invariant E). Pure function of Ledger state (plus the
    optional, bounded `claim_equivalence_arbiter` -- D-050C1.6 F4, see
    `_claims_dedup_equivalent`'s own docstring for exactly when it is
    consulted) -- no I/O of its own, no randomness, no dependency on
    wall-clock time or ASR timestamps."""
    clip_scores = _delivery_scores_by_clip(ledger)
    arbiter_log: list[dict] = []
    idea_resolutions: dict[str, RealizationResolution] = {}
    for idea_id, idea in ledger.ideas().items():
        candidate_ids = tuple(sorted(idea.realization_ids))
        if not candidate_ids:
            continue
        idea_resolutions[idea_id] = _resolve_one_idea(
            idea_id, candidate_ids, ledger, clip_scores,
            claim_equivalence_arbiter=claim_equivalence_arbiter, arbiter_log=arbiter_log,
        )
    orphan_reviews = resolve_orphan_realizations_shadow(ledger, claim_equivalence_arbiter=claim_equivalence_arbiter)
    return ResolverReport(
        idea_resolutions=idea_resolutions, orphan_reviews=orphan_reviews,
        arbiter_consultations=tuple(arbiter_log),
    )


# --- Diagnostics (additive, unread by any decision branch) ------------------

def build_realization_resolver_diagnostics(report: ResolverReport) -> dict:
    """JSON-safe view for `diagnostics["realization_resolver_shadow"]` --
    read-only, additive, a SEPARATE key from `diagnostics["semantic_
    ledger"]`. See module docstring's NO BEHAVIOR CUTOVER section: nothing
    downstream reads this key."""
    return {
        "schema_version": "cutsell.realization_resolver.v1",
        "total_ideas": report.total_ideas,
        "review_required_count": report.review_required_count,
        # D-050C1.6 F4: "record arbiter usage in resolver diagnostics".
        "arbiter_consultation_count": len(report.arbiter_consultations),
        "arbiter_consultations": [dict(entry) for entry in report.arbiter_consultations],
        "resolutions": {
            idea_id: {
                "candidate_realization_ids": list(res.candidate_realization_ids),
                "winner_realization_id": res.winner_realization_id,
                "composite_realization_ids": list(res.composite_realization_ids),
                "covered_canonical_claim_ids": list(res.covered_canonical_claim_ids),
                "missing_critical_claim_ids": list(res.missing_critical_claim_ids),
                "discarded_realization_ids": list(res.discarded_realization_ids),
                "retained_for_contextual_value": list(res.retained_for_contextual_value),
                "decision_status": res.decision_status,
                "decision_reason": res.decision_reason,
                "confidence": res.confidence,
            }
            for idea_id, res in report.idea_resolutions.items()
        },
        "orphan_reviews": [
            {
                "realization_id": o.realization_id, "discard_reason": o.discard_reason,
                "replacement_realization_id": o.replacement_realization_id,
                "replacement_verified": o.replacement_verified,
                "verdict": o.verdict, "decision_reason": o.decision_reason,
                # D-073: additive, diagnostic-only -- verification_method
                # distinguishes PATH A ("lexical") from PATH B ("semantic")
                # certifications; semantic_replacement_evidence is empty
                # whenever PATH B was never attempted or found nothing.
                "verification_method": o.verification_method,
                "semantic_replacement_evidence": dict(o.semantic_replacement_evidence),
            }
            for o in report.orphan_reviews
        ],
    }


# --- Parity report (current engine vs shadow resolver) -----------------------

SAME = "SAME"
CONTENT_SAFETY_IMPROVEMENT = "CONTENT_SAFETY_IMPROVEMENT"
CLAIM_DEDUP_DIFFERENCE = "CLAIM_DEDUP_DIFFERENCE"
COMPOSITE_DIFFERENCE = "COMPOSITE_DIFFERENCE"
DELIVERY_RANK_DIFFERENCE = "DELIVERY_RANK_DIFFERENCE"
POTENTIAL_REGRESSION = "POTENTIAL_REGRESSION"
REVIEW_REQUIRED_DIFFERENCE = "REVIEW_REQUIRED_DIFFERENCE"


@dataclass(frozen=True)
class ResolverParityEntry:
    semantic_idea_id: str
    category: str
    detail: str


def build_resolver_parity_report(
    report: ResolverReport, ledger: SemanticLedger,
) -> tuple[ResolverParityEntry, ...]:
    """Compares, per semantic idea, what the CURRENT engine actually
    decided against what this shadow resolver would have decided,
    classifying the difference into exactly one of the 7 named categories
    the D-050C1 directive requires. "Do not treat every difference as a
    bug" -- SAME/DELIVERY_RANK_DIFFERENCE/CONTENT_SAFETY_IMPROVEMENT are
    all benign or positive findings; only POTENTIAL_REGRESSION and
    REVIEW_REQUIRED_DIFFERENCE are the ones a D-050C2 cutover decision
    must weigh carefully.

    D-050C1.6 F6/F7: the engine side of this comparison is read from
    `SemanticIdeaRecord.engine_resolution_status` (one of the ENGINE_*
    constants in semantic_ledger.py, finalized from ground-truth
    realization state -- see `finalize_idea_engine_resolution`'s own
    docstring), NEVER from `current_winner_realization_id` alone. Two
    engine shapes get dedicated, conservative handling here rather than
    being compared as if they were a confident single winner:

    - `ENGINE_RESOLVED_COMPOSITE`: compared against the shadow's OWN
      composite/winner member set directly (F6's fix -- a composite that
      superseded an earlier single-winner decision is no longer silently
      compared as if that earlier decision were final).
    - `ENGINE_BLOCKED_UNRESOLVED` / `ENGINE_REVIEW_REQUIRED`: the engine
      itself never converged (e.g. `freeze_blocked` deliberately keeping
      multiple candidates pending human review, or an idea with nothing
      currently selected). F7's fix -- a shadow resolver that CAN
      converge here is never penalized as POTENTIAL_REGRESSION merely for
      reaching a confident answer the engine deliberately deferred;
      agreeing that it's unresolved is SAME, resolving it is
      CONTENT_SAFETY_IMPROVEMENT (still just informational -- shadow-only,
      never authoritative)."""
    entries = []
    ideas = ledger.ideas()
    for idea_id, resolution in report.idea_resolutions.items():
        idea = ideas[idea_id]
        engine_status = idea.engine_resolution_status
        engine_winner = idea.current_winner_realization_id
        engine_composite = idea.composite_realization_ids

        if engine_status in (ENGINE_BLOCKED_UNRESOLVED, ENGINE_REVIEW_REQUIRED):
            # F7: the engine never reached a final answer here -- never a
            # regression either way. Both unresolved -> SAME; shadow
            # reaches a confident, safe answer -> a positive finding, not
            # a bug (still shadow-only; nothing acts on it).
            if resolution.decision_status == REVIEW_REQUIRED:
                entries.append(ResolverParityEntry(
                    idea_id, SAME,
                    f"engine did not converge ({engine_status}) and shadow resolver also flags REVIEW_REQUIRED",
                ))
            else:
                shadow_pick = resolution.winner_realization_id or resolution.composite_realization_ids
                entries.append(ResolverParityEntry(
                    idea_id, CONTENT_SAFETY_IMPROVEMENT,
                    f"engine did not converge ({engine_status}, no confident final pick); shadow resolver "
                    f"reaches a confident, critically-complete resolution {shadow_pick!r}",
                ))
            continue

        if engine_status == ENGINE_RESOLVED_COMPOSITE:
            if resolution.decision_status == REVIEW_REQUIRED:
                entries.append(ResolverParityEntry(
                    idea_id, REVIEW_REQUIRED_DIFFERENCE,
                    f"engine formed a confident composite {engine_composite!r}; shadow resolver flags "
                    f"REVIEW_REQUIRED ({resolution.decision_reason})",
                ))
            elif resolution.decision_status == RESOLVED_COMPOSITE:
                if set(resolution.composite_realization_ids) == set(engine_composite):
                    entries.append(ResolverParityEntry(idea_id, SAME, "engine and shadow resolver agree (same composite)"))
                else:
                    entries.append(ResolverParityEntry(
                        idea_id, COMPOSITE_DIFFERENCE,
                        f"engine composite {engine_composite!r} vs shadow composite "
                        f"{resolution.composite_realization_ids!r} -- differing member sets, both critically complete",
                    ))
            else:
                # RESOLVED_WINNER: shadow found ONE realization sufficient
                # where the engine needed a composite.
                if resolution.winner_realization_id in engine_composite:
                    entries.append(ResolverParityEntry(
                        idea_id, COMPOSITE_DIFFERENCE,
                        f"engine required a composite {engine_composite!r}; shadow resolver's single winner "
                        f"{resolution.winner_realization_id!r} (one of the engine's own composite members) "
                        f"alone covers every critical requirement",
                    ))
                else:
                    entries.append(ResolverParityEntry(
                        idea_id, POTENTIAL_REGRESSION,
                        f"engine composite {engine_composite!r} vs shadow single winner "
                        f"{resolution.winner_realization_id!r} (not an engine composite member) -- "
                        f"verify no critical content was lost",
                    ))
            continue

        # engine_status == ENGINE_RESOLVED_WINNER (or, defensively, an
        # unrecognized/unfinalized status -- fail open the same
        # conservative way as BLOCKED_UNRESOLVED rather than risk a false
        # POTENTIAL_REGRESSION against a status this function doesn't
        # actually understand): a genuinely confident, final,
        # single-realization engine decision -- safe to compare directly
        # against the shadow's own pick.
        if engine_status != ENGINE_RESOLVED_WINNER:
            if resolution.decision_status != REVIEW_REQUIRED:
                entries.append(ResolverParityEntry(
                    idea_id, CONTENT_SAFETY_IMPROVEMENT,
                    f"engine resolution status {engine_status!r} not recognized as final; shadow resolver "
                    f"still reaches a confident, critically-complete resolution",
                ))
            else:
                entries.append(ResolverParityEntry(
                    idea_id, SAME, f"engine resolution status {engine_status!r} not recognized as final; "
                    f"shadow resolver also declines to resolve confidently",
                ))
            continue
        if resolution.decision_status == REVIEW_REQUIRED:
            entries.append(ResolverParityEntry(
                idea_id, REVIEW_REQUIRED_DIFFERENCE,
                f"engine confidently chose {engine_winner!r}; shadow resolver flags REVIEW_REQUIRED "
                f"({resolution.decision_reason})",
            ))
            continue
        if resolution.decision_status == RESOLVED_COMPOSITE:
            if engine_winner in resolution.composite_realization_ids and len(resolution.composite_realization_ids) > 1:
                entries.append(ResolverParityEntry(
                    idea_id, COMPOSITE_DIFFERENCE,
                    f"engine chose single winner {engine_winner!r}; shadow resolver requires a composite "
                    f"{resolution.composite_realization_ids!r} for full critical coverage",
                ))
            else:
                entries.append(ResolverParityEntry(
                    idea_id, POTENTIAL_REGRESSION,
                    f"engine winner {engine_winner!r} is not part of the shadow composite "
                    f"{resolution.composite_realization_ids!r} -- verify no critical content was lost",
                ))
            continue
        # RESOLVED_WINNER on both sides.
        if engine_winner == resolution.winner_realization_id:
            entries.append(ResolverParityEntry(idea_id, SAME, "engine and shadow resolver agree"))
            continue
        engine_covered = _engine_winner_covered_critical(idea_id, engine_winner, resolution, ledger)
        if engine_covered is False:
            entries.append(ResolverParityEntry(
                idea_id, CONTENT_SAFETY_IMPROVEMENT,
                f"engine winner {engine_winner!r} was missing a critical requirement group the shadow "
                f"resolver's winner {resolution.winner_realization_id!r} covers",
            ))
        elif engine_covered is True:
            entries.append(ResolverParityEntry(
                idea_id, DELIVERY_RANK_DIFFERENCE,
                f"engine winner {engine_winner!r} and shadow winner {resolution.winner_realization_id!r} "
                f"are both critically complete; difference is tie-break/delivery-evidence ordering only",
            ))
        else:
            entries.append(ResolverParityEntry(
                idea_id, CLAIM_DEDUP_DIFFERENCE,
                f"engine winner {engine_winner!r} vs shadow winner {resolution.winner_realization_id!r}: "
                f"difference traces to requirement-group dedup, not a raw coverage gap",
            ))
    return tuple(entries)


def _engine_winner_covered_critical(
    idea_id: str, engine_winner: str | None, resolution: RealizationResolution, ledger: SemanticLedger,
) -> bool | None:
    """Best-effort: True/False when the engine's own pick is among this
    resolution's known candidates (so its coverage is directly checkable
    against the SAME requirement groups the shadow resolver computed);
    `None` when the engine winner isn't a recognized candidate at all
    (e.g. a composite fragment or a realization the Ledger reconstruction
    couldn't map) -- reported as a claim-dedup-shaped difference rather
    than guessed at."""
    if engine_winner not in resolution.candidate_realization_ids:
        return None
    realizations = ledger.realizations()
    claims_by_id = ledger.claims()
    all_claims = [
        claims_by_id[cid] for rid in resolution.candidate_realization_ids
        for cid in realizations[rid].claim_ids if cid in claims_by_id
    ]
    groups = build_requirement_groups(all_claims)
    critical_ids = frozenset(g.group_id for g in groups if g.importance == _CRITICAL)
    covered = _covered_group_ids(realizations[engine_winner], groups)
    return critical_ids.issubset(covered)


# ---------------------------------------------------------------------------
# D-050C2: CONTROLLED AUTHORITY CUTOVER
# ---------------------------------------------------------------------------
#
# Everything above this line is shadow-only (D-050C1/D-050C1.5/D-050C1.6):
# `resolve_realizations_shadow`/`build_resolver_parity_report` compute and
# compare, but never write to a `DraftTimeline`. Everything below is what
# `resolver_mode.RESOLVER_MODE_AUTHORITATIVE` actually applies -- see
# `resolver_mode.py`'s own module docstring for the 3-state rollout
# contract (LEGACY/SHADOW/AUTHORITATIVE), and `universal_clean_cut.py`'s
# cutover-point comment for the ONE place this is ever called and the
# explicit EVIDENCE-ONLY-vs-AUTHORITATIVE list for every legacy module.
#
# NON-NEGOTIABLE SCOPE NOTE: `apply_authoritative_realization_resolution`
# can only ever act on realizations still PRESENT in the `DraftTimeline` it
# is handed (selected + alternates + discarded). A D-049 Case A shape --
# hybrid_editorial_chunks deleting a realization before it ever reached
# grouping -- has already vanished from the draft by the time this stage
# runs; this function cannot resurrect deleted content, only refuse to
# certify the result as safe. That is exactly what it does: any orphan
# discard `resolve_orphan_realizations_shadow` classifies REVIEW_REQUIRED
# (a hybrid_editorial semantic delete with no verified replacement -- the
# "true orphan" D-049 Case A is actually concerned about) forces the
# overall status to REVIEW_REQUIRED (Section 8's "delete safety"
# requirement, applied at the boundary of what this stage can actually
# see). D-050D1: an orphan discard classified PRE_GROUP_REJECTED (never
# reached semantic understanding at all -- ordinary deterministic
# clean_cut/provider-judgement cleanup) does NOT force this escalation --
# it was never a candidate for D-049 Case A's concern in the first place,
# and treating it as one only ever produced noise, not safety.

SEMANTICALLY_RESOLVED = "SEMANTICALLY_RESOLVED"
AUTHORITATIVE_REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True)
class AuthoritativeIdeaOutcome:
    """One idea's authoritative-application record -- the per-idea unit
    `build_authoritative_resolution_diagnostics` reports (Section 12:
    "make production diagnosis possible without reconstructing 20
    diagnostics dictionaries")."""

    semantic_idea_id: str
    decision_status: str
    winner_realization_id: str | None
    composite_realization_ids: tuple[str, ...]
    covered_canonical_claim_ids: tuple[str, ...]
    missing_critical_claim_ids: tuple[str, ...]
    discarded_realization_ids: tuple[str, ...]
    retained_for_contextual_value: tuple[str, ...]
    decision_reason: str
    legacy_winner_realization_id: str | None
    legacy_composite_realization_ids: tuple[str, ...]
    legacy_vs_authoritative_same: bool


@dataclass(frozen=True)
class AuthoritativeApplicationResult:
    draft: DraftTimeline
    status: str  # SEMANTICALLY_RESOLVED | REVIEW_REQUIRED
    idea_outcomes: tuple[AuthoritativeIdeaOutcome, ...]
    unresolved_orphan_realization_ids: tuple[str, ...]
    # D-089 Part B Section 12: one row per restoration placement unit
    # (`_place_restored_clips_at_story_position`), observability only.
    # Defaulted so every pre-D-089 construction site stays valid.
    story_placement: tuple[dict, ...] = ()


def _realization_id_of(clip) -> str:
    return str(getattr(clip, "realization_id", None) or clip.clip_id)


PLACEMENT_UNIT_COMPOSITE_BLOCK = "AUTHORITATIVE_COMPOSITE_BLOCK"
PLACEMENT_UNIT_WINNER_REPLACEMENT = "SINGLE_WINNER_REPLACEMENT"
PLACEMENT_UNIT_NO_ANCHOR_APPEND = "NO_ANCHOR_APPEND"


def _place_restored_clips_at_story_position(
    kept_selected: list, restored: list, original_selected, *,
    ideas_by_realization: Mapping[str, str],
    composite_order_by_idea: Mapping[str, tuple[str, ...]] | None = None,
    placement_log: list[dict] | None = None,
) -> list:
    """D-087 Section 12 / D-089 Part B (STORY ORDER): a clip this
    application moves INTO `selected` from `alternates`/`discarded` must
    land at its idea's canonical story position -- never at the end of the
    timeline, never inside another idea's composite, never between a
    departed winner's own predecessor and successor.

    D-088 live defect (run 33957582102): B (a composite member) was placed
    right after its sibling A, then the acne winner replacement was ALSO
    anchored "after the last original kept predecessor" (= A) and landed
    ahead of B -- splitting the authoritative composite A->B and the
    physical continuation "...resolvía con" -> "resorcina.".

    Placement is expressed as deterministic RESTORATION UNITS, applied to
    the CURRENT sequence in a stable order:

      * AUTHORITATIVE_COMPOSITE_BLOCK -- every selected member of an idea
        that (a) has an explicit resolver composite order, or (b) still has
        an originally-selected sibling, placed ATOMICALLY as one contiguous
        block in the resolver's member order (recording start only breaks
        ties / orders members with no explicit order). The block occupies
        the story position of its earliest originally-selected member.
      * SINGLE_WINNER_REPLACEMENT -- a restored idea whose originally-
        selected sibling(s) LEFT `selected` takes the departed winner's
        ACTUAL slot: immediately before the first ORIGINAL successor of the
        departed clip that is still in the sequence (never inside a placed
        block -- the block head is used instead); only when no successor
        survives, immediately after the last surviving original
        predecessor; else the front.
      * NO_ANCHOR_APPEND -- no idea context at all: appended (the pre-D-087
        behavior), in original bucket order.

    Units are processed blocks first (by anchor story position, then idea
    id), then replacements (by departed original index, then idea id), then
    appends (by recording start, then clip id) -- so output never depends
    on `restored`'s iteration order. Membership is untouched by
    construction: every input clip appears exactly once in the output.
    Sibling identity is the resolver's own idea membership
    (`ideas_by_realization`, from the Ledger) -- never text similarity.
    `placement_log` (Section 12 diagnostics) receives one row per unit."""
    composite_order_by_idea = dict(composite_order_by_idea or {})
    if not restored:
        return list(kept_selected)

    def _idea_of(clip):
        rid = _realization_id_of(clip)
        stamped = getattr(clip, "semantic_idea_id", None)
        return ideas_by_realization.get(rid) or (str(stamped) if stamped else None)

    def _ids(seq):
        return [c.clip_id for c in seq]

    original_order = list(original_selected)
    original_index = {c.clip_id: i for i, c in enumerate(original_order)}
    kept_ids = {c.clip_id for c in kept_selected}
    restored_ids = {c.clip_id for c in restored}
    seq = list(kept_selected)
    block_of: dict[str, str] = {}      # clip_id -> block idea id, once a block is placed
    block_members: dict[str, list[str]] = {}

    def _member_sort_key(idea_id):
        order = composite_order_by_idea.get(idea_id) or ()
        rank = {rid: i for i, rid in enumerate(order)}

        def key(clip):
            rid = _realization_id_of(clip)
            return (rank.get(rid, len(rank)), float(clip.start), clip.clip_id)
        return key

    restored_by_idea: dict[str, list] = {}
    unanchored: list = []
    for clip in restored:
        idea_id = _idea_of(clip)
        if idea_id is None:
            unanchored.append(clip)
        else:
            restored_by_idea.setdefault(idea_id, []).append(clip)

    blocks: list[tuple[tuple, str, list]] = []
    replacements: list[tuple[tuple, str, list, list]] = []
    for idea_id, clips in restored_by_idea.items():
        kept_members = [c for c in seq if _idea_of(c) == idea_id]
        departed = [
            c for c in original_order
            if c.clip_id not in kept_ids and c.clip_id not in restored_ids and _idea_of(c) == idea_id
        ]
        if kept_members or idea_id in composite_order_by_idea:
            anchor_pos = min((original_index.get(c.clip_id, len(original_order)) for c in kept_members), default=len(original_order))
            if not kept_members and departed:
                anchor_pos = original_index.get(departed[0].clip_id, len(original_order))
            blocks.append(((anchor_pos, idea_id), idea_id, sorted(kept_members + clips, key=_member_sort_key(idea_id))))
        elif departed:
            replacements.append(((original_index[departed[0].clip_id], idea_id), idea_id, sorted(clips, key=lambda c: (float(c.start), c.clip_id)), departed))
        else:
            unanchored.extend(clips)

    def _log(row):
        if placement_log is not None:
            placement_log.append(row)

    def _departed_slot(departed):
        """(insert_index, successor_id, predecessor_id, reason) against the CURRENT seq."""
        current_index = {c.clip_id: i for i, c in enumerate(seq)}
        departed_positions = [original_index[c.clip_id] for c in departed]
        first_departed = min(departed_positions)
        successor_id = predecessor_id = None
        for candidate in original_order[first_departed + 1:]:
            if candidate.clip_id in current_index:
                successor_id = candidate.clip_id
                break
        if successor_id is not None:
            idx = current_index[successor_id]
            block_idea = block_of.get(successor_id)
            if block_idea is not None:
                idx = min(current_index[m] for m in block_members[block_idea])  # never split a placed block
            return idx, successor_id, None, "before_surviving_original_successor"
        for candidate in reversed(original_order[:first_departed]):
            if candidate.clip_id in current_index:
                predecessor_id = candidate.clip_id
                break
        if predecessor_id is not None:
            idx = current_index[predecessor_id] + 1
            block_idea = block_of.get(predecessor_id)
            if block_idea is not None:
                idx = max(current_index[m] for m in block_members[block_idea]) + 1
            return idx, None, predecessor_id, "after_last_surviving_original_predecessor_fallback"
        return 0, None, None, "no_surviving_neighbor_front"

    # 1. Composite blocks, atomically.
    for _key, idea_id, members in sorted(blocks, key=lambda b: b[0]):
        member_ids = _ids(members)
        before = _ids(seq)
        kept_positions = [i for i, c in enumerate(seq) if c.clip_id in member_ids]
        departed_here = [
            c for c in original_order
            if c.clip_id not in kept_ids and c.clip_id not in restored_ids and _idea_of(c) == idea_id
        ]
        if kept_positions:
            insert_at = kept_positions[0]
            seq = [c for c in seq if c.clip_id not in member_ids]
            reason = "block_at_earliest_original_member_position"
            successor_id = predecessor_id = None
        elif departed_here:
            insert_at, successor_id, predecessor_id, reason = _departed_slot(departed_here)
        else:
            insert_at, successor_id, predecessor_id, reason = len(seq), None, None, "no_anchor_append"
        seq[insert_at:insert_at] = members
        for cid in member_ids:
            block_of[cid] = idea_id
        block_members[idea_id] = member_ids
        _log({
            "unit_type": PLACEMENT_UNIT_COMPOSITE_BLOCK, "semantic_idea_id": idea_id,
            "member_clip_ids": member_ids, "member_realization_ids": [_realization_id_of(c) for c in members],
            "authoritative_member_order": list(composite_order_by_idea.get(idea_id) or ()),
            "departed_clip_ids": _ids(departed_here),
            "departed_original_index": (original_index[departed_here[0].clip_id] if departed_here else None),
            "successor_anchor": successor_id, "predecessor_fallback": predecessor_id,
            "chosen_insertion_index": insert_at, "sequence_before": before, "sequence_after": _ids(seq),
            "contiguity_validated": None, "placement_reason": reason,
        })

    # 2. Winner replacements, into the departed winner's own slot.
    for _key, idea_id, clips, departed in sorted(replacements, key=lambda r: r[0]):
        before = _ids(seq)
        insert_at, successor_id, predecessor_id, reason = _departed_slot(departed)
        seq[insert_at:insert_at] = clips
        _log({
            "unit_type": PLACEMENT_UNIT_WINNER_REPLACEMENT, "semantic_idea_id": idea_id,
            "member_clip_ids": _ids(clips), "member_realization_ids": [_realization_id_of(c) for c in clips],
            "authoritative_member_order": [],
            "departed_clip_ids": _ids(departed), "departed_original_index": original_index[departed[0].clip_id],
            "successor_anchor": successor_id, "predecessor_fallback": predecessor_id,
            "chosen_insertion_index": insert_at, "sequence_before": before, "sequence_after": _ids(seq),
            "contiguity_validated": None, "placement_reason": reason,
        })

    # 3. No anchor at all: append (pre-D-087 behavior), deterministically.
    for clip in sorted(unanchored, key=lambda c: (float(c.start), c.clip_id)):
        before = _ids(seq)
        seq.append(clip)
        _log({
            "unit_type": PLACEMENT_UNIT_NO_ANCHOR_APPEND, "semantic_idea_id": _idea_of(clip),
            "member_clip_ids": [clip.clip_id], "member_realization_ids": [_realization_id_of(clip)],
            "authoritative_member_order": [], "departed_clip_ids": [], "departed_original_index": None,
            "successor_anchor": None, "predecessor_fallback": None,
            "chosen_insertion_index": len(seq) - 1, "sequence_before": before, "sequence_after": _ids(seq),
            "contiguity_validated": None, "placement_reason": "no_anchor_append",
        })

    # Contiguity validation (observability -- placement above is contiguous
    # by construction; this is the audit a future canary reads).
    final_index = {c.clip_id: i for i, c in enumerate(seq)}
    if placement_log is not None:
        for row in placement_log:
            if row["unit_type"] == PLACEMENT_UNIT_COMPOSITE_BLOCK:
                positions = sorted(final_index[cid] for cid in row["member_clip_ids"] if cid in final_index)
                row["contiguity_validated"] = bool(positions) and positions == list(range(positions[0], positions[0] + len(positions)))
            else:
                row["contiguity_validated"] = True
    return seq


def build_story_placement_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict:
    """D-089 Part B Section 12: JSON-safe view for `diagnostics
    ["authoritative_story_placement"]`."""
    rows = [dict(r) for r in rows]
    return {
        "schema_version": "cutsell.authoritative_story_placement.v1",
        "unit_count": len(rows),
        "composite_block_count": sum(1 for r in rows if r.get("unit_type") == PLACEMENT_UNIT_COMPOSITE_BLOCK),
        "winner_replacement_count": sum(1 for r in rows if r.get("unit_type") == PLACEMENT_UNIT_WINNER_REPLACEMENT),
        "append_count": sum(1 for r in rows if r.get("unit_type") == PLACEMENT_UNIT_NO_ANCHOR_APPEND),
        "all_blocks_contiguous": all(bool(r.get("contiguity_validated")) for r in rows if r.get("unit_type") == PLACEMENT_UNIT_COMPOSITE_BLOCK),
        "units": rows,
    }


def apply_authoritative_realization_resolution(
    draft: DraftTimeline, ledger: SemanticLedger, report: ResolverReport,
    *, claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
) -> AuthoritativeApplicationResult:
    """D-050C2 Section 3/4: THE ONE explicit point the resolver's decision
    is ever applied to a `DraftTimeline`. Atomic per semantic idea:

    D-073: `claim_equivalence_arbiter` is additive, defaulted to None
    (deterministic-only PATH B) -- threaded straight through to this
    function's own `resolve_orphan_realizations_shadow` call below, never
    consulted anywhere else in this function.

    - `RESOLVED_WINNER`: every clip belonging to the winning realization
      moves to `selected`; every OTHER candidate realization of that idea
      moves to `discarded` (if the resolver proved it safely redundant)
      or `alternates` (if it holds unique, non-critical content this
      resolver refuses to silently drop -- Invariant D/Section 5's "no
      unresolved contradiction is collapsed... discard requires safety").
    - `RESOLVED_COMPOSITE`: every clip belonging to a composite member
      moves to `selected`; the same discarded/alternates split as above
      applies to non-member candidates.
    - `REVIEW_REQUIRED`: this idea's realizations are NEVER touched --
      every clip stays in whatever bucket the (pre-authoritative,
      legacy-computed) draft already had it in, and the overall result
      status becomes `REVIEW_REQUIRED` (Section 4: "do NOT guess... mark
      the draft/freeze state so delivery cannot proceed silently").

    A realization with multiple physical fragments (a post-Boundary
    split) always moves as one unit -- D-050A's "physical split preserves
    realization identity" invariant is never violated by this function.

    Returns a NEW `DraftTimeline` (immutable, `draft` itself untouched)
    plus the overall `status` and full per-idea/orphan diagnostics.
    Raises nothing for a semantic disagreement -- ANY invariant failure
    this function itself detects becomes `REVIEW_REQUIRED` in the
    returned status, never an exception and never a silent legacy
    fallback (Section 6: "never hide semantic failure through
    fallback"). A genuine internal bug (a candidate realization_id this
    function cannot map back to any clip at all) is the one case still
    surfaced as `REVIEW_REQUIRED` with an explicit `evidence` note rather
    than crashing the whole request, since a render decision must never
    proceed on a state this function itself could not fully verify."""
    all_clips = [*draft.selected, *draft.alternates, *draft.discarded]
    clips_by_realization: dict[str, list] = {}
    for clip in all_clips:
        clips_by_realization.setdefault(_realization_id_of(clip), []).append(clip)

    ideas = ledger.ideas()
    final_bucket: dict[str, str] = {}  # realization_id -> "selected" | "discarded" | "alternates"
    idea_outcomes: list[AuthoritativeIdeaOutcome] = []
    any_review_required = False
    unmapped_realization_ids: list[str] = []

    for idea_id, resolution in report.idea_resolutions.items():
        idea = ideas.get(idea_id)
        legacy_winner = idea.current_winner_realization_id if idea else None
        legacy_composite = idea.composite_realization_ids if idea else ()

        if resolution.decision_status == REVIEW_REQUIRED:
            any_review_required = True
            # Untouched: no bucket assignment for this idea's realizations
            # at all -- they keep whatever the incoming draft already had.
            idea_outcomes.append(AuthoritativeIdeaOutcome(
                semantic_idea_id=idea_id, decision_status=resolution.decision_status,
                winner_realization_id=None, composite_realization_ids=(),
                covered_canonical_claim_ids=resolution.covered_canonical_claim_ids,
                missing_critical_claim_ids=resolution.missing_critical_claim_ids,
                discarded_realization_ids=(), retained_for_contextual_value=resolution.candidate_realization_ids,
                decision_reason=resolution.decision_reason,
                legacy_winner_realization_id=legacy_winner, legacy_composite_realization_ids=tuple(legacy_composite),
                legacy_vs_authoritative_same=False,
            ))
            continue

        winning_ids = frozenset(
            (resolution.winner_realization_id,) if resolution.winner_realization_id
            else resolution.composite_realization_ids
        )
        for rid in winning_ids:
            if rid not in clips_by_realization:
                unmapped_realization_ids.append(rid)
                continue
            final_bucket[rid] = "selected"
        for rid in resolution.discarded_realization_ids:
            if rid not in clips_by_realization:
                unmapped_realization_ids.append(rid)
                continue
            final_bucket[rid] = "discarded"
        for rid in resolution.retained_for_contextual_value:
            if rid not in clips_by_realization:
                continue  # legitimately absent -- nothing to preserve
            final_bucket[rid] = "alternates"

        if resolution.decision_status == RESOLVED_WINNER:
            same_as_legacy = legacy_winner == resolution.winner_realization_id
        else:
            same_as_legacy = set(legacy_composite) == set(resolution.composite_realization_ids)
        idea_outcomes.append(AuthoritativeIdeaOutcome(
            semantic_idea_id=idea_id, decision_status=resolution.decision_status,
            winner_realization_id=resolution.winner_realization_id,
            composite_realization_ids=resolution.composite_realization_ids,
            covered_canonical_claim_ids=resolution.covered_canonical_claim_ids,
            missing_critical_claim_ids=resolution.missing_critical_claim_ids,
            discarded_realization_ids=resolution.discarded_realization_ids,
            retained_for_contextual_value=resolution.retained_for_contextual_value,
            decision_reason=resolution.decision_reason,
            legacy_winner_realization_id=legacy_winner, legacy_composite_realization_ids=tuple(legacy_composite),
            legacy_vs_authoritative_same=same_as_legacy,
        ))

    # Section 8 (delete safety), applied at this function's own boundary --
    # see the module-level NON-NEGOTIABLE SCOPE NOTE above: an orphan
    # discard (D-049 Case A shape) this function cannot verify safe forces
    # REVIEW_REQUIRED rather than silently certifying the draft.
    orphan_reviews = resolve_orphan_realizations_shadow(ledger, claim_equivalence_arbiter=claim_equivalence_arbiter)
    unresolved_orphans = tuple(o.realization_id for o in orphan_reviews if o.verdict == REVIEW_REQUIRED)
    if unresolved_orphans:
        any_review_required = True

    if unmapped_realization_ids:
        # Internal-consistency failure (a resolution names a realization_id
        # this function cannot find a clip for) -- fail closed, never guess
        # which bucket it belongs in and never silently drop it either.
        any_review_required = True

    def _rebuild(clip):
        bucket = final_bucket.get(_realization_id_of(clip))
        return bucket  # None -> untouched (REVIEW_REQUIRED idea, or unmapped)

    new_selected, new_alternates, new_discarded = [], [], []
    restored: list = []  # clips entering `selected` from a non-selected bucket
    seen_ids = set()
    for clip, original_bucket in (
        [(c, "selected") for c in draft.selected]
        + [(c, "alternates") for c in draft.alternates]
        + [(c, "discarded") for c in draft.discarded]
    ):
        if clip.clip_id in seen_ids:
            continue
        seen_ids.add(clip.clip_id)
        bucket = _rebuild(clip) or original_bucket
        if bucket == "selected":
            if original_bucket == "selected":
                new_selected.append(clip)
            else:
                restored.append(clip)
        elif bucket == "alternates":
            new_alternates.append(clip)
        else:
            new_discarded.append(clip)
    placement_log: list[dict] = []
    new_selected = _place_restored_clips_at_story_position(
        new_selected, restored, draft.selected, ideas_by_realization={
            rid: idea_id for idea_id, idea in ideas.items() for rid in idea.realization_ids
        },
        # D-089 Part B: the resolver's OWN composite member order per idea --
        # an explicit authoritative order always outranks recording time.
        composite_order_by_idea={
            idea_id: tuple(res.composite_realization_ids)
            for idea_id, res in report.idea_resolutions.items()
            if res.decision_status == RESOLVED_COMPOSITE and res.composite_realization_ids
        },
        placement_log=placement_log,
    )

    status = AUTHORITATIVE_REVIEW_REQUIRED if any_review_required else SEMANTICALLY_RESOLVED
    new_draft = replace(draft, selected=tuple(new_selected), alternates=tuple(new_alternates), discarded=tuple(new_discarded))
    return AuthoritativeApplicationResult(
        draft=new_draft, status=status, idea_outcomes=tuple(idea_outcomes),
        unresolved_orphan_realization_ids=unresolved_orphans + tuple(unmapped_realization_ids),
        story_placement=tuple(placement_log),
    )


def build_authoritative_resolution_diagnostics(result: AuthoritativeApplicationResult, *, mode: str) -> dict:
    """JSON-safe view for `diagnostics["realization_resolver_authority"]`
    -- Section 12's observability requirement: every idea's semantic_idea_
    id, mode, candidates, authoritative winner/composite, canonical
    critical claims, covered claims, discarded realizations + reasons,
    legacy winner, and legacy-vs-authoritative agreement, in ONE place."""
    return {
        "schema_version": "cutsell.realization_resolver_authority.v1",
        "mode": mode,
        "status": result.status,
        "unresolved_orphan_realization_ids": list(result.unresolved_orphan_realization_ids),
        "ideas": [
            {
                "semantic_idea_id": outcome.semantic_idea_id,
                "decision_status": outcome.decision_status,
                "winner_realization_id": outcome.winner_realization_id,
                "composite_realization_ids": list(outcome.composite_realization_ids),
                "covered_canonical_claim_ids": list(outcome.covered_canonical_claim_ids),
                "missing_critical_claim_ids": list(outcome.missing_critical_claim_ids),
                "discarded_realization_ids": list(outcome.discarded_realization_ids),
                "retained_for_contextual_value": list(outcome.retained_for_contextual_value),
                "decision_reason": outcome.decision_reason,
                "legacy_winner_realization_id": outcome.legacy_winner_realization_id,
                "legacy_composite_realization_ids": list(outcome.legacy_composite_realization_ids),
                "legacy_vs_authoritative_same": outcome.legacy_vs_authoritative_same,
            }
            for outcome in result.idea_outcomes
        ],
    }


def build_semantic_preservation_proofs_diagnostics(
    proofs: Sequence[SemanticPreservationProof],
) -> dict:
    """D-076 Section 14: JSON-safe view for `diagnostics["semantic_
    preservation_proofs"]`. Takes the RAW output of `resolve_pre_group_
    semantic_preservation_shadow` -- every attempt, verified or not, so a
    rejection reason is always visible here, never silently dropped
    (Section 14's own requirement) -- scoped to PRE_GROUP_SEMANTIC_
    PRESERVATION specifically: LEXICAL_REPLACEMENT/SEMANTIC_REPLACEMENT's
    own full evidence (verified or rejected) is already fully exposed by
    the EXISTING `realization_resolver_shadow`/`realization_resolver_
    authority` diagnostics' own `orphan_reviews`, unchanged by this
    directive -- duplicating it here would be a second, redundant
    observability surface for the same underlying data. Diagnostic-only:
    nothing reads this key to make a decision."""
    rows = [
        {
            "discarded_id": proof.discarded_id,
            "preserving_id": proof.preserving_id,
            "proof_method": proof.proof_method,
            "preserved_claim_ids": list(proof.preserved_claim_ids),
            "nonrequired_omissions": [dict(o) for o in proof.nonrequired_omissions],
            "hard_gate_results": dict(proof.hard_gate_results),
            "arbiter_invoked": proof.arbiter_invoked,
            "verified": proof.verified,
            "rejection_reason": proof.rejection_reason,
            "relationship_evidence": proof.relationship_evidence,
            "candidate_discovery_method": proof.candidate_discovery_method,
        }
        for proof in proofs
    ]
    return {
        "schema_version": "cutsell.semantic_preservation_proofs.v1",
        "proof_count": len(rows),
        "verified_count": sum(1 for row in rows if row["verified"]),
        "proofs": rows,
    }


# ---------------------------------------------------------------------------
# D-050C3 Section 7: ONE typed resolved-state object. `apply_authoritative_
# realization_resolution` already produces the resolved `DraftTimeline` that
# CanonicalEditPlan/StoryValidator/FinalEditReviewer consume directly (in
# AUTHORITATIVE mode they run strictly AFTER this function, on its own
# output draft -- see universal_clean_cut.py's D-050C3 ordering) -- that
# draft.selected/discarded/alternates split IS the one resolved semantic
# state every downstream stage sees, so there is structurally no second,
# independently-reconstructed notion of winner/discard for them to disagree
# with. `AuthoritativeSemanticState` is the typed, flattened OBSERABILITY
# view of that same resolved state (semantic_idea_id, winner-or-composite,
# canonical critical claims, coverage, discarded realizations, replacement
# verification, review status, story order, provenance) -- built once,
# straight from `AuthoritativeApplicationResult` plus the Ledger it was
# computed from, so nothing downstream has to reconstruct it by hand from
# raw diagnostics dicts.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuthoritativeSemanticIdeaState:
    semantic_idea_id: str
    review_status: str  # RESOLVED_WINNER | RESOLVED_COMPOSITE | REVIEW_REQUIRED
    winner_or_composite_realization_ids: tuple[str, ...]
    canonical_critical_claim_ids: tuple[str, ...]
    covered_canonical_claim_ids: tuple[str, ...]
    missing_critical_claim_ids: tuple[str, ...]
    discarded_realization_ids: tuple[str, ...]
    retained_for_contextual_value: tuple[str, ...]
    replacement_verified: bool
    story_order_position: int | None
    provenance: Mapping[str, tuple[str, ...]]  # realization_id -> source_span_ids


@dataclass(frozen=True)
class AuthoritativeSemanticState:
    status: str  # SEMANTICALLY_RESOLVED | REVIEW_REQUIRED
    ideas: tuple[AuthoritativeSemanticIdeaState, ...]
    unresolved_orphan_realization_ids: tuple[str, ...]

    def idea(self, semantic_idea_id: str) -> AuthoritativeSemanticIdeaState | None:
        for entry in self.ideas:
            if entry.semantic_idea_id == semantic_idea_id:
                return entry
        return None


def build_authoritative_semantic_state(
    result: AuthoritativeApplicationResult, ledger: SemanticLedger,
) -> AuthoritativeSemanticState:
    """Builds the Section 7 typed contract from an already-computed
    `AuthoritativeApplicationResult`. Pure, read-only, no new decisions --
    every value here is a direct projection of `outcome`/`ledger`, never a
    fresh judgment call. `story_order_position` is this idea's rank among
    all RESOLVED ideas by their winning realization's earliest start time
    in the resolved draft (None for REVIEW_REQUIRED -- it never joined the
    resolved timeline). `provenance` maps each surviving realization_id to
    its `source_span_ids` as recorded in the Ledger (never guessed)."""
    realizations = ledger.realizations()

    def _earliest_start(realization_ids: tuple[str, ...]) -> float:
        starts = [realizations[rid].start for rid in realization_ids if rid in realizations]
        return min(starts) if starts else float("inf")

    provisional: list[tuple[float, AuthoritativeIdeaOutcome]] = []
    for outcome in result.idea_outcomes:
        winner_ids = outcome.composite_realization_ids or (
            (outcome.winner_realization_id,) if outcome.winner_realization_id else ()
        )
        provisional.append((_earliest_start(winner_ids), outcome))

    order_rank: dict[str, int] = {}
    resolved_sorted = sorted(
        (item for item in provisional if item[1].decision_status != REVIEW_REQUIRED),
        key=lambda item: item[0],
    )
    for position, (_, outcome) in enumerate(resolved_sorted):
        order_rank[outcome.semantic_idea_id] = position

    idea_states: list[AuthoritativeSemanticIdeaState] = []
    for _, outcome in provisional:
        winner_or_composite = outcome.composite_realization_ids or (
            (outcome.winner_realization_id,) if outcome.winner_realization_id else ()
        )
        canonical_critical_claim_ids = tuple(dict.fromkeys(
            (*outcome.covered_canonical_claim_ids, *outcome.missing_critical_claim_ids)
        ))
        provenance = {
            rid: realizations[rid].source_span_ids
            for rid in (*winner_or_composite, *outcome.discarded_realization_ids)
            if rid in realizations
        }
        idea_states.append(AuthoritativeSemanticIdeaState(
            semantic_idea_id=outcome.semantic_idea_id,
            review_status=outcome.decision_status,
            winner_or_composite_realization_ids=winner_or_composite,
            canonical_critical_claim_ids=canonical_critical_claim_ids,
            covered_canonical_claim_ids=outcome.covered_canonical_claim_ids,
            missing_critical_claim_ids=outcome.missing_critical_claim_ids,
            discarded_realization_ids=outcome.discarded_realization_ids,
            retained_for_contextual_value=outcome.retained_for_contextual_value,
            replacement_verified=(
                outcome.decision_status != REVIEW_REQUIRED and not outcome.missing_critical_claim_ids
            ),
            story_order_position=order_rank.get(outcome.semantic_idea_id),
            provenance=provenance,
        ))

    return AuthoritativeSemanticState(
        status=result.status, ideas=tuple(idea_states),
        unresolved_orphan_realization_ids=result.unresolved_orphan_realization_ids,
    )


def build_authoritative_semantic_state_diagnostics(state: AuthoritativeSemanticState) -> dict:
    """JSON-safe view for `diagnostics["authoritative_semantic_state"]`."""
    return {
        "schema_version": "cutsell.authoritative_semantic_state.v1",
        "status": state.status,
        "unresolved_orphan_realization_ids": list(state.unresolved_orphan_realization_ids),
        "ideas": [
            {
                "semantic_idea_id": idea.semantic_idea_id,
                "review_status": idea.review_status,
                "winner_or_composite_realization_ids": list(idea.winner_or_composite_realization_ids),
                "canonical_critical_claim_ids": list(idea.canonical_critical_claim_ids),
                "covered_canonical_claim_ids": list(idea.covered_canonical_claim_ids),
                "missing_critical_claim_ids": list(idea.missing_critical_claim_ids),
                "discarded_realization_ids": list(idea.discarded_realization_ids),
                "retained_for_contextual_value": list(idea.retained_for_contextual_value),
                "replacement_verified": idea.replacement_verified,
                "story_order_position": idea.story_order_position,
                "provenance": {rid: list(spans) for rid, spans in idea.provenance.items()},
            }
            for idea in state.ideas
        ],
    }
