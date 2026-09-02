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

from dataclasses import dataclass, field
from itertools import combinations
import re
from typing import Mapping, Sequence

from .claim_coverage_best_take import _is_low_information_incidental
from .final_sibling_grouping import _negations
from .semantic_claims import ClaimEquivalenceArbiter
from .semantic_ledger import (
    CanonicalClaimRecord,
    DELIVERY_SCORE_WINNER,
    ENGINE_BLOCKED_UNRESOLVED,
    ENGINE_RESOLVED_COMPOSITE,
    ENGINE_RESOLVED_WINNER,
    ENGINE_REVIEW_REQUIRED,
    RealizationRecord,
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


@dataclass(frozen=True)
class OrphanRealizationReview:
    """See module docstring's Hard Invariant E: a realization the current
    engine discarded before it ever reached grouping (no semantic_idea_id
    at all -- D-049 Case A's exact shape). This module can never confirm
    such a discard as safe on its own; it always reports a verdict here
    rather than silently agreeing."""

    realization_id: str
    discard_reason: str
    replacement_realization_id: str | None
    replacement_verified: bool
    verdict: str  # "REPLACEMENT_VERIFIED_SAFE" | "REVIEW_REQUIRED"
    decision_reason: str


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


def resolve_orphan_realizations_shadow(ledger: SemanticLedger) -> tuple[OrphanRealizationReview, ...]:
    """Hard Invariant E, standalone: a realization discarded before ever
    reaching grouping never enters the per-idea loop below (it has no
    `semantic_idea_id` to loop over), so it is walked here directly from
    the Ledger's own discard history. This is the module's proof of the
    D-049 Case A required shadow result -- see
    tests/test_cutsell_d050c1_realization_resolver.py's own Case A fixture."""
    reviews = []
    realizations = ledger.realizations()
    for discard in ledger.discards():
        record = realizations.get(discard.discarded_realization_id)
        if record is None or record.semantic_idea_id is not None:
            continue  # not an orphan -- either unknown, or reached grouping normally
        if discard.replacement_verified and discard.replacement_realization_id:
            verdict = "REPLACEMENT_VERIFIED_SAFE"
            reason = "ledger_confirms_verified_replacement_realization"
        else:
            verdict = REVIEW_REQUIRED
            reason = "pre_grouping_discard_with_no_verified_replacement_never_silently_confirmed"
        reviews.append(OrphanRealizationReview(
            realization_id=discard.discarded_realization_id, discard_reason=discard.reason,
            replacement_realization_id=discard.replacement_realization_id,
            replacement_verified=discard.replacement_verified, verdict=verdict, decision_reason=reason,
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

    def _pick_winner(pool: Sequence[str]) -> str:
        # Delivery quality (evidence) breaks ties only among candidates
        # already proven safe+complete; contextual richness (count of
        # covered non-critical groups) is the next tie-breaker; the
        # realization_id itself is the final, fully deterministic one.
        def sort_key(rid: str):
            score = _realization_delivery_score(realizations[rid], clip_scores)
            richness = len(coverage_by_id[rid] - critical_group_ids)
            return (-(score if score is not None else float("-inf")), -richness, rid)
        return sorted(pool, key=sort_key)[0]

    if complete_single:
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
    orphan_reviews = resolve_orphan_realizations_shadow(ledger)
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
