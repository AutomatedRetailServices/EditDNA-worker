"""Claim-coverage-aware Best-Take override -- D-038.

Runs immediately after `deterministic_best_take_authority.apply_
deterministic_best_take_authority`, before Final Story Coherence
Validation. Adds no new similarity/grouping heuristic of its own -- like
that module, it only reads `diagnostics["take_judge_groups"]` (the RankedTake
scores pipeline.py's take_judge.rank_takes already computed per retry-family
group) and the group's own current bucket assignment.

The question this module asks, for every genuine retry-family contest (2+
ranked members) with exactly one current winner: does that winner cover
every CRITICAL audience-facing claim (`semantic_claims.py`) found across
the GROUP'S OWN members? A visually/performance-clean take must never beat
a semantically complete one (RAW 33423953391: DeliveryScorer picked a take
missing the diagnosis-confirmation claim over the one that had it).

Resolution, bounded:
  1. If exactly one OTHER member covers every critical claim in the group
     (a strictly more complete realization), it becomes the new winner --
     the current winner is discarded, same KEEP/DISCARD-only move pattern
     `deterministic_best_take_authority.py` already uses (D-019: no SWAP).
  2. If no single member covers everything, but a PAIR of members --
     source-compatible, non-overlapping in time, not near-duplicates of
     each other -- together cover the full critical-claim set, both are
     kept as a narrow composite (ordered by recording time, matching
     `final_edit_reviewer._composite_order_findings`'s own requirement).
     This is a narrow, claim-coverage-triggered fallback, not a
     replacement for the general CompositeResolver upstream. Guarded
     against a real failure mode found while building this module's own
     CleanCutBench fixtures: two members whose UNIQUE contributions share a
     claim_type (e.g. both NEGATION) are more likely one idea's coarse
     paraphrase split across two attempts than genuinely complementary
     facts, so that pairing is never composited -- see the guard's own
     comment at the composite-pair loop for why, and why disjoint
     claim_types stay safe to composite.
  3. Anything broader than that (3+ members each carrying a different
     required claim, or no compatible pair) is left exactly as upstream
     decided it -- flagged in this module's own diagnostics for
     observability. The real backstop is `final_story_coherence_
     validation._lost_critical_claims`, which independently blocks Freeze
     on any critical claim still missing from the actual winning
     realization, regardless of whether this module could safely fix it.

Ambiguity fails open throughout, same posture as `deterministic_best_take_
authority.py`: any contest this module is not confident about is left
exactly as it was.

D-063 CRITICAL_COVERAGE_DOMINANCE (an already-ambiguous, 2+-selected
family): the three cases above all assume exactly ONE current winner. When
a retry-family group instead already has 2+ members selected (e.g. a prior
stage's conflicting semantic-winner evidence left both live), this module
used to skip it entirely as "not this module's job", falling straight
through to StoryValidator/FinalEditReviewer's existing
DUPLICATE_IDEA+UNRESOLVED_RETRY block -- this codebase's production analog
of the canonical CUTSELL_EDITORIAL_RESOLUTION_AND_HUMAN_ESCALATION_CONTRACT
doctrine's `REVIEW_REQUIRED_SEMANTIC` escalation (see
docs/CUTSELL_DECISIONS.md D-062.1's forensic and D-062.2/D-063). Before
that fallback, this module now checks whether exactly one of the
currently-selected candidates strictly dominates every other on CRITICAL-
claim coverage (`_critical_coverage_dominant_candidate`, reusing the SAME
`claim_coverage`/`resolve_ambiguous_coverage`/`ClaimEquivalenceArbiter`
machinery every other check in this module already uses, plus the SAME
`contradiction_signal.any_pair_contradicts` safety gate
`canonical_edit_plan.py`'s own composite-safety check already uses -- no
new heuristic, no new provider). If found, the dominant candidate becomes
the family's sole winner (KEEP/DISCARD only, D-019: the others move to
discard, never a SWAP). If not -- identical coverage, genuinely disjoint
unique claims, a factual contradiction between candidates, or the would-be
winner is itself a proven-incomplete/failed delivery -- the family is left
untouched, exactly as before this directive, and still falls through to
the existing block. See `docs/CUTSELL_EDITORIAL_RESOLUTION_AND_HUMAN_ESCALATION_CONTRACT.md`
Section 3 for the full CRITICAL_COVERAGE_DOMINANCE contract this
implements verbatim, and its Section 4 for why this check runs BEFORE any
semantic-winner-label evidence is ever consulted.

D-065/D-066 NEGATION SEMANTIC ROLE (claim-vs-claim hindsight equivalence):
D-064's forensic on a live D-063 blocker found that a candidate's own
CONTRASTIVE_HINDSIGHT_NEGATION clause (`semantic_claims.py`'s additive
`Claim.negation_role`, e.g. "no me parecian sospechosos... pero ahora si"
-- a before->after realization, not a standalone factual negation) can get
independently CRITICAL-classified while an equivalent, differently-worded
realization from ANOTHER candidate in the same family stays SUPPORTING,
making the two candidates' critical-claim sets look genuinely disjoint to
D-063's dominance check even though they express the same proposition.
Before computing dominance, this module now searches (`_find_hindsight_
alignment`) for a claim-vs-claim (never claim-vs-whole-candidate-text)
equivalence between a CONTRASTIVE_HINDSIGHT_NEGATION-eligible claim and a
same-family ACTION_EVENT/STATE_RESULT claim, gated by deterministic hard
exclusions (never a protected claim type, never any digit evidence,
Section 4's full protected-marker set) BEFORE any arbiter call, and
confirmed only by the SAME `ClaimEquivalenceArbiter` protocol every other
check in this module already uses (no new provider/model). A confirmed
alignment merges the two claims for coverage-unit purposes ONLY
(`_covered_claim_ids`'s own `hindsight_alignments` parameter) -- the raw
claims, their text, and their provenance are never rewritten or deleted.
D-063's own dominance rule (`_critical_coverage_dominant_candidate`) is
UNCHANGED by this: only the coverage SETS it consumes are corrected
upstream. See docs/CUTSELL_DECISIONS.md D-065 (design) and D-066
(implementation) for the full contract.
"""
from __future__ import annotations

from dataclasses import replace

from .contradiction_signal import any_pair_contradicts
from .final_sibling_grouping import _content, _numbers
from .semantic_atom_importance import _TEMPORAL_ASIDE_MARKERS, _clause_has_any, _looks_like_year
from .semantic_claims import (
    ACTION_EVENT,
    ClaimEquivalenceArbiter,
    ClauseRoleArbiter,
    CONTRASTIVE_HINDSIGHT_NEGATION,
    CORRECTION,
    CRITICAL,
    DIAGNOSIS_IDENTIFICATION,
    MEASUREMENT_QUANTITY,
    NEGATION,
    STATE_RESULT,
    _CAUSE_EFFECT_MARKERS,
    _CORRECTION_MARKERS,
    _CURRENCY_MARKERS,
    _DOSE_MARKERS,
    _MEASUREMENT_UNIT_MARKERS,
    _PERCENT_MARKERS,
    _RESULT_REPORTING_MARKERS,
    _STATE_RESULT_MARKERS,
    _STRONG_IDENTIFICATION_MARKERS,
    _UNIQUE_CONCLUSION_MARKERS,
    Claim,
    claim_coverage,
    dedupe_claims,
    extract_claims,
    resolve_ambiguous_coverage,
    _negation_role_hard_exclusion,
)

_COMPOSITE_OVERLAP_TOLERANCE_SEC = 0.05


# D-048 FIX 2 (D-047 Case 2 -- gastritis): a claim that is CRITICAL purely
# because it contains a bare negation or a bare number, with no
# independently substantive marker of its own, and additionally reads as an
# incidental temporal aside (a plain year sitting in an ordinary "durante
# .../en .../por ..." clause -- the exact shape semantic_atom_importance.py
# already treats as CONTEXTUAL for a missing NUMBER atom, D-031) is not, on
# its own, strong enough evidence to swap out an already-correct BestTake
# winner for a thinner candidate. This does NOT change classify_claim's own
# output, dedupe, or coverage math -- and it never touches StoryValidator's
# freeze-blocking posture (a negation still always blocks Freeze there,
# unchanged; that is a deliberate, general "WHEN UNCERTAIN, KEEP" backstop
# this module is not the place to weaken). It only answers a narrower
# question this module alone asks: is this SPECIFIC claim substantive
# enough to justify THIS module discarding a richer realization for a
# thinner one that merely happens to contain it verbatim?
def _is_low_information_incidental(claim: Claim) -> bool:
    """True when `claim` shows no independently substantive marker of its
    own (diagnosis/identification language, correction language, a
    genuinely unit/percent/currency/dose-qualified number, an explicit
    cause-effect connector, a unique-conclusion statistic, or explicit
    result-state language) AND its only CRITICAL signal is a bare negation
    or bare number riding on an incidental temporal aside. Protects
    genuinely critical negations/causal/treatment/diagnosis/identity claims
    (none of those patterns match) while catching a low-information,
    self-referential/temporal-filler remark that happens to trip the
    negation rule (D-047 Case 2's own real shape: "... en una temporada, en
    2023, no hay que preguntar.")."""
    if claim.importance != CRITICAL:
        return False
    text = claim.text
    if _clause_has_any(text, _STRONG_IDENTIFICATION_MARKERS + _RESULT_REPORTING_MARKERS):
        return False
    if _clause_has_any(text, _CORRECTION_MARKERS):
        return False
    if _numbers(text) and _clause_has_any(
        text, _PERCENT_MARKERS + _CURRENCY_MARKERS + _MEASUREMENT_UNIT_MARKERS + _DOSE_MARKERS
    ):
        return False
    if _clause_has_any(text, _CAUSE_EFFECT_MARKERS):
        return False
    if _clause_has_any(text, _UNIQUE_CONCLUSION_MARKERS):
        return False
    if _clause_has_any(text, _STATE_RESULT_MARKERS):
        return False
    # Nothing independently substantive found -- the claim is "critical"
    # purely via a bare negation and/or a bare (unit-less) number. Treat it
    # as incidental only when it ALSO carries a recognizable temporal-aside
    # shape (mirrors classify_number_atom's own CONTEXTUAL rule for a bare
    # year, D-031, applied here to the whole claim rather than one atom).
    has_year = any(_looks_like_year(token) for token in _numbers(text))
    return has_year and _clause_has_any(text, _TEMPORAL_ASIDE_MARKERS)


def _override_blocked_by_incidental_self_source_claims(
    missing: list[Claim],
    *,
    candidate_clip_id: str,
    other_coverers: dict,
    winner_clip,
    candidate_clip,
) -> bool:
    """D-048 FIX 2 SELF-SOURCE EXCLUSIVITY CHECK: the compound condition
    under which a single-candidate override should NOT happen, even though
    the candidate technically covers every critical claim the current
    winner is missing. All three must hold, matching the D-048 directive's
    own compound rule -- a single substantive, non-source-exclusive, or
    genuinely-richer-candidate missing claim is enough to keep the override
    eligible:

      1. every missing claim is source-exclusive to `candidate_clip_id`
         (no OTHER sibling in the retry family covers it either -- `other_
         coverers[claim_id]` is the set of member clip_ids, excluding the
         candidate itself, that independently cover that claim);
      2. every missing claim is itself low-information/incidental
         (`_is_low_information_incidental`);
      3. the candidate is not otherwise richer than the current winner --
         swapping to it would discard more of the winner's own content
         than the incidental claim(s) restore.

    A source-exclusive but genuinely critical claim (a unique diagnosis,
    negation, causal fact, treatment detail, etc. found nowhere else)
    still overrides -- exclusivity alone is never disqualifying, per the
    directive's own "do not make source exclusive an automatic discard
    either" instruction.
    """
    if not missing:
        return False
    if any(other_coverers.get(claim.claim_id) for claim in missing):
        return False  # independently corroborated elsewhere -- not source-exclusive
    if not all(_is_low_information_incidental(claim) for claim in missing):
        return False  # at least one missing claim is substantive on its own
    winner_content = len(_content(str(winner_clip.text or "")))
    candidate_content = len(_content(str(candidate_clip.text or "")))
    return candidate_content < winner_content


def _all_group_claims(members: list[tuple[str, object]], *, clause_role_arbiter: ClauseRoleArbiter | None = None):
    """Every claim (any importance) found across the group's own members,
    deduped across near-identical restatements between sibling attempts --
    the full pool `_group_critical_claims` filters down from, and (D-066)
    the pool `_find_hindsight_alignment` searches for a claim-vs-claim
    equivalence partner. D-040: `extract_claims` already splits a
    multi-clause sentence into its own CORE/SUPPORTING/CONTEXTUAL clauses,
    so a critical fact bundled with a merely-supporting reason surfaces as
    two separate claims here, not one."""
    all_claims = []
    for clip_id, clip in members:
        all_claims.extend(extract_claims(clip_id, str(clip.text or ""), clause_role_arbiter=clause_role_arbiter))
    return dedupe_claims(tuple(all_claims))


def _group_critical_claims(members: list[tuple[str, object]], *, clause_role_arbiter: ClauseRoleArbiter | None = None):
    """Every CRITICAL claim found across the group's own members -- see
    `_all_group_claims`'s own docstring for the full extraction/dedup
    contract this filters down from."""
    deduped = _all_group_claims(members, clause_role_arbiter=clause_role_arbiter)
    return tuple(c for c in deduped if c.importance == CRITICAL)


# --- D-065/D-066: negation semantic role -- claim-vs-claim hindsight
#     equivalence (feeds D-063 CRITICAL_COVERAGE_DOMINANCE's coverage sets,
#     never its own dominance rule) --------------------------------------

# A CONTRASTIVE_HINDSIGHT_NEGATION claim may only ever align with a
# non-protected reflective claim type -- ACTION_EVENT/STATE_RESULT are the
# general "plain statement" claim types this codebase already produces for
# an ordinary reflective clause (D-065 Section 6). DIAGNOSIS_IDENTIFICATION/
# CORRECTION/MEASUREMENT_QUANTITY are categorically excluded on the
# candidate side too -- diagnosis/correction/number safety, symmetric with
# the hard exclusions `semantic_claims._classify_negation_role` already
# applies when first deciding a claim is even hindsight-eligible.
_HINDSIGHT_ALIGNABLE_CLAIM_TYPES = frozenset({ACTION_EVENT, STATE_RESULT})
_HINDSIGHT_PROTECTED_CLAIM_TYPES = frozenset({DIAGNOSIS_IDENTIFICATION, CORRECTION, MEASUREMENT_QUANTITY})
# D-065 Section 6: below this floor, two claims' content-token overlap is
# too thin for a hindsight-paraphrase judgment to plausibly apply --
# confidently NOT equivalent, the arbiter is never consulted. Deliberately
# NOT `semantic_claims.claim_coverage` (that function's own negation-flip
# guard would cap this at `_DEFINITIVE_MISMATCH_COVERAGE_CAP` precisely
# BECAUSE one side is negated and the other is not -- exactly the shape
# this mechanism exists to recognize as equivalent, not reject). Raw
# content-token overlap is negation-agnostic by construction (`_content`
# already strips 2-3 character words including every negation marker in
# `_NEGATIONS`), so it is the correct, neutral floor check here.
_HINDSIGHT_ALIGNMENT_AMBIGUOUS_FLOOR = 0.10


def _hindsight_alignment_hard_gates_pass(negation_claim: Claim, candidate_claim: Claim) -> bool:
    """D-065 Section 5: deterministic hard gates, ALL of which must pass
    before any arbiter call is even considered for a claim-vs-claim
    CONTRASTIVE_HINDSIGHT_NEGATION alignment. Any failure -> NOT
    EQUIVALENT, arbiter NOT called."""
    if candidate_claim.claim_id == negation_claim.claim_id:
        return False
    if candidate_claim.claim_type in _HINDSIGHT_PROTECTED_CLAIM_TYPES:
        return False  # diagnosis/correction/measurement safety on the candidate side
    if candidate_claim.claim_type not in _HINDSIGHT_ALIGNABLE_CLAIM_TYPES:
        return False
    if _numbers(negation_claim.text) or _numbers(candidate_claim.text):
        return False  # number safety (defensive re-check; the role itself already excludes digit evidence)
    if _negation_role_hard_exclusion(candidate_claim.text):
        return False  # diagnosis/result-reporting/cause-effect/unique-conclusion/state-result/correction/percent/currency/measurement/dose safety
    return True


def _content_overlap_coefficient(left_tokens: frozenset, right_tokens: frozenset) -> float:
    """Szymkiewicz-Simpson overlap coefficient (shared fraction of the
    SMALLER side) over content tokens -- negation-agnostic by construction,
    see `_HINDSIGHT_ALIGNMENT_AMBIGUOUS_FLOOR`'s own docstring for why this
    is used here instead of `claim_coverage`."""
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def _find_hindsight_alignment(
    negation_claim: Claim,
    all_claims,
    *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None,
    arbiter_log: list[dict] | None = None,
) -> Claim | None:
    """D-065 Section 6: claim-vs-claim (never claim-vs-whole-candidate-
    text) equivalence search for a CONTRASTIVE_HINDSIGHT_NEGATION-eligible
    claim. Reuses the EXISTING `ClaimEquivalenceArbiter` protocol verbatim
    -- no new provider/model, no new arbiter class: "does `other`'s text
    preserve `negation_claim`'s meaning, even paraphrased" is exactly the
    same bounded question the arbiter already answers for claim-vs-
    realization-text coverage, applied here to one other CLAIM's text
    instead of a whole candidate's winning realization text. Fails closed
    (returns None) whenever no arbiter is available, the arbiter raises, or
    it does not explicitly return True -- same posture as every other
    arbiter consumer in this module. Candidates are visited in a stable,
    deterministic order (sorted by claim_id); the FIRST arbiter-confirmed
    match wins, never an arbitrary pick."""
    if negation_claim.negation_role != CONTRASTIVE_HINDSIGHT_NEGATION:
        return None
    for other in sorted(all_claims, key=lambda c: c.claim_id):
        if not _hindsight_alignment_hard_gates_pass(negation_claim, other):
            continue
        overlap = _content_overlap_coefficient(negation_claim.content_tokens, other.content_tokens)
        if overlap < _HINDSIGHT_ALIGNMENT_AMBIGUOUS_FLOOR:
            continue
        if claim_equivalence_arbiter is None:
            continue
        try:
            covered, confidence, reason = claim_equivalence_arbiter.claim_covered(negation_claim.text, other.text)
        except Exception:
            covered, confidence, reason = False, 0.0, "arbiter_exception"
        verdict = bool(covered) is True
        if arbiter_log is not None:
            arbiter_log.append({
                "negation_claim_id": negation_claim.claim_id,
                "candidate_claim_id": other.claim_id,
                "overlap": overlap,
                "verdict": verdict,
                "confidence": confidence,
                "reason": reason,
            })
        if verdict:
            return other
    return None


def _covered_claim_ids(
    claims, text: str, *, arbiter: ClaimEquivalenceArbiter | None,
    hindsight_alignments: dict | None = None,
) -> frozenset:
    """D-065/D-066: `hindsight_alignments` (claim_id -> aligned Claim,
    default None so every pre-existing call site is byte-identical) lets a
    CRITICAL claim ALSO count as covered when `text` covers its
    arbiter-confirmed hindsight-equivalent claim instead of covering the
    original claim's own (often thin, ambiguous-band) whole-text overlap
    directly -- this is exactly how candidate A's own reflective clause
    ends up satisfying candidate B's CONTRASTIVE_HINDSIGHT_NEGATION claim
    once the two are proven equivalent (D-065 Section 7). Never a
    replacement for the original coverage check -- tried first, the
    alignment is only ever a fallback."""
    covered = set()
    for claim in claims:
        coverage = claim_coverage(claim, text)
        if resolve_ambiguous_coverage(claim, text, coverage=coverage, arbiter=arbiter):
            covered.add(claim.claim_id)
            continue
        aligned = (hindsight_alignments or {}).get(claim.claim_id)
        if aligned is None:
            continue
        aligned_coverage = claim_coverage(aligned, text)
        if resolve_ambiguous_coverage(aligned, text, coverage=aligned_coverage, arbiter=arbiter):
            covered.add(claim.claim_id)
    return frozenset(covered)


def _critical_coverage_dominant_candidate(
    candidate_ids: list[str],
    all_clips: dict,
    critical_claims,
    *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
    hindsight_alignments: dict | None = None,
) -> str | None:
    """D-063 CRITICAL_COVERAGE_DOMINANCE. See module docstring and
    `docs/CUTSELL_EDITORIAL_RESOLUTION_AND_HUMAN_ESCALATION_CONTRACT.md`
    Section 3 for the full contract. Returns the one candidate id (among
    `candidate_ids`, all already-selected members of one ambiguous retry
    family) whose CRITICAL-claim coverage is a PROPER superset of every
    OTHER candidate's own CRITICAL-claim coverage -- covers everything
    every other candidate covers, plus at least one more -- and is safe to
    prefer outright:

      - not itself a proven-incomplete/failed delivery (`complete_idea is
        False`); CLAUDE.md "WHEN UNCERTAIN, KEEP" means unknown/unset
        completeness (`None`, the common case for hand-authored fixtures
        and most real clips) is never treated as a disqualifier, only an
        EXPLICIT `False` is;
      - does not factually contradict any other candidate in the family
        (`contradiction_signal.any_pair_contradicts`, the same safety gate
        `canonical_edit_plan.py`'s own composite-safety check already uses
        -- never a second, independently-derived contradiction heuristic).

    Returns `None` whenever no such single candidate exists: identical
    coverage between candidates (a proper superset can never hold when two
    sets are equal), each candidate covers something CRITICAL the other
    lacks (genuinely disjoint, no dominance either way), a contradiction is
    present, or the only structurally-dominant candidate is itself an
    unusable/incomplete delivery. The caller must then leave the family
    exactly as it was -- D-063 Section 5: "if neither dominates, retain
    current REVIEW_REQUIRED behavior".

    A CONTEXTUAL-only extra claim never makes a candidate "dominant" here:
    `critical_claims` is already filtered to CRITICAL importance only by
    the caller (`_group_critical_claims`), so coverage is compared over
    CRITICAL claims exclusively -- this needs no special-case code, it
    falls directly out of only ever comparing critical-claim coverage
    sets.

    `hindsight_alignments` (D-065/D-066, default None -- byte-identical to
    every pre-D-066 call) is forwarded straight to `_covered_claim_ids`:
    the dominance ALGORITHM below is completely unchanged by it, only the
    coverage sets it operates on are corrected upstream."""
    coverage_by_id = {
        cid: _covered_claim_ids(
            critical_claims, str(all_clips[cid].text or ""),
            arbiter=claim_equivalence_arbiter, hindsight_alignments=hindsight_alignments,
        )
        for cid in candidate_ids
    }
    dominant = [
        cid for cid in candidate_ids
        if all(coverage_by_id[other] < coverage_by_id[cid] for other in candidate_ids if other != cid)
    ]
    if len(dominant) != 1:
        return None
    winner_id = dominant[0]
    if all_clips[winner_id].complete_idea is False:
        return None  # D-063 Section 6: never blindly prefer a proven-incomplete/failed delivery
    if any_pair_contradicts([str(all_clips[cid].text or "") for cid in candidate_ids]):
        return None  # D-063 Section 3/4: safety is never overridden by coverage dominance
    return winner_id


def resolve_critical_coverage_dominance(
    members: list[tuple[str, object]],
    candidate_ids: list[str],
    *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
    clause_role_arbiter: ClauseRoleArbiter | None = None,
) -> tuple[str | None, tuple[dict, ...]]:
    """D-063/D-065/D-066 CRITICAL_COVERAGE_DOMINANCE, including its
    hindsight-alignment pre-step, factored out of `apply_claim_coverage_
    best_take`'s own already-multi-selected-family branch so every caller
    shares the exact same dominance decision -- never two independently
    maintained copies of the same contract (D-082 Section 6: "Reuse
    existing D-063 CRITICAL_COVERAGE_DOMINANCE. Do NOT reimplement it.").

    `members` supplies the full pool `_all_group_claims` extracts every
    claim from (any importance, any candidate -- needed so a hindsight
    alignment partner can be found even outside `candidate_ids`, exactly
    as the original inline call did with the group's own `members`).
    `candidate_ids` is the subset actually being compared for dominance
    (`apply_claim_coverage_best_take`'s own `current_winners`; D-082's
    non-decisive-semantic-label member set in `pipeline.py`).

    Returns `(dominant_id_or_None, hindsight_alignment_diagnostics)` --
    the diagnostics rows carry no `group_id` (the caller's own concern);
    callers that track a group id add it themselves, same as before this
    refactor.
    """
    all_clips = {cid: clip for cid, clip in members}
    dominance_all_claims = _all_group_claims(members, clause_role_arbiter=clause_role_arbiter)
    dominance_critical_claims = tuple(c for c in dominance_all_claims if c.importance == CRITICAL)
    if not dominance_critical_claims:
        return None, ()

    # D-065/D-066: for every CONTRASTIVE_HINDSIGHT_NEGATION-eligible
    # critical claim in this family, search for an arbiter-confirmed
    # claim-vs-claim equivalence among the family's own non-protected
    # reflective claims BEFORE computing dominance -- see module
    # docstring's own D-065/D-066 section.
    hindsight_alignments: dict[str, object] = {}
    hindsight_arbiter_log: list[dict] = []
    hindsight_diagnostics: list[dict] = []
    for claim in dominance_critical_claims:
        if claim.negation_role != CONTRASTIVE_HINDSIGHT_NEGATION:
            continue
        aligned = _find_hindsight_alignment(
            claim, dominance_all_claims,
            claim_equivalence_arbiter=claim_equivalence_arbiter,
            arbiter_log=hindsight_arbiter_log,
        )
        consultations = [
            row for row in hindsight_arbiter_log if row["negation_claim_id"] == claim.claim_id
        ]
        hindsight_diagnostics.append({
            "claim_id": claim.claim_id,
            "claim_text": claim.text,
            "negation_role": claim.negation_role,
            "aligned_claim_id": aligned.claim_id if aligned is not None else None,
            "aligned_claim_text": aligned.text if aligned is not None else None,
            "arbiter_invoked": bool(consultations),
            "arbiter_consultations": consultations,
            "coverage_unit_relation": "merged" if aligned is not None else "unmerged",
            "reason": (
                "claim_vs_claim_equivalence_arbiter_confirmed" if aligned is not None
                else "no_arbiter_confirmed_equivalent_reflective_claim_found"
            ),
        })
        if aligned is not None:
            hindsight_alignments[claim.claim_id] = aligned

    dominant_id = _critical_coverage_dominant_candidate(
        candidate_ids, all_clips, dominance_critical_claims,
        claim_equivalence_arbiter=claim_equivalence_arbiter,
        hindsight_alignments=hindsight_alignments,
    )
    return dominant_id, tuple(hindsight_diagnostics)


def critical_coverage_sets(
    members: list[tuple[str, object]],
    candidate_ids: list[str],
    *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
    clause_role_arbiter: ClauseRoleArbiter | None = None,
) -> dict[str, frozenset]:
    """D-082 Section 7 support: the per-candidate CRITICAL-claim coverage
    sets `resolve_critical_coverage_dominance` itself computes internally,
    exposed so a caller can distinguish WHY dominance returned None --
    identical coverage (a true tie, safe for delivery to decide) from
    genuinely disjoint/asymmetric coverage (distinct unique facts, must
    stay unresolved) -- without duplicating `_critical_coverage_dominant_
    candidate`'s own dominance algorithm or its safety gates. Returns an
    empty dict when there are no CRITICAL claims in the group at all
    (nothing to distinguish on)."""
    all_clips = {cid: clip for cid, clip in members}
    dominance_all_claims = _all_group_claims(members, clause_role_arbiter=clause_role_arbiter)
    critical_claims = tuple(c for c in dominance_all_claims if c.importance == CRITICAL)
    if not critical_claims:
        return {}
    return {
        cid: _covered_claim_ids(
            critical_claims, str(all_clips[cid].text or ""), arbiter=claim_equivalence_arbiter,
        )
        for cid in candidate_ids
    }


def _time_compatible(left, right) -> bool:
    if left.source_asset_id != right.source_asset_id:
        return False
    a, b = (left, right) if left.start <= right.start else (right, left)
    return float(b.start) >= float(a.end) - _COMPOSITE_OVERLAP_TOLERANCE_SEC


def apply_claim_coverage_best_take(
    draft, *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
    clause_role_arbiter: ClauseRoleArbiter | None = None,
):
    """See module docstring. Never invents a new grouping/similarity
    heuristic; only reads `take_judge_groups` and moves clips between the
    existing selected/discarded buckets (KEEP/DISCARD only, D-019)."""
    groups = list((draft.diagnostics or {}).get("take_judge_groups") or ())
    if not groups:
        return draft

    selected_by_id = {clip.clip_id: clip for clip in draft.selected}
    alternates_by_id = {clip.clip_id: clip for clip in draft.alternates}
    discarded_by_id = {clip.clip_id: clip for clip in draft.discarded}
    all_clips = {**selected_by_id, **alternates_by_id, **discarded_by_id}

    def bucket_of(clip_id: str) -> str:
        if clip_id in selected_by_id:
            return "select"
        if clip_id in alternates_by_id:
            return "swap"
        return "discard"

    new_selected = dict(selected_by_id)
    new_alternates = dict(alternates_by_id)
    new_discarded = dict(discarded_by_id)
    overrides: list[dict] = []
    composites: list[dict] = []
    unresolved_gaps: list[dict] = []
    suppressed_incidental_overrides: list[dict] = []
    dominance_resolutions: list[dict] = []
    hindsight_alignment_diagnostics: list[dict] = []

    def move(clip_id: str, target: str) -> None:
        clip = all_clips[clip_id]
        new_selected.pop(clip_id, None)
        new_alternates.pop(clip_id, None)
        new_discarded.pop(clip_id, None)
        {"select": new_selected, "swap": new_alternates, "discard": new_discarded}[target][clip_id] = \
            replace(clip, selected=(target == "select"))

    for group in groups:
        group_id = group.get("group_id")
        ranked = list(group.get("ranked") or ())
        member_ids = [str(row.get("clip_id") or "") for row in ranked]
        members = [(cid, all_clips[cid]) for cid in member_ids if cid in all_clips]
        if len(members) < 2:
            continue

        current_winners = [cid for cid, _clip in members if bucket_of(cid) == "select"]
        if len(current_winners) == 0:
            # Already-fully-lost family -- StoryValidator's existing
            # territory, unrelated to D-063 (scoped to the >=2 ambiguous
            # case only).
            continue
        if len(current_winners) >= 2:
            # D-063 CRITICAL_COVERAGE_DOMINANCE: before this ambiguous,
            # already-multi-selected family falls through unchanged to
            # StoryValidator/FinalEditReviewer's existing
            # DUPLICATE_IDEA+UNRESOLVED_RETRY block, check whether exactly
            # one currently-selected candidate strictly dominates every
            # other on CRITICAL-claim coverage. See module docstring,
            # `_critical_coverage_dominant_candidate`'s own docstring, and
            # (D-082) `resolve_critical_coverage_dominance`'s own docstring
            # for the full contract.
            dominant_id, hindsight_rows = resolve_critical_coverage_dominance(
                members, current_winners,
                claim_equivalence_arbiter=claim_equivalence_arbiter,
                clause_role_arbiter=clause_role_arbiter,
            )
            hindsight_alignment_diagnostics.extend(
                {**row, "group_id": group_id} for row in hindsight_rows
            )
            if dominant_id is not None:
                for cid in current_winners:
                    if cid != dominant_id:
                        move(cid, "discard")
                dominance_resolutions.append({
                    "group_id": group_id,
                    "winner_clip_id": dominant_id,
                    "discarded_clip_ids": sorted(cid for cid in current_winners if cid != dominant_id),
                    "reason": "critical_coverage_dominance",
                })
            # Whether resolved or not, this module's job on this
            # already-multi-selected family ends here -- D-063 Section 5:
            # "if neither dominates, retain current REVIEW_REQUIRED
            # behavior" (no forced pick, family left exactly as it was).
            continue
        winner_id = current_winners[0]
        winner_clip = all_clips[winner_id]

        critical_claims = _group_critical_claims(members, clause_role_arbiter=clause_role_arbiter)
        if not critical_claims:
            continue

        winner_covered = _covered_claim_ids(critical_claims, str(winner_clip.text or ""), arbiter=claim_equivalence_arbiter)
        missing = [c for c in critical_claims if c.claim_id not in winner_covered]
        if not missing:
            continue

        # 1. Does a single OTHER member cover every critical claim?
        full_coverage_candidate = None
        member_coverage: dict[str, frozenset] = {}
        for clip_id, clip in members:
            if clip_id == winner_id:
                continue
            covered = _covered_claim_ids(critical_claims, str(clip.text or ""), arbiter=claim_equivalence_arbiter)
            member_coverage[clip_id] = covered
            if full_coverage_candidate is None and covered == frozenset(c.claim_id for c in critical_claims):
                full_coverage_candidate = clip_id

        if full_coverage_candidate is not None:
            # D-048 FIX 2 (D-047 Case 2): before committing to this
            # override, check whether every missing claim is a source-
            # exclusive, low-information incidental aside that a richer
            # winner should not be discarded for -- see
            # _override_blocked_by_incidental_self_source_claims's own
            # docstring for the full three-part compound rule.
            other_coverers = {
                claim.claim_id: {
                    cid for cid, covered in member_coverage.items()
                    if cid != full_coverage_candidate and claim.claim_id in covered
                }
                for claim in missing
            }
            if _override_blocked_by_incidental_self_source_claims(
                missing,
                candidate_clip_id=full_coverage_candidate,
                other_coverers=other_coverers,
                winner_clip=winner_clip,
                candidate_clip=all_clips[full_coverage_candidate],
            ):
                # Suppressed: every missing claim was incidental and
                # source-exclusive, so the current winner is left exactly
                # as it was -- deliberately does NOT fall through to the
                # 2-piece composite check below, which would otherwise
                # inject the same low-value aside into the KEEP timeline
                # anyway via a different mechanism (compositing `candidate`
                # in alongside `winner` purely to "cover" the same
                # incidental claim this gate just judged not worth it).
                suppressed_incidental_overrides.append({
                    "group_id": group_id,
                    "winner_clip_id": winner_id,
                    "suppressed_new_winner_clip_id": full_coverage_candidate,
                    "reason": "missing_claims_are_incidental_and_source_exclusive",
                    "missing_claim_ids": [c.claim_id for c in missing],
                    "missing_claim_texts": [c.text for c in missing],
                })
                continue
            else:
                move(winner_id, "discard")
                move(full_coverage_candidate, "select")
                overrides.append({
                    "group_id": group_id,
                    "previous_winner_clip_id": winner_id,
                    "new_winner_clip_id": full_coverage_candidate,
                    "reason": "single_candidate_covers_all_critical_claims_previous_winner_did_not",
                    "missing_claim_ids": [c.claim_id for c in missing],
                    "missing_claim_texts": [c.text for c in missing],
                })
                continue

        # 2. Bounded 2-piece composite: do exactly two members, together,
        # cover everything, and are they safe to place side by side?
        claim_by_id = {c.claim_id: c for c in critical_claims}
        composite_found = False
        for i, (id_a, clip_a) in enumerate(members):
            for id_b, clip_b in members[i + 1:]:
                if id_a == winner_id and id_b == winner_id:
                    continue
                covered_a = _covered_claim_ids(critical_claims, str(clip_a.text or ""), arbiter=claim_equivalence_arbiter)
                covered_b = _covered_claim_ids(critical_claims, str(clip_b.text or ""), arbiter=claim_equivalence_arbiter)
                union = covered_a | covered_b
                if union != frozenset(c.claim_id for c in critical_claims):
                    continue
                # Guard against forcing two members into a false composite
                # when their UNIQUE contributions share a claim_type: two
                # same-typed claims (e.g. both NEGATION) are more likely a
                # coarse-classifier miss on one restated idea than genuinely
                # complementary distinct facts -- exactly how a paraphrased
                # retry family (each side worded its own negation slightly
                # differently) could otherwise get frozen as a fake
                # "composite" instead of correctly collapsing to one winner
                # via the existing semantic-equivalence-arbiter tie-break
                # this module never overrides. Disjoint claim_types (e.g. a
                # STATE_RESULT fact and a DIAGNOSIS_IDENTIFICATION fact) are
                # structurally unlikely to be the same proposition, so those
                # stay compositable. No arbiter exists yet for "are these
                # two claims equivalent" specifically (a different, pairwise
                # question from ClaimEquivalenceArbiter's own coverage
                # check) -- honest gap, fails closed to unresolved_gaps.
                unique_to_a = covered_a - covered_b
                unique_to_b = covered_b - covered_a

                def _unique_contribution_is_incidental(unique_ids, contributor_id) -> bool:
                    # D-048 FIX 2: the same self-source-exclusivity gate as
                    # the single-candidate override path (see
                    # _override_blocked_by_incidental_self_source_claims),
                    # applied to a composite pairing -- a claim only one
                    # side contributes must not force that side into the
                    # KEEP timeline merely because it is incidental AND no
                    # other sibling in the family covers it either.
                    if not unique_ids:
                        return False
                    for cid in unique_ids:
                        claim = claim_by_id[cid]
                        if not _is_low_information_incidental(claim):
                            return False
                        corroborated_elsewhere = any(
                            other_id != contributor_id and cid in _covered_claim_ids(
                                (claim,), str(other_clip.text or ""), arbiter=claim_equivalence_arbiter,
                            )
                            for other_id, other_clip in members
                        )
                        if corroborated_elsewhere:
                            return False
                    return True

                if _unique_contribution_is_incidental(unique_to_a, id_a) or _unique_contribution_is_incidental(unique_to_b, id_b):
                    continue
                types_a = {claim_by_id[cid].claim_type for cid in unique_to_a}
                types_b = {claim_by_id[cid].claim_type for cid in unique_to_b}
                if types_a & types_b:
                    continue
                if not _time_compatible(clip_a, clip_b):
                    continue
                first, second = (clip_a, clip_b) if clip_a.start <= clip_b.start else (clip_b, clip_a)
                move(first.clip_id, "select")
                move(second.clip_id, "select")
                for clip_id, _clip in members:
                    if clip_id not in (first.clip_id, second.clip_id) and bucket_of(clip_id) != "discard":
                        move(clip_id, "discard")
                composites.append({
                    "group_id": group_id,
                    "clip_ids": [first.clip_id, second.clip_id],
                    "reason": "claim_coverage_complementary",
                    "covered_claim_ids": sorted(union),
                })
                composite_found = True
                break
            if composite_found:
                break

        if not composite_found:
            unresolved_gaps.append({
                "group_id": group_id,
                "winner_clip_id": winner_id,
                "missing_claim_ids": [c.claim_id for c in missing],
                "missing_claim_texts": [c.text for c in missing],
                "reason": "no_single_or_paired_candidate_safely_covers_every_critical_claim",
            })

    if not (
        overrides or composites or unresolved_gaps or suppressed_incidental_overrides
        or dominance_resolutions or hindsight_alignment_diagnostics
    ):
        return draft

    def _order(clip):
        return (clip.source_order, float(clip.start), float(clip.end), clip.clip_id)

    selected = tuple(sorted(new_selected.values(), key=_order))
    alternates = tuple(sorted(new_alternates.values(), key=_order))
    discarded = tuple(sorted(new_discarded.values(), key=_order))

    diagnostics = dict(draft.diagnostics or {})
    diagnostics["claim_coverage_best_take"] = {
        "status": "applied",
        "overrides": overrides,
        "composites": composites,
        "unresolved_gaps": unresolved_gaps,
        # D-048 FIX 2: single-candidate overrides this module deliberately
        # did NOT apply because every missing claim was source-exclusive
        # and low-information/incidental relative to the current winner --
        # see _override_blocked_by_incidental_self_source_claims.
        "suppressed_incidental_overrides": suppressed_incidental_overrides,
        # D-063: already-ambiguous (2+ selected) families resolved via
        # CRITICAL_COVERAGE_DOMINANCE -- see
        # `_critical_coverage_dominant_candidate`'s own docstring. Empty
        # whenever no such family existed, or none had a single dominant
        # candidate (still falls through to the existing
        # DUPLICATE_IDEA+UNRESOLVED_RETRY block unchanged).
        "dominance_resolutions": dominance_resolutions,
        # D-065/D-066: which CONTRASTIVE_HINDSIGHT_NEGATION-eligible claim
        # was checked against which same-family reflective claim, whether
        # the arbiter was invoked, its verdict, and the resulting coverage-
        # unit relation -- never exposed to any user-facing UI. Empty
        # whenever no already-ambiguous (2+ selected) family contained a
        # CONTRASTIVE_HINDSIGHT_NEGATION-eligible claim this run.
        "hindsight_alignments": hindsight_alignment_diagnostics,
    }
    return replace(draft, selected=selected, alternates=alternates, discarded=discarded, diagnostics=diagnostics)
