"""Final Story / Coherence Validation -- Clean Cut Core V1.

Runs after Best-Take authority resolves decisive retry-family contests and
before Selection Freeze. It is the last semantic authority allowed to touch
membership; Boundary must never repair a semantic membership mistake (see
CLAUDE.md / docs/CUTSELL_DECISIONS.md).

Clean Cut Core V1 product scope: SELECT/KEEP vs DISCARD only. SWAP is out of
scope until explicitly reintroduced. This module is also where that becomes
final and irreversible for the winning timeline: whatever the upstream
authorities left in ``alternates`` is folded into ``discarded`` here, because
nothing that isn't SELECT belongs in the one winning edit.

What this module checks, deterministically, from evidence the pipeline
already computed (no new heuristic invented for this pass):

  - KEEP/DISCARD only: alternates always folds into discarded.
  - Unresolved retry families: a take_judge_groups entry (a genuine retry
    contest -- 2+ ranked members) that STILL has 2+ members in the final
    selected set means Best-Take authority did not resolve it (the
    score gap was too thin to be decisive). This is exactly the "unresolved
    final-story coherence" case the architecture reserves for a bounded
    semantic arbiter, not a new guard: if a semantic_equivalence_arbiter is
    available, the residual members are asked pairwise whether they are the
    same intended idea; a confirmed match keeps only the take_judge's own
    top-ranked member and discards the rest. Fails open -- no arbiter, or an
    arbiter that cannot confidently confirm sameness, leaves the family
    exactly as it was, flagged in diagnostics for human review rather than
    silently resolved without evidence.
  - Missing story ending: flags (does not auto-restore) when the
    chronologically-last kept take in a source was discarded and nothing
    selected follows it in that source -- a possible dropped CTA/closing
    beat. Observability only; auto-restoring here would risk overriding a
    legitimate composer/review trim on no stronger evidence than position.

  - Contradiction invariant: within a genuine retry-family group (2+ ranked
    members -- i.e. already-established competing attempts of ONE idea),
    two members that both remain selected but disagree on a number or an
    explicit negation (reusing final_sibling_grouping's own _numbers/
    _negations extractors -- the same signals that module already requires
    to MATCH before it will merge two takes) are factually incompatible
    variants, not two acceptable phrasings. This is not a semantic-judgment
    question an arbiter should guess at; it is evidence-based and
    deterministic. Such a family is never auto-resolved here -- it sets
    freeze_blocked so the caller does not proceed to Selection Freeze with a
    self-contradictory statement in the winning timeline.
  - Idea coverage: every take_judge_groups entry represents one intended
    idea/retry contest. If a group ends up with ZERO members in the final
    selected set, that idea vanished from the winning edit entirely --
    high-confidence, deterministic, and exactly the "missing required idea"
    failure class Selection Freeze must never see silently. This also sets
    freeze_blocked.
  - Lost semantic atoms (general coverage ledger): the take_judge_groups
    check above is blind to anything deleted upstream of grouping entirely
    -- most importantly hybrid_session_cleanup's per-clip failed/BTS
    classification, which runs before IdeaClusterer ever sees the
    candidate and has no idea-coverage awareness of its own. This check
    instead compares every discarded clip's own content directly against
    the union of the FINAL selected text -- independent of which stage
    discarded it, or whether it was ever grouped at all -- so it catches
    unique-fact/idea loss regardless of which upstream authority (or
    combination of legacy hybrid_* authorities, see D-021) caused it. Any
    missing number/negation atom is flagged unconditionally; a broader
    loss of content vocabulary is flagged only past a volume+coverage
    floor, so a genuinely redundant, correctly-discarded retry (which
    shares most of its topic vocabulary with the surviving winner) is not
    mistaken for real information loss. This also sets freeze_blocked. See
    _lost_semantic_atoms's own docstring for the concrete failure case
    (RAW 33345946000) that motivated it.

Not implemented in V1 (documented gap, not silently skipped): general
(non-numeric/negation) factual contradiction between two takes that are
not already an established retry-family pair -- comparing arbitrary
unrelated texts would trivially "contradict" on every number/negation
mismatch, which is not a general-purpose check's job. The lost-semantic-
atoms check above closes the exhaustive-unique-fact-loss gap that used to
be listed here, but is lexical-overlap-based, not paraphrase-aware: a
fact restated with entirely different vocabulary could still be
mis-flagged as lost. That failure mode blocks freeze for human review
rather than silently producing a bad video, which is the conservative
direction CLAUDE.md's "WHEN UNCERTAIN, KEEP" rule asks for.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import Mapping

from .contracts import effective_parent_semantic_clip_id
from .contradiction_signal import any_pair_contradicts, detect_text_contradiction
from .final_sibling_grouping import _content, _negations, _numbers
from .semantic_atom_importance import (
    SemanticAtomImportanceArbiter,
    blocks_freeze,
    classify_negation_atom,
    classify_number_atom,
    resolve_uncertain_with_arbiter,
)
from .semantic_claims import (
    CRITICAL as CLAIM_CRITICAL,
    COVERAGE_THRESHOLD,
    ClaimEquivalenceArbiter,
    ClauseRoleArbiter,
    claim_coverage,
    dedupe_claims,
    extract_claims,
    resolve_ambiguous_coverage,
)
from .semantic_idea_equivalence import (
    IdeaEquivalencePair,
    IdeaEquivalenceRequest,
    SemanticEquivalenceArbiter,
    same_idea_by_pair_index,
    safe_check_idea_equivalence,
)


def _fold_alternates_into_discarded(draft):
    if not draft.alternates:
        return draft
    discarded = tuple(
        sorted(
            (*draft.discarded, *(replace(clip, selected=False) for clip in draft.alternates)),
            key=lambda clip: (clip.source_order, clip.start, clip.end, clip.clip_id),
        )
    )
    return replace(draft, alternates=(), discarded=discarded)


def _claim_coverage_composite_group_ids(diagnostics: Mapping[str, object]) -> frozenset[str]:
    """group_ids `claim_coverage_best_take.py` (D-038) already resolved as
    its own narrow 2-piece composite fallback. A group it resolved this way
    legitimately has 2+ members still selected on purpose -- it is not an
    unresolved retry-family contest, and must never be handed to
    `_resolve_residual_family` below, which would otherwise use the very
    same semantic_equivalence_arbiter that grouped these members together
    in the first place to collapse the composite back down to one winner,
    silently destroying the claim-coverage fix (same trap
    `canonical_edit_plan._composite_piece_ids` already accounts for)."""
    composites = ((diagnostics or {}).get("claim_coverage_best_take") or {}).get("composites") or ()
    return frozenset(str(row.get("group_id") or "") for row in composites if isinstance(row, dict))


def _members_contradiction_free(still_selected: list[dict], take_by_id: dict[str, object]) -> bool:
    """D-056.3 Section 4: the shared contradiction contract, applied to an
    upstream-claimed "resolved composite" before StoryValidator agrees to
    drop it from unresolved-family bookkeeping. A composite is only ever
    VALIDATED SAFE -- eligible to resolve the family -- when none of its
    still-selected members factually contradict each other. Members whose
    text cannot be resolved are skipped (never invented), matching this
    file's existing fail-open posture elsewhere; this never widens what
    counts as a conflict, only narrows what evidence is available to find
    one."""
    texts = [
        str(take.text)
        for row in still_selected
        if (take := take_by_id.get(str(row.get("clip_id") or ""))) is not None
    ]
    return not any_pair_contradicts(texts)


def _residual_multi_select_groups(draft, take_by_id: dict[str, object]) -> list[dict]:
    selected_ids = {clip.clip_id for clip in draft.selected}
    diagnostics = draft.diagnostics or {}
    groups = list(diagnostics.get("take_judge_groups") or ())
    exempt_group_ids = _claim_coverage_composite_group_ids(diagnostics)
    residual = []
    for group in groups:
        ranked = list(group.get("ranked") or ())
        still_selected = [row for row in ranked if str(row.get("clip_id") or "") in selected_ids]
        if len(still_selected) < 2:
            continue
        if str(group.get("group_id") or "") in exempt_group_ids:
            # D-056.3 root-defect fix: an upstream composite mechanism
            # (e.g. claim_coverage_best_take.py's own narrow 2-piece
            # fallback) claims this family is resolved -- but StoryValidator
            # never takes that on faith merely because a composite object
            # exists (D-056.2 Run B/C: `tg_539b31f663aaf9e13f`/
            # `tg_f4b9e7c1fe3e28a1af`, both a factually-contradictory pair
            # accepted as "resolved" by an upstream mechanism that never
            # checked for contradiction at all). Only a composite whose
            # still-selected members are NOT contradictory (same shared
            # primitive `_contradiction_findings` below uses) may be
            # exempted. A contradictory "composite" falls straight through
            # to residual tracking, exactly as if no upstream mechanism had
            # ever claimed it was resolved.
            if _members_contradiction_free(still_selected, take_by_id):
                continue
        residual.append({**group, "ranked": ranked, "still_selected": still_selected})
    return residual


def _resolve_residual_family(
    group: dict,
    take_by_id: dict[str, object],
    arbiter: SemanticEquivalenceArbiter | None,
) -> tuple[list[str], list[dict], list[dict]]:
    """Return (clip_ids_to_discard, audit_rows, contradiction_rows) for one
    still-ambiguous retry family. Empty on fail-open (no arbiter, or nothing
    confirmed).

    A pair the arbiter confirms as the same idea is NOT automatically safe to
    collapse: "same intended idea" is a different question from "factually
    compatible enough to silently keep one and drop the other." A pair that
    disagrees on a number or explicit negation is reported as a
    contradiction and excluded from the union-find collapse entirely --
    both members stay selected, unresolved, for the caller's freeze-blocking
    contradiction invariant to catch.
    """
    still_selected = group["still_selected"]
    if arbiter is None or len(still_selected) < 2:
        return [], [], []

    ordered = sorted(still_selected, key=lambda row: -float(row.get("score") or 0.0))
    pairs_meta = list(combinations(range(len(ordered)), 2))
    request = IdeaEquivalenceRequest(pairs=tuple(
        IdeaEquivalencePair(
            left_text=str(take_by_id.get(ordered[i]["clip_id"]).text if take_by_id.get(ordered[i]["clip_id"]) else ""),
            right_text=str(take_by_id.get(ordered[j]["clip_id"]).text if take_by_id.get(ordered[j]["clip_id"]) else ""),
        )
        for i, j in pairs_meta
    ))
    if not request.pairs:
        return [], []
    result = safe_check_idea_equivalence(arbiter, request)
    decisions = same_idea_by_pair_index(result)
    if not decisions:
        return [], [], []

    # Union-find over the ordered members: any confirmed same-idea pair that
    # is ALSO factually compatible collapses to keeping only the
    # higher-ranked (index 0 after sort) member of that connected component.
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    audit: list[dict] = []
    contradictions: list[dict] = []
    any_confirmed = False
    for pair_index, (i, j) in enumerate(pairs_meta):
        decision = decisions.get(pair_index)
        if decision is None:
            continue
        same_idea, confidence, reason = decision
        if not same_idea:
            continue
        left_take, right_take = take_by_id.get(ordered[i]["clip_id"]), take_by_id.get(ordered[j]["clip_id"])
        left_text = left_take.text if left_take is not None else ""
        right_text = right_take.text if right_take is not None else ""
        signal = detect_text_contradiction(left_text, right_text)
        if signal.has_conflict:
            # Confirmed same idea, but factually incompatible -- do not
            # collapse; leave both selected, unresolved, for the caller's
            # freeze-blocking contradiction check to catch.
            contradictions.append({
                "group_id": group.get("group_id"),
                "left_clip_id": ordered[i]["clip_id"],
                "right_clip_id": ordered[j]["clip_id"],
                "number_conflict": signal.number_conflict,
                "negation_conflict": signal.negation_conflict,
            })
            continue
        any_confirmed = True
        ra, rb = find(i), find(j)
        if ra != rb:
            keeper = min(ra, rb)
            loser = max(ra, rb)
            parent[loser] = keeper
        audit.append({
            "left_clip_id": ordered[i]["clip_id"],
            "right_clip_id": ordered[j]["clip_id"],
            "confidence": round(confidence, 4),
            "reason": reason,
        })

    if not any_confirmed:
        return [], [], contradictions

    clusters: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        clusters.setdefault(find(index), []).append(index)

    to_discard: list[str] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        # ordered is already sorted by score descending, so the
        # lowest-index member of each cluster is the take_judge's own
        # top-ranked pick within it.
        keeper_index = min(members)
        for member_index in members:
            if member_index != keeper_index:
                to_discard.append(ordered[member_index]["clip_id"])

    return to_discard, audit, contradictions


def _contradiction_findings(draft, take_by_id: dict[str, object]) -> list[dict]:
    """Detect factually-incompatible members still co-selected within the
    SAME retry-family group. Scoped to established retry families only --
    comparing arbitrary unrelated texts would trivially "contradict" on
    every number/negation mismatch, which is not this check's job."""
    selected_ids = {clip.clip_id for clip in draft.selected}
    findings: list[dict] = []
    for group in (draft.diagnostics or {}).get("take_judge_groups") or ():
        ranked = list(group.get("ranked") or ())
        still_selected = [row for row in ranked if str(row.get("clip_id") or "") in selected_ids]
        if len(still_selected) < 2:
            continue
        for left, right in combinations(still_selected, 2):
            left_take = take_by_id.get(left["clip_id"])
            right_take = take_by_id.get(right["clip_id"])
            if left_take is None or right_take is None:
                continue
            signal = detect_text_contradiction(left_take.text, right_take.text)
            if signal.has_conflict:
                findings.append({
                    "group_id": group.get("group_id"),
                    "left_clip_id": left["clip_id"],
                    "right_clip_id": right["clip_id"],
                    "number_conflict": signal.number_conflict,
                    "negation_conflict": signal.negation_conflict,
                })
    return findings


def _missing_idea_coverage(draft) -> list[dict]:
    """Every take_judge_groups entry is one intended idea/retry contest.
    Flag any whose members are ALL absent from the final selected set --
    that idea vanished from the winning edit entirely.

    D-046 FIX A: "absent from the final selected set" must also recognize a
    winning member that a post-selection physical split (e.g.
    post_selection_interior_gap_trim) later divided into fragments carrying
    a different `clip_id` -- those fragments' own `parent_semantic_clip_id`
    (D-036 provenance, general and not Video00-specific) still names the
    original member. Without this, a genuinely-surviving realization was
    misreported as having "vanished" and incorrectly blocked Freeze -- see
    D-045 Case A. Only `draft.selected` is consulted for this linkage, so a
    fragment of an actually-discarded clip cannot revive it."""
    selected_ids = {clip.clip_id for clip in draft.selected}
    selected_parent_ids = {
        pid for clip in draft.selected
        if (pid := effective_parent_semantic_clip_id(clip)) is not None
    }
    missing: list[dict] = []
    for group in (draft.diagnostics or {}).get("take_judge_groups") or ():
        ranked = list(group.get("ranked") or ())
        member_ids = [str(row.get("clip_id") or "") for row in ranked]
        covered = any(
            cid in selected_ids or cid in selected_parent_ids
            for cid in member_ids
        )
        if member_ids and not covered:
            missing.append({"group_id": group.get("group_id"), "member_clip_ids": member_ids})
    return missing


# ---------------------------------------------------------------------------
# D-061 Phase 1: SAME-IDEA PARAPHRASE CREDIT for _lost_semantic_atoms
# ---------------------------------------------------------------------------
#
# Root defect (docs/CUTSELL_DECISIONS.md D-060's forensic): `_lost_semantic_
# atoms` is deliberately whole-video/bag-of-words/idea-membership-blind (see
# its own docstring below) -- correct for its original purpose of catching
# content deleted BEFORE grouping ever saw it, but that same blindness makes
# it flag a discarded clip as UNIQUE_FACT_LOST even when a SELECTED member of
# its own retry family already carries the identical proposition in
# different words. D-060's live shape: two clips merged by the semantic-
# equivalence arbiter at 0.9 confidence ("Both reflect on unrecognized
# symptoms viewed in retrospect") shared almost no raw vocabulary, so the
# discarded one scored only 0.4 whole-video coverage and blocked Freeze for
# content that was never actually lost.
#
# Fix: before treating a content-loss trigger as blocking, check whether the
# discarded clip reached semantic grouping at all (is a member of a genuine,
# 2+-member `take_judge_groups` retry contest) and has a SELECTED sibling in
# that exact group. If so, reuse existing evidence -- no new paid call: an
# existing `semantic_idea_equivalence` merge record for this exact
# (discarded, winner) pair at high confidence already answers the question
# an arbiter would otherwise be asked again for nothing.
#
# Deliberately NOT also tried: a purely deterministic idea-scoped word-
# overlap fallback (comparing the discarded clip only against its own
# winner's text, instead of the whole video). It looks like it should catch
# an exact/near-duplicate that never got a separate arbiter call, but it is
# mathematically inert here: `winner_content` is always a SUBSET of
# `kept_content` (the winner is itself part of the selected/kept timeline),
# so idea-scoped coverage can never exceed whole-video coverage. Since this
# credit path is only ever consulted after whole-video coverage has ALREADY
# failed the same 0.45 floor, the idea-scoped version is guaranteed to fail
# it too -- it would never once fire. A genuine near-duplicate is, in fact,
# already handled correctly by the existing whole-video check with no fix
# needed at all: the winner's own presence in `kept_text` guarantees high
# overlap for anything that closely resembles it. Real low-word-overlap
# paraphrases (the actual D-060 shape) can only be told apart from a
# genuinely additive fact by semantic judgment, which is exactly what the
# arbiter-evidence path above reuses -- not reinvented as a second, useless
# heuristic here.
#
# A clip that never reached grouping at all (PRE_GROUP_REJECTED, an
# ungrouped candidate, or a hybrid_session_cleanup delete with no verified
# replacement -- exactly the gap this check exists to catch, see the RAW
# 33345946000 papillary-cancer case in the docstring below) has no entry in
# `clip_id_to_group` and this credit never applies -- current fail-closed
# behavior is completely unchanged for that class of clip. A genuinely
# additive fact inside the same idea group also clears nothing (no pairwise
# arbiter record confirms it) and remains blocking, unchanged.
_SAME_IDEA_HIGH_CONFIDENCE_THRESHOLD = 0.85  # matches D-058 Phase 2's own bar


def _clip_id_to_group_members(groups) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Map clip_id -> (group_id, all member clip_ids) from take_judge_
    groups' own `ranked` lists -- the FINAL, post-merge/post-split retry-
    family membership Best Take actually contested over (D-058 Phase 1's
    split_incohesive_retry_groups already ran upstream of this). Only
    genuine 2+-member contests count; a singleton is not a retry family."""
    mapping: dict[str, tuple[str, tuple[str, ...]]] = {}
    for group in groups or ():
        group_id = str(group.get("group_id") or "")
        member_ids = tuple(
            str(row.get("clip_id") or "") for row in (group.get("ranked") or ()) if row.get("clip_id")
        )
        if len(member_ids) < 2:
            continue
        for clip_id in member_ids:
            mapping[clip_id] = (group_id, member_ids)
    return mapping


def _semantic_equivalence_confidence(diagnostics, left_id: str, right_id: str) -> float:
    """Highest confidence any EXISTING `semantic_idea_equivalence` merge
    record already established for this exact pair -- reused, never
    recomputed. 0.0 when no such record exists for this pair. This is
    'existing semantic-equivalence evidence the engine already possesses'
    (D-061 Phase 1): a call already paid for during grouping, never asked
    again here."""
    merges = ((diagnostics or {}).get("semantic_idea_equivalence") or {}).get("merges") or ()
    pair = {left_id, right_id}
    best = 0.0
    for merge in merges:
        if {str(merge.get("left_clip_id") or ""), str(merge.get("right_clip_id") or "")} == pair:
            try:
                best = max(best, float(merge.get("confidence") or 0.0))
            except (TypeError, ValueError):
                continue
    return best


def _same_idea_paraphrase_credit(
    clip, member_ids: tuple[str, ...], selected_ids: set, diagnostics,
) -> tuple[bool, str | None]:
    """Does a SELECTED member of `clip`'s own retry family already carry
    the same proposition, per EXISTING semantic-equivalence evidence? Never
    invents a new heuristic and never places a new paid call -- see the
    module comment above for why this is the only evidence source
    consulted (a deterministic idea-scoped overlap fallback is inert by
    construction and is deliberately not implemented)."""
    winner_ids = [cid for cid in member_ids if cid != clip.clip_id and cid in selected_ids]
    if not winner_ids:
        return False, None
    if len(_content(str(clip.text or ""))) < 5:
        return False, None
    for winner_id in winner_ids:
        if _semantic_equivalence_confidence(
            diagnostics, clip.clip_id, winner_id
        ) >= _SAME_IDEA_HIGH_CONFIDENCE_THRESHOLD:
            return True, "same_idea_semantic_equivalence"
    return False, None


def _lost_semantic_atoms(
    draft, *,
    semantic_atom_importance_arbiter: SemanticAtomImportanceArbiter | None = None,
    semantic_preservation_proofs: Mapping[str, object] | None = None,
) -> list[dict]:
    """General coverage ledger over the ACTUAL final KEEP timeline.

    ``_missing_idea_coverage`` above is scoped to ``take_judge_groups`` --
    the retry families IdeaClusterer actually formed. It is blind to any
    candidate deleted upstream of grouping entirely, most importantly
    ``hybrid_session_cleanup.apply_hybrid_session_cleanup`` (pipeline.py
    Pass 2): a per-clip failed/BTS classifier that runs BEFORE IdeaClusterer
    (safe_group_takes_by_sessions / reconcile_semantic_idea_equivalence)
    ever sees the candidate, and has no idea-coverage awareness of its own
    -- it judges one take in isolation, with no concept of whether it is
    the sole carrier of an audience-facing fact. A clip it deletes never
    enters any take_judge group, so a whole idea can vanish this way while
    ``_missing_idea_coverage`` reports nothing missing, because as far as
    that check can see, the idea never existed in the first place.
    (RAW 33345946000, head 0ea0adf: the papillary-cancer diagnosis
    confirmation and the pimples/rash story beat were both
    high_confidence_semantic hybrid deletions; freeze_blocked was false.)

    This check instead compares every discarded clip's own content
    directly against the union of the final selected text -- independent
    of which stage discarded it, or whether it was ever grouped at all.

    D-031: a missing number/negation atom is no longer flagged as blocking
    unconditionally. RAW 33402023395 found the old unconditional rule too
    blunt: it blocked Freeze over a discarded clip's incidental year
    ("...en 2023.") that the Human Gold oracle itself does not preserve in
    its own equivalent delivery -- the audience-facing idea (endoscopy ->
    diagnosis -> medication) was already fully intact. Each missing atom is
    now run through `semantic_atom_importance.classify_*` -- a negation is
    always CRITICAL; a number is CRITICAL when its own clip's text carries
    a percentage/price/measurement/dose/correction-language marker, or
    UNCERTAIN with no such marker (which still blocks -- WHEN UNCERTAIN,
    KEEP), and only CONTEXTUAL for a bare, plausible-year-shaped number in
    an ordinary temporal-aside clause. `blocking` on each finding reflects
    this: True (as before) for a genuinely critical/uncertain atom or the
    broader content-loss signal below; False only when every missing atom
    on that clip classified as CONTEXTUAL. The broader loss of ordinary
    content vocabulary (a whole idea's worth of unrelated words, not a
    specific atom) is UNCHANGED and always blocking once it clears both a
    volume and a coverage floor -- this reclassification is scoped to
    number/negation atoms only, not to that coarser signal.

    D-061 Phase 1: the broader content-vocabulary trigger above is now
    checked against same-idea paraphrase credit (see the module comment and
    `_same_idea_paraphrase_credit` above) before it is allowed to block --
    a POST-GROUPING discarded clip whose own retry family already has a
    SELECTED member covering the same proposition is relabeled
    `SEMANTICALLY_COVERED_BY_SELECTED_REALIZATION` and stops blocking,
    still recorded (never silently dropped) with the evidence that
    suppressed it. Number/negation atom handling above is completely
    untouched: `missing_critical` is computed against the whole selected
    timeline's own vocabulary already, so an atom this idea's own winner
    genuinely carries was never "missing" in the first place, and a
    genuinely missing one still blocks regardless of idea membership.
    """
    kept_text = " ".join(str(clip.text or "") for clip in draft.selected)
    kept_content = _content(kept_text)
    kept_critical = _numbers(kept_text) | _negations(kept_text)
    selected_ids = {clip.clip_id for clip in draft.selected}
    clip_id_to_group = _clip_id_to_group_members((draft.diagnostics or {}).get("take_judge_groups"))

    findings: list[dict] = []
    for clip in draft.discarded:
        text = str(clip.text or "")
        if len(text.split()) < 3:
            # Too short to safely judge as carrying a distinct idea, a
            # standalone critical fact, or even a filler reaction ("no
            # wait", "okay stop") -- avoids flagging bare BTS scraps.
            continue
        own_content = _content(text)
        own_numbers = _numbers(text)
        own_negations = _negations(text)
        own_critical = own_numbers | own_negations
        missing_critical = sorted(own_critical - kept_critical)
        # The broader content-vocabulary check needs enough own content to
        # judge reliably; a critical atom (number/negation) is meaningful
        # evidence even in an otherwise short clip, so it is never gated by
        # this floor.
        missing_content = sorted(own_content - kept_content) if len(own_content) >= 5 else []
        coverage = 1.0 - (len(missing_content) / max(1, len(own_content) or 1))
        content_loss = len(own_content) >= 5 and len(missing_content) >= 4 and coverage < 0.45
        if not (missing_critical or content_loss):
            continue

        # D-061 Phase 1: same-idea paraphrase credit -- only ever suppresses
        # the broader content_loss signal above, never touches
        # missing_critical/classifications (see docstring paragraph above).
        suppressed_reason = None
        if content_loss:
            group = clip_id_to_group.get(clip.clip_id)
            if group is not None:
                _group_id, member_ids = group
                credited, evidence_kind = _same_idea_paraphrase_credit(
                    clip, member_ids, selected_ids, draft.diagnostics,
                )
                if credited:
                    content_loss = False
                    suppressed_reason = evidence_kind

        classifications = [
            classify_negation_atom(atom) if atom in own_negations else classify_number_atom(atom, text)
            for atom in missing_critical
        ]
        classifications = resolve_uncertain_with_arbiter(
            classifications, source_text=text, kept_text=kept_text,
            arbiter=semantic_atom_importance_arbiter,
        )
        blocking = content_loss or any(blocks_freeze(c.importance) for c in classifications)

        # D-076: a verified SEMANTIC_PRESERVATION_PROOF from the Unified
        # Resolver (LEXICAL_REPLACEMENT/SEMANTIC_REPLACEMENT for a
        # hybrid_editorial_chunks discard already certified by PATH A/PATH
        # B, or PRE_GROUP_SEMANTIC_PRESERVATION for a discard that never
        # reached grouping at all -- see realization_resolver.py's own
        # SemanticPreservationProof/build_semantic_preservation_proofs
        # docstrings) may ALSO suppress an otherwise-blocking finding, or
        # simply upgrade this row's own classification/evidence when it
        # was already non-blocking for an unrelated reason (e.g. its only
        # missing atom was already CONTEXTUAL) -- Section 6's own "must
        # not be silently dropped from diagnostics" requires surfacing a
        # genuine proof even then, never only when it flips a verdict.
        # Only consulted when GROUPED_SAME_IDEA (`_same_idea_paraphrase_
        # credit` above) did NOT already claim this row -- that mechanism
        # keeps its own precedence untouched. StoryValidator only ever
        # CONSUMES this proof -- one dict lookup -- it never discovers a
        # candidate, extracts a claim, ranks a retry, or invokes an
        # arbiter itself for this decision. A proof that is present but
        # not `verified` changes nothing: current fail-closed behavior
        # remains exactly as before this directive.
        preserving_realization_id = None
        preserved_claim_ids: tuple = ()
        nonrequired_omissions: tuple = ()
        if semantic_preservation_proofs is not None and suppressed_reason is None:
            proof = semantic_preservation_proofs.get(clip.clip_id)
            if proof is not None and bool(getattr(proof, "verified", False)):
                blocking = False
                suppressed_reason = str(getattr(proof, "proof_method", "") or "")
                preserving_realization_id = getattr(proof, "preserving_id", None)
                preserved_claim_ids = tuple(getattr(proof, "preserved_claim_ids", ()) or ())
                nonrequired_omissions = tuple(getattr(proof, "nonrequired_omissions", ()) or ())

        row = {
            "clip_id": clip.clip_id,
            "text": text,
            "missing_critical_atoms": missing_critical,
            "atom_classifications": [
                {
                    "atom": c.atom, "atom_type": c.atom_type, "importance": c.importance,
                    "evidence": c.evidence, "resolved_by": c.resolved_by,
                }
                for c in classifications
            ],
            "missing_content_token_count": len(missing_content),
            "own_content_token_count": len(own_content),
            "coverage_against_final_keep": round(coverage, 4),
            "blocking": blocking,
            # D-061 Phase 3: distinguish REAL_CONTENT_LOSS from a finding
            # that was suppressed because same-idea semantic equivalence
            # already established coverage -- never silently hidden, even
            # in the rare case a separate missing-critical-atom signal
            # still blocks the row for an unrelated reason.
            "classification": (
                "SEMANTICALLY_COVERED_BY_SELECTED_REALIZATION" if suppressed_reason and not blocking
                else "REAL_CONTENT_LOSS"
            ),
        }
        if suppressed_reason is not None:
            row["content_loss_suppressed_by"] = suppressed_reason
        if preserving_realization_id is not None:
            row["preserving_realization_id"] = preserving_realization_id
        if preserved_claim_ids:
            row["preserved_claim_ids"] = list(preserved_claim_ids)
        if nonrequired_omissions:
            row["nonrequired_omissions"] = [dict(o) for o in nonrequired_omissions]
        findings.append(row)
    return findings


def _lost_critical_claims(
    draft, *,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
    clause_role_arbiter: ClauseRoleArbiter | None = None,
    # D-079 Phase 2: an optional, pre-built canonical_claim_id -> verified
    # SemanticPreservationProof index (realization_resolver.py's
    # `build_preserved_claim_id_index`) -- structurally can only be
    # supplied by a caller that already built a Semantic Ledger
    # (AUTHORITATIVE mode's second pass; see universal_clean_cut.py's own
    # call site comment, same scoping D-076 already established for
    # `_lost_semantic_atoms`'s own `semantic_preservation_proofs`
    # parameter). None everywhere else -- current fail-closed behavior is
    # then byte-for-byte unchanged (D-079 Phase 1's own "if canonical
    # identity cannot be established, current fail-closed behavior
    # remains"). Deliberately CLAIM-scoped, never idea- or clip-scoped --
    # see that function's own docstring for why a coarse `GROUPED_SAME_
    # IDEA`/clip-level credit is structurally never enough to appear here.
    critical_claim_preservation_index: Mapping[str, object] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Per-Idea claim coverage (D-038) -- the backstop for `claim_coverage_
    best_take.py`'s own best-effort override. Unlike `_lost_semantic_atoms`
    above (which compares a discarded clip's vocabulary against the ENTIRE
    final KEEP timeline's bag of words), this compares each retry-family
    group's own critical claims directly against ONLY that idea's own
    winning realization -- so a claim can never be falsely satisfied merely
    because its words happen to also appear in a different, unrelated
    selected clip elsewhere in the video. RAW 33423953391: "cancer" /
    "thyroid" / "biopsy" recurred in unrelated clips about earlier
    screening, which incorrectly satisfied whole-video vocabulary coverage
    for a discarded diagnosis-confirmation clip -- this check's entire job
    is to not be fooled that way.

    This runs even when `claim_coverage_best_take.py` already tried to fix
    the family: that module can fail to find a safe single-candidate or
    paired-composite resolution, and this is the real, independent, always-
    on gate that must never let a critical claim reach Freeze silently
    missing regardless of whether an earlier stage could repair it.

    Returns (findings, confirmations). D-061 Phase 3: `resolve_ambiguous_
    coverage`'s ambiguous band (semantic_claims.AMBIGUOUS_COVERAGE_FLOOR <=
    coverage < COVERAGE_THRESHOLD) can only ever resolve to covered via the
    arbiter -- by that function's own construction, coverage in this band
    with no arbiter or a False/exception verdict always fails open to NOT
    covered. So every claim that lands in this band and does NOT become a
    finding was necessarily confirmed by `claim_equivalence_arbiter`; that
    evidence is recorded in `confirmations` for observability (never
    silently dropped) rather than only being visible as an absence from
    `findings`. A confidently-covered claim (coverage >= COVERAGE_THRESHOLD,
    no arbiter ever consulted) is not recorded here -- nothing was
    suppressed, there is no hidden evidence to surface.
    """
    groups = list((draft.diagnostics or {}).get("take_judge_groups") or ())
    if not groups:
        return [], []
    all_clips = {clip.clip_id: clip for clip in (*draft.selected, *draft.discarded, *draft.alternates)}
    selected_ids = {clip.clip_id for clip in draft.selected}

    findings: list[dict] = []
    confirmations: list[dict] = []
    for group in groups:
        ranked = list(group.get("ranked") or ())
        member_ids = [str(row.get("clip_id") or "") for row in ranked]
        members = [(cid, all_clips[cid]) for cid in member_ids if cid in all_clips]
        if len(members) < 2:
            continue
        winners = [cid for cid, _clip in members if cid in selected_ids]
        if not winners:
            # The whole idea vanished -- _missing_idea_coverage's job, not
            # this check's (which only judges an idea that DID survive).
            continue

        all_claims = []
        for clip_id, clip in members:
            all_claims.extend(extract_claims(clip_id, str(clip.text or ""), clause_role_arbiter=clause_role_arbiter))
        critical_claims = [c for c in dedupe_claims(tuple(all_claims)) if c.importance == CLAIM_CRITICAL]
        if not critical_claims:
            continue

        winning_realization_text = " ".join(str(all_clips[cid].text or "") for cid in winners)
        for claim in critical_claims:
            coverage = claim_coverage(claim, winning_realization_text)
            covered = resolve_ambiguous_coverage(
                claim, winning_realization_text, coverage=coverage, arbiter=claim_equivalence_arbiter,
            )
            if covered:
                if coverage < COVERAGE_THRESHOLD:
                    confirmations.append({
                        "idea_id": group.get("group_id"),
                        "claim_id": claim.claim_id,
                        "claim_type": claim.claim_type,
                        "claim_text": claim.text,
                        "importance": claim.importance,
                        "source_clip_id": claim.source_clip_id,
                        "winning_clip_ids": list(winners),
                        "coverage_against_winning_realization": round(coverage, 4),
                        "owning_authority": "BestTakeResolver",
                        "resolution": "claim_equivalence_arbiter_confirmed",
                    })
                continue
            # D-079 Phase 1/2: canonical claim identity. `claim.
            # canonical_claim_id` (semantic_claims.extract_claims, minted
            # via the SAME `mint_canonical_claim_id(claim_type,
            # content_tokens)` the Ledger's own `CanonicalClaimRecord`
            # uses -- see canonical_identity.py) is a deterministic,
            # PROPOSITION-scoped identity: a pure function of this claim's
            # own type+content, never of idea id or clip id. When the
            # Semantic Ledger independently extracted the SAME source
            # clip's SAME clause, it produces the IDENTICAL id -- this is
            # what makes the lookup below safe without any new minting
            # scheme. `_lost_critical_claims`'s own extraction here is
            # UNCHANGED; only this additional check is new.
            proof = None
            if critical_claim_preservation_index is not None:
                proof = critical_claim_preservation_index.get(claim.canonical_claim_id)
            # D-079 Phase 2's own hard requirement: consumable ONLY when
            # `proof.verified` AND the exact canonical claim id being
            # validated is genuinely a member of `proof.preserved_claim_
            # ids` (guaranteed by construction here -- that is exactly
            # how the index above was built -- re-checked explicitly for
            # defense-in-depth, never trusted implicitly). A `None` result
            # (no such id in the index at all -- GROUPED_SAME_IDEA proofs,
            # which carry no `preserved_claim_ids`, can never appear here;
            # neither can an unverified proof, filtered out when the index
            # was built) means Phase 1's own "canonical identity cannot be
            # established" escape hatch: fall through to the existing,
            # unchanged CRITICAL_CLAIM_LOST finding below.
            if (
                proof is not None
                and bool(getattr(proof, "verified", False))
                and claim.canonical_claim_id in (getattr(proof, "preserved_claim_ids", ()) or ())
            ):
                confirmations.append({
                    "idea_id": group.get("group_id"),
                    "claim_id": claim.claim_id,
                    "canonical_claim_id": claim.canonical_claim_id,
                    "claim_type": claim.claim_type,
                    "claim_text": claim.text,
                    "importance": claim.importance,
                    "source_clip_id": claim.source_clip_id,
                    "winning_clip_ids": list(winners),
                    "coverage_against_winning_realization": round(coverage, 4),
                    "owning_authority": "BestTakeResolver",
                    "resolution": "semantic_preservation_proof_consumed",
                    "claim_preservation_consumed": True,
                    "proof_method": str(getattr(proof, "proof_method", "") or ""),
                    "preserving_realization_id": getattr(proof, "preserving_id", None),
                })
                continue
            findings.append({
                "idea_id": group.get("group_id"),
                "claim_id": claim.claim_id,
                "canonical_claim_id": claim.canonical_claim_id,
                "claim_type": claim.claim_type,
                "claim_text": claim.text,
                "importance": claim.importance,
                "source_clip_id": claim.source_clip_id,
                "winning_clip_ids": list(winners),
                "coverage_against_winning_realization": round(coverage, 4),
                "owning_authority": "BestTakeResolver",
                "blocking": True,
            })
    return findings, confirmations


def apply_final_story_coherence_validation(
    draft,
    *,
    semantic_equivalence_arbiter: SemanticEquivalenceArbiter | None = None,
    semantic_atom_importance_arbiter: SemanticAtomImportanceArbiter | None = None,
    claim_equivalence_arbiter: ClaimEquivalenceArbiter | None = None,
    clause_role_arbiter: ClauseRoleArbiter | None = None,
    # D-076: an optional, pre-built map of clip_id -> SemanticPreservation
    # Proof (realization_resolver.py) -- structurally can only be supplied
    # by a caller that already built a Semantic Ledger (AUTHORITATIVE
    # mode's second pass; see universal_clean_cut.py's own call site
    # comment). None everywhere else -- current fail-closed behavior is
    # then byte-for-byte unchanged.
    semantic_preservation_proofs: Mapping[str, object] | None = None,
    # D-079 Phase 2: an optional, pre-built canonical_claim_id -> verified
    # SemanticPreservationProof index -- see `_lost_critical_claims`'s own
    # identically-named parameter docstring. None everywhere else.
    critical_claim_preservation_index: Mapping[str, object] | None = None,
):
    """Last semantic authority before Selection Freeze. See module docstring."""
    draft = _fold_alternates_into_discarded(draft)

    take_by_id = {clip.clip_id: clip for clip in (*draft.selected, *draft.discarded)}
    residual = _residual_multi_select_groups(draft, take_by_id)

    resolved_families: list[dict] = []
    unresolved_families: list[dict] = []
    discard_ids: set[str] = set()
    residual_contradictions: list[dict] = []

    for group in residual:
        to_discard, audit, contradictions = _resolve_residual_family(
            group, take_by_id, semantic_equivalence_arbiter,
        )
        residual_contradictions.extend(contradictions)
        if to_discard:
            discard_ids.update(to_discard)
            resolved_families.append({
                "group_id": group.get("group_id"),
                "discarded_clip_ids": to_discard,
                "merges": audit,
            })
        else:
            unresolved_families.append({
                "group_id": group.get("group_id"),
                "still_selected_clip_ids": [row["clip_id"] for row in group["still_selected"]],
            })

    if discard_ids:
        keep_selected = tuple(
            clip for clip in draft.selected if clip.clip_id not in discard_ids
        )
        newly_discarded = tuple(
            replace(clip, selected=False)
            for clip in draft.selected
            if clip.clip_id in discard_ids
        )
        discarded = tuple(sorted(
            (*draft.discarded, *newly_discarded),
            key=lambda clip: (clip.source_order, clip.start, clip.end, clip.clip_id),
        ))
        draft = replace(draft, selected=keep_selected, discarded=discarded)

    # Missing-story-ending observability check: the chronologically-last kept
    # take (by source_order/start across everything that survived attempt
    # reconstruction) was discarded and nothing selected follows it in that
    # source. Flag only -- never auto-restore on position evidence alone.
    all_takes = sorted(
        (*draft.selected, *draft.discarded),
        key=lambda clip: (clip.source_order, clip.start, clip.end, clip.clip_id),
    )
    possible_missing_ending = False
    if all_takes:
        selected_ids = {clip.clip_id for clip in draft.selected}
        last_by_source: dict[str, object] = {}
        for clip in all_takes:
            last_by_source[clip.source_asset_id] = clip
        for last_clip in last_by_source.values():
            if last_clip.clip_id not in selected_ids:
                possible_missing_ending = True
                break

    # Contradiction invariant and idea-coverage tracking run on the state
    # AFTER residual-family resolution above, using the take_by_id map
    # extended with anything freshly discarded by that step.
    take_by_id = {clip.clip_id: clip for clip in (*draft.selected, *draft.discarded)}
    contradiction_findings = residual_contradictions + [
        finding for finding in _contradiction_findings(draft, take_by_id)
        if finding not in residual_contradictions
    ]
    missing_idea_coverage = _missing_idea_coverage(draft)
    lost_semantic_atoms = _lost_semantic_atoms(
        draft, semantic_atom_importance_arbiter=semantic_atom_importance_arbiter,
        semantic_preservation_proofs=semantic_preservation_proofs,
    )
    lost_critical_claims, claim_coverage_confirmations = _lost_critical_claims(
        draft, claim_equivalence_arbiter=claim_equivalence_arbiter, clause_role_arbiter=clause_role_arbiter,
        critical_claim_preservation_index=critical_claim_preservation_index,
    )
    # D-031: a lost_semantic_atoms finding only blocks Freeze when its own
    # `blocking` field says so (a genuinely critical/uncertain atom, or the
    # broader content-loss signal) -- a CONTEXTUAL-only atom loss (e.g. an
    # incidental year) is recorded for observability but does not itself
    # block. contradiction_findings/missing_idea_coverage are unaffected
    # and remain unconditionally blocking, unchanged. D-038: any lost
    # critical claim always blocks -- by construction, only CRITICAL-
    # importance claims are ever recorded here at all.
    freeze_blocked = (
        bool(contradiction_findings)
        or bool(missing_idea_coverage)
        or any(row.get("blocking", True) for row in lost_semantic_atoms)
        or bool(lost_critical_claims)
    )

    diagnostics = dict(draft.diagnostics or {})
    diagnostics["final_story_coherence_validation"] = {
        "status": "applied",
        "alternates_folded_into_discard": True,
        "residual_family_count": len(residual),
        "resolved_family_count": len(resolved_families),
        "resolved_families": resolved_families,
        "unresolved_family_count": len(unresolved_families),
        "unresolved_families": unresolved_families,
        "possible_missing_story_ending": possible_missing_ending,
        "contradiction_findings": contradiction_findings,
        "missing_idea_coverage": missing_idea_coverage,
        "lost_semantic_atoms": lost_semantic_atoms,
        "lost_critical_claims": lost_critical_claims,
        # D-061 Phase 3: observability-only, additive, never wired into
        # freeze_blocked -- claims that landed in the genuinely ambiguous
        # coverage band and were confirmed covered by claim_equivalence_
        # arbiter (the only way to resolve True in that band; see
        # _lost_critical_claims's own docstring). Distinguishes REAL_
        # CONTENT_LOSS (in lost_critical_claims) from SEMANTICALLY_COVERED_
        # BY_SELECTED_REALIZATION (here) without ever silently hiding the
        # evidence that a suppression happened.
        "claim_coverage_confirmations": claim_coverage_confirmations,
        "freeze_blocked": freeze_blocked,
        "not_implemented": [
            "general_non_numeric_non_negation_contradiction_detection",
        ],
    }
    return replace(draft, diagnostics=diagnostics)
