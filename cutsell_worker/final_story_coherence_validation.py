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

from .final_sibling_grouping import _content, _negations, _numbers
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


def _residual_multi_select_groups(draft) -> list[dict]:
    selected_ids = {clip.clip_id for clip in draft.selected}
    groups = list((draft.diagnostics or {}).get("take_judge_groups") or ())
    residual = []
    for group in groups:
        ranked = list(group.get("ranked") or ())
        still_selected = [row for row in ranked if str(row.get("clip_id") or "") in selected_ids]
        if len(still_selected) >= 2:
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
        left_numbers, right_numbers = _numbers(left_text), _numbers(right_text)
        left_negations, right_negations = _negations(left_text), _negations(right_text)
        number_conflict = bool(left_numbers) and bool(right_numbers) and left_numbers != right_numbers
        negation_conflict = bool(left_negations) != bool(right_negations)
        if number_conflict or negation_conflict:
            # Confirmed same idea, but factually incompatible -- do not
            # collapse; leave both selected, unresolved, for the caller's
            # freeze-blocking contradiction check to catch.
            contradictions.append({
                "group_id": group.get("group_id"),
                "left_clip_id": ordered[i]["clip_id"],
                "right_clip_id": ordered[j]["clip_id"],
                "number_conflict": number_conflict,
                "negation_conflict": negation_conflict,
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
            left_numbers, right_numbers = _numbers(left_take.text), _numbers(right_take.text)
            left_negations, right_negations = _negations(left_take.text), _negations(right_take.text)
            number_conflict = bool(left_numbers) and bool(right_numbers) and left_numbers != right_numbers
            negation_conflict = bool(left_negations) != bool(right_negations)
            if number_conflict or negation_conflict:
                findings.append({
                    "group_id": group.get("group_id"),
                    "left_clip_id": left["clip_id"],
                    "right_clip_id": right["clip_id"],
                    "number_conflict": number_conflict,
                    "negation_conflict": negation_conflict,
                })
    return findings


def _missing_idea_coverage(draft) -> list[dict]:
    """Every take_judge_groups entry is one intended idea/retry contest.
    Flag any whose members are ALL absent from the final selected set --
    that idea vanished from the winning edit entirely."""
    selected_ids = {clip.clip_id for clip in draft.selected}
    missing: list[dict] = []
    for group in (draft.diagnostics or {}).get("take_judge_groups") or ():
        ranked = list(group.get("ranked") or ())
        member_ids = [str(row.get("clip_id") or "") for row in ranked]
        if member_ids and not any(cid in selected_ids for cid in member_ids):
            missing.append({"group_id": group.get("group_id"), "member_clip_ids": member_ids})
    return missing


def _lost_semantic_atoms(draft) -> list[dict]:
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
    Any number/negation atom (a critical fact) absent from the final KEEP
    text is flagged unconditionally. A broader loss of ordinary content
    vocabulary is flagged only once it clears both a volume and a coverage
    floor, so a genuinely redundant retry of an idea the KEEP timeline
    already covers -- which shares most of its topic vocabulary with the
    winner -- is not mistaken for real information loss.
    """
    kept_text = " ".join(str(clip.text or "") for clip in draft.selected)
    kept_content = _content(kept_text)
    kept_critical = _numbers(kept_text) | _negations(kept_text)

    findings: list[dict] = []
    for clip in draft.discarded:
        text = str(clip.text or "")
        if len(text.split()) < 3:
            # Too short to safely judge as carrying a distinct idea, a
            # standalone critical fact, or even a filler reaction ("no
            # wait", "okay stop") -- avoids flagging bare BTS scraps.
            continue
        own_content = _content(text)
        own_critical = _numbers(text) | _negations(text)
        missing_critical = sorted(own_critical - kept_critical)
        # The broader content-vocabulary check needs enough own content to
        # judge reliably; a critical atom (number/negation) is meaningful
        # evidence even in an otherwise short clip, so it is never gated by
        # this floor.
        missing_content = sorted(own_content - kept_content) if len(own_content) >= 5 else []
        coverage = 1.0 - (len(missing_content) / max(1, len(own_content) or 1))
        content_loss = len(own_content) >= 5 and len(missing_content) >= 4 and coverage < 0.45
        if missing_critical or content_loss:
            findings.append({
                "clip_id": clip.clip_id,
                "text": text,
                "missing_critical_atoms": missing_critical,
                "missing_content_token_count": len(missing_content),
                "own_content_token_count": len(own_content),
                "coverage_against_final_keep": round(coverage, 4),
            })
    return findings


def apply_final_story_coherence_validation(
    draft, *, semantic_equivalence_arbiter: SemanticEquivalenceArbiter | None = None,
):
    """Last semantic authority before Selection Freeze. See module docstring."""
    draft = _fold_alternates_into_discarded(draft)

    residual = _residual_multi_select_groups(draft)
    take_by_id = {clip.clip_id: clip for clip in (*draft.selected, *draft.discarded)}

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
    lost_semantic_atoms = _lost_semantic_atoms(draft)
    freeze_blocked = bool(contradiction_findings) or bool(missing_idea_coverage) or bool(lost_semantic_atoms)

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
        "freeze_blocked": freeze_blocked,
        "not_implemented": [
            "general_non_numeric_non_negation_contradiction_detection",
        ],
    }
    return replace(draft, diagnostics=diagnostics)
