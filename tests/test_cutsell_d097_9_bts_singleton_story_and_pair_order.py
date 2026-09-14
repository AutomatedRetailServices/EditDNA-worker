"""D-097.9 -- two gaps RAW 34045158712 (head c85c3ab) exposed while proving
D-097.8 on the rendered video.

R11 -- the D-097.8 R10 outcome (a corroborated lone `bts` take dropped as
`single_bts_unusable`; the run dropped "¡Vamos!", which both QA references
also remove) was recorded exactly like a D-097.B all-failed family, so
`story_completeness` became `incomplete_no_usable_realization` and the
delivery gate refused a technically clean MP4 (Freeze PASS, QC PASS on
attempt 1) as an incomplete story. Removing recording-process material is
the product working, not a missing idea: the judge row now carries the
BASIS of the no-usable outcome, the story-completeness derivation counts
only dropped IDEA families, and the StoryValidator/stage_status list the
bts singleton (never silent) without marking the story incomplete. A
dropped idea family behaves exactly as D-097.B ruled.

R12 -- `editorial_slot_resolution_install` wrapped
`take_grouping_provider._rank_candidate_pairs` at import with a
coverage-first re-order (one pair per group before any group gets a
second): a second authority over the arbiter budget order that silently
defeated the D-097.8 R9 ranking. Offline replay of the run's 49 takes: the
wrapper promoted zero-evidence neighbours (score 0.02-0.03) into the 14-pair
budget and pushed a 0.95 same-opening retry pair to 12th. The wrapper is
retired from the active path; the ranked budget is now recorded with its
scores so the order authority is visible on every run.
"""
from __future__ import annotations

from cutsell_worker import take_grouping_provider as tgp
from cutsell_worker.contracts import CandidateTake, DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.editorial_slot_resolution_install import (
    _coverage_first_pair_order,
    install_editorial_slot_resolution,
)
from cutsell_worker.final_story_coherence_validation import (
    _no_usable_realization_groups,
    apply_final_story_coherence_validation,
)
from cutsell_worker.semantic_idea_equivalence import (
    IdeaEquivalenceDecision,
    IdeaEquivalenceResult,
    SemanticEquivalenceGatePolicy,
)
from cutsell_worker.universal_clean_cut import derive_story_completeness


# ------------------------------------------------------------------ R11 rows

def _row(group_id, clip_id, basis, *, label="bts"):
    return {
        "group_id": group_id,
        "selected_clip_id": "",
        "semantic_best_take_reason": basis,
        "semantic_candidates": [{"clip_id": clip_id, "label": label, "confidence": 0.95}],
        "ranked": [{"clip_id": clip_id, "score": 0.5, "reason": "watch_listen_baseline"}],
        "no_usable_realization": True,
        "no_usable_realization_basis": basis,
        "member_usability": {clip_id: {"delete_recommended": True, "deterministic_unusable": True,
                                       "local_failure_corroborated": True, "ranker_reason": ""}},
    }


BTS_ROW = _row("tg_bts", "c_bts", "single_bts_unusable")
IDEA_ROW = _row("tg_idea", "c_idea", "no_usable_realization", label="failed")


def test_a_corroborated_bts_singleton_leaves_the_story_complete():
    story = derive_story_completeness([BTS_ROW])
    assert story["story_completeness"] == "complete"
    assert story["bts_singleton_ids"] == ["tg_bts"] and story["idea_family_ids"] == []
    assert [r["group_id"] for r in story["dropped_families"]] == ["tg_bts"]  # listed, never silent


def test_a_dropped_idea_family_still_makes_the_story_incomplete_d097_b():
    story = derive_story_completeness([IDEA_ROW])
    assert story["story_completeness"] == "incomplete_no_usable_realization"
    assert story["idea_family_ids"] == ["tg_idea"]


def test_mixed_rows_are_incomplete_because_of_the_idea_not_the_bts():
    story = derive_story_completeness([BTS_ROW, IDEA_ROW])
    assert story["story_completeness"] == "incomplete_no_usable_realization"
    assert story["idea_family_ids"] == ["tg_idea"] and story["bts_singleton_ids"] == ["tg_bts"]


def test_a_row_without_a_basis_is_read_as_a_dropped_idea():
    legacy = dict(IDEA_ROW)
    legacy.pop("no_usable_realization_basis")
    assert derive_story_completeness([legacy])["story_completeness"] == "incomplete_no_usable_realization"


def test_rows_that_found_a_winner_are_ignored():
    kept = dict(IDEA_ROW, no_usable_realization=False, selected_clip_id="c_idea")
    story = derive_story_completeness([kept])
    assert story["story_completeness"] == "complete" and story["dropped_families"] == []


def _clip(clip_id, text, *, selected, start, end):
    return DraftClip(clip_id=clip_id, source_asset_id="src", source_order=0, start=start, end=end,
                     text=text, caption_text=text, selected=selected, attempt_id=clip_id)


def test_the_story_validator_names_the_basis_and_never_blocks():
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(_clip("c_keep", "The biopsy confirmed it was papillary thyroid cancer.", selected=True, start=0.0, end=4.0),),
        alternates=(),
        discarded=(_clip("c_bts", "Let's go!", selected=False, start=4.5, end=5.3),),
        diagnostics={"take_judge_groups": [BTS_ROW]},
    )
    assert _no_usable_realization_groups(draft)[0]["basis"] == "single_bts_unusable"
    validated = apply_final_story_coherence_validation(draft)
    report = validated.diagnostics["final_story_coherence_validation"]
    row = next(f for f in report["lost_semantic_atoms"] if f["clip_id"] == "c_bts")
    assert row["kind"] == "LOST_IN_NO_USABLE_REALIZATION_FAMILY" and row["basis"] == "single_bts_unusable"
    assert row["blocking"] is False and report["freeze_blocked"] is False


# ------------------------------------------------------------ R12 pair order

def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(clip_id=clip_id, source_asset_id="src", source_order=0, start=start, end=end,
                         text=text, complete_idea=complete)


HAIR = _take("hair", 226.7, 233.2, "My hair was falling out whenever I washed it and I blamed the stress.")
ABANDONED = _take("aband", 236.2, 244.2, "I had stomach problems for a while where they did an endoscopy and diagnosed me with", complete=False)
ASIDE = _take("aside", 245.4, 251.6, "I had stomach problems one season, in 2023, no need to ask.")
CLEAN = _take("clean", 258.9, 269.4, "I had digestion problems where they did an endoscopy and said I had gastritis, nothing severe, three months of pills.")
VACCINE = _take("vacc", 276.1, 283.7, "I do not want to sound like a conspiracy but all these symptoms started after the pandemic vaccine.")


def _takes(*takes):
    return {t.clip_id: t for t in takes}


def test_the_install_no_longer_wraps_the_ranking_authority():
    install_editorial_slot_resolution()
    install_editorial_slot_resolution()
    assert not getattr(tgp._rank_candidate_pairs, "_cutsell_editorial_slot_coverage", False)
    assert tgp._rank_candidate_pairs.__name__ == "_rank_candidate_pairs"


def _score(take_map, pair):
    return tgp._pair_priority_score(take_map[pair[2]], take_map[pair[3]],
                                    gap_sec=tgp._group_gap((pair[2],), (pair[3],), take_map))


def test_first_pass_is_descending_priority_score_and_zero_evidence_pairs_never_jump_evidence_pairs():
    take_map = _takes(HAIR, ABANDONED, ASIDE, CLEAN, VACCINE)
    groups = tuple((cid,) for cid in ("hair", "aband", "aside", "clean", "vacc"))
    pairs = tgp._cross_group_candidate_pairs(groups, take_map, maximum_gap_sec=30.0)
    marked = tgp._rank_candidate_pairs_with_marks(pairs, take_map)
    first_pass = [_score(take_map, p) for p, deferred in marked if not deferred]
    deferred = [_score(take_map, p) for p, was_deferred in marked if was_deferred]
    assert first_pass == sorted(first_pass, reverse=True)
    assert deferred == sorted(deferred, reverse=True)
    assert tuple(p for p, _ in marked) == tuple(tgp._rank_candidate_pairs(pairs, take_map))
    order = [(p[2], p[3]) for p, _ in marked]
    assert order.index(("aband", "clean")) < order.index(("hair", "aside"))
    assert order.index(("aside", "clean")) < order.index(("hair", "aside"))
    # The retired wrapper put the zero-evidence neighbour ahead of shared-content pairs.
    diversified = [(p[2], p[3]) for p in _coverage_first_pair_order(tgp._rank_candidate_pairs(pairs, take_map))]
    assert diversified.index(("hair", "aside")) < diversified.index(("aside", "clean"))


def test_a_dense_cluster_cannot_starve_a_distinct_paraphrase_pair_d042_concern():
    cluster = [
        _take(f"d{i}", 250.0 + i * 5.0, 254.0 + i * 5.0,
              f"story beat number {i} about a separate unrelated subject and event")
        for i in range(8)
    ]
    late_a = _take("late_a", 295.4, 314.6, "This is my experience, I am the only one in my family with this type of cancer and only a percentage of cancers are hereditary.")
    late_b = _take("late_b", 319.4, 334.2, "I am the first in my family with this type of cancer and science backs that only a percentage of cancers are hereditary.")
    takes = (*cluster, late_a, late_b)
    take_map = _takes(*takes)
    groups = tuple((t.clip_id,) for t in takes)
    pairs = tgp._cross_group_candidate_pairs(groups, take_map, maximum_gap_sec=100.0)
    ranked = tgp._rank_candidate_pairs(pairs, take_map)
    budget = [(p[2], p[3]) for p in ranked[:14]]
    assert ("late_a", "late_b") in budget
    # ... and no cluster take was asked about more than the cap in the first pass.
    marked = tgp._rank_candidate_pairs_with_marks(pairs, take_map)
    from collections import Counter
    usage = Counter()
    for p, deferred in marked:
        if not deferred:
            usage[p[0]] += 1
            usage[p[1]] += 1
    assert max(usage.values()) <= tgp._PAIR_BUDGET_PER_GROUP_CAP


class _RecordingArbiter:
    def __init__(self):
        self.pairs_asked = []

    def check(self, request):
        self.pairs_asked.extend((p.left_text, p.right_text) for p in request.pairs)
        decisions = tuple(
            IdeaEquivalenceDecision(pair_index=i, same_idea=False, confidence=0.9, reason="test declines")
            for i, _ in enumerate(request.pairs)
        )
        return IdeaEquivalenceResult(decisions=decisions, provider="fake", model="fake", requested=True,
                                     available=True, estimated_input_tokens=10, estimated_output_tokens=5)


def test_a_two_pair_budget_is_spent_on_the_two_strongest_pairs_and_recorded():
    # D-097.12 (approved behavior change, bounded stomach-family encargo):
    # ABANDONED (incomplete) now resolves against BOTH ASIDE and CLEAN by
    # deterministic evidence (`incomplete_attempt_completed_by_retry`) before
    # the arbiter is ever asked -- ASIDE shares "stomach"/"problems" with
    # ABANDONED beyond this fixture's two-word English opening ("I had"),
    # unlike the Spanish original this rule targets, where the equivalent
    # words sit INSIDE the two-word opening itself. Both pairs are now
    # spent on deterministic evidence, not the arbiter's bounded budget, so
    # this test asserts that supersession and re-anchors its original R9
    # claim (budget spent on the two strongest REMAINING pairs, in
    # descending score, no zero-evidence neighbour) on the two pairs that
    # still reach the arbiter.
    install_editorial_slot_resolution()
    takes = (HAIR, ABANDONED, ASIDE, CLEAN, VACCINE)
    groups = tuple((t.clip_id,) for t in takes)
    arbiter = _RecordingArbiter()
    _groups, diag = tgp.reconcile_semantic_idea_equivalence(
        groups, takes, arbiter, policy=SemanticEquivalenceGatePolicy(max_pairs_per_request=2),
    )
    restart_merged = {frozenset((m["left_clip_id"], m["right_clip_id"])) for m in diag.get("restart_evidence_merges") or ()}
    assert frozenset(("aband", "clean")) in restart_merged
    assert frozenset(("aband", "aside")) in restart_merged
    assert diag["checked_pair_count"] == 2
    assert diag["pair_order_authority"] == tgp._PAIR_ORDER_AUTHORITY
    budget = diag["ranked_pair_budget"]
    assert [row["priority_score"] for row in budget] == sorted(row["priority_score"] for row in budget)[::-1]
    assert all(row["group_cap_deferred"] is False for row in budget)
    asked = {frozenset(pair) for pair in arbiter.pairs_asked}
    assert frozenset((ASIDE.text, CLEAN.text)) in asked  # the two strongest pairs left after determinism
    assert frozenset((HAIR.text, ABANDONED.text)) in asked
    assert frozenset((HAIR.text, ASIDE.text)) not in asked  # a zero-evidence neighbour never spends a slot
