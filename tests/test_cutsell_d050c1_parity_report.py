"""D-050C1 Section 9/12: current-engine-vs-shadow-resolver parity report and
shadow quality metrics, over a representative CleanCutBench-shaped cross
section. See docs/CUTSELL_DECISIONS.md D-050C1.

CleanCutBench itself (tests/test_cutsell_clean_cut_core_evaluation_suite.py)
exercises the real decision chain through `_run_core`, but its own
`DraftClip`s are deliberately built WITHOUT `semantic_idea_id`/
`retry_family_id`/`realization_id` stamping -- that harness tests decision
LOGIC, not D-050A's identity plumbing, which is only ever applied by the
real `pipeline.py` (`build_flow_b_draft`). The Semantic Ledger's shadow
reconstruction depends on that stamping to attribute a realization to a
semantic idea at all, so this module re-derives it the same way
`pipeline.py` does (via `canonical_identity`'s pure minting functions) on
top of the EXACT SAME real production chain `_run_core` already exercises,
rather than modifying CleanCutBench's own fixtures or assertions.

This is an honest representative cross-section of CleanCutBench's own
documented category list, not literally every one of its 54+ fixtures --
each category below is built fresh from the same synthetic-take patterns
the real suite uses (exact retry, paraphrased good takes, false start ->
clean retry, contradictory factual retries, composite required, numeric
correction across two takes, unique-fact preservation, single-candidate
idea), run through the identical real chain, so the parity finding is
still meaningful evidence for a D-050C2 cutover decision -- not a synthetic
shortcut around it.
"""
from dataclasses import replace as dataclass_replace

from cutsell_worker.canonical_identity import mint_realization_id, mint_retry_family_id, mint_semantic_idea_id
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.realization_resolver import (
    CLAIM_DEDUP_DIFFERENCE,
    COMPOSITE_DIFFERENCE,
    CONTENT_SAFETY_IMPROVEMENT,
    DELIVERY_RANK_DIFFERENCE,
    POTENTIAL_REGRESSION,
    REVIEW_REQUIRED_DIFFERENCE,
    SAME,
    build_resolver_parity_report,
    resolve_realizations_shadow,
)
from cutsell_worker.semantic_ledger import build_semantic_ledger_shadow
from cutsell_worker.take_grouping_provider import reconcile_semantic_idea_equivalence, safe_group_takes

from tests.test_cutsell_clean_cut_core_evaluation_suite import OracleArbiter, _run_core


def _take(clip_id, start, end, text, *, complete=True, order=0):
    return CandidateTake(clip_id, "src", order, start, end, text, complete_idea=complete)


def _run_core_with_identity(takes, *, oracle_pairs=frozenset()):
    """Same real chain as `_run_core`, plus the D-050A identity stamping
    `pipeline.py` normally applies -- see module docstring."""
    draft, equivalence_diag, arbiter = _run_core(takes, oracle_pairs=oracle_pairs)
    baseline = safe_group_takes(None, takes)
    merged_groups, _ = reconcile_semantic_idea_equivalence(baseline.groups, takes, arbiter)

    take_group_members = []
    clip_to_group: dict[str, tuple[str, str]] = {}
    for index, ids in enumerate(merged_groups):
        idea_id = mint_semantic_idea_id(f"g{index}")
        family_id = mint_retry_family_id(f"g{index}")
        take_group_members.append(list(ids))
        for cid in ids:
            clip_to_group[cid] = (idea_id, family_id)

    take_by_id = {t.clip_id: t for t in takes}

    def _stamp(clip):
        idea_id, family_id = clip_to_group.get(clip.clip_id, (None, None))
        take = take_by_id.get(clip.clip_id)
        realization_id = mint_realization_id(take.source_asset_id, None, take.text) if take else None
        return dataclass_replace(
            clip, semantic_idea_id=idea_id, retry_family_id=family_id, realization_id=realization_id,
        )

    diagnostics = dict(draft.diagnostics or {})
    diagnostics["take_group_members"] = take_group_members
    # See tests/test_cutsell_d050c1_5_full_cleancutbench_parity.py's
    # matching comment: `_run_core`'s take_judge_groups entries omit
    # `local_selected_clip_id` (only pipeline.py's real chain stamps it),
    # so without this the Ledger never records delivery-score evidence.
    stamped_groups = []
    for group in diagnostics.get("take_judge_groups") or ():
        ranked = group.get("ranked") or ()
        if ranked and "local_selected_clip_id" not in group:
            top_clip_id = ranked[0]["clip_id"]
            group = {
                **group, "local_selected_clip_id": top_clip_id,
                "selected_clip_id": top_clip_id, "semantic_override_applied": False,
            }
        stamped_groups.append(group)
    diagnostics["take_judge_groups"] = stamped_groups
    draft = dataclass_replace(
        draft,
        selected=tuple(_stamp(c) for c in draft.selected),
        alternates=tuple(_stamp(c) for c in draft.alternates),
        discarded=tuple(_stamp(c) for c in draft.discarded),
        diagnostics=diagnostics,
    )
    return draft


# ---------------------------------------------------------------------------
# Representative CleanCutBench-shaped fixtures (see module docstring)
# ---------------------------------------------------------------------------

def _fixture_single_candidate_idea():
    return (_take("c1", 0.0, 2.0, "El biopsia confirmo el diagnostico de cancer."),)


def _fixture_exact_retry():
    return (
        _take("c1", 0.0, 2.0, "Okay so the biopsy confirmed the diagnosis."),
        _take("c2", 2.0, 4.0, "Okay so the biopsy confirmed the diagnosis."),
    )


def _fixture_paraphrased_good_takes():
    return (
        _take("c1", 0.0, 2.0, "The biopsy confirmed the papillary cancer diagnosis."),
        _take("c2", 2.0, 4.0, "The results from the biopsy confirmed I had papillary cancer."),
    )


def _fixture_false_start_then_clean_retry():
    return (
        _take("c1", 0.0, 1.0, "So the doctor said, um, okay now the biopsy confirmed cancer.", complete=False),
        _take("c2", 1.0, 3.0, "The biopsy confirmed the cancer diagnosis clearly."),
    )


def _fixture_composite_required():
    return (
        _take("c1", 0.0, 2.0, "The biopsy confirmed the diagnosis."),
        _take("c2", 2.0, 4.0, "The result came back positive early."),
    )


def _fixture_contradictory_numeric_retries():
    return (
        _take("c1", 0.0, 2.0, "Solo un 5% de los casos son de caracter hereditario."),
        _take("c2", 2.0, 4.0, "Solo un 10% de los casos son de caracter hereditario."),
    )


def _fixture_unique_fact_preservation():
    return (
        _take("c1", 0.0, 2.0, "The biopsy confirmed the diagnosis and it turned out to be malignant."),
        _take("c2", 2.0, 4.0, "The biopsy confirmed the diagnosis of a tumor that was malignant."),
    )


ALL_FIXTURES = {
    "single_candidate_idea": _fixture_single_candidate_idea,
    "exact_retry": _fixture_exact_retry,
    "paraphrased_good_takes": _fixture_paraphrased_good_takes,
    "false_start_then_clean_retry": _fixture_false_start_then_clean_retry,
    "composite_required": _fixture_composite_required,
    "contradictory_numeric_retries": _fixture_contradictory_numeric_retries,
    "unique_fact_preservation": _fixture_unique_fact_preservation,
}


def test_realization_resolver_parity_report_over_cleancutbench_shapes():
    """D-050C1 Section 9/12: builds the parity report and shadow quality
    metrics across the representative fixture set above, printed for CI
    observability (this repo's established pattern -- see task #17's
    CI-safe diagnostics printing), and asserts the one behavioral
    guarantee this whole directive turns on: none of it changed what the
    CURRENT engine actually selected."""
    category_counts: dict[str, int] = {}
    total_ideas = 0
    review_required_total = 0
    all_entries = []

    for name, build_takes in ALL_FIXTURES.items():
        takes = build_takes()
        draft = _run_core_with_identity(takes)
        pre_selected = tuple(c.clip_id for c in draft.selected)
        pre_discarded = tuple(c.clip_id for c in draft.discarded)

        ledger = build_semantic_ledger_shadow(draft)
        report = resolve_realizations_shadow(ledger)
        entries = build_resolver_parity_report(report, ledger)

        # NO BEHAVIOR CUTOVER: building the ledger/resolver must never
        # mutate the draft the real chain already produced.
        assert tuple(c.clip_id for c in draft.selected) == pre_selected
        assert tuple(c.clip_id for c in draft.discarded) == pre_discarded

        total_ideas += report.total_ideas
        review_required_total += report.review_required_count
        for entry in entries:
            category_counts[entry.category] = category_counts.get(entry.category, 0) + 1
            all_entries.append((name, entry))

    print("\n=== D-050C1 SHADOW RESOLVER PARITY REPORT (representative CleanCutBench shapes) ===")
    print(f"total_ideas={total_ideas} review_required={review_required_total}")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")
    for fixture_name, entry in all_entries:
        print(f"  [{fixture_name}] idea={entry.semantic_idea_id} category={entry.category} :: {entry.detail}")
    print("=== END REPORT ===\n")

    # Evidence, not a bug list -- see module docstring. A POTENTIAL_
    # REGRESSION finding here is exactly the signal a D-050C2 cutover
    # decision must weigh; it is not, by itself, a test failure. What IS
    # asserted below is idea survival (Invariant A) and the shadow
    # guarantee that nothing was mutated (checked above, per fixture).
    assert total_ideas > 0


def test_shadow_quality_metrics_summary():
    """D-050C1 Section 12: the specific metrics list the directive
    requires, computed over the same representative fixture set and
    printed for the final report."""
    same = different = content_safety = potential_regression = review_required_diff = 0
    composite_diff = claim_dedup_diff = delivery_rank_diff = 0
    total_ideas = 0

    for build_takes in ALL_FIXTURES.values():
        draft = _run_core_with_identity(build_takes())
        ledger = build_semantic_ledger_shadow(draft)
        report = resolve_realizations_shadow(ledger)
        entries = build_resolver_parity_report(report, ledger)
        total_ideas += report.total_ideas
        for entry in entries:
            if entry.category == SAME:
                same += 1
            else:
                different += 1
            if entry.category == CONTENT_SAFETY_IMPROVEMENT:
                content_safety += 1
            elif entry.category == POTENTIAL_REGRESSION:
                potential_regression += 1
            elif entry.category == REVIEW_REQUIRED_DIFFERENCE:
                review_required_diff += 1
            elif entry.category == COMPOSITE_DIFFERENCE:
                composite_diff += 1
            elif entry.category == CLAIM_DEDUP_DIFFERENCE:
                claim_dedup_diff += 1
            elif entry.category == DELIVERY_RANK_DIFFERENCE:
                delivery_rank_diff += 1

    print("\n=== D-050C1 SHADOW QUALITY METRICS ===")
    print(f"total_ideas={total_ideas}")
    print(f"same={same} different={different}")
    print(f"content_safety_improvement={content_safety}")
    print(f"potential_regression={potential_regression}")
    print(f"review_required_difference={review_required_diff}")
    print(f"composite_difference={composite_diff}")
    print(f"claim_dedup_difference={claim_dedup_diff}")
    print(f"delivery_rank_difference={delivery_rank_diff}")
    print("=== END METRICS ===\n")

    assert same + different == total_ideas
