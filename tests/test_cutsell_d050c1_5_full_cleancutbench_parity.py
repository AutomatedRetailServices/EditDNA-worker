"""D-050C1.5: FULL current-engine-vs-shadow-resolver parity across EVERY
CleanCutBench fixture (not a representative subset). See
docs/CUTSELL_DECISIONS.md D-050C1 and D-050C1.5.

HOW THIS RUNS EVERY FIXTURE WITHOUT MODIFYING THE CANONICAL SUITE:
`tests/test_cutsell_clean_cut_core_evaluation_suite.py`'s 54 `test_*`
functions are ordinary no-argument functions that each build their own
synthetic takes inline and call the module-private `_run_core` exactly
once (a few call it twice). Rather than duplicating any fixture's take
construction here -- a real transcription-error risk across 54+ cases --
this module monkeypatches `_run_core` for the duration of one pass over
every real test function, capturing each call's `(takes, oracle_pairs,
claim_equivalence_arbiter, draft, arbiter)` under that test's own name,
then restores the original. Every captured fixture ran through the
REAL, unmodified production chain with the REAL suite's own assertions
still executing (so a broken fixture surfaces as a normal pytest failure
here too, not just silently skipped).

`_run_core`'s own `DraftClip`s are never stamped with `semantic_idea_id`/
`retry_family_id`/`realization_id` (that harness tests decision LOGIC, not
D-050A's identity plumbing -- see test_cutsell_d050c1_parity_report.py's
own module docstring for the same note), so `_stamp_identity` re-derives
it the same way `pipeline.py` does, from the SAME captured takes/arbiter,
and cross-checks that stamping changed nothing about which clip_ids ended
up selected/discarded before ever building a Ledger from it.
"""
from dataclasses import replace as dataclass_replace

import tests.test_cutsell_clean_cut_core_evaluation_suite as bench
from cutsell_worker.canonical_identity import mint_realization_id, mint_retry_family_id, mint_semantic_idea_id
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

_ALL_CATEGORIES = (
    SAME, CONTENT_SAFETY_IMPROVEMENT, CLAIM_DEDUP_DIFFERENCE, COMPOSITE_DIFFERENCE,
    DELIVERY_RANK_DIFFERENCE, REVIEW_REQUIRED_DIFFERENCE, POTENTIAL_REGRESSION,
)


def _stamp_identity(draft, takes, arbiter):
    """D-050A identity stamping, re-derived from the same real grouping
    call `_run_core` itself made -- see module docstring."""
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
            complete_idea=take.complete_idea if take else None,
        )

    diagnostics = dict(draft.diagnostics or {})
    diagnostics["take_group_members"] = take_group_members
    # `_run_core`'s take_judge_groups entries carry only {"group_id",
    # "ranked"} -- `local_selected_clip_id`/`selected_clip_id` are stamped
    # by the REAL production pipeline.py (`local_selected_clip_id =
    # ranked[0].clip_id`, see pipeline.py:252), not by this cheaper test
    # harness. Without it, `build_semantic_ledger_shadow` never records a
    # DELIVERY_SCORE_WINNER decision at all, so the resolver's delivery-
    # evidence tie-break silently sees no evidence for every fixture in
    # this sweep -- reproduce the same stamp here so this parity run
    # actually exercises that path the way production does.
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
    return dataclass_replace(
        draft,
        selected=tuple(_stamp(c) for c in draft.selected),
        alternates=tuple(_stamp(c) for c in draft.alternates),
        discarded=tuple(_stamp(c) for c in draft.discarded),
        diagnostics=diagnostics,
    )


def _collect_every_fixture_call():
    """Runs EVERY CleanCutBench `test_*` function for real (their own
    assertions still execute -- a broken fixture fails here exactly as it
    would under plain pytest), capturing every `_run_core` call it makes.
    Zero duplication of fixture construction; zero modification of the
    canonical suite file (monkeypatch is installed and torn down within
    this one function)."""
    captured: list[dict] = []
    original_run_core = bench._run_core

    def _capturing_run_core(takes, *, oracle_pairs=frozenset(), claim_equivalence_arbiter=None):
        draft, equivalence_diag, arbiter = original_run_core(
            takes, oracle_pairs=oracle_pairs, claim_equivalence_arbiter=claim_equivalence_arbiter,
        )
        captured.append({"takes": takes, "draft": draft, "arbiter": arbiter, "fixture_id": None})
        return draft, equivalence_diag, arbiter

    fixture_names = sorted(name for name in dir(bench) if name.startswith("test_"))
    bench._run_core = _capturing_run_core
    try:
        for name in fixture_names:
            before = len(captured)
            getattr(bench, name)()
            for entry in captured[before:]:
                entry["fixture_id"] = name
    finally:
        bench._run_core = original_run_core
    return captured, fixture_names


def _run_full_parity():
    captured, fixture_names = _collect_every_fixture_call()
    rows: list[tuple[str, str, str, str]] = []
    counts = {category: 0 for category in _ALL_CATEGORIES}
    total_ideas = 0
    mutation_mismatches: list[str] = []
    review_required_orphans = 0

    for entry in captured:
        takes = entry["takes"]
        arbiter = entry["arbiter"]
        original_draft = entry["draft"]
        stamped = _stamp_identity(original_draft, takes, arbiter)

        # Section 7 proof, per-fixture: stamping identity changed nothing
        # about which clip_ids the REAL chain selected/discarded.
        if {c.clip_id for c in original_draft.selected} != {c.clip_id for c in stamped.selected}:
            mutation_mismatches.append(f"{entry['fixture_id']}: selected set changed by identity stamping")
        if {c.clip_id for c in original_draft.discarded} != {c.clip_id for c in stamped.discarded}:
            mutation_mismatches.append(f"{entry['fixture_id']}: discarded set changed by identity stamping")

        ledger = build_semantic_ledger_shadow(stamped)
        report = resolve_realizations_shadow(ledger)
        review_required_orphans += sum(1 for o in report.orphan_reviews if o.verdict == "REVIEW_REQUIRED")
        total_ideas += report.total_ideas

        for parity_entry in build_resolver_parity_report(report, ledger):
            counts[parity_entry.category] += 1
            rows.append((entry["fixture_id"], parity_entry.semantic_idea_id, parity_entry.category, parity_entry.detail))

    return {
        "rows": rows, "counts": counts, "total_ideas": total_ideas,
        "fixture_names": fixture_names, "fixture_count": len(fixture_names),
        "mutation_mismatches": mutation_mismatches,
        "review_required_orphans": review_required_orphans,
    }


def test_full_cleancutbench_shadow_parity():
    result = _run_full_parity()
    rows = result["rows"]
    counts = result["counts"]
    total_ideas = result["total_ideas"]

    print("\n=== D-050C1.5 FULL CLEANCUTBENCH SHADOW PARITY ===")
    print(f"total_fixtures={result['fixture_count']} total_semantic_ideas={total_ideas}")
    print(f"total_parity_rows={len(rows)}")  # may exceed total_ideas across a couple of multi-call fixtures
    for category in _ALL_CATEGORIES:
        count = counts[category]
        pct = (100.0 * count / len(rows)) if rows else 0.0
        print(f"  {category}: {count} ({pct:.1f}%)")
    print(f"orphan_reviews_review_required={result['review_required_orphans']}")
    print("--- per-row detail ---")
    for fixture_id, idea_id, category, detail in rows:
        print(f"  [{fixture_id}] idea={idea_id} category={category} :: {detail}")
    print("=== END FULL PARITY REPORT ===\n")

    # Section 7 (no behavior change): identity stamping must never alter
    # which clip_ids the real chain selected/discarded, for ANY fixture.
    assert result["mutation_mismatches"] == []
    assert total_ideas > 0
    assert result["fixture_count"] == 54
