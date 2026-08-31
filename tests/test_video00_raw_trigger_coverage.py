"""D-039: the Video00 RAW workflow's push `paths:` filter must cover every
file in the active Clean Cut Core V1 canonical component map (docs/
CUTSELL_DECISIONS.md D-021, extended by D-024/025/026/027/037/038) --
otherwise a semantic- or physical-output-changing commit can land without
ever getting a controlled RAW automatically (exactly what happened to the
D-038 negation-guard fix and the D-039 grouping fix themselves: neither
touched a file the workflow was watching at the time).

This test does not re-derive the canonical map; it pins the same file list
this session's own audit produced, grouped by component to mirror D-021's
own table, so a future change to either the map or the workflow surfaces a
concrete diff here rather than silently drifting apart. Adding a new file to
an existing canonical component's implementation should add it to BOTH this
list and the workflow in the same change (see the workflow file's own
D-039 comment for the parallel grouping).
"""
from pathlib import Path

import yaml

WORKFLOW_PATH = Path(__file__).resolve().parent.parent / ".github/workflows/cutsell-video00-raw-v5-auto-microtrim.yml"

# Every file whose change can materially alter Video00's semantic or
# physical output, per D-021's canonical component map plus D-024/025/026/
# 027/037/038's additions to it. Deliberately excludes docs-only files,
# tests-only files, unrelated infrastructure, and the dormant Sales/TikTok
# Shop extension points (CanonicalEditPlan.annotations and neighbors).
REQUIRED_CANONICAL_PATHS = {
    # Entry points / harness
    "cutsell_worker/__init__.py",
    "cutsell_worker/brain_runtime.py",
    "cutsell_worker/serverless_handler.py",
    "cutsell_worker/universal_clean_cut.py",
    "cutsell_worker/universal_clean_cut_validation.py",
    "cutsell_worker/editorial_guardrails_v2.py",
    # AttemptReconstructor
    "cutsell_worker/attempt_reconstruction.py",
    # IdeaClusterer + SemanticArbiter
    "cutsell_worker/take_grouping_provider.py",
    "cutsell_worker/take_grouping.py",
    "cutsell_worker/semantic_idea_equivalence.py",
    "cutsell_worker/semantic_idea_equivalence_google.py",
    # RetryFamilyResolver (evidence)
    "cutsell_worker/final_sibling_grouping.py",
    "cutsell_worker/global_session_sibling_bridge.py",
    "cutsell_worker/session_boundaries.py",
    # DeliveryScorer
    "cutsell_worker/take_judge.py",
    # BestTakeResolver + its upstream Hybrid-vote input
    "cutsell_worker/deterministic_best_take_authority.py",
    "cutsell_worker/selection_phase_authority.py",
    "cutsell_worker/selection_conflicted_bridge_guard.py",
    # D-038: per-Idea semantic claim coverage
    "cutsell_worker/semantic_claims.py",
    "cutsell_worker/claim_coverage_best_take.py",
    "cutsell_worker/semantic_atom_importance.py",
    # Whole-video Unified Selection reasoner (rollback target)
    "cutsell_worker/unified_selection_reasoner.py",
    "cutsell_worker/unified_selection_google.py",
    # CompositeResolver's real chain of installers (D-023)
    "cutsell_worker/composite_resolver.py",
    "cutsell_worker/semantic_fragment_guard.py",
    "cutsell_worker/hybrid_story_guard.py",
    "cutsell_worker/hybrid_alternate_integrity.py",
    "cutsell_worker/hybrid_cross_group_retry_integrity.py",
    "cutsell_worker/incomplete_bridge_retry_authority.py",
    "cutsell_worker/hybrid_failed_continuation_integrity.py",
    "cutsell_worker/hybrid_retry_winner_authority.py",
    "cutsell_worker/hybrid_gold_reconciliation.py",
    "cutsell_worker/failed_prefix_completion_rescue.py",
    "cutsell_worker/final_delivery_integrity.py",
    "cutsell_worker/terminal_delivery_reconciliation.py",
    "cutsell_worker/hybrid_failed_soft_restore.py",
    "cutsell_worker/hybrid_unavailable_retry_fallback.py",
    "cutsell_worker/hybrid_complementary_delivery_guard.py",
    "cutsell_worker/hybrid_semantic_complementary_rescue.py",
    "cutsell_worker/hybrid_semantic_composite_bridge.py",
    "cutsell_worker/hybrid_composite_best_take.py",
    "cutsell_worker/hybrid_semantic_conflict_arbitration.py",
    "cutsell_worker/hybrid_session_cleanup.py",
    "cutsell_worker/hybrid_performance_retry_restore_guard.py",
    "cutsell_worker/post_selection_interior_gap_trim.py",
    "cutsell_worker/post_selection_incomplete_bridge_authority.py",
    "cutsell_worker/post_selection_complementary_family_stabilizer.py",
    "cutsell_worker/post_selection_edge_only_boundary.py",
    "cutsell_worker/semantic_best_take_integrity.py",
    # StoryValidator + CanonicalEditPlan + FinalEditReviewer + repair loop +
    # causal/story order validator
    "cutsell_worker/final_story_coherence_validation.py",
    "cutsell_worker/canonical_edit_plan.py",
    "cutsell_worker/final_edit_reviewer.py",
    "cutsell_worker/repair_loop.py",
    "cutsell_worker/causal_order_validator.py",
    # SelectionFreeze
    "cutsell_worker/selection_boundary_contract.py",
    # BoundaryEngine
    "cutsell_worker/final_boundary_authority.py",
    "cutsell_worker/human_boundary_polish_v5.py",
    "cutsell_worker/speech_visual_microtrim.py",
    "cutsell_worker/boundary_retry_tail_guard.py",
    "cutsell_worker/speech_safe_dead_air_guard.py",
    "cutsell_worker/terminal_sentence_boundary_guard.py",
    # Renderer + live render QC + physical-repair gate
    "cutsell_worker/render_plan.py",
    "cutsell_worker/render.py",
    "cutsell_worker/live_render_qc.py",
    "cutsell_worker/post_render_watch_listen_qc.py",
    # Shared identity/data model
    "cutsell_worker/contracts.py",
    "cutsell_worker/source_identity.py",
}


def _workflow_trigger_paths() -> set[str]:
    with open(WORKFLOW_PATH, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    on_key = "on" if "on" in doc else True  # PyYAML on older releases parses bare `on:` as boolean True
    return set(doc[on_key]["push"]["paths"])


def test_workflow_yaml_parses_and_targets_the_right_branch():
    with open(WORKFLOW_PATH, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    on_key = "on" if "on" in doc else True
    assert doc[on_key]["push"]["branches"] == ["cutsell/mobile-v1-clean"]
    assert "workflow_dispatch" in doc[on_key]


def test_every_canonical_active_module_is_covered_by_the_raw_trigger():
    trigger_paths = _workflow_trigger_paths()
    missing = REQUIRED_CANONICAL_PATHS - trigger_paths
    assert not missing, (
        f"These canonical Clean Cut Core V1 files can change Video00's output "
        f"but are not in the RAW workflow's push path filter: {sorted(missing)}"
    )


def test_required_canonical_paths_all_exist_on_disk():
    # Catches a typo in either this list or the workflow -- a path that
    # cannot possibly match anything is a silent, permanent trigger gap.
    repo_root = WORKFLOW_PATH.parent.parent.parent
    missing = [p for p in REQUIRED_CANONICAL_PATHS if not (repo_root / p).is_file()]
    assert missing == []


def test_trigger_set_excludes_docs_and_tests_only_paths():
    trigger_paths = _workflow_trigger_paths()
    assert not any(p.startswith("docs/") for p in trigger_paths)
    assert not any(p.startswith("tests/") for p in trigger_paths)
