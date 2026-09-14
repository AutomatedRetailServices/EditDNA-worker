"""composite_resolver.py -- CompositeResolver's single, directly-callable
entry point (D-023). See that module's own docstring for the full ordered
chain and why it replaces 14 separate import-time monkeypatch installers.

These tests target composite_resolver.py directly, in addition to (not
instead of) each contributing module's own existing tests, which already
cover each step's matching/threshold logic in isolation and are unchanged
by this consolidation (proven by the full suite staying green at the same
pass count before and after). What is unique to test here is the
COMPOSITION itself: that steps chain in the documented order with the
right data threaded between them -- especially cross-module diagnostics
coupling (hybrid_failed_soft_restore reads hybrid_cross_group_retry_
integrity's diagnostics entry by name) that a hand-transcription error in
the composition would most likely break.
"""
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.composite_resolver import (
    apply_composite_family_stabilization,
    apply_composite_group_split,
    apply_composite_resolution,
)
from cutsell_worker.hybrid_editorial import EditorialDecision, EditorialJudgeResult
from cutsell_worker.hybrid_session_cleanup import HybridSessionCleanupResult
from cutsell_worker.session_boundaries import TakeGroupingProviderResult
from cutsell_worker.providers import ProviderStatus


def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id, source_asset_id="src", source_order=0,
        start=start, end=end, text=text, complete_idea=complete,
    )


class _MappingJudge:
    """Same shape as tests/test_cutsell_hybrid_session_cleanup.py's MappingJudge."""

    def __init__(self, labels):
        self.labels = labels

    def judge(self, session):
        return EditorialJudgeResult(
            decisions=tuple(
                EditorialDecision(candidate.clip_id, *self.labels[candidate.clip_id], "test")
                for candidate in session.candidates
            ),
            provider="fake", model="flash-lite", requested=True, available=True,
            estimated_input_tokens=100, estimated_output_tokens=50,
        )


def test_base_step_alone_reaches_final_conflict_arbitration_unchanged():
    # No candidate here should trip any of the 19 restore/rescue steps --
    # this proves the chain composes without raising and returns the base
    # step's own decision when nothing downstream has evidence to act on.
    takes = (
        _take("a", 0.0, 3.0, "the product arrived in great condition"),
        _take("b", 4.0, 7.0, "the packaging was also very sturdy"),
    )
    judge = _MappingJudge({"a": ("keep", 0.90), "b": ("keep", 0.90)})

    result, split_ids = apply_composite_resolution(takes, None, judge)

    assert {t.clip_id for t in result.kept} == {"a", "b"}
    assert result.deleted == ()
    assert split_ids == frozenset()


def test_cross_group_retry_integrity_feeds_failed_soft_restore_by_diagnostics_name():
    # hybrid_failed_soft_restore (step 9) reads the
    # "hybrid_cross_group_retry_integrity" diagnostics entry (step 5) BY
    # NAME to decide what to un-delete. This is the exact cross-module
    # coupling a hand-transcription error in the composition would break
    # silently (wrong order, or step 5 renamed/omitted) -- so this pins it
    # explicitly rather than relying only on each module's own isolated test.
    #
    # "weak" is a complete take whose ONLY authoritative peer ("authoritative")
    # covers it, so step 5 deletes it as a cross-group-covered retry with
    # confidence 0.80 (< the 0.90 floor step 9 restores below).
    weak = _take("weak", 0.0, 3.0, "so basically this cream works great for dry skin overall")
    authoritative = _take(
        "authoritative", 4.0, 7.0,
        "so basically this cream works great for dry skin overall and also reduces redness",
    )
    judge = _MappingJudge({
        "weak": ("failed", 0.80),
        "authoritative": ("winner", 0.95),
    })

    result, _ = apply_composite_resolution((weak, authoritative), None, judge)

    # Restored by hybrid_failed_soft_restore because the cross-group delete's
    # own recorded confidence (0.80) is below its 0.90 destructive-authority
    # floor -- proving both step 5 and step 9 ran, in that order, and step 9
    # could see step 5's diagnostics.
    kept_ids = {t.clip_id for t in result.kept}
    assert "weak" in kept_ids
    diag_names = [
        key
        for row in result.diagnostics
        if isinstance(row, dict)
        for key in row
    ]
    assert "hybrid_cross_group_retry_integrity" in diag_names
    assert "hybrid_failed_soft_restore" in diag_names


def test_apply_composite_group_split_forces_singleton_groups():
    grouping = TakeGroupingProviderResult(
        groups=(("a", "b", "c"),), status=ProviderStatus("test", True, True, "applied"), reason="",
    )
    takes = (_take("a", 0.0, 1.0, "x"), _take("b", 1.0, 2.0, "y"), _take("c", 2.0, 3.0, "z"))

    out = apply_composite_group_split(grouping, takes, frozenset({"a", "c"}))

    # Sorted by each resulting group's earliest natural clip position.
    assert out.groups == (("a",), ("b",), ("c",))
    assert "composite_resolver_group_split:2" in out.reason


def test_apply_composite_group_split_is_a_noop_without_split_ids():
    grouping = TakeGroupingProviderResult(
        groups=(("a", "b"),), status=ProviderStatus("test", True, True, "applied"), reason="",
    )
    takes = (_take("a", 0.0, 1.0, "x"), _take("b", 1.0, 2.0, "y"))

    out = apply_composite_group_split(grouping, takes, frozenset())

    assert out is grouping


def test_stray_legacy_install_call_after_chain_is_built_cannot_affect_it():
    # "Legacy downstream hook cannot mutate final membership": once
    # composite_resolver's chain has been built (which happens lazily, on
    # first use -- see _get_take_level_chain), it is a private, cached
    # reference. Even if some future code accidentally calls one of the 19
    # consolidated hooks' own install_*() again afterward (they still
    # exist, unchanged, for their own monkeypatch-based tests -- see
    # D-023), that stray call can only rewrap the CURRENT global
    # hybrid_session_cleanup.apply_hybrid_session_cleanup attribute; it
    # cannot reach composite_resolver's already-cached chain.
    from cutsell_worker import composite_resolver as cr
    from cutsell_worker import hybrid_session_cleanup, session_boundaries
    from cutsell_worker.hybrid_composite_best_take import install_hybrid_composite_best_take

    takes = (
        _take("a", 0.0, 3.0, "the product arrived in great condition"),
        _take("b", 4.0, 7.0, "the packaging was also very sturdy"),
    )
    judge = _MappingJudge({"a": ("keep", 0.90), "b": ("keep", 0.90)})

    # Warm the cache first (idempotent if some earlier test already did).
    before, _ = apply_composite_resolution(takes, None, judge)
    cached_chain = cr._take_level_chain
    assert cached_chain is not None

    original_cleanup = hybrid_session_cleanup.apply_hybrid_session_cleanup
    original_grouping = session_boundaries.safe_group_takes_by_sessions
    try:
        install_hybrid_composite_best_take()
        # The stray install DID monkeypatch the global attributes...
        assert hybrid_session_cleanup.apply_hybrid_session_cleanup is not original_cleanup

        # ...but composite_resolver's cached chain reference is untouched,
        # and apply_composite_resolution's behavior is unaffected.
        assert cr._take_level_chain is cached_chain
        after, _ = apply_composite_resolution(takes, None, judge)
        assert {t.clip_id for t in after.kept} == {t.clip_id for t in before.kept}
    finally:
        hybrid_session_cleanup.apply_hybrid_session_cleanup = original_cleanup
        session_boundaries.safe_group_takes_by_sessions = original_grouping


def test_apply_composite_family_stabilization_delegates_and_is_a_noop_without_swaps():
    from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION

    clip = DraftClip(
        clip_id="a", source_asset_id="src", source_order=0, start=0.0, end=1.0,
        text="hello", caption_text="hello", selected=True,
    )
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(clip,), alternates=(), discarded=(), diagnostics={},
    )

    out = apply_composite_family_stabilization(draft)

    assert out is draft


# ---------------------------------------------------------------------------
# D-053 Section 11 / Section 12 category 10: "SemanticComputePlan
# diagnostics survive wrapper chain" -- several of the 19 chain hooks
# reconstruct HybridSessionCleanupResult via a keyword constructor call that
# never named semantic_compute_plan, silently reverting it to None the
# moment such a hook fires (discovered live in the D-052 stability battery).
# The fix is the hybrid_session_cleanup._LAST_SEMANTIC_COMPUTE_PLAN
# ContextVar, read-and-cleared by composite_resolver._restore_semantic_
# compute_plan -- mirrors _composite_split_ids's own established pattern.
# ---------------------------------------------------------------------------

def test_semantic_compute_plan_survives_the_full_19_hook_chain(monkeypatch):
    monkeypatch.setenv("CUTSELL_SEMANTIC_COMPUTE_PLANNER", "1")
    # Same fixture as the cross-group-retry-integrity test above -- proven
    # to actually route through multiple chain hooks (steps 5 and 9), at
    # least one of which reconstructs HybridSessionCleanupResult via a
    # keyword constructor call that predates the semantic_compute_plan
    # field and would otherwise silently drop it.
    weak = _take("weak", 0.0, 3.0, "so basically this cream works great for dry skin overall")
    authoritative = _take(
        "authoritative", 4.0, 7.0,
        "so basically this cream works great for dry skin overall and also reduces redness",
    )
    judge = _MappingJudge({
        "weak": ("failed", 0.80),
        "authoritative": ("winner", 0.95),
    })

    result, _ = apply_composite_resolution((weak, authoritative), None, judge)

    assert result.semantic_compute_plan is not None
    diag_names = [
        key
        for row in result.diagnostics
        if isinstance(row, dict)
        for key in row
    ]
    # Confirms the chain actually ran through the multi-hook path this test
    # depends on, not merely the base step alone.
    assert "hybrid_cross_group_retry_integrity" in diag_names
    assert "hybrid_failed_soft_restore" in diag_names


def test_semantic_compute_plan_absent_when_flag_off_even_through_the_chain(monkeypatch):
    monkeypatch.delenv("CUTSELL_SEMANTIC_COMPUTE_PLANNER", raising=False)
    takes = (
        _take("a", 0.0, 3.0, "the product arrived in great condition"),
        _take("b", 4.0, 7.0, "the packaging was also very sturdy"),
    )
    judge = _MappingJudge({"a": ("keep", 0.90), "b": ("keep", 0.90)})

    result, _ = apply_composite_resolution(takes, None, judge)

    assert result.semantic_compute_plan is None


def test_restore_semantic_compute_plan_reattaches_a_dropped_plan():
    # Direct unit test of the fix itself: simulate a hook that reconstructed
    # the result via a keyword call that dropped semantic_compute_plan
    # (result.semantic_compute_plan is None) while the ContextVar still
    # holds the plan the base call actually produced.
    from cutsell_worker import hybrid_session_cleanup
    from cutsell_worker.composite_resolver import _restore_semantic_compute_plan
    from cutsell_worker.semantic_compute_planner import SemanticComputePlan

    fake_plan = SemanticComputePlan(
        work_items=(), planned_calls=(), deferred_optional_calls=(),
        planned_cost_usd=0.0, cost_ceiling_usd=0.0075,
        required_p0_cost_usd=0.0, required_p1_cost_usd=0.0, optional_p2_cost_usd=0.0,
    )
    token = hybrid_session_cleanup._LAST_SEMANTIC_COMPUTE_PLAN.set(fake_plan)
    try:
        dropped = HybridSessionCleanupResult(
            kept=(), deleted=(), requested_chunk_count=0, available_chunk_count=0,
            diagnostics=(), semantic_decisions=(),
        )
        restored = _restore_semantic_compute_plan(dropped)
        assert restored.semantic_compute_plan is fake_plan
        # Never touches kept/deleted/diagnostics/semantic_decisions.
        assert restored.kept == dropped.kept
        assert restored.deleted == dropped.deleted
        assert restored.diagnostics == dropped.diagnostics
        assert restored.semantic_decisions == dropped.semantic_decisions
    finally:
        hybrid_session_cleanup._LAST_SEMANTIC_COMPUTE_PLAN.reset(token)


def test_restore_semantic_compute_plan_never_overwrites_a_plan_the_hook_kept():
    from cutsell_worker import hybrid_session_cleanup
    from cutsell_worker.composite_resolver import _restore_semantic_compute_plan
    from cutsell_worker.semantic_compute_planner import SemanticComputePlan

    contextvar_plan = SemanticComputePlan(
        work_items=(), planned_calls=(), deferred_optional_calls=(),
        planned_cost_usd=0.0, cost_ceiling_usd=0.0075,
        required_p0_cost_usd=0.0, required_p1_cost_usd=0.0, optional_p2_cost_usd=0.0,
    )
    kept_plan = SemanticComputePlan(
        work_items=(), planned_calls=(), deferred_optional_calls=(),
        planned_cost_usd=0.005, cost_ceiling_usd=0.0075,
        required_p0_cost_usd=0.0, required_p1_cost_usd=0.0, optional_p2_cost_usd=0.0,
    )
    token = hybrid_session_cleanup._LAST_SEMANTIC_COMPUTE_PLAN.set(contextvar_plan)
    try:
        result_with_plan = HybridSessionCleanupResult(
            kept=(), deleted=(), requested_chunk_count=0, available_chunk_count=0,
            diagnostics=(), semantic_decisions=(), semantic_compute_plan=kept_plan,
        )
        restored = _restore_semantic_compute_plan(result_with_plan)
        assert restored.semantic_compute_plan is kept_plan
        assert restored is result_with_plan
    finally:
        hybrid_session_cleanup._LAST_SEMANTIC_COMPUTE_PLAN.reset(token)


def test_restore_semantic_compute_plan_clears_the_contextvar_after_reading():
    # Never leaks a stale plan into a later, unrelated call -- same
    # discipline _composite_split_ids already applies to its two ContextVars.
    from cutsell_worker import hybrid_session_cleanup
    from cutsell_worker.composite_resolver import _restore_semantic_compute_plan
    from cutsell_worker.semantic_compute_planner import SemanticComputePlan

    fake_plan = SemanticComputePlan(
        work_items=(), planned_calls=(), deferred_optional_calls=(),
        planned_cost_usd=0.0, cost_ceiling_usd=0.0075,
        required_p0_cost_usd=0.0, required_p1_cost_usd=0.0, optional_p2_cost_usd=0.0,
    )
    hybrid_session_cleanup._LAST_SEMANTIC_COMPUTE_PLAN.set(fake_plan)
    dropped = HybridSessionCleanupResult(
        kept=(), deleted=(), requested_chunk_count=0, available_chunk_count=0,
        diagnostics=(), semantic_decisions=(),
    )
    _restore_semantic_compute_plan(dropped)
    assert hybrid_session_cleanup._LAST_SEMANTIC_COMPUTE_PLAN.get() is None
