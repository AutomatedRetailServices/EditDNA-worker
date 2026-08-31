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
    # No candidate here should trip any of the 14 restore/rescue steps --
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
